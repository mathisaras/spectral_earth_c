import copy
import math
import os
from collections.abc import Sequence
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import hydra
import wandb

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
import kornia.augmentation as K
import random

from src.transforms.normalize_mm import SingleSensorNormalizer
from src.transforms.standardize_mm import SingleSensorStandardizer


class DINOModule(LightningModule):
    """
    DINO module.
    """
    def __init__(
        self,
        backbone_config: Dict[str, Any],
        sensor_config: Dict[str, Any],
        hidden_dim: float = 2048,
        bottleneck_dim: float = 256,
        output_dim: int = 32768,
        lr: float = 9.6,
        warmup_epochs: int = 20,
        weight_decay: float = 1e-6,
        momentum: float = 0.9,
        warmup_teacher_temp_epochs: int = 10,
        multicrop: bool = False,
        n_views: int = 0,  # Number of extra local views when multicrop is enabled.
        global_crop_scale: Sequence[float] = (0.4, 1.0),
        local_crop_scale: Sequence[float] = (0.05, 0.4),
        standardize: bool = False,
        standardization_mode: str = "bandwise",
        standardization_stats_path: Optional[str] = None,
        standardization_eps: float = 1e-6,
        student_local_input_drop_ratio: float = 0.0,
        student_local_input_drop_min_ratio: Optional[float] = None,
        student_local_input_drop_max_ratio: Optional[float] = None,
        student_local_input_drop_strategy: str = "random",
        student_local_input_drop_group_size: int = 1,
        student_local_input_drop_fill_mode: str = "transformed_zero",
        student_local_spectral_mask_ratio: float = 0.0,
        student_local_spectral_mask_min_ratio: Optional[float] = None,
        student_local_spectral_mask_max_ratio: Optional[float] = None,
        student_local_spectral_mask_strategy: str = "random",
    ) -> None:
        super().__init__()
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.size = sensor_config["img_size"]
        self.multicrop = multicrop
        self.n_views = n_views
        self.sensor_config = sensor_config
        self.in_channels = self.sensor_config["num_bands"]

        if len(global_crop_scale) != 2 or len(local_crop_scale) != 2:
            raise ValueError("global_crop_scale and local_crop_scale must have length 2")
        self.global_crop_scale = tuple(float(v) for v in global_crop_scale)
        self.local_crop_scale = tuple(float(v) for v in local_crop_scale)
        if not (0.0 < self.global_crop_scale[0] <= self.global_crop_scale[1] <= 1.0):
            raise ValueError(
                f"global_crop_scale must satisfy 0 < min <= max <= 1, got {self.global_crop_scale}"
            )
        if not (0.0 < self.local_crop_scale[0] <= self.local_crop_scale[1] <= 1.0):
            raise ValueError(
                f"local_crop_scale must satisfy 0 < min <= max <= 1, got {self.local_crop_scale}"
            )

        if student_local_input_drop_strategy not in {"random", "contiguous"}:
            raise ValueError(
                "student_local_input_drop_strategy must be one of "
                "['random', 'contiguous']"
            )
        if int(student_local_input_drop_group_size) < 1:
            raise ValueError(
                "student_local_input_drop_group_size must be >= 1, got "
                f"{student_local_input_drop_group_size}"
            )
        if student_local_input_drop_fill_mode not in {"zero", "transformed_zero"}:
            raise ValueError(
                "student_local_input_drop_fill_mode must be one of "
                "['zero', 'transformed_zero']"
            )
        if student_local_spectral_mask_strategy not in {"random", "contiguous"}:
            raise ValueError(
                "student_local_spectral_mask_strategy must be one of "
                "['random', 'contiguous']"
            )

        (
            self.student_local_input_drop_min_ratio,
            self.student_local_input_drop_max_ratio,
        ) = self._resolve_ratio_range(
            name="student_local_input_drop",
            fixed_ratio=student_local_input_drop_ratio,
            min_ratio=student_local_input_drop_min_ratio,
            max_ratio=student_local_input_drop_max_ratio,
        )
        self.student_local_input_drop_strategy = str(student_local_input_drop_strategy)
        self.student_local_input_drop_group_size = int(student_local_input_drop_group_size)
        self.student_local_input_drop_fill_mode = str(student_local_input_drop_fill_mode)
        (
            self.student_local_spectral_mask_min_ratio,
            self.student_local_spectral_mask_max_ratio,
        ) = self._resolve_ratio_range(
            name="student_local_spectral_mask",
            fixed_ratio=student_local_spectral_mask_ratio,
            min_ratio=student_local_spectral_mask_min_ratio,
            max_ratio=student_local_spectral_mask_max_ratio,
        )
        self.student_local_spectral_mask_strategy = str(student_local_spectral_mask_strategy)
        self._transformed_zero_cache: Dict[tuple[str, str], Tensor] = {}

        # Same normalization logic as MMNormalizer (scale / harmonize / thermal / sar)
        self.normalizer = SingleSensorNormalizer(sensor_config=self.sensor_config)

        # Optional standardization over normalized reflectance.
        self.standardizer: Optional[nn.Module] = None
        if standardize:
            if not standardization_stats_path:
                raise ValueError(
                    "standardize=True requires `standardization_stats_path` "
                    "(sensor stats yaml/index/directory)."
                )
            sensor_name = str(self.sensor_config.get("name", "")).upper()
            if not sensor_name:
                raise ValueError("sensor_config.name is required when standardize=True")
            self.standardizer = SingleSensorStandardizer.from_stats_file(
                stats_path=standardization_stats_path,
                sensor_name=sensor_name,
                mode=standardization_mode,
                eps=standardization_eps,
            )

        # Compute augmentation parameters.
        ks = self.size // 10 // 2 * 2 + 1
        global_size = self.size
        # Keep local-view output size equal to model input size.
        # Hiera variants use a fixed token-grid positional embedding and do not
        # accept reduced-resolution local crops (token count mismatch).
        # Multicrop signal still comes from smaller crop_scale, then resize.
        local_size = self.size
        local_ks = local_size // 10 // 2 * 2 + 1

        # Build global augmentation pipeline.
        global_pipeline = [
            self.normalizer,
            self.standardizer if self.standardizer is not None else nn.Identity(),
            K.RandomResizedCrop(size=(global_size, global_size), scale=self.global_crop_scale),
            K.RandomGaussianBlur(kernel_size=(ks, ks), sigma=(0.1, 2), p=0.5),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
        ]
        # Build local augmentation pipeline.
        local_pipeline = [
            self.normalizer,
            self.standardizer if self.standardizer is not None else nn.Identity(),
            K.RandomResizedCrop(size=(local_size, local_size), scale=self.local_crop_scale),
            K.RandomGaussianBlur(kernel_size=(local_ks, local_ks), sigma=(0.1, 2), p=0.5),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
        ]
        self.augmentation1 = K.AugmentationSequential(*global_pipeline, data_keys=["input"])
        # When multicrop is enabled, augmentation2 will generate local views.
        self.augmentation2 = K.AugmentationSequential(*local_pipeline, data_keys=["input"])

        # Create backbone.
        backbone: nn.Module = hydra.utils.instantiate(backbone_config)

        self.student_backbone = backbone
        self.teacher_backbone = copy.deepcopy(backbone)
        self.student_supports_spectral_mask = self._backbone_supports_spectral_mask(self.student_backbone)
        if self.student_local_spectral_mask_max_ratio > 0.0 and not self.student_supports_spectral_mask:
            raise ValueError(
                "student_local_spectral_mask_max_ratio > 0 requires a backbone with "
                "spectral-mask support (e.g. TwoStagesHiera)."
            )
        # Create DINO projection heads.

        # Get num_features from the instantiated backbone
        num_features = backbone(torch.randn(1, self.in_channels, self.size, self.size)).shape[-1]

        self.student_head = DINOProjectionHead(num_features, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=1)
        self.teacher_head = DINOProjectionHead(num_features, hidden_dim, bottleneck_dim, output_dim)
        # Freeze teacher parameters.
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=output_dim, warmup_teacher_temp_epochs=warmup_teacher_temp_epochs)
        self.avg_output_std = 0.0
        
        # Radiometric augmentation parameters
        self.radiometric_aug_p = 0.5
        self.radiometric_brightness_range = (0.8, 1.2)
        self.radiometric_bias_range = (-0.1, 0.1)
        
        # RGB indices for visualization
        self.rgb_indices = list(self.sensor_config.get("rgb_indices", [0, 1, 2]))

    @staticmethod
    def _backbone_supports_spectral_mask(backbone: nn.Module) -> bool:
        patch_embed = getattr(backbone, "patch_embed", None)
        return (
            patch_embed is not None
            and hasattr(patch_embed, "spectral_mask_token")
            and hasattr(backbone, "tokens_spatial_shape")
        )

    @staticmethod
    def _resolve_ratio_range(
        name: str,
        fixed_ratio: float,
        min_ratio: Optional[float],
        max_ratio: Optional[float],
    ) -> Tuple[float, float]:
        fixed_ratio = float(fixed_ratio)
        if min_ratio is None and max_ratio is None:
            min_value = fixed_ratio
            max_value = fixed_ratio
        else:
            min_value = fixed_ratio if min_ratio is None else float(min_ratio)
            max_value = fixed_ratio if max_ratio is None else float(max_ratio)

        if not 0.0 <= min_value < 1.0:
            raise ValueError(f"{name}_min_ratio must be in [0, 1), got {min_value}")
        if not 0.0 <= max_value < 1.0:
            raise ValueError(f"{name}_max_ratio must be in [0, 1), got {max_value}")
        if min_value > max_value:
            raise ValueError(
                f"{name}_min_ratio must be <= {name}_max_ratio, got "
                f"{min_value} > {max_value}"
            )
        return min_value, max_value

    @staticmethod
    def _sample_ratio(min_ratio: float, max_ratio: float, device: torch.device) -> float:
        if max_ratio <= 0.0:
            return 0.0
        if min_ratio == max_ratio:
            return min_ratio
        return float(torch.empty(1, device=device).uniform_(min_ratio, max_ratio).item())

    def _forward_backbone(
        self,
        backbone: nn.Module,
        x: Tensor,
        spectral_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if spectral_mask is not None:
            if not self._backbone_supports_spectral_mask(backbone):
                raise ValueError(
                    "spectral_mask was provided, but the selected backbone does not "
                    "support spectral masking."
                )
            return backbone(x, spectral_mask=spectral_mask)
        return backbone(x)

    def forward(self, x: Tensor, spectral_mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the student network."""
        y = self._forward_backbone(self.student_backbone, x, spectral_mask=spectral_mask).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x: Tensor) -> Tensor:
        """Forward pass through the teacher network."""
        y = self._forward_backbone(self.teacher_backbone, x, spectral_mask=None).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z
    
    def _apply_radiometric_augmentation(self, tensor: Tensor) -> Tensor:
        """Apply radiometric augmentation (brightness + bias) after normalization."""
        if random.random() > self.radiometric_aug_p:
            return tensor
        
        B, C, H, W = tensor.shape
        device = tensor.device
        
        # Sample scalar params (same for all channels)
        brightness = torch.empty(1, device=device).uniform_(
            self.radiometric_brightness_range[0], 
            self.radiometric_brightness_range[1]
        )
        bias = torch.empty(1, device=device).uniform_(
            self.radiometric_bias_range[0], 
            self.radiometric_bias_range[1]
        )
        
        # Apply: output = tensor * brightness + bias
        augmented = tensor * brightness + bias

        
        return augmented

    def _sample_keep_mask(
        self,
        batch_size: int,
        num_units: int,
        mask_ratio: float,
        strategy: str,
        device: torch.device,
        min_keep: int = 1,
    ) -> Tensor:
        if num_units <= 0:
            raise ValueError(f"num_units must be > 0, got {num_units}")

        keep_units = int(round(num_units * (1.0 - mask_ratio)))
        keep_units = min(max(keep_units, min_keep), num_units)
        if keep_units == num_units:
            return torch.ones(batch_size, num_units, dtype=torch.bool, device=device)

        if strategy == "random":
            noise = torch.rand(batch_size, num_units, device=device)
            ids_keep = torch.argsort(noise, dim=1)[:, :keep_units]
            keep = torch.zeros(batch_size, num_units, dtype=torch.bool, device=device)
            keep.scatter_(1, ids_keep, True)
            return keep

        if strategy == "contiguous":
            max_start = num_units - keep_units
            starts = torch.randint(0, max_start + 1, (batch_size,), device=device)
            idx = torch.arange(num_units, device=device).view(1, -1)
            return (idx >= starts.view(-1, 1)) & (idx < (starts + keep_units).view(-1, 1))

        raise ValueError(f"Unsupported keep-mask strategy '{strategy}'")

    def _get_transformed_zero_fill(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        cache_key = (str(device), str(dtype))
        cached = self._transformed_zero_cache.get(cache_key)
        if cached is not None:
            return cached

        zeros = torch.zeros(1, self.in_channels, 1, 1, device=device, dtype=torch.float32)
        fill = self.normalizer(zeros)
        if self.standardizer is not None:
            fill = self.standardizer(fill)
        fill = fill.to(dtype=dtype)
        self._transformed_zero_cache[cache_key] = fill
        return fill

    def _apply_student_local_input_drop(self, tensor: Tensor) -> Tuple[Tensor, float]:
        sampled_ratio = self._sample_ratio(
            min_ratio=self.student_local_input_drop_min_ratio,
            max_ratio=self.student_local_input_drop_max_ratio,
            device=tensor.device,
        )
        if sampled_ratio <= 0.0:
            return tensor, 0.0

        batch_size, channels, _, _ = tensor.shape
        group_size = self.student_local_input_drop_group_size
        groups = [
            list(range(start, min(start + group_size, channels)))
            for start in range(0, channels, group_size)
        ]
        keep_groups = self._sample_keep_mask(
            batch_size=batch_size,
            num_units=len(groups),
            mask_ratio=sampled_ratio,
            strategy=self.student_local_input_drop_strategy,
            device=tensor.device,
            min_keep=1,
        )

        keep_channels = torch.zeros(batch_size, channels, dtype=torch.bool, device=tensor.device)
        for group_idx, group_channels in enumerate(groups):
            keep_channels[:, group_channels] = keep_groups[:, group_idx].unsqueeze(1)

        keep_channels = keep_channels.unsqueeze(-1).unsqueeze(-1)
        if self.student_local_input_drop_fill_mode == "zero":
            fill = torch.zeros(1, channels, 1, 1, device=tensor.device, dtype=tensor.dtype)
        else:
            fill = self._get_transformed_zero_fill(device=tensor.device, dtype=tensor.dtype)

        dropped = torch.where(keep_channels, tensor, fill)
        realized_ratio = 1.0 - float(keep_channels.float().mean().item())
        return dropped, realized_ratio

    def _to_display_rgb(
        self,
        image: Tensor,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> np.ndarray:
        """Convert a transformed tensor to a displayable RGB image.

        Views are logged after normalization / optional standardization, so raw values
        are not suitable for direct display in W&B. We robustly rescale the selected
        channels for visualization only.
        """
        image = image.detach().cpu().float()
        channels = image.shape[0]

        rgb_indices = [idx for idx in self.rgb_indices if 0 <= idx < channels]
        if not rgb_indices:
            rgb_indices = list(range(min(3, channels)))
        rgb = image[rgb_indices]
        if rgb.shape[0] == 1:
            rgb = rgb.repeat(3, 1, 1)
        elif rgb.shape[0] == 2:
            rgb = torch.cat([rgb, rgb[:1]], dim=0)
        elif rgb.shape[0] > 3:
            rgb = rgb[:3]

        rgb_np = rgb.permute(1, 2, 0).numpy()
        lo = float(np.percentile(rgb_np, percentile_low))
        hi = float(np.percentile(rgb_np, percentile_high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(rgb_np)
        return np.clip((rgb_np - lo) / (hi - lo), 0.0, 1.0)

    def _build_student_local_spectral_mask(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[Tensor], float]:
        sampled_ratio = self._sample_ratio(
            min_ratio=self.student_local_spectral_mask_min_ratio,
            max_ratio=self.student_local_spectral_mask_max_ratio,
            device=device,
        )
        if sampled_ratio <= 0.0:
            return None, 0.0
        if not self.student_supports_spectral_mask:
            return None, 0.0

        patch_embed = self.student_backbone.patch_embed
        groups = int(getattr(patch_embed, "G", 0))
        if groups <= 0:
            return None, 0.0

        spatial_tokens = int(math.prod(getattr(self.student_backbone, "tokens_spatial_shape")))
        keep_groups = self._sample_keep_mask(
            batch_size=batch_size,
            num_units=groups,
            mask_ratio=sampled_ratio,
            strategy=self.student_local_spectral_mask_strategy,
            device=device,
            min_keep=1,
        )
        spectral_mask = keep_groups.unsqueeze(1).expand(batch_size, spatial_tokens, groups).contiguous()
        realized_ratio = 1.0 - float(keep_groups.float().mean().item())
        return spectral_mask, realized_ratio

    def _get_image_tensors(self, batch: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        """Get (x1, x2) from batch. Supports MM Zarr (sensor name key) or legacy image/image1/image2."""
        sensor_key = self.sensor_config.get("name")
        if sensor_key and sensor_key in batch and isinstance(batch[sensor_key], Tensor):
            x = batch[sensor_key].float()
            assert x.size(1) == self.in_channels
            return x, x
        if "image1" in batch and "image2" in batch:
            x1 = batch["image1"].float()
            x2 = batch["image2"].float()
            assert x1.size(1) == self.in_channels
            return x1, x2
        x = batch["image"].float()
        assert x.size(1) == self.in_channels
        return x, x

    def training_step(self, batch, batch_idx) -> Tensor:
        # Update teacher momentum using cosine schedule (span full training).
        momentum = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        # Get input views: MM Zarr uses sensor name key (e.g. "ENMAP"); legacy uses "image" or "image1"/"image2".
        x1, x2 = self._get_image_tensors(batch)

        with torch.no_grad():
            x1 = self.augmentation1(x1)
            x2 = self.augmentation1(x2)
            # Apply radiometric augmentation AFTER normalization
            x1 = self._apply_radiometric_augmentation(x1)
            x2 = self._apply_radiometric_augmentation(x2)

        global_views = [
            {"x": x1, "spectral_mask": None},
            {"x": x2, "spectral_mask": None},
        ]
        views = list(global_views)
        local_views = []
        if self.multicrop:
            local_views = []
            local_input_drop_ratios = []
            local_spectral_mask_ratios = []
            for i in range(self.n_views):
                x, _ = self._get_image_tensors(batch)
                x_aug = self.augmentation2(x)
                # Apply radiometric augmentation to local views too
                x_aug = self._apply_radiometric_augmentation(x_aug)
                x_aug, realized_input_drop = self._apply_student_local_input_drop(x_aug)
                spectral_mask, realized_spectral_mask = self._build_student_local_spectral_mask(
                    batch_size=x_aug.shape[0],
                    device=x_aug.device,
                )
                local_views.append({"x": x_aug, "spectral_mask": spectral_mask})
                local_input_drop_ratios.append(realized_input_drop)
                local_spectral_mask_ratios.append(realized_spectral_mask)
            views = global_views + local_views
            if local_input_drop_ratios:
                self.log(
                    "train_local_input_drop_ratio",
                    float(sum(local_input_drop_ratios) / len(local_input_drop_ratios)),
                    on_step=True,
                    on_epoch=False,
                )
            if local_spectral_mask_ratios:
                self.log(
                    "train_local_spectral_mask_ratio",
                    float(sum(local_spectral_mask_ratios) / len(local_spectral_mask_ratios)),
                    on_step=True,
                    on_epoch=False,
                )

        teacher_out = [self.forward_teacher(view["x"]) for view in global_views]
        student_out = [self.forward(view["x"], spectral_mask=view["spectral_mask"]) for view in views]

        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss", loss)

        with torch.no_grad():
            features = self.student_backbone(global_views[0]["x"]).flatten(start_dim=1)
            norm_features = F.normalize(features, dim=1)
            output_std = torch.std(norm_features, dim=0).mean().item()
            self.avg_output_std = 0.9 * self.avg_output_std + 0.1 * output_std
            self.log("train_ssl_std", self.avg_output_std)
        
        # Visualize global views periodically for debugging
        if self.current_epoch % 10 == 0 and batch_idx == 0:
            self._log_visualization(global_views[0]["x"], global_views[1]["x"], local_views=local_views)

        return loss
    
    def _log_visualization(
        self,
        x1: Tensor,
        x2: Tensor,
        local_views: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Log global and local views with display-only robust normalization."""
        if not self.logger:
            return
        
        # Take first sample from batch
        img1_np = self._to_display_rgb(x1[0])
        img2_np = self._to_display_rgb(x2[0])
        
        # Concatenate views side by side
        combined = np.concatenate([img1_np, img2_np], axis=1)
        
        self.logger.experiment.log({
            "viz/dino_global_views": [wandb.Image(combined, caption="Global View 1 | Global View 2")]
        }, step=self.global_step)

        if local_views:
            local_images = []
            for idx, local_view in enumerate(local_views[:4], start=1):
                local_images.append(self._to_display_rgb(local_view["x"][0]))
            if local_images:
                local_combined = np.concatenate(local_images, axis=1)
                self.logger.experiment.log(
                    {
                        "viz/dino_local_views": [
                            wandb.Image(local_combined, caption=" | ".join(f"Local View {i}" for i in range(1, len(local_images) + 1)))
                        ]
                    },
                    step=self.global_step,
                )

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self) -> Tuple[list[Optimizer], list]:
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1 / self.warmup_epochs, total_iters=self.warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs),
            ],
            milestones=[self.warmup_epochs],
        )
        return [optimizer], [lr_scheduler]
