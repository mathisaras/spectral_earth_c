from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import hydra
import kornia.augmentation as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.errors import InstantiationException
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.losses.sigreg import SIGReg
from src.transforms.normalize_mm import SingleSensorNormalizer
from src.transforms.standardize_mm import SingleSensorStandardizer


class LeJEPAProjectionHead(nn.Module):
    """Simple MLP projection head used by LeJEPA."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 3,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        if num_layers == 1:
            self.net = nn.Linear(input_dim, output_dim)
            return

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            is_last = i == (len(dims) - 2)
            bias = not use_bn or is_last
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            if not is_last:
                if use_bn:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


LEJEPA_INVARIANCE_MODES = ("coupled_center", "detached_global_center")


def compute_lejepa_invariance_loss(
    proj: Tensor,
    *,
    n_global: int,
    invariance_mode: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute LeJEPA's invariance term and its global/local components."""
    if proj.ndim != 3:
        raise ValueError(f"Expected proj [V, B, D], got {tuple(proj.shape)}")
    if not (1 <= int(n_global) <= int(proj.shape[0])):
        raise ValueError(
            f"n_global must be in [1, {int(proj.shape[0])}], got {n_global}."
        )

    mode = str(invariance_mode).lower()
    if mode not in LEJEPA_INVARIANCE_MODES:
        raise ValueError(
            f"invariance_mode must be one of {LEJEPA_INVARIANCE_MODES}, got {invariance_mode}."
        )

    global_proj = proj[:n_global]
    local_proj = proj[n_global:]
    center = global_proj.mean(dim=0, keepdim=True)
    zero = global_proj.new_zeros(())

    if mode == "coupled_center":
        inv_total = (center - proj).square().mean()
        inv_global = (center - global_proj).square().mean()
        inv_local = (center - local_proj).square().mean() if local_proj.numel() > 0 else zero
        return inv_total, inv_global, inv_local

    detached_center = center.detach()
    inv_global = (detached_center - global_proj).square().mean()
    inv_local = (detached_center - local_proj).square().mean() if local_proj.numel() > 0 else zero
    inv_total = inv_global + inv_local
    return inv_total, inv_global, inv_local


class LeJEPAModule(LightningModule):
    """
    LeJEPA module for single-sensor SSL on SpectralEarth MM-Zarr batches.

    Loss:
      lejepa_loss = lambda * sigreg(proj) + (1 - lambda) * invariance(proj)
      invariance_mode="coupled_center":
        center = mean(global_proj), matched to all views with full gradient coupling.
      invariance_mode="detached_global_center":
        center = stopgrad(mean(global_proj)), with explicit global/global and
        local/global matching terms.
    """

    def __init__(
        self,
        backbone_config: DictConfig,
        sensor_config: DictConfig,
        output_dim: int = 128,
        projector_hidden_dim: int = 2048,
        projector_num_layers: int = 3,
        projector_use_bn: bool = True,
        lamb: float = 0.5,
        invariance_mode: str = "coupled_center",
        lr: float = 1.0e-4,
        warmup_epochs: int = 20,
        max_epochs: int = 200,
        weight_decay: float = 1.0e-5,
        multicrop: bool = True,
        n_views: int = 6,
        global_crop_scale: Tuple[float, float] = (0.3, 1.0),
        local_crop_scale: Tuple[float, float] = (0.05, 0.3),
        local_resize_to_global: bool = False,
        local_view_size: Optional[int] = None,
        local_view_size_ratio: Optional[float] = None,
        # Blur settings (kept configurable across sensors/resolutions).
        # Defaults map to kernel~7 for 128px inputs and scale with resolution.
        blur_kernel_ratio: float = 7.0 / 128.0,
        global_blur_kernel_size: Optional[int] = None,
        local_blur_kernel_size: Optional[int] = None,
        global_blur_p: float = 0.5,
        local_blur_p: float = 0.5,
        blur_sigma: Tuple[float, float] = (0.1, 2.0),
        radiometric_aug_p: float = 0.5,
        radiometric_brightness_range: Tuple[float, float] = (0.8, 1.2),
        radiometric_bias_range: Tuple[float, float] = (-0.1, 0.1),
        sequence_pool: str = "auto",  # auto | mean | cls
        # Optional mean/std standardization on top of normalization.
        standardize: bool = False,
        standardization_mode: str = "bandwise",
        standardization_stats_path: Optional[str] = None,
        standardization_eps: float = 1.0e-6,
        # Optional local-student-view spectral corruption.
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
        # SIGReg
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_num_vectors: int = 256,
        sigreg_gather_distributed: bool = False,
    ) -> None:
        super().__init__()
        # Avoid key collisions with DataModule hparams when Lightning merges
        # model/datamodule hyperparameters for logger backends (e.g. wandb).
        self.save_hyperparameters(
            ignore=[
                "backbone_config",
                "sensor_config",
                "radiometric_aug_p",
                "standardize",
                "standardization_mode",
                "standardization_stats_path",
                "standardization_eps",
            ]
        )

        if not (0.0 <= lamb <= 1.0):
            raise ValueError(f"`lamb` must be in [0, 1], got {lamb}.")
        invariance_mode = str(invariance_mode).lower()
        if invariance_mode not in LEJEPA_INVARIANCE_MODES:
            raise ValueError(
                f"`invariance_mode` must be one of {LEJEPA_INVARIANCE_MODES}, got {invariance_mode}."
            )
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}.")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {max_epochs}.")

        self.sensor_config = sensor_config
        self.size = int(sensor_config["img_size"])
        self.in_channels = int(sensor_config["num_bands"])
        self.sequence_pool = str(sequence_pool).lower()
        if self.sequence_pool not in {"auto", "mean", "cls"}:
            raise ValueError(
                f"sequence_pool must be one of ['auto', 'mean', 'cls'], got {sequence_pool}."
            )

        self.multicrop = bool(multicrop)
        self.n_views = int(n_views) if multicrop else 0
        self.lamb = float(lamb)
        self.invariance_mode = invariance_mode
        if student_local_input_drop_strategy not in {"random", "contiguous"}:
            raise ValueError(
                "student_local_input_drop_strategy must be one of ['random', 'contiguous']"
            )
        if int(student_local_input_drop_group_size) < 1:
            raise ValueError(
                f"student_local_input_drop_group_size must be >= 1, got {student_local_input_drop_group_size}."
            )
        if student_local_input_drop_fill_mode not in {"zero", "transformed_zero"}:
            raise ValueError(
                "student_local_input_drop_fill_mode must be one of ['zero', 'transformed_zero']"
            )
        if student_local_spectral_mask_strategy not in {"random", "contiguous"}:
            raise ValueError(
                "student_local_spectral_mask_strategy must be one of ['random', 'contiguous']"
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

        self.normalizer = SingleSensorNormalizer(sensor_config=self.sensor_config)
        self.standardizer: Optional[nn.Module] = None
        if standardize:
            if not standardization_stats_path:
                raise ValueError(
                    "standardize=True requires `standardization_stats_path` "
                    "(sensor stats yaml/index/directory)."
                )
            sensor_name = str(self.sensor_config.get("name", "")).upper()
            if not sensor_name:
                raise ValueError(
                    "sensor_config.name is required when standardize=True."
                )
            self.standardizer = SingleSensorStandardizer.from_stats_file(
                stats_path=standardization_stats_path,
                sensor_name=sensor_name,
                mode=standardization_mode,
                eps=standardization_eps,
            )

        if local_view_size is not None and int(local_view_size) <= 0:
            raise ValueError(
                f"local_view_size must be > 0 when provided, got {local_view_size}."
            )
        if local_view_size_ratio is not None and not (0.0 < float(local_view_size_ratio) <= 1.0):
            raise ValueError(
                "local_view_size_ratio must be in (0, 1] when provided, "
                f"got {local_view_size_ratio}."
            )
        if float(blur_kernel_ratio) <= 0.0:
            raise ValueError(
                f"blur_kernel_ratio must be > 0, got {blur_kernel_ratio}."
            )
        if not (0.0 <= float(global_blur_p) <= 1.0):
            raise ValueError(f"global_blur_p must be in [0, 1], got {global_blur_p}.")
        if not (0.0 <= float(local_blur_p) <= 1.0):
            raise ValueError(f"local_blur_p must be in [0, 1], got {local_blur_p}.")
        if float(blur_sigma[0]) <= 0.0 or float(blur_sigma[1]) <= 0.0:
            raise ValueError(f"blur_sigma values must be > 0, got {blur_sigma}.")
        if float(blur_sigma[0]) > float(blur_sigma[1]):
            raise ValueError(
                f"blur_sigma must satisfy min<=max, got {blur_sigma}."
            )

        def _resolve_kernel_size(size: int, override: Optional[int]) -> int:
            if override is not None:
                ks = int(override)
                if ks <= 0:
                    raise ValueError(
                        f"Blur kernel override must be > 0, got {override}."
                    )
            else:
                ks = max(3, int(round(float(size) * float(blur_kernel_ratio))))
            if ks % 2 == 0:
                ks += 1
            return ks

        # Local view sizing (fully configurable):
        # 1) local_resize_to_global=True  -> local_size = global_size
        # 2) local_view_size set          -> explicit absolute size
        # 3) local_view_size_ratio set    -> ratio of global size
        # 4) otherwise                    -> LeJEPA-style fallback ratio (98/224)
        global_size = self.size
        if local_resize_to_global:
            local_size = self.size
        elif local_view_size is not None:
            local_size = max(int(local_view_size), 16)
        else:
            ratio = float(local_view_size_ratio) if local_view_size_ratio is not None else (98.0 / 224.0)
            local_size = max(int(round(self.size * ratio)), 16)

        global_ks = _resolve_kernel_size(global_size, global_blur_kernel_size)
        local_ks = _resolve_kernel_size(local_size, local_blur_kernel_size)
        prep = [
            self.normalizer,
            self.standardizer if self.standardizer is not None else nn.Identity(),
        ]

        self.global_augmentation = K.AugmentationSequential(
            *prep,
            K.RandomResizedCrop(
                size=(global_size, global_size), scale=global_crop_scale
            ),
            K.RandomGaussianBlur(
                kernel_size=(global_ks, global_ks),
                sigma=blur_sigma,
                p=float(global_blur_p),
            ),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            data_keys=["input"],
        )

        self.local_augmentation = K.AugmentationSequential(
            *prep,
            K.RandomResizedCrop(size=(local_size, local_size), scale=local_crop_scale),
            K.RandomGaussianBlur(
                kernel_size=(local_ks, local_ks),
                sigma=blur_sigma,
                p=float(local_blur_p),
            ),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            data_keys=["input"],
        )

        self.radiometric_aug_p = float(radiometric_aug_p)
        self.radiometric_brightness_range = radiometric_brightness_range
        self.radiometric_bias_range = radiometric_bias_range

        self.backbone = self._instantiate_backbone(backbone_config)
        self.backbone_supports_spectral_mask = self._backbone_supports_spectral_mask(self.backbone)
        if self.student_local_spectral_mask_max_ratio > 0.0 and not self.backbone_supports_spectral_mask:
            raise ValueError(
                "student_local_spectral_mask_max_ratio > 0 requires a backbone with "
                "spectral-mask support (e.g. TwoStagesHiera)."
            )
        feature_dim = self._infer_backbone_feature_dim()
        self.projection_head = LeJEPAProjectionHead(
            input_dim=feature_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=output_dim,
            num_layers=projector_num_layers,
            use_bn=projector_use_bn,
        )

        self.sigreg = SIGReg(
            knots=sigreg_knots,
            t_max=sigreg_t_max,
            num_vectors=sigreg_num_vectors,
            gather_distributed=sigreg_gather_distributed,
        )

        self.avg_output_std = 0.0

    def _instantiate_backbone(self, backbone_config: DictConfig) -> nn.Module:
        try:
            return hydra.utils.instantiate(backbone_config)
        except (TypeError, InstantiationException) as err:
            # Some configs used by TIMM include `global_pool`, which custom
            # backbones may not accept. Retry without it.
            if "global_pool" not in str(err):
                raise
            cfg = OmegaConf.create(OmegaConf.to_container(backbone_config, resolve=False))
            if "global_pool" in cfg:
                del cfg["global_pool"]
                return hydra.utils.instantiate(cfg)
            raise

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
                f"{name}_min_ratio must be <= {name}_max_ratio, got {min_value} > {max_value}."
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
        x: Tensor,
        spectral_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if spectral_mask is not None:
            if not self._backbone_supports_spectral_mask(self.backbone):
                raise ValueError(
                    "spectral_mask was provided, but the selected backbone does not support spectral masking."
                )
            return self.backbone(x, spectral_mask=spectral_mask)
        return self.backbone(x)

    def _extract_backbone_features(
        self, x: Tensor, spectral_mask: Optional[Tensor] = None
    ) -> Tensor:
        y = self._forward_backbone(x, spectral_mask=spectral_mask)
        if isinstance(y, (tuple, list)):
            if len(y) == 0:
                raise RuntimeError("Backbone returned an empty tuple/list.")
            y = y[0]

        if not isinstance(y, Tensor):
            raise RuntimeError(
                f"Backbone output must be a Tensor, got {type(y).__name__}."
            )

        if y.ndim == 2:
            return y

        if y.ndim == 3:
            if self.sequence_pool == "mean":
                return y.mean(dim=1)
            if self.sequence_pool == "cls":
                return y[:, 0]
            # auto
            num_prefix_tokens = getattr(self.backbone, "num_prefix_tokens", 0)
            if isinstance(num_prefix_tokens, int) and num_prefix_tokens > 0:
                return y[:, 0]
            return y.mean(dim=1)

        if y.ndim == 4:
            return F.adaptive_avg_pool2d(y, output_size=(1, 1)).flatten(start_dim=1)

        return y.flatten(start_dim=1)

    def _infer_backbone_feature_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.randn(1, self.in_channels, self.size, self.size)
            feat = self._extract_backbone_features(dummy)
        if feat.ndim != 2:
            raise RuntimeError(
                f"Expected 2D backbone features [B, D], got shape {tuple(feat.shape)}."
            )
        return int(feat.shape[-1])

    def _apply_radiometric_augmentation(self, tensor: Tensor) -> Tensor:
        if random.random() > self.radiometric_aug_p:
            return tensor
        device = tensor.device
        brightness = torch.empty(1, device=device).uniform_(
            self.radiometric_brightness_range[0],
            self.radiometric_brightness_range[1],
        )
        bias = torch.empty(1, device=device).uniform_(
            self.radiometric_bias_range[0],
            self.radiometric_bias_range[1],
        )
        return tensor * brightness + bias

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
        if not self.backbone_supports_spectral_mask:
            return None, 0.0

        patch_embed = self.backbone.patch_embed
        groups = int(getattr(patch_embed, "G", 0))
        if groups <= 0:
            return None, 0.0

        spatial_tokens = int(math.prod(getattr(self.backbone, "tokens_spatial_shape")))
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
        # Temporal-view mode from datamodule (`num_temporal_views=2`).
        if "image1" in batch and "image2" in batch:
            x1 = batch["image1"].float()
            x2 = batch["image2"].float()
            if x1.ndim != 4 or x2.ndim != 4:
                raise ValueError(
                    f"image1/image2 must be 4D tensors, got {x1.ndim}D/{x2.ndim}D."
                )
            if x1.size(1) != self.in_channels or x2.size(1) != self.in_channels:
                raise ValueError(
                    f"image1/image2 channel mismatch. Expected C={self.in_channels}, "
                    f"got {x1.size(1)}/{x2.size(1)}."
                )
            return x1, x2

        sensor_name = str(self.sensor_config.get("name", ""))
        for key in (sensor_name, sensor_name.upper(), sensor_name.lower()):
            if key and key in batch and isinstance(batch[key], Tensor):
                x = batch[key].float()
                if x.ndim != 4:
                    raise ValueError(
                        f"Sensor tensor '{key}' must be 4D [B,C,H,W], got shape {tuple(x.shape)}."
                    )
                if x.size(1) != self.in_channels:
                    raise ValueError(
                        f"Sensor tensor '{key}' channel mismatch. "
                        f"Expected C={self.in_channels}, got {x.size(1)}."
                    )
                return x, x

        if "image" in batch and isinstance(batch["image"], Tensor):
            x = batch["image"].float()
            if x.ndim != 4:
                raise ValueError(
                    f"'image' tensor must be 4D [B,C,H,W], got shape {tuple(x.shape)}."
                )
            if x.size(1) != self.in_channels:
                raise ValueError(
                    f"'image' channel mismatch. Expected C={self.in_channels}, got {x.size(1)}."
                )
            return x, x

        # Last-resort fallback for custom loaders: first 4D tensor in batch.
        for key, value in batch.items():
            if isinstance(value, Tensor) and value.ndim == 4:
                if value.size(1) != self.in_channels:
                    continue
                x = value.float()
                return x, x

        raise KeyError(
            "LeJEPAModule could not find an image tensor in batch. "
            "Expected one of: sensor key, image, image1/image2."
        )

    def _build_views(
        self, x1: Tensor, x2: Tensor
    ) -> Tuple[list[Dict[str, Optional[Tensor]]], list[float], list[float]]:
        with torch.no_grad():
            global_1 = self._apply_radiometric_augmentation(self.global_augmentation(x1))
            global_2 = self._apply_radiometric_augmentation(self.global_augmentation(x2))

        views: list[Dict[str, Optional[Tensor]]] = [
            {"x": global_1, "spectral_mask": None},
            {"x": global_2, "spectral_mask": None},
        ]
        local_input_drop_ratios: list[float] = []
        local_spectral_mask_ratios: list[float] = []
        if self.multicrop and self.n_views > 0:
            for i in range(self.n_views):
                src = x1 if (i % 2 == 0) else x2
                with torch.no_grad():
                    local_view = self._apply_radiometric_augmentation(
                        self.local_augmentation(src)
                    )
                local_view, realized_input_drop = self._apply_student_local_input_drop(local_view)
                spectral_mask, realized_spectral_mask = self._build_student_local_spectral_mask(
                    batch_size=local_view.shape[0],
                    device=local_view.device,
                )
                views.append({"x": local_view, "spectral_mask": spectral_mask})
                local_input_drop_ratios.append(realized_input_drop)
                local_spectral_mask_ratios.append(realized_spectral_mask)
        return views, local_input_drop_ratios, local_spectral_mask_ratios

    def _should_log_view_viz(self, batch_idx: int) -> bool:
        if int(batch_idx) != 0:
            return False
        if int(self.current_epoch) % 10 != 0:
            return False
        if int(getattr(self, "global_rank", 0)) != 0:
            return False
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return False
        if not hasattr(self.logger.experiment, "log"):
            return False
        return True

    def _get_vis_channels(self, num_channels: int) -> list[int]:
        configured = self.sensor_config.get("rgb_indices", None)
        if isinstance(configured, (list, tuple)) and len(configured) >= 3:
            idx = [int(i) for i in configured[:3]]
            if min(idx) >= 0 and max(idx) < int(num_channels):
                return idx
        if num_channels >= 3:
            return [0, 1, 2]
        if num_channels == 2:
            return [0, 1, 0]
        return [0, 0, 0]

    @staticmethod
    def _normalize_for_display(x: Tensor) -> Tensor:
        # x: [3, H, W], robust per-channel normalization for stable visualization.
        eps = 1.0e-6
        flat = x.reshape(x.shape[0], -1)
        q_low = torch.quantile(flat, 0.02, dim=1, keepdim=True)
        q_high = torch.quantile(flat, 0.98, dim=1, keepdim=True)
        denom = (q_high - q_low).clamp_min(eps)
        norm = (flat.clamp(min=q_low, max=q_high) - q_low) / denom
        return norm.reshape_as(x).clamp_(0.0, 1.0)

    def _to_display_rgb(
        self,
        view: Tensor,
        sample_idx: int = 0,
        spatial_mask: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if view.ndim != 4 or int(sample_idx) >= int(view.shape[0]):
            return None
        sample = view[sample_idx].detach()
        vis_idx = self._get_vis_channels(sample.shape[0])
        rgb = sample[vis_idx]
        rgb = self._normalize_for_display(rgb)

        if spatial_mask is None:
            return rgb

        if spatial_mask.ndim != 3 or int(sample_idx) >= int(spatial_mask.shape[0]):
            return rgb

        sample_mask = spatial_mask[sample_idx].detach().float()
        sample_mask = F.interpolate(
            sample_mask.view(1, 1, *sample_mask.shape),
            size=(rgb.shape[1], rgb.shape[2]),
            mode="nearest",
        ).squeeze(0)

        overlay = rgb.clone()
        masked = sample_mask.squeeze(0) > 0.5
        if masked.any():
            overlay[:, masked] = overlay[:, masked] * 0.4
            overlay[0, masked] = overlay[0, masked] + 0.6
        return overlay.clamp_(0.0, 1.0)

    def _build_views_panel(
        self,
        views: Sequence[Tensor],
        sample_idx: int = 0,
        spatial_masks: Optional[Sequence[Optional[Tensor]]] = None,
    ) -> Optional[Tensor]:
        if spatial_masks is None:
            spatial_masks = [None] * len(views)
        elif len(spatial_masks) != len(views):
            raise ValueError(
                "spatial_masks must have the same length as views, "
                f"got {len(spatial_masks)} masks for {len(views)} views."
            )

        tiles = []
        for view, spatial_mask in zip(views, spatial_masks):
            tile = self._to_display_rgb(
                view=view,
                sample_idx=sample_idx,
                spatial_mask=spatial_mask,
            )
            if tile is not None:
                tiles.append(tile)
        if not tiles:
            return None

        target_h = max(int(tile.shape[1]) for tile in tiles)
        target_w = max(int(tile.shape[2]) for tile in tiles)
        resized_tiles: list[Tensor] = []
        for tile in tiles:
            if int(tile.shape[1]) != target_h or int(tile.shape[2]) != target_w:
                tile = F.interpolate(
                    tile.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            resized_tiles.append(tile)

        sep = torch.ones(
            3,
            target_h,
            2,
            dtype=resized_tiles[0].dtype,
            device=resized_tiles[0].device,
        )
        row: list[Tensor] = []
        for i, tile in enumerate(resized_tiles):
            if i > 0:
                row.append(sep)
            row.append(tile)
        return torch.cat(row, dim=2).permute(1, 2, 0).contiguous()

    def _log_view_visualizations(
        self,
        *,
        global_views: Sequence[Tensor],
        local_views: Sequence[Tensor],
        batch_idx: int,
        global_spatial_masks: Optional[Sequence[Optional[Tensor]]] = None,
    ) -> None:
        if not self._should_log_view_viz(batch_idx=batch_idx):
            return
        try:
            import wandb
        except Exception:
            return

        log_dict: Dict[str, Any] = {}
        viz_namespace = str(getattr(self, "_viz_namespace", "lejepa"))
        global_panel = self._build_views_panel(
            global_views,
            sample_idx=0,
            spatial_masks=global_spatial_masks,
        )
        if global_panel is not None:
            global_caption = f"epoch={int(self.current_epoch)} global_views={len(global_views)}"
            if global_spatial_masks and any(mask is not None for mask in global_spatial_masks):
                global_caption = f"{global_caption} red=spatial_mask"
            log_dict[f"viz/{viz_namespace}/global_views"] = [
                wandb.Image(
                    global_panel.detach().cpu().numpy(),
                    caption=global_caption,
                )
            ]

        local_panel = self._build_views_panel(local_views, sample_idx=0)
        if local_panel is not None:
            log_dict[f"viz/{viz_namespace}/local_views"] = [
                wandb.Image(
                    local_panel.detach().cpu().numpy(),
                    caption=f"epoch={int(self.current_epoch)} local_views={len(local_views)}",
                )
            ]

        if log_dict:
            self.logger.experiment.log(log_dict, step=int(self.global_step))

    def forward(self, x: Tensor, spectral_mask: Optional[Tensor] = None) -> Tensor:
        features = self._extract_backbone_features(x, spectral_mask=spectral_mask)
        return self.projection_head(features)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        x1, x2 = self._get_image_tensors(batch)
        views, local_input_drop_ratios, local_spectral_mask_ratios = self._build_views(x1, x2)
        if local_input_drop_ratios:
            self.log(
                "train_local_input_drop_ratio",
                float(sum(local_input_drop_ratios) / len(local_input_drop_ratios)),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        if local_spectral_mask_ratios:
            self.log(
                "train_local_spectral_mask_ratio",
                float(sum(local_spectral_mask_ratios) / len(local_spectral_mask_ratios)),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

        projections = []
        first_view_features: Optional[Tensor] = None
        for i, view in enumerate(views):
            features = self._extract_backbone_features(view["x"], spectral_mask=view["spectral_mask"])
            if i == 0:
                first_view_features = features
            projections.append(self.projection_head(features))

        proj = torch.stack(projections, dim=0)  # [V_all, B, D]
        # LeJEPA paper alignment:
        # center is computed from global views only, then matched to all views.
        n_global = min(2, proj.shape[0])
        inv_loss, inv_loss_global, inv_loss_local = compute_lejepa_invariance_loss(
            proj,
            n_global=n_global,
            invariance_mode=self.invariance_mode,
        )
        sigreg_loss = self.sigreg(proj)
        loss = self.lamb * sigreg_loss + (1.0 - self.lamb) * inv_loss
        weighted_sigreg_loss = self.lamb * sigreg_loss

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_loss_inv", inv_loss, prog_bar=False, sync_dist=True)
        self.log("train_loss_inv_global", inv_loss_global, prog_bar=False, sync_dist=True)
        self.log("train_loss_inv_local", inv_loss_local, prog_bar=False, sync_dist=True)
        self.log("train_loss_sigreg", sigreg_loss, prog_bar=False, sync_dist=True)
        self.log("train_loss_sigreg_weighted", weighted_sigreg_loss, prog_bar=False, sync_dist=True)
        self.log("train_num_views", float(len(views)), prog_bar=False, sync_dist=True)
        self.log("train_num_global_views", float(n_global), prog_bar=False, sync_dist=True)

        self._log_view_visualizations(
            global_views=[view["x"] for view in views[:n_global]],
            local_views=[view["x"] for view in views[n_global:]],
            batch_idx=batch_idx,
        )

        if first_view_features is not None:
            with torch.no_grad():
                norm_features = F.normalize(first_view_features.detach(), dim=1)
                output_std = torch.std(norm_features, dim=0).mean().item()
                self.avg_output_std = 0.9 * self.avg_output_std + 0.1 * output_std
                self.log("train_ssl_std", self.avg_output_std, prog_bar=False, sync_dist=True)

        return loss

    def configure_optimizers(self) -> Tuple[list[Optimizer], list]:
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        trainer = getattr(self, "_trainer", None)
        max_epochs = int(getattr(trainer, "max_epochs", self.hparams.max_epochs))
        max_epochs = max(max_epochs, 1)
        warmup_epochs = int(self.hparams.warmup_epochs)

        if warmup_epochs > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=1.0 / warmup_epochs,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, max_epochs - warmup_epochs),
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        return [optimizer], [scheduler]
