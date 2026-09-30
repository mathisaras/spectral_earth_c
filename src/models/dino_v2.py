import copy
import random
from typing import Dict, Any, Tuple, Optional, List
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import kornia.augmentation as K
import hydra
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from hydra.errors import InstantiationException

# Lightly Imports
from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules import MaskedVisionTransformerTIMM
from lightly.models.utils import (
    random_block_mask,
    update_drop_path_rate,
    update_momentum,
    deactivate_requires_grad,
)
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule

# -----------------------------------------------------------------------------
# 1. VERSION COMPATIBILITY
# -----------------------------------------------------------------------------
try:
    from lightly.models.modules import DINOv2ProjectionHead
except ImportError:
    from lightly.models.modules import DINOProjectionHead as DINOv2ProjectionHead

# Custom Imports
from src.transforms.normalize_mm import SingleSensorNormalizer
from .model_utils import instantiate_component
from .hiera_ssl_adapter import HieraMaskedBackboneAdapter, is_hiera_style_backbone

# -----------------------------------------------------------------------------
# 2. HELPER CLASSES
# -----------------------------------------------------------------------------
class DINOv2Head(nn.Module):
    def __init__(self, dino_head: nn.Module, ibot_head: nn.Module) -> None:
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head

def percentile_clip_and_normalize(
    x: torch.Tensor,
    lower_q: float = 2.0,
    upper_q: float = 98.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Clip each image‐band to percentiles and rescale to [0,1].
    Args: x: Tensor (C, H, W)
    """
    C, H, W = x.shape
    x_out = torch.empty_like(x)
    for c in range(C):
        band = x[c].detach().cpu().numpy().ravel()
        # removing NaNs for percentile calc if necessary
        band = band[~np.isnan(band)]
        if len(band) > 0:
            p_low, p_high = np.percentile(band, [lower_q, upper_q])
        else:
            p_low, p_high = 0.0, 1.0
            
        img = x[c]
        img = torch.clamp(img, p_low, p_high)
        img = (img - p_low) / (p_high - p_low + eps)
        x_out[c] = img
    return x_out
# -----------------------------------------------------------------------------
# 3. MAIN MODULE
# -----------------------------------------------------------------------------
class DINOv2Module(LightningModule):
    def __init__(
        self,
        backbone_config: Dict[str, Any],
        sensor_config: Dict[str, Any],
        # Model Params
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        # Training Params
        lr: float = 2.0e-4, 
        min_lr: float = 1.0e-6,
        max_epochs: int = 100,
        warmup_epochs: int = 20,
        weight_decay: float = 0.04,
        weight_decay_end: float = 0.4,
        freeze_last_layer_epochs: int = 1,
        # Multicrop
        n_local_views: int = 6,
        global_crop_scale: Tuple[float, float] = (0.32, 1.0),
        local_crop_scale: Tuple[float, float] = (0.05, 0.32),
        # Schedules
        student_temp: float = 0.1,
        teacher_temp_warmup: float = 0.04,
        teacher_temp: float = 0.07,
        teacher_temp_warmup_steps: int = 30000, 
        # Radiometric
        radiometric_aug_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.sensor_config = sensor_config
        self.size = sensor_config["img_size"]
        self.in_channels = self.sensor_config.get("num_bands", 3)
        
        self.n_local_views = n_local_views
        self.max_epochs = max_epochs
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.weight_decay_end = weight_decay_end
        
        # ---------------------------------------------------------------------
        # Augmentations
        # ---------------------------------------------------------------------
        self.normalizer = SingleSensorNormalizer(sensor_config=self.sensor_config)
        self.radiometric_aug_p = radiometric_aug_p
        self.radiometric_brightness_range = (0.8, 1.2)
        self.radiometric_bias_range = (-0.1, 0.1)

        # Kernel sizes for blur (Global vs Local)
        global_ks = self.size // 10 // 2 * 2 + 1
        local_size =  self.size // 4
        local_ks = local_size // 10 // 2 * 2 + 1

        self.global_augmentation = K.AugmentationSequential(
            self.normalizer,
            K.RandomResizedCrop(size=(self.size, self.size), scale=global_crop_scale),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.RandomGaussianBlur(kernel_size=(global_ks, global_ks), sigma=(0.1, 2.0), p=0.5),
            data_keys=["input"]
        )

        self.local_augmentation = K.AugmentationSequential(
            self.normalizer,        
            K.RandomResizedCrop(size=(local_size, local_size), scale=local_crop_scale),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.RandomGaussianBlur(kernel_size=(local_ks, local_ks), sigma=(0.1, 2.0), p=0.5),
            data_keys=["input"]
        )

        # ---------------------------------------------------------------------
        # Backbones
        # ---------------------------------------------------------------------
        try:
            vit_backbone = hydra.utils.instantiate(backbone_config, dynamic_img_size=True)
        except (TypeError, InstantiationException) as err:
            # Hiera backbones don't accept dynamic_img_size; retry without it.
            if "dynamic_img_size" in str(err):
                print("[DINOv2] Warning: Backbone does not accept dynamic_img_size. Local crops might fail.")
                vit_backbone = hydra.utils.instantiate(backbone_config)
            else:
                raise

        use_hiera_adapter = is_hiera_style_backbone(vit_backbone)
        if use_hiera_adapter:
            self.student_backbone = HieraMaskedBackboneAdapter(
                backbone=vit_backbone,
                include_cls_token=True,
                global_pool="",
            )
        else:
            self.student_backbone = MaskedVisionTransformerTIMM(
                vit=vit_backbone,
                pos_embed_initialization="skip",
            )

        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        deactivate_requires_grad(self.teacher_backbone)

        if hasattr(self.student_backbone, "embed_dim"):
            self.embed_dim = self.student_backbone.embed_dim
        elif hasattr(vit_backbone, "embed_dim"):
            self.embed_dim = vit_backbone.embed_dim
        elif hasattr(vit_backbone, "num_features"):
            self.embed_dim = vit_backbone.num_features
        else:
            raise ValueError("Backbone must expose an embedding dimension.")

        # Fixed-size backbones (e.g. Hiera) cannot process smaller local crops.
        if self.n_local_views > 0 and not getattr(vit_backbone, "dynamic_img_size", False):
            print(
                "[DINOv2] Info: local crops disabled because backbone does not support dynamic image sizes."
            )
            self.n_local_views = 0

        # ---------------------------------------------------------------------
        # Heads
        # ---------------------------------------------------------------------
        self.student_head = DINOv2Head(
            dino_head=DINOv2ProjectionHead(self.embed_dim, hidden_dim, bottleneck_dim, output_dim),
            ibot_head=DINOv2ProjectionHead(self.embed_dim, hidden_dim, bottleneck_dim, output_dim),
        )
        self.teacher_head = DINOv2Head(
            dino_head=DINOv2ProjectionHead(self.embed_dim, hidden_dim, bottleneck_dim, output_dim),
            ibot_head=DINOv2ProjectionHead(self.embed_dim, hidden_dim, bottleneck_dim, output_dim),
        )
        deactivate_requires_grad(self.teacher_head)

        # ---------------------------------------------------------------------
        # Losses
        # ---------------------------------------------------------------------
        self.dino_criterion = DINOLoss(
            output_dim=output_dim,
            student_temp=student_temp
        )
        self.ibot_criterion = IBOTPatchLoss(output_dim=output_dim)
        self.koleo_criterion = KoLeoLoss()

        self.avg_output_std = 0.0

    # -------------------------------------------------------------------------
    # Forward Methods
    # -------------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Inference forward pass (Student DINO head)."""
        features = self.student_backbone.encode(x)
        cls_token = features[:, 0]
        return self.student_head.dino_head(cls_token)

    def forward_teacher(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Teacher forward: Returns (CLS token, All tokens)."""
        features = self.teacher_backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(self, x: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Student forward.
        If mask is provided (B, N_tokens), masked tokens are replaced with [MASK].
        Sequence length is preserved.
        """
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------
    def _apply_radiometric_augmentation(self, tensor: Tensor) -> Tensor:
        if random.random() > self.radiometric_aug_p:
            return tensor
        device = tensor.device
        brightness = torch.empty(1, device=device).uniform_(*self.radiometric_brightness_range)
        bias = torch.empty(1, device=device).uniform_(*self.radiometric_bias_range)
        return tensor * brightness + bias

    def _get_image_tensors(self, batch: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        if "image1" in batch and "image2" in batch:
            return batch["image1"].float(), batch["image2"].float()
        
        sensor_name = self.sensor_config.get("name")
        if sensor_name:
            if sensor_name in batch:
                x = batch[sensor_name].float()
                return x, x
            if sensor_name.upper() in batch:
                x = batch[sensor_name.upper()].float()
                return x, x
        
        for k, v in batch.items():
            if isinstance(v, Tensor) and v.ndim == 4:
                if k in ['patch_id', 'mask', 'labels']: continue
                return v.float(), v.float()
        raise ValueError(f"Could not find valid image tensor in batch. Keys: {list(batch.keys())}")

    def _log_visualization(self, images: Tensor, block_mask: Tensor):
        """
        Visualizes the first image in the batch and its iBOT mask.
        """
        # 1. Select first sample
        img = images[0].detach().cpu() # (C, H, W)
        mask = block_mask[0].detach().cpu().float() # (H_grid, W_grid)

        # 2. Calculate Coverage (Debug Statistic)
        coverage = mask.mean().item()
        self.log("val/mask_coverage", coverage) # Check this in WandB! Should be ~0.4 to 0.6

        # 3. Prepare RGB
        rgb_indices = self.sensor_config.get("rgb_indices", [0, 1, 2])
        if max(rgb_indices) >= img.shape[0]:
            rgb_indices = [0, 1, 2] if img.shape[0] > 2 else [0]
            
        img_vis = img[rgb_indices, :, :]
        if img_vis.shape[0] == 1: img_vis = img_vis.repeat(3, 1, 1)
        
        img_vis = percentile_clip_and_normalize(img_vis) # (3, H, W) in [0,1]
        
        # 4. Upscale Mask
        # Use nearest to keep sharp block edges, but output is float 0.0 or 1.0
        mask_upscaled = F.interpolate(
            mask.view(1, 1, *mask.shape),
            size=(img.shape[1], img.shape[2]),
            mode='nearest'
        ).squeeze() 

        # 5. Create Overlay
        img_np = img_vis.permute(1, 2, 0).numpy()
        mask_np = mask_upscaled.numpy()
        
        overlay = img_np.copy()
        # Robust threshold: anything > 0.5 is masked
        is_masked = mask_np > 0.5
        
        # Apply red tint to masked areas
        if is_masked.any():
            overlay[is_masked] = overlay[is_masked] * 0.4 + np.array([0.8, 0.0, 0.0]) * 0.6

        # 6. Log
        if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log({
                "val/input_rgb": [wandb.Image(img_np, caption=f"Input (Coverage: {coverage:.2f})")],
                "val/overlay":   [wandb.Image(overlay, caption="Red = Dropped")]
            })

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        # 1. Inputs
        x1_raw, x2_raw = self._get_image_tensors(batch)
        
        # 2. Augmentations
        with torch.no_grad():
            global_1 = self._apply_radiometric_augmentation(self.global_augmentation(x1_raw))
            global_2 = self._apply_radiometric_augmentation(self.global_augmentation(x2_raw))
            global_views = [global_1, global_2]
            
            local_views = []
            for _ in range(self.n_local_views):
                lx = self._apply_radiometric_augmentation(self.local_augmentation(x1_raw))
                local_views.append(lx)

        all_global_views = torch.cat(global_views)
        all_local_views = torch.cat(local_views) if local_views else None
        
        # 3. Mask Generation
        B_global = all_global_views.shape[0]
        # Get grid size for the global view resolution
        H_grid, W_grid = self.teacher_backbone.vit.patch_embed.grid_size
        
        
        # block_mask: (B, H, W). True = Masked/Replaced.
        block_mask = random_block_mask(size=(B_global, H_grid, W_grid), device=all_global_views.device, min_num_masks_per_block=8, min_image_mask_ratio=0.2)
        
        # Create full mask for Backbone: (B, N_patches+1).
        # We prepend FALSE for the CLS token (never masked).
        mask_with_cls = torch.cat([
            torch.zeros(B_global, 1, dtype=torch.bool, device=block_mask.device), 
            block_mask.flatten(start_dim=1)
        ], dim=1)

        # 4. Teacher Forward (Full Global Views)
        with torch.no_grad():
            teacher_cls_token, teacher_features = self.forward_teacher(all_global_views)
            teacher_cls_out = self.teacher_head.dino_head(teacher_cls_token)
            # Pass full sequence to head. We will filter later.
            teacher_ibot_full = self.teacher_head.ibot_head(teacher_features)

        # 5. Student Forward (Global Views with Masking)
        # Pass the mask_with_cls (1025) to avoid mismatch error.
        # Backbone replaces masked tokens with [MASK]. Output is (B, 1025, D).
        student_global_cls_token, student_global_features = self.forward_student(
            all_global_views, mask=mask_with_cls
        )
        student_global_cls_out = self.student_head.dino_head(student_global_cls_token)
        # Pass full sequence to head.
        student_ibot_full = self.student_head.ibot_head(student_global_features)

        # 5b. Local Views (optional for fixed-size backbones)
        if all_local_views is not None:
            student_local_cls_token, _ = self.forward_student(all_local_views, mask=None)
            student_local_cls_out = self.student_head.dino_head(student_local_cls_token)
            student_cls_all = torch.cat([student_global_cls_out, student_local_cls_out])
            student_out_chunked = student_cls_all.chunk(2 + self.n_local_views)
        else:
            student_out_chunked = student_global_cls_out.chunk(2)

        # 6. Prepare iBOT Loss Inputs
        # We remove the CLS token from projections (index 0) and the mask (index 0)
        # because iBOT Patch loss is calculated on patches only.
        
        # patch_mask: (B, N_patches). True where [MASK] token is.
        patch_mask = mask_with_cls[:, 1:] 
        
        # features: (B, N_patches, D)
        teacher_patches = teacher_ibot_full[:, 1:, :] 
        student_patches = student_ibot_full[:, 1:, :] 

        # Filter: Select only the masked tokens for loss calculation
        # Teacher provides target for masked spots. Student provides prediction at masked spots.
        teacher_masked_probs = teacher_patches[patch_mask]
        student_masked_preds = student_patches[patch_mask]

        # 7. Losses
        # A. DINO Loss
        teacher_out_chunked = teacher_cls_out.chunk(2)

        teacher_temp = linear_warmup_schedule(
            step=self.trainer.global_step,
            warmup_steps=self.hparams.teacher_temp_warmup_steps,
            start_value=self.hparams.teacher_temp_warmup,
            end_value=self.hparams.teacher_temp,
        )

        dino_loss = self.dino_criterion(
            teacher_out=teacher_out_chunked,
            student_out=student_out_chunked,
            epoch=self.current_epoch,
            teacher_temp=teacher_temp
        )

        # B. iBOT Loss
        # We pass the flattened, filtered tensors.
        # Important: We must also pass the 3D block_mask for normalization inside the loss.
        ibot_loss = self.ibot_criterion(
            teacher_out=teacher_masked_probs,
            student_out=student_masked_preds,
            mask=block_mask, # (B, H, W) -- used to count masked patches per image
            teacher_temp=teacher_temp,
        )

        # C. KoLeo Loss
        koleo_loss = 0.0
        if student_global_cls_token.shape[0] > 1:
            for t in student_global_cls_token.chunk(2):
                koleo_loss += self.koleo_criterion(t)
            koleo_loss = 0.1 * koleo_loss

        total_loss = dino_loss + ibot_loss + koleo_loss

        # 8. Logging
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_dino_loss", dino_loss)
        self.log("train_ibot_loss", ibot_loss)
        self.log("train_koleo_loss", koleo_loss)
        self.log("teacher_temp", teacher_temp)

        with torch.no_grad():
            norm_features = F.normalize(student_global_cls_token, dim=1)
            output_std = torch.std(norm_features, dim=0).mean().item()
            self.avg_output_std = 0.9 * self.avg_output_std + 0.1 * output_std
            self.log("train_ssl_std", self.avg_output_std)
        
        if batch_idx == 0 and (self.current_epoch % 10 == 0): # Log every epoch
        #if batch_idx % 100 == 0:
             # Visualize the first global view and its corresponding mask
             # all_global_views shape: (2*B, C, H, W)
             # block_mask shape: (2*B, H_grid, W_grid)
             self._log_visualization(all_global_views, block_mask)

        return total_loss

    # -------------------------------------------------------------------------
    # Optimization Hooks
    # -------------------------------------------------------------------------
    def on_train_batch_end(self, outputs, batch, batch_idx):
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.996,
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        if hasattr(self.student_backbone.vit, "drop_path_rate"):
            update_drop_path_rate(
                self.student_backbone.vit,
                drop_path_rate=0.1, 
                mode="uniform",
            )
    
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if self.current_epoch < self.hparams.freeze_last_layer_epochs:
            for param_group in optimizer.param_groups:
                if "last_layer" in param_group and param_group["last_layer"]:
                    param_group["lr"] = 0.0

        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.hparams.weight_decay,
            end_value=self.hparams.weight_decay_end,
        )
        for group in optimizer.param_groups:
            if group.get("weight_decay_scheduler", False):
                group["weight_decay"] = weight_decay

    def configure_optimizers(self) -> Tuple[list[Optimizer], list]:
        regularized = []
        not_regularized = []
        last_layer_params = []

        for name, param in self.student_backbone.named_parameters():
            if not param.requires_grad: continue
            if name.endswith(".bias") or "norm" in name or "gamma" in name:
                not_regularized.append(param)
            else:
                regularized.append(param)

        for name, param in self.student_head.named_parameters():
            if not param.requires_grad:
                continue
            if "last_layer" in name:
                last_layer_params.append(param)
            elif name.endswith(".bias") or "norm" in name or "gamma" in name:
                not_regularized.append(param)
            else:
                regularized.append(param)

        param_groups = [
            {"params": regularized, "weight_decay_scheduler": True},
            {"params": not_regularized, "weight_decay": 0.0, "weight_decay_scheduler": False},
            {"params": last_layer_params, "weight_decay_scheduler": True, "last_layer": True}
        ]

        optimizer = AdamW(param_groups, lr=self.lr)

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1e-4, total_iters=self.warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=self.min_lr),
            ],
            milestones=[self.warmup_epochs],
        )
        return [optimizer], [lr_scheduler]
