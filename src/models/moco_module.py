import os
from collections.abc import Sequence
from typing import Tuple, Dict, Any

import kornia.augmentation as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch import Tensor
import numpy as np
import random
from lightning.pytorch import LightningModule
import hydra
from typing  import Dict, Any, Tuple

from src.transforms.normalize_mm import SingleSensorNormalizer

from .model_utils import instantiate_component

# ------------------------------------------------------------------------------
# Baseline augmentation (sensor normalization: same logic as MMNormalizer)
# ------------------------------------------------------------------------------
def moco_augmentations(size: int, normalizer: nn.Module) -> Tuple[nn.Module, nn.Module]:
    """
    Create the baseline MoCo augmentation pipeline without any color augmentations,
    using the given normalizer (SingleSensorNormalizer from sensor config).
    """
    max_scale = 0.2 if size < 32 else 1.0
    ks = (size // 10 // 2 * 2) + 1

    base_pipeline = [
        normalizer,
        K.RandomResizedCrop(size=(size, size), scale=(0.08, max_scale)),
        K.RandomGaussianBlur(kernel_size=(ks, ks), sigma=(0.1, 2), p=0.5),
        K.RandomHorizontalFlip(),
        K.RandomVerticalFlip(),
    ]
    aug = K.AugmentationSequential(*base_pipeline, data_keys=["input"])
    return aug, aug

# ------------------------------------------------------------------------------
# MoCo Module (v2 only)
# ------------------------------------------------------------------------------
class MoCoModule(LightningModule):
    def __init__(
        self,
        backbone_config: Dict[str, Any],
        sensor_config: Dict[str, Any],
        layers: int = 2,            # Fixed to 2 for v2.
        hidden_dim: int = 2048,      # Recommended for v2.
        output_dim: int = 128,       # Recommended for v2.
        lr: float = 0.001,
        warmup_epochs: int = 20,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        memory_bank_size: int = 65536,
        moco_momentum: float = 0.999,
        gather_distributed: bool = False,
        freeze_patch_embed: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.sensor_config = sensor_config
        self.backbone_config = backbone_config
        self.freeze_patch_embed = freeze_patch_embed
        self.temperature = temperature
        self.memory_bank_size = memory_bank_size
        self.moco_momentum = moco_momentum
        self.gather_distributed = gather_distributed
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Get the img size from the sensor config
        self.size = self.sensor_config["img_size"]
        # Same normalization logic as MMNormalizer (scale / harmonize / thermal / sar)
        self.normalizer = SingleSensorNormalizer(sensor_config=self.sensor_config)
        self.augmentation1, self.augmentation2 = moco_augmentations(self.size, self.normalizer)

        self._build_model()

        self.criterion = NTXentLoss(
            temperature=self.temperature, 
            memory_bank_size=(self.memory_bank_size, self.output_dim), 
            gather_distributed=self.gather_distributed,
        )
        self.avg_output_std = 0.0
        
        # Radiometric augmentation parameters
        self.radiometric_aug_p = 0.5
        self.radiometric_brightness_range = (0.8, 1.2)
        self.radiometric_bias_range = (-0.1, 0.1)

    def _build_model(self) -> None:
        """Builds the backbone, momentum backbone, and projection heads."""
        self.backbone = hydra.utils.instantiate(self.backbone_config)
        self.backbone_momentum = hydra.utils.instantiate(self.backbone_config)

        deactivate_requires_grad(self.backbone_momentum)

        if self.freeze_patch_embed:
            target_vit_module = None
            if hasattr(self.backbone, "vit_core") and hasattr(self.backbone.vit_core, "patch_embed"):
                target_vit_module = self.backbone.vit_core
            elif hasattr(self.backbone, "model") and hasattr(self.backbone.model, "patch_embed"):
                target_vit_module = self.backbone.model
            elif hasattr(self.backbone, "patch_embed"):
                target_vit_module = self.backbone
            
            if target_vit_module and hasattr(target_vit_module.patch_embed, "proj"):
                print(f"Freezing patch_embed projection in {target_vit_module.__class__.__name__}")
                target_vit_module.patch_embed.proj.weight.requires_grad = False
                if target_vit_module.patch_embed.proj.bias is not None:
                    target_vit_module.patch_embed.proj.bias.requires_grad = False
            elif self.freeze_patch_embed:
                print(
                    f"Warning: 'freeze_patch_embed' is True, but 'patch_embed.proj' was not found "
                    f"in backbone {self.backbone.__class__.__name__} or its potential submodules. "
                    f"Patch embedding was not frozen."
                )

        if hasattr(self.backbone, 'num_features'):
            num_backbone_features = self.backbone.num_features
        else:
            raise AttributeError(f"Cannot determine output features for backbone {self.backbone.__class__.__name__}. "
                                 f"It must have a 'num_features' attribute.")

        if not isinstance(num_backbone_features, int) or num_backbone_features <= 0:
            raise ValueError(f"Failed to get a valid positive integer for backbone output features. Got: {num_backbone_features}")

        self.projection_head = MoCoProjectionHead(
            num_backbone_features, self.hidden_dim, self.output_dim, self.layers, batch_norm=True
        )
        self.projection_head_momentum = MoCoProjectionHead(
            num_backbone_features, self.hidden_dim, self.output_dim, self.layers, batch_norm=True
        )
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the main encoder.
        Returns the projection (q) and the backbone features.
        """
        features = self.backbone(x)
        q = self.projection_head(features)
        return q, features

    def forward_momentum(self, x: Tensor) -> Tensor:
        """
        Forward pass through the momentum encoder.
        """
        features = self.backbone_momentum(x)
        q = self.projection_head_momentum(features)
        return q
    
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

    def _get_image_tensors(self, batch: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        """Get (x1, x2) from batch. Supports MM Zarr (sensor name key) or legacy image/image1/image2."""
        sensor_key = self.sensor_config.get("name")
        if sensor_key and sensor_key in batch and isinstance(batch[sensor_key], Tensor):
            x = batch[sensor_key].float()
            return x, x
        if "image1" in batch and "image2" in batch:
            return batch["image1"].float(), batch["image2"].float()
        x = batch["image"].float()
        return x, x

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        x1, x2 = self._get_image_tensors(batch)

        with torch.no_grad():
            x1 = self.augmentation1(x1)
            x2 = self.augmentation2(x2)
            # Apply radiometric augmentation AFTER normalization
            x1 = self._apply_radiometric_augmentation(x1)
            x2 = self._apply_radiometric_augmentation(x2)

        q, features = self.forward(x1)
        
        # Cosine momentum schedule: momentum increases from moco_momentum to 1.0 over training
        momentum = cosine_schedule(
            self.current_epoch,
            self.warmup_epochs,
            self.moco_momentum,
            1.0,
        )
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        
        with torch.no_grad():
            k = self.forward_momentum(x2)
        
        loss = self.criterion(q, k)

        # Compute mean normalized standard deviation of the backbone features.
        output = features.detach()
        output = F.normalize(output, dim=1)
        output_std = torch.std(output, dim=0).mean()
        self.avg_output_std = 0.9 * self.avg_output_std + 0.1 * output_std.item()

        self.log("train_ssl_std", self.avg_output_std)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self) -> Tuple[list[Optimizer], list]:
        """AdamW + warmup + CosineAnnealingLR (same pattern as DINO/MAE)."""
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