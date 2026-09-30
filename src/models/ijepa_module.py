import math
import torch
from torch import Tensor
from typing import List, Tuple, Optional

class JEPAMaskGenerator:
    """
    Generates I-JEPA masks on-the-fly. 
    """
    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        enc_mask_scale: Tuple[float, float] = (0.5, 1.0),
        pred_mask_scale: Tuple[float, float] = (0.2, 0.5),
        aspect_ratio: Tuple[float, float] = (0.3, 3.0),
        nenc: int = 1,
        npred: int = 4,
        min_keep: int = 4,
        allow_overlap: bool = False,
    ):
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.patch_size = patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height: h -= 1
        while w >= self.width: w -= 1
        return (h, w)

    def _sample_block_mask(self, b_size, generator, acceptable_regions=None):
        h, w = b_size
        tries = 0
        timeout = 20
        valid_mask = False
        
        mask = torch.zeros((self.height, self.width), dtype=torch.int32)
        
        while not valid_mask:
            # FIX: Use the generator here!
            top = torch.randint(0, self.height - h, (1,), generator=generator)
            left = torch.randint(0, self.width - w, (1,), generator=generator)
            
            mask.fill_(0)
            mask[top : top + h, left : left + w] = 1
            
            if acceptable_regions is not None:
                for k in range(max(len(acceptable_regions) - tries, 0)):
                    mask *= acceptable_regions[k]
            
            non_zero = torch.nonzero(mask.flatten()).squeeze()
            valid_mask = len(non_zero) > self.min_keep if non_zero.ndim > 0 else False
            
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = 20
        
        mask_complement = torch.ones_like(mask)
        mask_complement[top : top + h, left : left + w] = 0
        
        return non_zero, mask_complement

    def __call__(self, batch_size: int, device: torch.device, seed: int = None) -> Tuple[List[Tensor], List[Tensor]]:
        # Create a local generator for this batch
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        else:
            g.manual_seed(torch.randint(0, 10**9, (1,)).item())

        p_size = self._sample_block_size(g, self.pred_mask_scale, self.aspect_ratio)
        e_size = self._sample_block_size(g, self.enc_mask_scale, (1.0, 1.0))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width

        # No fork_rng needed anymore, we pass 'g' everywhere
        for _ in range(batch_size):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                # FIX: Pass generator 'g'
                mask, mask_C = self._sample_block_mask(p_size, generator=g)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            
            acceptable_regions = masks_C if not self.allow_overlap else None
            
            masks_e = []
            for _ in range(self.nenc):
                # FIX: Pass generator 'g'
                mask, _ = self._sample_block_mask(e_size, generator=g, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
                
            collated_masks_pred.append(masks_p)
            collated_masks_enc.append(masks_e)

        # Collate logic
        def collate_list(mask_list, min_k):
            n_masks_per_sample = len(mask_list[0])
            out_list = []
            for i in range(n_masks_per_sample):
                stacked = torch.stack([m[i][:min_k] for m in mask_list])
                out_list.append(stacked.to(device, non_blocking=True))
            return out_list

        final_masks_pred = collate_list(collated_masks_pred, min_keep_pred)
        final_masks_enc = collate_list(collated_masks_enc, min_keep_enc)

        return final_masks_enc, final_masks_pred


import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from omegaconf import DictConfig

from lightning.pytorch import LightningModule

# Lightly Imports
from lightly.models import utils
from lightly.models.modules.ijepa import IJEPABackbone, IJEPAPredictor
from lightly.utils.scheduler import cosine_schedule

# Custom Imports
from .model_utils import instantiate_component

def percentile_clip_and_normalize(
    x: torch.Tensor,
    lower_q: float = 2.0,
    upper_q: float = 98.0,
    eps: float = 1e-6
) -> torch.Tensor:
    # 1. Capture original dimensionality
    original_ndim = x.ndim
    
    # 2. Ensure 4D for batch processing
    if original_ndim == 3:
        x = x.unsqueeze(0)
    
    B, C, H, W = x.shape
    x_out = torch.empty_like(x)
    
    # 3. Process
    for b in range(B):
        for c in range(C):
            band = x[b, c].detach().cpu().numpy().ravel()
            p_low, p_high = np.percentile(band, [lower_q, upper_q])
            img = x[b, c]
            img = torch.clamp(img, p_low, p_high)
            img = (img - p_low) / (p_high - p_low + eps)
            x_out[b, c] = img
            
    # 4. Restore original dimensionality
    if original_ndim == 3:
        return x_out.squeeze(0)
        
    return x_out

# ---------------------------------------------------------------------------
# [Insert JEPAMaskGenerator Class from above here]
# ---------------------------------------------------------------------------

class IJEPAModule(LightningModule):
    def __init__(
        self,
        backbone_config: DictConfig,
        sensor_config: DictConfig,
        # Mask Configuration
        patch_size: int = 16, # Ideally this matches backbone, but we can override
        enc_mask_scale: Tuple[float, float] = (0.75, 1.0),
        pred_mask_scale: Tuple[float, float] = (0.15, 0.5),
        # Predictor Config
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        predictor_num_heads: int = 12,
        # Optimizer
        lr: float = 1.5e-4,
        weight_decay: float = 0.04,
        max_epochs: int = 300,
        warmup_epochs: int = 40,
        ema_momentum_start: float = 0.996,
        ema_momentum_end: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['backbone_config', 'sensor_config'])
        
        self.sensor_config = sensor_config
        self.image_size = sensor_config["img_size"]
        self.rgb_indices: List[int] = list(self.sensor_config.get("rgb_indices", [0, 1, 2]))

        # 1. Instantiate Backbone
        print(f"IJEPAModule: Instantiating backbone with config: {backbone_config}")
        vit_backbone = instantiate_component(config=backbone_config)
        
        self.backbone_dim = vit_backbone.hidden_dim 
        # Ensure patch size consistency
        if hasattr(vit_backbone, 'patch_size'):
            self.patch_size = vit_backbone.patch_size
        else:
            self.patch_size = patch_size

        # 2. Setup Student
        self.student_encoder = IJEPABackbone.from_vit(vit_backbone)
        
        # 3. Setup Predictor
        num_patches = (self.image_size // self.patch_size) ** 2
        self.predictor = IJEPAPredictor(
            seq_length=num_patches,
            num_layers=predictor_depth,
            num_heads=predictor_num_heads,
            hidden_dim=predictor_embed_dim,
            predictor_embed_dim=predictor_embed_dim, 
            num_patches=num_patches,
            mlp_dim=self.backbone_dim,
            dropout=0.0,
            attention_dropout=0.0,
        )

        # 4. Setup Teacher
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

        # 5. Setup Mask Generator (Internal)
        self.mask_generator = JEPAMaskGenerator(
            input_size=(self.image_size, self.image_size),
            patch_size=self.patch_size,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            # Defaults for others can be exposed in __init__ if needed
        )

        self.criterion = nn.SmoothL1Loss()

    def _get_images_from_batch(self, batch: Dict[str, Any]) -> Tensor:
        preferred_key = str(self.sensor_config.get("name", "")).upper()
        if preferred_key in batch and isinstance(batch[preferred_key], torch.Tensor):
            return batch[preferred_key]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.ndim == 4:
                return v
        raise KeyError("IJEPAModule could not find a 4D sensor tensor in batch.")

    def on_train_batch_start(self, batch, batch_idx):
        m = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.hparams.ema_momentum_start,
            end_value=self.hparams.ema_momentum_end,
        )
        utils.update_momentum(self.student_encoder, self.teacher_encoder, m=m)

    def forward_target(self, imgs, masks_enc, masks_pred):
        with torch.no_grad():
            h = self.teacher_encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),)) 
            B = len(imgs)
            h = utils.apply_masks(h, masks_pred)
            h = utils.repeat_interleave_batch(h, B, repeat=len(masks_enc))
            return h

    def training_step(self, batch, batch_idx):
        # 1. Extract Images (Your DataModule returns a Dict)
        imgs = self._get_images_from_batch(batch).float()
        
        # 2. GENERATE MASKS ON THE FLY
        # This happens after DataModule GPU augmentation, so sizes are correct.
        rank = self.trainer.global_rank
        seed = batch_idx + (self.current_epoch * 100000) + (rank * 10000)
        
        # 3. Generate Masks
        masks_enc, masks_pred = self.mask_generator(
            batch_size=imgs.shape[0], 
            device=imgs.device,
            seed=seed  # <--- Pass the seed here
        )

        # 3. I-JEPA Logic
        z_context = self.student_encoder(imgs, masks_enc)
        z_pred = self.predictor(z_context, masks_enc, masks_pred)
        h_target = self.forward_target(imgs, masks_enc, masks_pred)

        # 4. Loss
        loss = self.criterion(z_pred, h_target)
        self.log("train_loss", loss, prog_bar=True)

        #if batch_idx == 0 and self.current_epoch % 10 == 0:
        if batch_idx % 100 == 0:
            self._log_visualization(imgs, masks_enc, masks_pred)

        return loss

    def _log_visualization(self, imgs, masks_enc, masks_pred):
        img = imgs[0]
        # 1. Prepare Base RGB Image
        img_vis = img[self.rgb_indices, :, :]
        img_vis = percentile_clip_and_normalize(img_vis)
        img_np = img_vis.permute(1, 2, 0).cpu().numpy()

        # 2. Helper to convert Indices -> Density Map -> Upscaled Image
        grid_h = grid_w = self.image_size // self.patch_size
        num_patches = grid_h * grid_w

        def tokens_to_mask_overlay(indices):
            mask_flat = torch.zeros(num_patches, device=indices.device)
            mask_flat[indices.long()] = 1.0
            mask_grid = mask_flat.reshape(grid_h, grid_w)
            mask_img = F.interpolate(
                mask_grid.view(1, 1, grid_h, grid_w),
                size=(self.image_size, self.image_size),
                mode='nearest'
            ).squeeze().cpu().numpy()
            return mask_img

        # 3. Build panels: [Original] + [Context_0, ...] + [Target_0, Target_1, ...]
        panels = []

        # Original
        panels.append(img_np.copy())

        # All context views (patches the student sees)
        for i in range(len(masks_enc)):
            mask_enc_indices = masks_enc[i][0]
            context_map = tokens_to_mask_overlay(mask_enc_indices)
            context_vis = img_np.copy()
            context_vis[context_map == 0] = context_vis[context_map == 0] * 0.25
            panels.append(context_vis)

        # All target views (patches to predict)
        for i in range(len(masks_pred)):
            mask_pred_indices = masks_pred[i][0]
            target_map = tokens_to_mask_overlay(mask_pred_indices)
            target_vis = img_np.copy()
            target_mask_bool = (target_map == 1)
            target_vis[target_mask_bool, 0] = np.clip(target_vis[target_mask_bool, 0] + 0.5, 0, 1)
            target_vis[target_mask_bool, 1] *= 0.5
            target_vis[target_mask_bool, 2] *= 0.5
            panels.append(target_vis)

        # 4. Concatenate horizontally with thin white separators (2px)
        sep_w = 2
        sep = np.ones((self.image_size, sep_w, 3), dtype=img_np.dtype)
        row_parts = [panels[0]]
        for p in panels[1:]:
            row_parts.append(sep)
            row_parts.append(p)
        combined = np.concatenate(row_parts, axis=1)

        labels = ["Original"] + [f"Context_{i}" for i in range(len(masks_enc))] + [f"Target_{i}" for i in range(len(masks_pred))]
        caption = " | ".join(labels)
        if self.logger:
            self.logger.experiment.log({
                "viz/masks_row": [wandb.Image(combined, caption=caption)]
            }, step=self.global_step)

    def configure_optimizers(self) -> Tuple[list[Optimizer], list]:
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return [optimizer], [scheduler]