from typing import Tuple, Union, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import hydra
from ..backbones.base_backbone import BaseBackbone
from omegaconf import DictConfig, OmegaConf
# Import MAE modules from lightly.
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM

from .model_utils import instantiate_component
import numpy as np
import wandb

def percentile_clip_and_normalize(
    x: torch.Tensor,
    lower_q: float = 2.0,
    upper_q: float = 98.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Clip each image‐band to the [lower_q, upper_q] percentiles and rescale to [0,1].

    Args:
        x:        Tensor of shape (B, C, H, W)
        lower_q:  Lower percentile (e.g. 2.0 for 2%)
        upper_q:  Upper percentile (e.g. 98.0 for 98%)
        eps:      Small value to avoid division by zero

    Returns:
        Tensor of same shape, clipped & linearly scaled to [0,1]
    """
    B, C, H, W = x.shape
    x_out = torch.empty_like(x)
    for b in range(B):
        for c in range(C):
            # move to CPU / numpy for percentile calc
            band = x[b, c].detach().cpu().numpy().ravel()
            p_low, p_high = np.percentile(band, [lower_q, upper_q])
            # clamp & normalize
            img = x[b, c]
            img = torch.clamp(img, p_low, p_high)
            img = (img - p_low) / (p_high - p_low + eps)
            x_out[b, c] = img
    return x_out


####################################################### ########################
# MAEModule
###############################################################################
class MAEModule(LightningModule):
    def __init__(
        self,
        backbone_config: DictConfig,
        sensor_config: DictConfig,
        image_size: int = 128,
        decoder_dim: int = 512,
        mask_ratio: float = 0.75,
        weight_decay: float = 0.01,
        decoder_depth: int = 1,
        decoder_num_heads: int = 16,
        decoder_mlp_ratio: int = 4,
        decoder_dropout: float = 0.0,
        decoder_attention_dropout: float = 0.0,
        lr: float = 0.001,
        T_max: int = 100,
        eta_min: float = 0.0,
        norm_pix: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['backbone_config', 'sensor_config'])

        self.mask_ratio = mask_ratio
        self.lr = lr
        self.T_max = T_max
        self.eta_min = eta_min
        self.norm_pix = norm_pix
        self.sensor_config = sensor_config
        self.image_size = image_size
        self.weight_decay = weight_decay
        # Get necessary values from sensor_config (new YAML-style structure)
        # Expected keys:
        #   - num_bands: int
        #   - rgb_indices: list[int] (optional, for visualization)
        in_channels = int(self.sensor_config["num_bands"])
        self.rgb_indices: List[int] = list(self.sensor_config.get("rgb_indices", [0, 1, 2]))

        # Create the backbone.
        print(f"MAEModule: Instantiating backbone with config: {backbone_config}")

        vit = instantiate_component(config=backbone_config)
        
        
        self.vit_patch_size = backbone_config.patch_size

        # Create the encoder (using MaskedVisionTransformerTIMM) on the backbone. Let the backbone handle the positional embeddings.
        self.encoder = MaskedVisionTransformerTIMM(vit=vit, pos_embed_initialization="skip", weight_initialization="skip")
        
        self.sequence_length = self.encoder.sequence_length

        self.decoder = MAEDecoderTIMM(
                in_chans=in_channels,
                num_patches=vit.patch_embed.num_patches,
                patch_size=self.vit_patch_size,
                decoder_depth=decoder_depth,
                decoder_num_heads=decoder_num_heads,
                embed_dim=vit.embed_dim,
                decoder_embed_dim=decoder_dim,
                mlp_ratio=decoder_mlp_ratio,
                proj_drop_rate=decoder_dropout,
                attn_drop_rate=decoder_attention_dropout,
            )

        self.criterion = nn.MSELoss()

        

    def forward_encoder(self, images: Tensor, idx_keep: Tensor = None) -> Tensor:
        return self.encoder.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred


    def _get_images_from_batch(self, batch: Dict[str, Any]) -> Tensor:
        """
        Extract the single-sensor image tensor from the new multi-modal batch dict.

        Assumes the SpectralEarthMMZarrDataModule has already:
          - Selected a single timestamp per sample.
          - Applied all augmentations & normalization on-device.
        """
        # Prefer the explicitly configured sensor name if present.
        preferred_key = str(self.sensor_config.get("name", "")).upper()
        if preferred_key in batch and isinstance(batch[preferred_key], torch.Tensor):
            return batch[preferred_key]

        # Fallback: first 4D tensor key that looks like a sensor.
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.ndim == 4:
                return v

        raise KeyError("MAEModule could not find a 4D sensor tensor in batch.")

    def training_step(self, batch, batch_idx) -> Tensor:
        images = self._get_images_from_batch(batch).float()
        batch_size = images.shape[0]
        # Create a random token mask.
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        patches = utils.patchify(images, self.vit_patch_size)
        target = utils.get_at_index(patches, idx_mask - 1)

        if self.norm_pix:
            mean_patch = target.mean(dim=-1, keepdim=True)
            var_patch = target.var(dim=-1, keepdim=True)
            target = (target - mean_patch) / (var_patch + 1e-6).sqrt()
        loss = self.criterion(x_pred, target)
        self.log("train_loss", loss)

        current_epoch = self.current_epoch
        
        #if batch_idx % 200 == 0:
        if batch_idx == 0 and current_epoch % 10 == 0:
            num_images = min(8, images.shape[0])
            rgb_idx = self.rgb_indices

            # 1) undo patch-level normalization
            if self.norm_pix:
                x_pred = x_pred * (var_patch + 1e-6).sqrt() + mean_patch

            # 2) reconstruct back into [0,1] space
            recon = self.reconstruct_images_flat(
                x_pred, idx_mask, idx_keep, images, self.vit_patch_size
            )

            # 3) pick out the pseudo-RGB channels
            orig_rgb  = images[:num_images, rgb_idx, :, :]
            recon_rgb = recon[:num_images,        rgb_idx, :, :]

            # Print mean value of orig_rgb and recon_rgb
            print(f"Orig RGB mean: {orig_rgb.abs().mean()}, Recon RGB mean: {recon_rgb.abs().mean()}")

            # 4) clip to [2%,98%] for better contrast
            orig_rgb  = percentile_clip_and_normalize(orig_rgb,  lower_q=1.0, upper_q=99.0)
            recon_rgb = percentile_clip_and_normalize(recon_rgb, lower_q=1.0, upper_q=99.0)

            # 5) reverse sensor MinMax (if you still want raw-range values)
            #minv, maxv = self.sensor_config["min_val"], self.sensor_config["max_val"]
            #orig_rgb  = self.reverse_normalize_and_minmax(orig_rgb,  minv, maxv)
            #recon_rgb = self.reverse_normalize_and_minmax(recon_rgb, minv, maxv)

            # 6) permute & to numpy
            orig_np  = orig_rgb.permute(0,2,3,1).detach().cpu().numpy()
            recon_np = recon_rgb.permute(0,2,3,1).detach().cpu().numpy()

            # 7) log!
            wandb_imgs_orig  = [wandb.Image(img) for img in orig_np]
            wandb_imgs_recon = [wandb.Image(img) for img in recon_np]
            self.logger.experiment.log({
                "Original Images":      wandb_imgs_orig,
                "Reconstructed Images": wandb_imgs_recon
            }, step=self.global_step)
                
        return loss

    def reverse_normalize_and_minmax(self, image, min_val, max_val):
        image = image * (max_val - min_val) + min_val
        return image

    
    def min_max_normalize(self, images):
        # Apply min-max normalization to each image separately
        B, C, H, W = images.shape
        # First clip the 5% and 95% percentile values for each band per image
        for b in range(B):
            for c in range(C):
                min_val = images[b, c].flatten().kthvalue(int(0.02 * H * W)).values
                max_val = images[b, c].flatten().kthvalue(int(0.98 * H * W)).values
                images[b, c] = torch.clamp(images[b, c], min_val, max_val)

        # Normalize each image separately
        images_flat = images.view(B, C, -1)
        min_vals = images_flat.min(dim=2, keepdim=True)[0].view(B, C, 1, 1)
        max_vals = images_flat.max(dim=2, keepdim=True)[0].view(B, C, 1, 1)
        normalized_images = (images - min_vals) / (max_vals - min_vals)
        return normalized_images#.clamp(0, 1)  # Ensure values are in [0, 1]
    

    
    def patchify(self, images: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Converts a batch of input images into patches.

        Args:
            images:
                Images tensor with shape (batch_size, channels, height, width)
            patch_size:
                Patch size in pixels. Image width and height must be multiples of
                the patch size.

        Returns:
            Patches tensor with shape (batch_size, num_patches, channels * patch_size ** 2)
            where num_patches = image_width / patch_size * image_height / patch_size.

        """
        # N, C, H, W = (batch_size, channels, height, width)
        N, C, H, W = images.shape
        assert H == W and H % patch_size == 0

        patch_h = patch_w = H // patch_size
        num_patches = patch_h * patch_w
        patches = images.reshape(shape=(N, C, patch_h, patch_size, patch_w, patch_size))
        patches = torch.einsum("nchpwq->nhwpqc", patches)
        patches = patches.reshape(shape=(N, num_patches, patch_size**2 * C))
        return patches
    
    def unpatchify(self, x, in_chans, patch_size):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], in_chans, h * p, h * p))
        return imgs
    
    def reconstruct_images_flat(self, x_pred, idx_mask, idx_keep, images, patch_size):
        batch_size, C, H, W = images.shape

        # Calculate total number of patches
        num_patches_H = H // patch_size
        num_patches_W = W // patch_size
        num_patches = num_patches_H * num_patches_W

        # Initialize an empty tensor for the reconstructed images
        reconstructed_images = torch.zeros(batch_size, C, H, W, device=x_pred.device)

        for i in range(batch_size):
            # Flatten original patches for the i-th image
            image_patches_flat = self.patchify(images[i].unsqueeze(0), patch_size)[0]

            # Initialize a flat tensor for reconstructed patches
            image_reconstructed_flat = torch.zeros_like(image_patches_flat)

            # Adjust indices for masked and unmasked patches
            idx_mask_adj = idx_mask[i] - 1
            idx_keep_adj = idx_keep[i] - 1

            # Place unmasked (original) patches directly into the reconstructed array
            image_reconstructed_flat[idx_keep_adj, :] = image_patches_flat[idx_keep_adj, :]

            # Place predicted patches for the masked areas
            image_reconstructed_flat[idx_mask_adj, :] = x_pred[i].to(image_reconstructed_flat.dtype)

            reconstructed_images[i] = self.unpatchify(image_reconstructed_flat.unsqueeze(0), C, patch_size) 

        return reconstructed_images


    def configure_optimizers(self) -> Tuple[list[Optimizer], list]:
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.eta_min)
        return [optimizer], [scheduler]
