"""
Multi-Modal Augmentations for Spectral Earth data.

Supports:
- Spatial augmentations (flip, crop, resize)
- Radiometric augmentations (brightness, contrast/bias)
- Consistency options: apply same or different transforms per sensor
- Probability-based application

All augmentations use pure PyTorch to avoid Kornia's type casting issues.
"""

import random
from typing import Dict, Any, Tuple, Optional, List, Set

import torch
import torch.nn.functional as F


class SpatialAugmentation:
    """
    Spatial augmentations for multi-modal Earth observation data.
    
    Supports:
    - Random horizontal flip
    - Random vertical flip
    - Random resized crop
    
    Args:
        p: Probability of applying augmentation (0 = never, 1 = always)
        horizontal_flip: Enable horizontal flipping
        vertical_flip: Enable vertical flipping
        random_crop: Enable random resized cropping
        crop_scale: Scale range for random crop (min, max) as fraction of original
        output_size: Output size after crop. If None, resize back to original size.
        consistent: If True, apply same transform to all sensors.
                   If False, each sensor gets independent random transforms.
    
    Note:
        When consistent=True, the same flip/crop parameters are used for all sensors.
        This is important for multi-modal learning where spatial alignment matters.
        When consistent=False, each sensor is augmented independently (useful for
        regularization but breaks spatial correspondence).
    """
    
    METADATA_KEYS: Set[str] = {'patch_id', '_group_name', '_available_sensors'}
    
    def __init__(
        self,
        p: float = 0.5,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        random_crop: bool = False,
        crop_scale: Tuple[float, float] = (0.4, 1.0),
        output_size: Optional[int] = None,
        consistent: bool = True,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if not 0.0 < crop_scale[0] <= crop_scale[1] <= 1.0:
            raise ValueError(f"crop_scale must satisfy 0 < min <= max <= 1, got {crop_scale}")
        
        self.p = p
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_crop = random_crop
        self.crop_scale = crop_scale
        self.output_size = output_size
        self.consistent = consistent
    
    def _get_sensor_keys(self, batch: Dict[str, Any]) -> List[str]:
        """Extract sensor keys from batch."""
        return [
            k for k in batch.keys()
            if k not in self.METADATA_KEYS
            and '_mask' not in k
            and not k.startswith('_')
            and isinstance(batch[k], torch.Tensor)
            and batch[k].ndim == 4  # (B, C, H, W)
        ]
    
    def _sample_flip_params(self) -> Tuple[bool, bool]:
        """Sample flip parameters."""
        flip_h = self.horizontal_flip and random.random() < 0.5
        flip_v = self.vertical_flip and random.random() < 0.5
        return flip_h, flip_v
    
    def _sample_crop_params(self, H: int, W: int) -> Tuple[int, int, int, int]:
        """Sample random crop parameters (top, left, crop_h, crop_w)."""
        scale = random.uniform(self.crop_scale[0], self.crop_scale[1])
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)
        return top, left, crop_h, crop_w
    
    def _apply_flip(self, tensor: torch.Tensor, flip_h: bool, flip_v: bool) -> torch.Tensor:
        """Apply flip to tensor (B, C, H, W)."""
        if flip_h:
            tensor = torch.flip(tensor, dims=[-1])
        if flip_v:
            tensor = torch.flip(tensor, dims=[-2])
        return tensor
    
    def _apply_crop(
        self, 
        tensor: torch.Tensor, 
        top: int, 
        left: int, 
        crop_h: int, 
        crop_w: int,
        original_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Apply crop and optionally resize back to original size."""
        B, C, H, W = tensor.shape
        
        # Scale crop params if tensor has different size than reference
        if original_size is not None:
            ref_H, ref_W = original_size
            scale_h = H / ref_H
            scale_w = W / ref_W
            top = int(top * scale_h)
            left = int(left * scale_w)
            crop_h = int(crop_h * scale_h)
            crop_w = int(crop_w * scale_w)
            # Clamp to valid range
            top = min(top, H - crop_h)
            left = min(left, W - crop_w)
        
        # Crop
        cropped = tensor[:, :, top:top+crop_h, left:left+crop_w]
        
        # Resize back to original if requested
        out_size = self.output_size or H
        if cropped.shape[-1] != out_size or cropped.shape[-2] != out_size:
            cropped = F.interpolate(cropped, size=(out_size, out_size), mode='bilinear', align_corners=False)
        
        return cropped
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply spatial augmentation to batch."""
        # Check probability
        if random.random() > self.p:
            return batch
        
        sensors = self._get_sensor_keys(batch)
        if not sensors:
            return batch
        
        # Sample parameters once if consistent
        if self.consistent:
            flip_h, flip_v = self._sample_flip_params()
            # Use first sensor as reference for crop params
            ref_tensor = batch[sensors[0]]
            _, _, ref_H, ref_W = ref_tensor.shape
            if self.random_crop:
                crop_params = self._sample_crop_params(ref_H, ref_W)
                original_size = (ref_H, ref_W)
            else:
                crop_params = None
                original_size = None
        
        for sensor in sensors:
            tensor = batch[sensor]
            
            # Sample independent params if not consistent
            if not self.consistent:
                flip_h, flip_v = self._sample_flip_params()
                _, _, H, W = tensor.shape
                if self.random_crop:
                    crop_params = self._sample_crop_params(H, W)
                    original_size = None  # No scaling needed
                else:
                    crop_params = None
                    original_size = None
            
            # Apply flip
            tensor = self._apply_flip(tensor, flip_h, flip_v)
            
            # Apply crop
            if self.random_crop and crop_params is not None:
                _, _, H, W = batch[sensor].shape  # Original shape before any transform
                tensor = self._apply_crop(
                    tensor, 
                    crop_params[0], crop_params[1], crop_params[2], crop_params[3],
                    original_size=original_size if self.consistent else None
                )
            
            batch[sensor] = tensor
        
        return batch
    
    def __repr__(self) -> str:
        return (
            f"SpatialAugmentation(p={self.p}, h_flip={self.horizontal_flip}, "
            f"v_flip={self.vertical_flip}, crop={self.random_crop}, "
            f"crop_scale={self.crop_scale}, consistent={self.consistent})"
        )


class RadiometricAugmentation:
    """
    Radiometric augmentations (brightness/contrast) for multi-modal data.
    
    Applies:
    - Brightness adjustment: tensor * brightness_factor
    - Contrast/bias adjustment: tensor + bias
    
    Final formula: output = tensor * brightness + bias
    
    Args:
        p: Probability of applying augmentation (0 = never, 1 = always)
        brightness_range: Range for brightness multiplier (min, max).
                         E.g., (0.8, 1.2) means 80% to 120% of original.
        bias_range: Range for additive bias (min, max).
                   E.g., (-0.1, 0.1) means shift by -0.1 to +0.1.
        per_channel: If True, sample different params per channel.
                    If False, same params for all channels in a sensor.
        consistent: If True, apply same transform to all sensors.
                   If False, each sensor gets independent random transforms.
        clamp: If True, clamp output to [0, 1]. Recommended for normalized data.
    
    Note:
        This augmentation simulates sensor calibration differences, atmospheric
        effects, and other radiometric variations between acquisitions.
    """
    
    METADATA_KEYS: Set[str] = {'patch_id', '_group_name', '_available_sensors'}
    
    def __init__(
        self,
        p: float = 0.5,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        bias_range: Tuple[float, float] = (-0.1, 0.1),
        per_channel: bool = False,
        consistent: bool = True,
        clamp: bool = False,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        
        self.p = p
        self.brightness_range = brightness_range
        self.bias_range = bias_range
        self.per_channel = per_channel
        self.consistent = consistent
        self.clamp = clamp
    
    def _get_sensor_keys(self, batch: Dict[str, Any]) -> List[str]:
        """Extract sensor keys from batch."""
        return [
            k for k in batch.keys()
            if k not in self.METADATA_KEYS
            and '_mask' not in k
            and not k.startswith('_')
            and isinstance(batch[k], torch.Tensor)
            and batch[k].ndim == 4  # (B, C, H, W)
        ]
    
    def _sample_params(self, num_channels: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample brightness and bias parameters."""
        if self.per_channel:
            brightness = torch.empty(num_channels, device=device).uniform_(
                self.brightness_range[0], self.brightness_range[1]
            )
            bias = torch.empty(num_channels, device=device).uniform_(
                self.bias_range[0], self.bias_range[1]
            )
            # Reshape for broadcasting: (C,) -> (1, C, 1, 1)
            brightness = brightness.view(1, -1, 1, 1)
            bias = bias.view(1, -1, 1, 1)
        else:
            brightness = torch.empty(1, device=device).uniform_(
                self.brightness_range[0], self.brightness_range[1]
            )
            bias = torch.empty(1, device=device).uniform_(
                self.bias_range[0], self.bias_range[1]
            )
        
        return brightness, bias
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply radiometric augmentation to batch."""
        # Check probability
        if random.random() > self.p:
            return batch
        
        sensors = self._get_sensor_keys(batch)
        if not sensors:
            return batch
        
        # Sample parameters once if consistent
        # Note: when consistent, we use scalar params (not per-channel)
        # because different sensors have different channel counts
        if self.consistent:
            device = batch[sensors[0]].device
            brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
            bias = random.uniform(self.bias_range[0], self.bias_range[1])
            shared_brightness = brightness
            shared_bias = bias
        
        for sensor in sensors:
            tensor = batch[sensor]
            
            if self.consistent:
                # Use shared scalar params
                tensor = tensor * shared_brightness + shared_bias
            else:
                # Sample independent params for this sensor
                brightness, bias = self._sample_params(tensor.shape[1], tensor.device)
                tensor = tensor * brightness + bias
            
            # Clamp to valid range
            if self.clamp:
                tensor = torch.clamp(tensor, 0.0, 1.0)
            
            batch[sensor] = tensor
        
        return batch
    
    def __repr__(self) -> str:
        return (
            f"RadiometricAugmentation(p={self.p}, brightness={self.brightness_range}, "
            f"bias={self.bias_range}, per_channel={self.per_channel}, "
            f"consistent={self.consistent}, clamp={self.clamp})"
        )


class AugmentationPipeline:
    """
    Combines multiple augmentations into a single callable.
    
    Args:
        spatial: SpatialAugmentation instance or None
        radiometric: RadiometricAugmentation instance or None
    
    Example:
        pipeline = AugmentationPipeline(
            spatial=SpatialAugmentation(p=0.5, consistent=True),
            radiometric=RadiometricAugmentation(p=0.3, brightness_range=(0.9, 1.1))
        )
        batch = pipeline(batch)
    """
    
    def __init__(
        self,
        spatial: Optional[SpatialAugmentation] = None,
        radiometric: Optional[RadiometricAugmentation] = None,
    ):
        self.spatial = spatial
        self.radiometric = radiometric
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentation pipeline to batch."""
        if self.spatial is not None:
            batch = self.spatial(batch)
        if self.radiometric is not None:
            batch = self.radiometric(batch)
        return batch
    
    def __repr__(self) -> str:
        parts = []
        if self.spatial is not None:
            parts.append(f"spatial={self.spatial}")
        if self.radiometric is not None:
            parts.append(f"radiometric={self.radiometric}")
        return f"AugmentationPipeline({', '.join(parts)})"
