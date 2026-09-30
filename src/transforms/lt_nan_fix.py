"""
Landsat Thermal NaN Interpolation Transform.

Handles missing/invalid pixels in Landsat Thermal data that are encoded
as a specific value (typically 0) before normalization.
"""

from typing import Dict, Any

import torch
import torch.nn.functional as F
import torch.nn as nn


class LTNaNInterpolateOld:
    """
    Interpolate NaN values in Landsat Thermal (LT) data.
    
    Missing/invalid thermal pixels are often encoded as 0 in the raw data.
    This transform detects these pixels and fills them using bilinear
    interpolation from surrounding valid pixels.
    
    This should be applied BEFORE normalization to work with raw DN values.
    
    Args:
        nan_value: Value that encodes NaN/invalid pixels. Default is 0.0.
        sensor_key: Key in batch dict for LT data. Default is 'LT'.
    
    Example:
        transform = LTNaNInterpolate(nan_value=0.0)
        batch = {'LT': tensor, 'EMIT': tensor, ...}
        batch = transform(batch)  # LT tensor now has NaN pixels filled
    """
    
    def __init__(self, nan_value: float = 0.0, sensor_key: str = 'LT'):
        self.nan_value = nan_value
        self.sensor_key = sensor_key
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply NaN interpolation to LT data in the batch.
        
        Args:
            batch: Dictionary containing sensor tensors
            
        Returns:
            Modified batch with LT NaN pixels interpolated
        """
        if self.sensor_key not in batch:
            return batch
        
        lt = batch[self.sensor_key]
        
        if not isinstance(lt, torch.Tensor):
            return batch
        
        # Create mask of NaN pixels (True = valid, False = NaN)
        valid_mask = (lt != self.nan_value)
        
        # Check if there are any NaN pixels
        if valid_mask.all():
            return batch
        
        # Interpolate NaN pixels
        batch[self.sensor_key] = self._interpolate_nans(lt, valid_mask)
        
        return batch
    
    def _interpolate_nans(self, tensor: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Fill NaN pixels using inpainting-style interpolation.
        
        Uses an iterative dilation approach: repeatedly replace NaN pixels
        with the mean of their valid neighbors until all pixels are filled.
        
        Args:
            tensor: Input tensor (B, C, H, W)
            valid_mask: Boolean mask where True = valid pixel
            
        Returns:
            Tensor with NaN pixels filled
        """
        B, C, H, W = tensor.shape
        result = tensor.clone()
        mask = valid_mask.clone()
        
        # Iteratively fill NaN pixels using neighbor averaging
        # This is a simple but effective approach for sparse NaN patterns
        max_iterations = max(H, W)  # Worst case: need to propagate across entire image
        
        for _ in range(max_iterations):
            if mask.all():
                break
            
            # For each NaN pixel, compute mean of valid neighbors
            # Use convolution with a 3x3 kernel for efficiency
            
            # Sum of valid neighbor values
            # Pad to handle borders
            padded = F.pad(result * mask.float(), (1, 1, 1, 1), mode='replicate')
            padded_mask = F.pad(mask.float(), (1, 1, 1, 1), mode='constant', value=0)
            
            # 3x3 neighbor sum (excluding center)
            kernel = torch.tensor([
                [1., 1., 1.],
                [1., 0., 1.],
                [1., 1., 1.]
            ], device=tensor.device, dtype=tensor.dtype).view(1, 1, 3, 3)
            
            # Sum values and counts for each channel
            neighbor_sum = torch.zeros_like(result)
            neighbor_count = torch.zeros_like(result)
            
            for c in range(C):
                neighbor_sum[:, c:c+1] = F.conv2d(
                    padded[:, c:c+1], kernel, padding=0
                )
                neighbor_count[:, c:c+1] = F.conv2d(
                    padded_mask[:, c:c+1], kernel, padding=0
                )
            
            # Compute mean where we have neighbors
            has_neighbors = neighbor_count > 0
            nan_pixels = ~mask
            
            # Fill NaN pixels that have valid neighbors
            fillable = nan_pixels & has_neighbors
            
            if not fillable.any():
                # No more fillable pixels, but still have NaN
                # Fall back to global mean
                for b in range(B):
                    for c in range(C):
                        channel_mask = mask[b, c]
                        if not channel_mask.all() and channel_mask.any():
                            mean_val = result[b, c][channel_mask].mean()
                            result[b, c][~channel_mask] = mean_val
                break
            
            # Fill with neighbor mean
            result = torch.where(
                fillable,
                neighbor_sum / neighbor_count.clamp(min=1),
                result
            )
            
            # Update mask
            mask = mask | fillable
        
        return result
    
    def __repr__(self) -> str:
        return f"LTNaNInterpolate(nan_value={self.nan_value}, sensor_key='{self.sensor_key}')"


class LTNaNInterpolate(nn.Module):
    """
    GPU-native Masked Pyramid Inpainting.
    
    Solves the "dark patch" issue by correctly ignoring invalid pixels during downsampling.
    This approximates linear/smooth interpolation for large holes efficiently.
    """
    def __init__(self, nan_value: float = 0.0, sensor_key: str = 'LT', levels: int = 5):
        super().__init__()
        self.nan_value = nan_value
        self.sensor_key = sensor_key
        self.levels = levels 

    def _masked_downsample(self, x, mask):
        """
        Downsamples 2x2 by averaging ONLY valid pixels.
        Standard AvgPool would treat 0s as data, darkening the image.
        """
        # Sum of 2x2 block
        # divisor_override=1 prevents internal division; we divide manually later
        x_sum = F.avg_pool2d(x, kernel_size=2, stride=2, divisor_override=1)
        mask_sum = F.avg_pool2d(mask, kernel_size=2, stride=2, divisor_override=1)
        
        # Avoid div by zero
        # Where mask_sum is 0 (no valid pixels in 2x2), result stays 0
        safe_mask = torch.clamp(mask_sum, min=1e-6)
        
        # New Valid Value = Sum of Values / Count of Valid Pixels
        x_down = x_sum / safe_mask
        
        # New Mask = 1.0 if we had ANY valid data, 0.0 otherwise
        mask_down = (mask_sum > 0).float()
        
        # Zero out invalid areas again to keep math clean
        x_down = x_down * mask_down
        
        return x_down, mask_down

    def forward(self, batch_dict):
        if self.sensor_key not in batch_dict: return batch_dict
        x = batch_dict[self.sensor_key]
        
        if not isinstance(x, torch.Tensor): return batch_dict

        # 1. Define Valid Mask (1=Valid, 0=Hole)
        if torch.isnan(torch.tensor(self.nan_value)):
            mask = (~torch.isnan(x)).float()
            x = torch.nan_to_num(x, nan=0.0)
        else:
            mask = (x != self.nan_value).float()
            x = x * mask

        # Optimization: Return early if fully valid
        if mask.min() == 1.0: return batch_dict

        # 2. Build Pyramid (Analysis)
        # We store (image, mask) pairs at each scale
        pyramid = [(x, mask)]
        
        curr_x, curr_m = x, mask
        
        for _ in range(self.levels):
            # Masked Downsample
            curr_x, curr_m = self._masked_downsample(curr_x, curr_m)
            pyramid.append((curr_x, curr_m))

        # 3. Collapse Pyramid (Synthesis)
        # Start from the coarsest level and pull data UP
        reconstructed = pyramid[-1][0]
        
        # If the coarsest level still has holes (very rare), fill them with the global mean
        if (reconstructed == 0).any():
            global_mean = reconstructed.sum() / ( (reconstructed!=0).float().sum() + 1e-6 )
            reconstructed[reconstructed == 0] = global_mean

        # Go up the pyramid
        for i in range(self.levels - 1, -1, -1):
            target_x, target_m = pyramid[i]
            
            # Upsample the lower-res reconstruction to current size
            # 'bilinear' creates the smooth linear gradient you want
            up_x = F.interpolate(reconstructed, size=target_x.shape[-2:], mode='bilinear', align_corners=False)
            
            # Blend:
            # If we have valid data at this level (target_m=1), use it.
            # If we have a hole (target_m=0), use the upsampled guess.
            reconstructed = target_x * target_m + up_x * (1 - target_m)

        batch_dict[self.sensor_key] = reconstructed
        return batch_dict