"""
DESIS Band Trimming Transform.

Removes noisy edge bands from DESIS hyperspectral data.
The first and last bands are typically noisy due to sensor characteristics.
"""

from typing import Dict, Any, Optional

import torch


class DESISBandTrim:
    """
    Remove noisy edge bands from DESIS hyperspectral data.
    
    DESIS has 235 bands, but the first and last bands are often noisy.
    This transform removes them to improve data quality.
    
    Default: Remove first 10 and last 10 bands (235 -> 215 bands).
    
    This should be applied BEFORE normalization.
    
    Args:
        trim_start: Number of bands to remove from the start. Default is 10.
        trim_end: Number of bands to remove from the end. Default is 10.
        sensor_key: Key in batch dict for DESIS data. Default is 'DESIS'.
    
    Example:
        transform = DESISBandTrim(trim_start=10, trim_end=10)
        batch = {'DESIS': tensor_235_bands, 'EMIT': tensor, ...}
        batch = transform(batch)  # DESIS now has 215 bands
    """
    
    def __init__(
        self, 
        trim_start: int = 10, 
        trim_end: int = 10, 
        sensor_key: str = 'DESIS'
    ):
        if trim_start < 0:
            raise ValueError(f"trim_start must be >= 0, got {trim_start}")
        if trim_end < 0:
            raise ValueError(f"trim_end must be >= 0, got {trim_end}")
        
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.sensor_key = sensor_key

    def _resolve_sensor_key(self, batch: Dict[str, Any]) -> Optional[str]:
        """Resolve sensor key case-insensitively to support mixed key conventions."""
        if self.sensor_key in batch:
            return self.sensor_key

        target = str(self.sensor_key).upper()
        for key in batch.keys():
            if isinstance(key, str) and key.upper() == target:
                return key
        return None
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply band trimming to DESIS data in the batch.
        
        Args:
            batch: Dictionary containing sensor tensors
            
        Returns:
            Modified batch with DESIS bands trimmed
        """
        sensor_key = self._resolve_sensor_key(batch)
        if sensor_key is None:
            return batch
        
        desis = batch[sensor_key]
        
        if not isinstance(desis, torch.Tensor):
            return batch
        
        # Supported shapes:
        # - (B, C, H, W)
        # - (B, T, C, H, W)
        if desis.ndim == 4:
            band_dim = 1
        elif desis.ndim == 5:
            band_dim = 2
        else:
            return batch
        
        num_bands = int(desis.shape[band_dim])
        
        # Ensure we're not trimming more bands than available
        if self.trim_start + self.trim_end >= num_bands:
            raise ValueError(
                f"Cannot trim {self.trim_start} + {self.trim_end} = {self.trim_start + self.trim_end} bands "
                f"from DESIS with only {num_bands} bands"
            )
        
        # Calculate the slice indices
        start_idx = self.trim_start
        end_idx = num_bands - self.trim_end if self.trim_end > 0 else num_bands
        
        # Trim the bands
        slices = [slice(None)] * desis.ndim
        slices[band_dim] = slice(start_idx, end_idx)
        batch[sensor_key] = desis[tuple(slices)]
        
        return batch
    
    def __repr__(self) -> str:
        return f"DESISBandTrim(trim_start={self.trim_start}, trim_end={self.trim_end}, sensor_key='{self.sensor_key}')"
