"""
Multi-Modal Normalizer for Earth Observation sensors.

Supports config-driven normalization with backward compatibility.

SingleSensorNormalizer: same logic as MMNormalizer but for a single tensor (B, C, H, W),
used by single-sensor SSL (DINO, MoCo) so Landsat 8, Sentinel-1, etc. use the same
formulas as the multi-modal pipeline.
"""

import warnings
from typing import Dict, Optional, Any

import torch
import torch.nn as nn


class SingleSensorNormalizer(nn.Module):
    """
    Apply the same physical normalization as MMNormalizer to a single tensor.

    Takes a sensor config dict (with a 'normalization' key) and normalizes the input
    tensor to [0, 1] using the same formulas: scale, harmonize, thermal, sar.
    Used by DINO/MoCo and other single-sensor baselines so normalization is consistent
    with the multi-modal pipeline (MMNormalizer).

    Args:
        sensor_config: Dict for one sensor, e.g. from SensorRegistry.get("ENMAP").
                      Must contain "normalization" with "type" and params (scale,
                      slope, bias, temp_min, etc. depending on type).

    Example:
        config = SensorRegistry.get("ENMAP")  # or "LO", "S1", "LT", ...
        normalizer = SingleSensorNormalizer(sensor_config=config)
        x = normalizer(x)  # (B, C, H, W) -> [0, 1]
    """
    def __init__(self, sensor_config: Dict[str, Any]):
        super().__init__()
        self.sensor_config = sensor_config
        norm_cfg = sensor_config.get('normalization', {})
        norm_type = norm_cfg.get('type', 'scale')

        if norm_type == 'scale':
            scale = norm_cfg.get('scale', 10000.0)
            self.register_buffer('scale', torch.tensor(scale))
        elif norm_type == 'harmonize':
            self.register_buffer('slope', torch.tensor(norm_cfg.get('slope', 1.0)))
            self.register_buffer('bias', torch.tensor(norm_cfg.get('bias', 0.0)))
            self.register_buffer('scale', torch.tensor(norm_cfg.get('scale', 10000.0)))
        elif norm_type == 'thermal':
            self.register_buffer('slope', torch.tensor(norm_cfg.get('slope', 0.00341802)))
            self.register_buffer('add', torch.tensor(norm_cfg.get('add', 149.0)))
            self.register_buffer('temp_min', torch.tensor(norm_cfg.get('temp_min', 250.0)))
            self.register_buffer('temp_max', torch.tensor(norm_cfg.get('temp_max', 350.0)))
        elif norm_type == 'sar':
            self.register_buffer('min_val', torch.tensor(norm_cfg.get('min', -50.0)))
            self.register_buffer('max_val', torch.tensor(norm_cfg.get('max', 0.0)))
        else:
            raise ValueError(f"SingleSensorNormalizer: unknown normalization type '{norm_type}'")

        self._norm_type = norm_type

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor (B, C, H, W) to [0, 1] using the sensor's formula."""
        if not torch.is_floating_point(x):
            x = x.float()

        if self._norm_type == 'scale':
            out = x / self.scale.to(x.device)
        elif self._norm_type == 'harmonize':
            out = (x * self.slope.to(x.device) + self.bias.to(x.device)) / self.scale.to(x.device)
        elif self._norm_type == 'thermal':
            kelvin = (x * self.slope.to(x.device)) + self.add.to(x.device)
            out = (kelvin - self.temp_min.to(x.device)) / (self.temp_max.to(x.device) - self.temp_min.to(x.device))
        elif self._norm_type == 'sar':
            out = (x - self.min_val.to(x.device)) / (self.max_val.to(x.device) - self.min_val.to(x.device))
        else:
            return x

        return torch.clamp(out, 0.0, 1.0)


class MMNormalizer(nn.Module):
    """
    Physically-aware normalization for Multi-Modal Earth Observation.
    Maps all modalities to a common [0, 1] range based on physical units.
    
    Can be initialized in two ways:
    1. No args (backward compatible): Uses hardcoded defaults for known sensors
    2. With sensor_configs: Uses normalization params from sensor config dicts
    
    Args:
        sensor_configs: Optional dict mapping sensor names to their config dicts.
                       Each config should have a 'normalization' key with params.
                       If None, uses hardcoded defaults.
    
    Example:
        # Backward compatible (hardcoded defaults)
        normalizer = MMNormalizer()
        
        # Config-driven
        from src.utils.sensor_registry import SensorRegistry
        configs = SensorRegistry.get_all(["EMIT", "ENMAP", "DESIS"])
        normalizer = MMNormalizer(sensor_configs=configs)
    """
    
    # Default normalization configs (for backward compatibility)
    DEFAULT_CONFIGS = {
        'S2': {'normalization': {'type': 'scale', 'scale': 10000.0}},
        'ENMAP': {'normalization': {'type': 'scale', 'scale': 10000.0}},
        'DESIS': {'normalization': {'type': 'scale', 'scale': 10000.0}},
        'EMIT': {'normalization': {'type': 'scale', 'scale': 32767.0}},
        'LO': {'normalization': {'type': 'harmonize', 'slope': 0.275, 'bias': -2000.0, 'scale': 10000.0}},
        'LT': {'normalization': {'type': 'thermal', 'slope': 0.00341802, 'add': 149.0, 'temp_min': 250.0, 'temp_max': 350.0}},
        'S1': {'normalization': {'type': 'sar', 'min': -50.0, 'max': 0.0}},
    }
    
    def __init__(self, sensor_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        super().__init__()
        
        # Merge provided configs with defaults (provided takes precedence)
        self.sensor_configs = dict(self.DEFAULT_CONFIGS)
        if sensor_configs:
            for name, cfg in sensor_configs.items():
                # Normalize name to uppercase
                self.sensor_configs[name.upper()] = cfg
        
        # Register buffers for all normalization parameters
        # This ensures they move to GPU automatically
        self._register_normalization_buffers()
    
    def _register_normalization_buffers(self):
        """Register normalization constants as buffers for GPU compatibility."""
        for sensor_name, cfg in self.sensor_configs.items():
            norm_cfg = cfg.get('normalization', {})
            norm_type = norm_cfg.get('type', 'scale')
            
            prefix = sensor_name.lower()
            
            if norm_type == 'scale':
                self.register_buffer(f'{prefix}_scale', torch.tensor(norm_cfg.get('scale', 10000.0)))
                
            elif norm_type == 'harmonize':
                self.register_buffer(f'{prefix}_slope', torch.tensor(norm_cfg.get('slope', 1.0)))
                self.register_buffer(f'{prefix}_bias', torch.tensor(norm_cfg.get('bias', 0.0)))
                self.register_buffer(f'{prefix}_scale', torch.tensor(norm_cfg.get('scale', 10000.0)))
                
            elif norm_type == 'thermal':
                self.register_buffer(f'{prefix}_slope', torch.tensor(norm_cfg.get('slope', 0.00341802)))
                self.register_buffer(f'{prefix}_add', torch.tensor(norm_cfg.get('add', 149.0)))
                self.register_buffer(f'{prefix}_temp_min', torch.tensor(norm_cfg.get('temp_min', 250.0)))
                self.register_buffer(f'{prefix}_temp_max', torch.tensor(norm_cfg.get('temp_max', 350.0)))
                
            elif norm_type == 'sar':
                self.register_buffer(f'{prefix}_min', torch.tensor(norm_cfg.get('min', -50.0)))
                self.register_buffer(f'{prefix}_max', torch.tensor(norm_cfg.get('max', 0.0)))
    
    def _get_buffer(self, sensor_name: str, param_name: str) -> torch.Tensor:
        """Get a registered buffer by sensor and parameter name."""
        buffer_name = f'{sensor_name.lower()}_{param_name}'
        return getattr(self, buffer_name)
    
    def _normalize_sensor(self, tensor: torch.Tensor, sensor_name: str) -> torch.Tensor:
        """
        Normalize a single sensor tensor using its config.
        
        Args:
            tensor: Input tensor (B, C, H, W)
            sensor_name: Sensor name (uppercase)
            
        Returns:
            Normalized tensor in [0, 1] range
        """
        cfg = self.sensor_configs.get(sensor_name)
        if cfg is None:
            return None  # Signal that sensor is unknown
        
        norm_cfg = cfg.get('normalization', {})
        norm_type = norm_cfg.get('type', 'scale')
        
        if norm_type == 'scale':
            scale = self._get_buffer(sensor_name, 'scale')
            return tensor / scale
        
        elif norm_type == 'harmonize':
            slope = self._get_buffer(sensor_name, 'slope')
            bias = self._get_buffer(sensor_name, 'bias')
            scale = self._get_buffer(sensor_name, 'scale')
            harmonized = (tensor * slope) + bias
            return harmonized / scale
        
        elif norm_type == 'thermal':
            slope = self._get_buffer(sensor_name, 'slope')
            add = self._get_buffer(sensor_name, 'add')
            temp_min = self._get_buffer(sensor_name, 'temp_min')
            temp_max = self._get_buffer(sensor_name, 'temp_max')
            kelvin = (tensor * slope) + add
            return (kelvin - temp_min) / (temp_max - temp_min)
        
        elif norm_type == 'sar':
            sar_min = self._get_buffer(sensor_name, 'min')
            sar_max = self._get_buffer(sensor_name, 'max')
            return (tensor - sar_min) / (sar_max - sar_min)
        
        else:
            warnings.warn(f"MMNormalizer: Unknown normalization type '{norm_type}' for sensor '{sensor_name}'")
            return None
    
    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a dictionary batch in-place.
        
        Args:
            batch_dict: Dictionary containing sensor tensors and metadata.
                       Keys like 'patch_id', '_group_name', and '*_mask' are skipped.
        
        Returns:
            The same dictionary with sensor tensors normalized to [0, 1].
        """
        keys = list(batch_dict.keys())  # Copy keys to avoid iteration issues
        
        for k in keys:
            # Skip metadata and masks
            if k in ('patch_id', '_group_name', '_available_sensors') or '_mask' in k:
                continue
            if k.startswith('_'):
                continue
            
            tensor = batch_dict[k]
            
            if not isinstance(tensor, torch.Tensor):
                continue
            
            # Ensure float for normalization
            if not torch.is_floating_point(tensor):
                tensor = tensor.float()
            
            # Normalize
            sensor_name = k.upper()
            normalized = self._normalize_sensor(tensor, sensor_name)
            
            if normalized is None:
                # Unknown sensor - warn and skip
                warnings.warn(
                    f"MMNormalizer: Unknown sensor key '{k}' - passing through unnormalized. "
                    f"Known sensors: {list(self.sensor_configs.keys())}"
                )
                continue
            
            # Clamp to [0, 1] and assign back
            batch_dict[k] = torch.clamp(normalized, 0.0, 1.0)
        
        return batch_dict
