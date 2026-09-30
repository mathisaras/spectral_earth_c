"""
Sensor Drop Transform for robustness training.

Randomly drops sensors from a batch to train models that can handle
partial/missing modalities.
"""

import random
from typing import Dict, Any, List, Set


class SensorDropTransform:
    """
    Randomly drop sensors from a batch for robustness training.
    
    This augmentation helps models learn to handle missing modalities by
    randomly removing some sensors from the batch during training.
    
    The drop is performed at the BATCH level, meaning all samples in the
    batch share the same sensor mask. This ensures consistent tensor shapes.
    
    Args:
        drop_prob: Probability of performing sensor drop on each batch.
                  0.0 = never drop, 1.0 = always drop
        min_sensors: Minimum number of sensors to keep after dropping.
                    Must be >= 1.
    
    Example:
        transform = SensorDropTransform(drop_prob=0.3, min_sensors=1)
        
        # Batch with 3 sensors
        batch = {'EMIT': ..., 'ENMAP': ..., 'DESIS': ..., 'patch_id': [...]}
        
        # After transform, might become:
        batch = {'EMIT': ..., 'patch_id': [...], '_available_sensors': ['EMIT']}
    """
    
    # Keys that are metadata, not sensor data
    METADATA_KEYS: Set[str] = {'patch_id', '_group_name', '_available_sensors'}
    
    def __init__(self, drop_prob: float = 0.3, min_sensors: int = 1):
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}")
        if min_sensors < 1:
            raise ValueError(f"min_sensors must be >= 1, got {min_sensors}")
        
        self.drop_prob = drop_prob
        self.min_sensors = min_sensors
    
    def _get_sensor_keys(self, batch: Dict[str, Any]) -> List[str]:
        """Extract sensor keys from batch (exclude metadata)."""
        sensors = []
        for k in batch.keys():
            # Skip metadata keys
            if k in self.METADATA_KEYS:
                continue
            # Skip mask keys
            if '_mask' in k:
                continue
            # Skip internal keys (start with _)
            if k.startswith('_'):
                continue
            sensors.append(k)
        return sensors
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply sensor drop augmentation to a batch.
        
        Args:
            batch: Dictionary containing sensor tensors and metadata
            
        Returns:
            Modified batch with some sensors potentially removed.
            Adds '_available_sensors' key listing remaining sensors.
        """
        # Check if we should drop this batch
        if random.random() > self.drop_prob:
            # No drop - just add available sensors list
            sensors = self._get_sensor_keys(batch)
            batch['_available_sensors'] = sensors
            return batch
        
        # Get available sensors in this batch
        sensors = self._get_sensor_keys(batch)
        
        # Can't drop if we're already at or below minimum
        if len(sensors) <= self.min_sensors:
            batch['_available_sensors'] = sensors
            return batch
        
        # Randomly select how many sensors to keep
        n_keep = random.randint(self.min_sensors, len(sensors))
        
        # Randomly select which sensors to keep
        keep_sensors = set(random.sample(sensors, n_keep))
        
        # Remove dropped sensors
        for s in sensors:
            if s not in keep_sensors:
                del batch[s]
        
        # Record which sensors are available
        batch['_available_sensors'] = sorted(list(keep_sensors))
        
        return batch
    
    def __repr__(self) -> str:
        return f"SensorDropTransform(drop_prob={self.drop_prob}, min_sensors={self.min_sensors})"
