"""
Spectral Earth Multi-Modal Zarr DataModule.

Supports:
- Sensor filtering: Only use specific sensors (filters groups accordingly)
- Group filtering: Explicitly select which groups to use
- Sensor drop augmentation: Randomly drop sensors for robustness training
- Spatial augmentations: Flip, crop with consistency options
- Radiometric augmentations: Brightness/contrast with consistency options
- Config-driven normalization: Uses sensor configs from YAML files
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import warnings

import zarr
import torch.distributed as dist
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch import Tensor

from src.datasets.spectral_earth_mm_zarr import ZarrSpectralEarthMMDataset
from src.samplers.block_distributed import ZarrBlockDistributedSampler
from src.loaders.group_mixing import SyncedGroupLoader
from src.transforms.normalize_mm import MMNormalizer
from src.utils.sensor_registry import SensorRegistry




def _identity_collate(x):
    """Identity collate - dataset already returns batched dict."""
    return x


class SpectralEarthMMZarrDataModule(LightningDataModule):
    """
    Lightning DataModule for Spectral Earth Multi-Modal Zarr dataset.
    
    Handles distributed training with synchronized group mixing.
    
    Args:
        zarr_path: Path to the Zarr archive
        batch_size: Batch size per GPU
        num_workers: Max number of workers per group (scaled by size/weight)
        sensors: List of sensors to use (filters groups). None = all sensors.
        groups: List of groups to use. None = all groups (after sensor filtering).
        sensor_drop_prob: Probability of dropping sensors during training (0 = disabled)
        min_sensors: Minimum sensors to keep when dropping
        sampling_weights: Dict of group -> weight. None = sqrt(group_size) default.
        pin_memory: Pin memory for faster GPU transfer
        
        # Spatial Augmentation
        spatial_aug_p: Probability of spatial augmentation (0 = disabled)
        spatial_aug_flip_h: Enable horizontal flip
        spatial_aug_flip_v: Enable vertical flip
        spatial_aug_crop: Enable random resized crop
        spatial_aug_crop_scale: Crop scale range (min, max)
        spatial_aug_consistent: Apply same transform to all sensors
        
        # Radiometric Augmentation
        radiometric_aug_p: Probability of radiometric augmentation (0 = disabled)
        radiometric_aug_brightness: Brightness multiplier range (min, max)
        radiometric_aug_bias: Additive bias range (min, max)
        radiometric_aug_per_channel: Apply different params per channel
        radiometric_aug_consistent: Apply same transform to all sensors
        num_temporal_views: Number of temporal views to return per sample (1 or 2).
            When 2, returns two different timestamps as separate views for SSL (DINO/MoCo).
        normalize: If True, apply MMNormalizer to sensor tensors. If False, leave data raw
            (e.g. for DINO/MoCo where the model does its own normalization).
        standardize: If True, apply mean/std standardization after normalization.
        standardize_modalities: Optional list of sensor names to standardize.
            If null, standardize all sensor tensors in the batch.
        standardization_mode: 'bandwise' or 'global' statistics.
        standardization_stats_path: Path to sensor stats yaml, index.yaml, or directory.
        standardization_eps: Numerical stability epsilon for std clamp.
    """
    
    # Group -> Sensors mapping
    # Defines which sensors are available in each group
    GROUP_SENSORS: Dict[str, List[str]] = {
        "emit_only": ["EMIT"],
        "enmap_only": ["ENMAP"],
        "desis_only": ["DESIS"],
        "pair_emit_enmap": ["EMIT", "ENMAP"],
        "pair_emit_desis": ["EMIT", "DESIS"],
        "pair_enmap_desis": ["ENMAP", "DESIS"],
        "triplet": ["EMIT", "ENMAP", "DESIS"],
    }
    COMMON_SENSORS: List[str] = ["S2", "S1", "LT", "LO"]
    
    def __init__(
        self,
        zarr_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        sensors: Optional[List[str]] = None,
        groups: Optional[List[str]] = None,
        sensor_drop_prob: float = 0.0,
        min_sensors: int = 1,
        sampling_weights: Optional[Dict[str, float]] = None,
        pin_memory: bool = True,
        # Spatial Augmentation
        spatial_aug_p: float = 0.0,
        spatial_aug_flip_h: bool = True,
        spatial_aug_flip_v: bool = True,
        spatial_aug_crop: bool = False,
        spatial_aug_crop_scale: Tuple[float, float] = (0.4, 1.0),
        spatial_aug_consistent: bool = True,
        # Radiometric Augmentation
        radiometric_aug_p: float = 0.0,
        radiometric_aug_brightness: Tuple[float, float] = (0.8, 1.2),
        radiometric_aug_bias: Tuple[float, float] = (-0.1, 0.1),
        radiometric_aug_per_channel: bool = False,
        radiometric_aug_consistent: bool = True,
        num_temporal_views: int = 1,
        normalize: bool = True,
        standardize: bool = False,
        standardize_modalities: Optional[List[str]] = None,
        standardization_mode: str = "bandwise",
        standardization_stats_path: Optional[str] = None,
        standardization_eps: float = 1e-6,
        **kwargs: Any,
    ):
        super().__init__()
        # Keep datamodule/model hparams keyspaces non-overlapping for Lightning's
        # automatic hyperparameter merge/logging.
        self.save_hyperparameters(ignore=["standardize_modalities"])
        
        self.zarr_path = zarr_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sensors = sensors
        self.groups = groups
        self.sensor_drop_prob = sensor_drop_prob
        self.min_sensors = min_sensors
        self.sampling_weights = sampling_weights
        self.num_temporal_views = num_temporal_views
        self.normalize = normalize
        self.standardize = standardize
        self.standardize_modalities = (
            [str(sensor).upper() for sensor in standardize_modalities]
            if standardize_modalities is not None
            else None
        )
        self.standardization_mode = standardization_mode
        self.standardization_stats_path = standardization_stats_path
        self.standardization_eps = standardization_eps
        
        # Spatial augmentation config
        self.spatial_aug_p = spatial_aug_p
        self.spatial_aug_flip_h = spatial_aug_flip_h
        self.spatial_aug_flip_v = spatial_aug_flip_v
        self.spatial_aug_crop = spatial_aug_crop
        self.spatial_aug_crop_scale = spatial_aug_crop_scale
        self.spatial_aug_consistent = spatial_aug_consistent
        
        # Radiometric augmentation config
        self.radiometric_aug_p = radiometric_aug_p
        self.radiometric_aug_brightness = radiometric_aug_brightness
        self.radiometric_aug_bias = radiometric_aug_bias
        self.radiometric_aug_per_channel = radiometric_aug_per_channel
        self.radiometric_aug_consistent = radiometric_aug_consistent
        
        # Initialize transforms (will be set up in setup())
        self.normalizer = None
        self.standardizer = None
        self.sensor_drop = None
        self.lt_nan_fix = None
        self.desis_band_trim = None
        self.spatial_aug = None
        self.radiometric_aug = None
        
        # Track if we need to split temporal views for SSL
        self._split_temporal_views = num_temporal_views > 1
        
    def setup(self, stage: Optional[str] = None):
        """Set up transforms based on configuration."""
        # Load sensor configs for normalization
        if self.sensors:
            sensor_configs = SensorRegistry.get_all(self.sensors)
        else:
            # Load all known sensor configs
            available = SensorRegistry.list_available()
            sensor_configs = SensorRegistry.get_all(available)
        
        self.normalizer = MMNormalizer(sensor_configs=sensor_configs)

        # Optional standardized input on top of existing normalization.
        # Kept default-off for retro-compatibility.
        if self.standardize:
            if not self.standardization_stats_path:
                raise ValueError(
                    "standardize=True requires `standardization_stats_path` "
                    "(sensor stats yaml, index yaml, or directory)."
                )
            if not self.normalize:
                warnings.warn(
                    "standardize=True with normalize=False. Stats are expected over normalized inputs."
                )
            if self.standardize_modalities is not None:
                known_modalities = {
                    str(sensor).upper() for sensor in sensor_configs.keys()
                }
                unknown = sorted(
                    {
                        str(sensor).upper()
                        for sensor in self.standardize_modalities
                        if str(sensor).upper() not in known_modalities
                    }
                )
                if unknown:
                    raise ValueError(
                        "standardize_modalities contains unknown sensors: "
                        f"{unknown}. Available: {sorted(known_modalities)}"
                    )
            from src.transforms.standardize_mm import MMBatchStandardizer

            default_sensor_name = None
            if self.sensors is not None and len(self.sensors) == 1:
                default_sensor_name = self.sensors[0]

            self.standardizer = MMBatchStandardizer(
                stats_path=self.standardization_stats_path,
                mode=self.standardization_mode,
                eps=self.standardization_eps,
                default_sensor_name=default_sensor_name,
                include_sensors=self.standardize_modalities,
            )
        
        # Set up sensor drop transform if enabled
        if self.sensor_drop_prob > 0:
            from src.transforms.sensor_drop import SensorDropTransform
            self.sensor_drop = SensorDropTransform(
                drop_prob=self.sensor_drop_prob,
                min_sensors=self.min_sensors
            )
        
        # Set up LT NaN interpolation if LT sensor is used
        sensors_to_check = [s.upper() for s in self.sensors] if self.sensors else []
        if not sensors_to_check or 'LT' in sensors_to_check:
            try:
                from src.transforms.lt_nan_fix import LTNaNInterpolate
                lt_config = SensorRegistry.get('LT')
                nan_value = lt_config.get('nan_value', 0.0)
                self.lt_nan_fix = LTNaNInterpolate(nan_value=nan_value)
            except FileNotFoundError:
                # LT config not available, skip
                pass
        
        # Set up DESIS band trimming if DESIS sensor is used
        if not sensors_to_check or 'DESIS' in sensors_to_check:
            try:
                from src.transforms.desis_band_trim import DESISBandTrim
                desis_config = SensorRegistry.get('DESIS')
                band_trim_cfg = desis_config.get('band_trim', {})
                trim_start = band_trim_cfg.get('trim_start', 10)
                trim_end = band_trim_cfg.get('trim_end', 10)
                self.desis_band_trim = DESISBandTrim(
                    trim_start=trim_start,
                    trim_end=trim_end
                )
            except FileNotFoundError:
                # DESIS config not available, skip
                pass
        
        # Set up spatial augmentation if enabled
        if self.spatial_aug_p > 0:
            from src.transforms.augmentations import SpatialAugmentation
            self.spatial_aug = SpatialAugmentation(
                p=self.spatial_aug_p,
                horizontal_flip=self.spatial_aug_flip_h,
                vertical_flip=self.spatial_aug_flip_v,
                random_crop=self.spatial_aug_crop,
                crop_scale=self.spatial_aug_crop_scale,
                consistent=self.spatial_aug_consistent,
            )
        
        # Set up radiometric augmentation if enabled
        if self.radiometric_aug_p > 0:
            from src.transforms.augmentations import RadiometricAugmentation
            self.radiometric_aug = RadiometricAugmentation(
                p=self.radiometric_aug_p,
                brightness_range=self.radiometric_aug_brightness,
                bias_range=self.radiometric_aug_bias,
                per_channel=self.radiometric_aug_per_channel,
                consistent=self.radiometric_aug_consistent,
            )
    
    def _filter_groups_by_sensors(self, all_groups: List[str]) -> List[str]:
        """
        Filter groups to only those containing at least one requested sensor.
        
        Args:
            all_groups: List of all available groups
            
        Returns:
            Filtered list of groups
        """
        if self.sensors is None:
            return all_groups
        
        requested = set(s.upper() for s in self.sensors)
        filtered = []
        
        for g in all_groups:
            group_sensors = set(self.GROUP_SENSORS.get(g, [])) | set(self.COMMON_SENSORS)
            # Keep group if it has at least one requested sensor
            if group_sensors & requested:
                filtered.append(g)
        
        return filtered
    
    def _filter_groups_explicit(self, all_groups: List[str]) -> List[str]:
        """
        Filter groups to only those explicitly requested.
        
        Args:
            all_groups: List of all available groups
            
        Returns:
            Filtered list of groups
        """
        if self.groups is None:
            return all_groups
        
        requested = set(self.groups)
        filtered = [g for g in all_groups if g in requested]
        
        # Warn about requested groups that don't exist
        missing = requested - set(filtered)
        if missing:
            warnings.warn(
                f"Requested groups not found in Zarr: {missing}. "
                f"Available groups: {all_groups}"
            )
        
        return filtered
    
    def _compute_default_weights(self, group_sizes: Dict[str, int]) -> Dict[str, float]:
        """
        Compute default sampling weights using sqrt(group_size).
        
        This balances between:
        - Size-proportional: Large groups get more samples
        - Equal: Small groups still get meaningful representation
        """
        return {g: math.sqrt(size) for g, size in group_sizes.items()}

    def train_dataloader(self):
        root = zarr.open_group(self.zarr_path, mode='r')
        all_groups = sorted(list(root.group_keys()))
        
        # Step 1: Filter by sensors
        active_groups = self._filter_groups_by_sensors(all_groups)
        
        # Step 2: Filter by explicit group list
        active_groups = self._filter_groups_explicit(active_groups)
        
        # Step 3: Filter based on sampling weights (if provided)
        if self.sampling_weights:
            active_groups = [g for g in active_groups if g in self.sampling_weights]
        
        if not active_groups:
            raise RuntimeError(
                f"No groups remaining after filtering. "
                f"Sensors filter: {self.sensors}, Groups filter: {self.groups}, "
                f"Weights filter: {list(self.sampling_weights.keys()) if self.sampling_weights else None}"
            )
        
        # Get group sizes
        group_sizes = {}
        for g in active_groups:
            keys = [k for k in root[g].array_keys() if k != 'patch_id' and '_mask' not in k]
            if keys:
                group_sizes[g] = root[g][keys[0]].shape[0]
            else:
                group_sizes[g] = 0
        
        # Compute weights
        if self.sampling_weights:
            weights = {g: self.sampling_weights[g] for g in active_groups}
        else:
            weights = self._compute_default_weights(group_sizes)
        
        # Filter out groups that are too small for distributed training
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        min_samples = self.batch_size * world_size
        
        valid_groups = []
        for g in active_groups:
            if group_sizes[g] >= min_samples:
                valid_groups.append(g)
            else:
                warnings.warn(
                    f"Skipping group '{g}' with {group_sizes[g]} samples "
                    f"(need >= {min_samples} for batch_size={self.batch_size}, world_size={world_size})"
                )
        
        if not valid_groups:
            raise RuntimeError(
                f"No groups have enough samples for distributed training. "
                f"Need at least {min_samples} samples per group."
            )
        
        active_groups = valid_groups
        weights = {g: weights[g] for g in active_groups}
        
        # Log final configuration
        if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
            print(f"[DataModule] Active groups: {active_groups}")
            print(f"[DataModule] Sensors filter: {self.sensors}")
            print(f"[DataModule] Sensor drop prob: {self.sensor_drop_prob}")
        
        # Calculate worker scaling factors
        max_size = max(group_sizes[g] for g in active_groups) if active_groups else 1
        max_weight = max(weights.values()) if weights else 1
            
        loaders = {}
        for g in active_groups:
            ds = ZarrSpectralEarthMMDataset(self.zarr_path, g, num_temporal_views=self.num_temporal_views)
            
            sampler = ZarrBlockDistributedSampler(
                ds, 
                batch_size=self.batch_size, 
                drop_last=True
            )
            
            # Scale workers by BOTH size and weight.
            # Respect explicit num_workers=0 for single-process debugging/analysis.
            size_factor = group_sizes[g] / max_size
            weight_factor = weights[g] / max_weight
            combined_factor = max(size_factor, weight_factor)
            if self.num_workers <= 0:
                scaled_workers = 0
            else:
                scaled_workers = max(1, int(self.num_workers * combined_factor))
            
            loaders[g] = DataLoader(
                ds,
                sampler=sampler,
                batch_size=None,
                num_workers=scaled_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=_identity_collate,
                persistent_workers=scaled_workers > 0,
            )
            
        return SyncedGroupLoader(loaders, weights)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply transforms after batch is transferred to GPU."""
        # 0. Optional sensor key filtering:
        #    If `self.sensors` is specified, we drop any sensor tensors
        #    from the batch that are not in the requested list.
        #    This is in addition to group-level filtering and ensures
        #    that models only ever see the requested modalities.
        if self.sensors is not None:
            requested = {s.upper() for s in self.sensors}
            keys = list(batch.keys())
            for k in keys:
                # Skip metadata and masks
                if k in ('patch_id', '_group_name', '_available_sensors'):
                    continue
                if '_mask' in k or k.startswith('_'):
                    continue
                value = batch[k]
                # Only treat tensor entries as sensor data
                if hasattr(value, "shape"):
                    if k.upper() not in requested:
                        del batch[k]

        # 0.5. Handle temporal multi-view: split into image1/image2 for SSL (DINO/MoCo)
        if self._split_temporal_views:
            sensor_keys = [
                k
                for k in batch.keys()
                if isinstance(batch[k], Tensor)
                and k not in ("patch_id", "_group_name", "_available_sensors")
                and "_mask" not in k
                and not k.startswith("_")
            ]

            # Temporal multi-view splitting is only valid for single-sensor runs.
            if len(sensor_keys) != 1:
                raise ValueError(
                    "num_temporal_views > 1 is only supported when exactly one sensor tensor is present "
                    f"in the batch. Found {len(sensor_keys)} sensors: {sensor_keys}. "
                    "Use data.sensors=[<single_sensor>] for temporal-view SSL."
                )

            s_key = sensor_keys[0]
            tensor = batch[s_key]
            if tensor.ndim != 5 or tensor.shape[1] != self.num_temporal_views:
                raise ValueError(
                    f"Expected temporal multi-view tensor for '{s_key}' with shape "
                    f"[B, {self.num_temporal_views}, C, H, W], got {tuple(tensor.shape)}."
                )

            # Split into two views expected by DINO/MoCo-style objectives.
            batch["image1"] = tensor[:, 0]  # (B, C, H, W)
            batch["image2"] = tensor[:, 1] if self.num_temporal_views >= 2 else tensor[:, 0]

            # Remove original sensor key so downstream code only sees image1/image2.
            del batch[s_key]

        # 1. Pre-normalization sensor-specific fixes.
        #    These transforms operate on raw data before normalization.
        
        # 1a. DESIS band trimming (removes noisy edge bands)
        if self.desis_band_trim is not None:
            batch = self.desis_band_trim(batch)
        
        # 1b. LT NaN interpolation
        if self.lt_nan_fix is not None:
            batch = self.lt_nan_fix(batch)
        
        # 2. Normalize all sensors (optional; set normalize=False for DINO/MoCo etc.)
        if self.normalize and self.normalizer is not None:
            batch = self.normalizer(batch)

        # 2.5. Optional mean/std standardization on top of normalization.
        if self.standardizer is not None:
            batch = self.standardizer(batch)
        
        # 3. Data augmentations (only during training)
        is_training = self.trainer is not None and self.trainer.training
        
        if is_training:
            # 3a. Spatial augmentation (flip, crop)
            if self.spatial_aug is not None:
                batch = self.spatial_aug(batch)
            
            # 3b. Radiometric augmentation (brightness, bias)
            if self.radiometric_aug is not None:
                batch = self.radiometric_aug(batch)
        
        # 4. Sensor drop augmentation (only during training)
        if is_training and self.sensor_drop is not None:
            batch = self.sensor_drop(batch)
        
        return batch
