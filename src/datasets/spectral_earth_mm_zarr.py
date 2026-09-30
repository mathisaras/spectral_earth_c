import warnings
import torch
import numpy as np
import zarr
from torch.utils.data import Dataset
from typing import List, Dict, Any, Union

class ZarrSpectralEarthMMDataset(Dataset):
    """
    Production Dataset for Spectral Earth Multi-Modal (Zarr).
    
    Features:
    - Lazy Loading (Pickle-safe).
    - Block Reading (Optimized for ZarrBlockDistributedSampler).
    - Random Timestamp Selection (Mask-aware).
    - Type Casting (Everything to Float32 for training stability).
    """
    def __init__(self, zarr_path: str, group_name: str, num_temporal_views: int = 1):
        self.zarr_path = zarr_path
        self.group_name = group_name
        self.num_temporal_views = num_temporal_views
        
        # 1. Metadata Init (Main Process)
        # We verify existence but DO NOT store the Zarr Group object
        try:
            root = zarr.open_group(zarr_path, mode='r')
            if group_name not in root:
                raise ValueError(f"Group '{group_name}' not found in {zarr_path}")
            
            group = root[group_name]
            
            # Detect Sensors
            self.sensor_names = [
                k for k in group.array_keys() 
                if "_mask" not in k and k != "patch_id"
            ]
            
            # Detect Length
            self.length = group[self.sensor_names[0]].shape[0]
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize dataset from {zarr_path}: {e}")
        
        # 2. Worker Handles (Initialized Lazily)
        self.group = None
        self.arrays = None
        self.masks = None
        self.ids = None
        
        # 3. Track warnings to avoid spam (warn once per sensor)
        self._warned_sensors = set()

    def _ensure_open(self):
        """Called inside worker process to open file handles."""
        if self.group is None:
            # Mode 'r' allows concurrent reads
            root = zarr.open_group(self.zarr_path, mode='r')
            self.group = root[self.group_name]
            
            # Cache handles for speed
            self.arrays = {s: self.group[s] for s in self.sensor_names}
            self.masks = {s: self.group[f"{s}_mask"] for s in self.sensor_names}
            self.ids = self.group['patch_id']

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: Union[int, List[int], slice]) -> Dict[str, Any]:
        """
        Supports getting a single item OR a slice/batch of items.
        The BatchSampler yields a LIST of indices.
        """
        self._ensure_open()
        
        # 1. Normalize Indexing
        # Convert contiguous list indices to slice for optimal Zarr performance
        # Slice indexing is more efficient than vindex for contiguous reads
        
        if isinstance(idx, list) and len(idx) > 1:
            # Check if indices are contiguous
            if idx == list(range(idx[0], idx[-1] + 1)):
                idx = slice(idx[0], idx[-1] + 1)
        
        batch = {}
        
        # 2. Load Metadata (Patch IDs)
        # Zarr returns numpy array of objects/strings
        raw_ids = self.ids[idx]
        
        # Robust conversion to Python strings
        if isinstance(raw_ids, np.ndarray):
            batch['patch_id'] = raw_ids.tolist()
        else:
            # Single item case
            batch['patch_id'] = [raw_ids]
            
        # 3. Load Sensor Data
        for s in self.sensor_names:
            # A. Read Raw Data + Mask
            # Shapes: (B, 4, C, H, W)
            data_np = self.arrays[s][idx]
            mask_np = self.masks[s][idx]
            
            # Convert to Tensor (Zero-copy if possible)
            data = torch.from_numpy(data_np)
            mask = torch.from_numpy(mask_np)
            
            # Ensure mask is boolean-like (handles various storage formats)
            # Common formats: bool, uint8 (0/1 or 0/255), int8, float
            if mask.dtype == torch.bool:
                pass  # Already boolean
            else:
                # Throw an error, mask should be boolean
                raise ValueError(f"Mask for sensor '{s}' is not boolean. Shape: {mask.shape}, dtype: {mask.dtype}")
            
            # B. Process Temporal Dimension
            # We have (B, 4, ...) but want (B, ...) 
            # We must pick one valid timestamp per sample in the batch.
            
            # Case 1: Batch Loading
            if data.ndim == 5:  # (B, T, C, H, W)
                B, T, C, H, W = data.shape
                
                if self.num_temporal_views == 1:
                    # Single view: (B, C, H, W)
                    batch_tensor = torch.empty((B, C, H, W), dtype=data.dtype)
                    for i in range(B):
                        sample_data = data[i]  # (T, C, H, W)
                        sample_mask = mask[i]  # (T,)
                        valid_indices = torch.nonzero(sample_mask, as_tuple=True)[0]
                        if len(valid_indices) > 0:
                            choice = valid_indices[torch.randint(0, len(valid_indices), (1,)).item()]
                            batch_tensor[i] = sample_data[choice]
                        else:
                            if s not in self._warned_sensors:
                                self._warned_sensors.add(s)
                                sample_id = batch['patch_id'][i] if i < len(batch.get('patch_id', [])) else f"idx={i}"
                                warnings.warn(
                                    f"[{self.group_name}] Sensor '{s}' has samples with no valid timestamps "
                                    f"(e.g., sample '{sample_id}'). Using fallback."
                                )
                            batch_tensor[i] = sample_data[0]
                else:
                    # Multi-view: (B, num_views, C, H, W)
                    # IMPORTANT: Batch dimension is always consistent. If a sample has fewer valid
                    # timestamps than num_temporal_views, we sample WITH REPLACEMENT (duplicate timestamps).
                    # This ensures all samples have shape (num_views, C, H, W) for consistent batching.
                    batch_tensor = torch.empty((B, self.num_temporal_views, C, H, W), dtype=data.dtype)
                    for i in range(B):
                        sample_data = data[i]  # (T, C, H, W)
                        sample_mask = mask[i]  # (T,)
                        valid_indices = torch.nonzero(sample_mask, as_tuple=True)[0].tolist()
                        if len(valid_indices) > 0:
                            # Sample without replacement if possible, with replacement if not enough valid
                            if len(valid_indices) >= self.num_temporal_views:
                                # Enough valid timestamps: sample different ones
                                choices = torch.tensor(valid_indices)[torch.randperm(len(valid_indices))[:self.num_temporal_views]]
                            else:
                                # Not enough valid timestamps - sample WITH REPLACEMENT
                                # This means the same timestamp may appear multiple times to fill all views
                                choices = torch.tensor([valid_indices[torch.randint(0, len(valid_indices), (1,)).item()] 
                                                       for _ in range(self.num_temporal_views)])
                            for v in range(self.num_temporal_views):
                                batch_tensor[i, v] = sample_data[choices[v]]
                        else:
                            if s not in self._warned_sensors:
                                self._warned_sensors.add(s)
                                sample_id = batch['patch_id'][i] if i < len(batch.get('patch_id', [])) else f"idx={i}"
                                warnings.warn(
                                    f"[{self.group_name}] Sensor '{s}' has samples with no valid timestamps "
                                    f"(e.g., sample '{sample_id}'). Using fallback."
                                )
                            # No valid timestamps: use first timestamp (possibly invalid) for all views
                            for v in range(self.num_temporal_views):
                                batch_tensor[i, v] = sample_data[0]
                
            # Case 2: Single Item Loading (Fallback)
            else:
                # (T, C, H, W) -> need (1, C, H, W) or (1, num_views, C, H, W)
                valid_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
                if len(valid_indices) > 0:
                    if self.num_temporal_views == 1:
                        choice = valid_indices[torch.randint(0, len(valid_indices), (1,)).item()]
                        batch_tensor = data[choice].unsqueeze(0)  # (1, C, H, W)
                    else:
                        if len(valid_indices) >= self.num_temporal_views:
                            choices = torch.tensor(valid_indices)[torch.randperm(len(valid_indices))[:self.num_temporal_views]]
                        else:
                            choices = torch.tensor([valid_indices[torch.randint(0, len(valid_indices), (1,)).item()] 
                                                   for _ in range(self.num_temporal_views)])
                        batch_tensor = torch.stack([data[c] for c in choices]).unsqueeze(0)  # (1, num_views, C, H, W)
                else:
                    if s not in self._warned_sensors:
                        self._warned_sensors.add(s)
                        sample_id = batch['patch_id'][0] if batch.get('patch_id') else "unknown"
                        warnings.warn(
                            f"[{self.group_name}] Sensor '{s}' has samples with no valid timestamps "
                            f"(e.g., sample '{sample_id}'). Using fallback."
                        )
                    if self.num_temporal_views == 1:
                        batch_tensor = data[0].unsqueeze(0)  # (1, C, H, W)
                    else:
                        batch_tensor = data[0].unsqueeze(0).unsqueeze(0).repeat(1, self.num_temporal_views, 1, 1, 1)  # (1, num_views, C, H, W)
            
            # C. Cast to Float32 (Standard for DL training)
            # Zarr stores int16/uint16 to save space. Models need floats.
            if batch_tensor.dtype != torch.float32:
                batch_tensor = batch_tensor.float()
                
       

            batch[s] = batch_tensor

        return batch