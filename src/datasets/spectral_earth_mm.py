import os
import torch
import pandas as pd
import numpy as np
import rasterio
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from collections import defaultdict
from typing import List, Dict, Optional, Any

class SpectralEarthMMDataset(Dataset):
    """
    Spectral Earth Multi-Modal (MM) Dataset.
    
    - Loads Single-Timestamp Random Views.
    - Uses Min-Max Normalization from Config (0-1 Scaling).
    - Groups data for Homogeneous Batch Sampling.
    """

    def __init__(
        self,
        index_path: str,
        root_dir: str,
        sensors: List[Dict[str, Any]],
        normalize: bool = False,
        default_img_size: int = 128,
        transform: Optional[Any] = None,
    ):
        self.root_dir = root_dir
        self.default_img_size = default_img_size
        self.transform = transform
        self.normalize = normalize
        
        # 1. Parse Sensor Configs
        self.sensor_configs = {s['name']: s for s in sensors}
        self.sensor_names = list(self.sensor_configs.keys())
        
        # 2. Load Index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        self.index_df = pd.read_csv(index_path)
        self.samples = self.index_df.to_dict('records')
        
        # 3. Build Groups for Batch Sampler
        print(f"SpectralEarthMM: Grouping {len(self.samples)} samples by config...")
        self.groups = self._build_groups()

    def _build_groups(self) -> Dict[str, List[int]]:
        """Classifies every sample into a Config Bucket."""
        groups = defaultdict(list)
        for idx, row in enumerate(self.samples):
            e = row.get('count_EMIT', 0) > 0
            n = row.get('count_ENMAP', 0) > 0
            d = row.get('count_DESIS', 0) > 0
            
            if e and n and d: key = "triplet"
            elif e and n:     key = "pair_emit_enmap"
            elif e and d:     key = "pair_emit_desis"
            elif n and d:     key = "pair_enmap_desis"
            elif e:           key = "emit_only"
            elif n:           key = "enmap_only"
            elif d:           key = "desis_only"
            else:             key = "no_hsi"
            
            groups[key].append(idx)
        return dict(groups)

    def _load_tiff(self, path: str) -> Optional[torch.Tensor]:
        if not os.path.exists(path): return None
        try:
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)
                data = np.nan_to_num(data)
                return torch.from_numpy(data)
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.samples[idx]
        patch_id = row['patch_id']
        patch_dir = os.path.join(self.root_dir, patch_id)
        
        sample = {'patch_id': patch_id}

        for s_name in self.sensor_names:
            if row.get(f"count_{s_name}", 0) == 0:
                continue

            cfg = self.sensor_configs[s_name]
            expected_c = cfg.get('bands', -1)
            target_size = cfg.get('img_size', self.default_img_size)
            
            # Retrieve Min/Max for Normalization
            min_val = cfg.get('min_val', 0.0)
            max_val = cfg.get('max_val', 10000.0) # Default to 10k (typical for S2/EnMAP)
            
            file_str = row.get(s_name, "")
            
            if isinstance(file_str, str) and len(file_str) > 0:
                filenames = file_str.split(';')
                valid_files = [f for f in filenames if f != ""]
                
                if not valid_files: continue

                # Random Selection
                selected_fname = random.choice(valid_files)
                fpath = os.path.join(patch_dir, selected_fname)
                
                tens = self._load_tiff(fpath)
                
                if tens is not None:
                    # Checks
                    if expected_c > 0 and tens.shape[0] != expected_c: 
                        print(f"Expected {expected_c} bands, got {tens.shape[0]} for {s_name}")
                        continue
                    if tens.shape[1] != target_size or tens.shape[2] != target_size: 
                        print(f"Expected {target_size}x{target_size}, got {tens.shape[1]}x{tens.shape[2]} for {s_name}")
                        continue
                    
                    # --- MIN-MAX NORMALIZATION ---
                    if self.normalize:
                        # Scale to [0, 1]
                        tens = (tens - min_val) / (max_val - min_val + 1e-6)
                        # Clamp to ensure stability (handle sun glint / bright clouds)
                        tens = torch.clamp(tens, 0.0, 1.0)
                    
                    sample[s_name] = tens

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.samples)
    
    def plot_sample(self, sample: Dict[str, Any], save_path: Optional[str] = None):
        """
        Visualizes the loaded tensors with Robust Contrast Stretching.
        Works for both Raw (0-10000) and Normalized (0-1) data.
        """
        # Filter keys to find valid sensor data
        valid_keys = [k for k in sample.keys() if k != "patch_id" and k in self.sensor_names]
        
        if not valid_keys:
            print(f"Sample {sample.get('patch_id', 'Unknown')} contains no loaded sensor data.")
            return

        # Setup Plot Grid
        n = len(valid_keys)
        cols = min(n, 7)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        if n == 1: axes = [axes]
        axes = np.array(axes).flatten()
        
        for i, s_name in enumerate(valid_keys):
            ax = axes[i]
            data = sample[s_name].clone().detach().cpu() # (C, H, W)
            
            # Get RGB indices from config, default to [0] if missing
            rgb = self.sensor_configs[s_name].get('rgb_indices', [0])
            
            # Prepare Image for Display
            if len(rgb) == 3 and data.shape[0] >= 3:
                # RGB Mode
                img = data[rgb].numpy().transpose(1, 2, 0) # (H, W, 3)
                is_gray = False
            else:
                # Grayscale Mode (First defined band)
                band_idx = rgb[0] if rgb else 0
                if band_idx >= data.shape[0]: band_idx = 0 # Safety fallback
                img = data[band_idx].numpy() # (H, W)
                is_gray = True

            # --- ROBUST NORMALIZATION (2% - 98% Stretch) ---
            # This handles raw data (0-10000) and normalized data (0-1) automatically
            img = np.nan_to_num(img) # Safety for NaNs
            
            # Compute percentiles ignoring NaNs/Infs
            p2 = np.percentile(img, 2)
            p98 = np.percentile(img, 98)
            
            # Apply Stretch
            if p98 - p2 > 1e-6:
                img = (img - p2) / (p98 - p2)
            else:
                # If image is flat (constant value), subtract min to make it 0
                img = img - p2
                
            # Clip to valid range [0, 1] for matplotlib
            img = np.clip(img, 0.0, 1.0)
            # -----------------------------------------------
            
            # Plot
            if is_gray:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            
            ax.set_title(s_name)
            ax.axis('off')
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Patch: {sample['patch_id']}", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()