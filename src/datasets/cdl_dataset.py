import os
from typing import Callable, Optional, List, Dict, Tuple, Any
import random

import torch
from torch import Tensor
# import numpy as np # No longer directly used here
# import rasterio # Handled by base
import matplotlib.pyplot as plt
from torchgeo.datasets.cdl import CDL as TorchGeoCDL # Rename to avoid class name conflict

from .base_segmentation_dataset import BaseSegmentationDataset

class CDLDataset(BaseSegmentationDataset):
    """Generic Cropland Data Layer (CDL) dataset.

    This dataset is designed to be generic across different sensors.
    It expects images and CDL masks, with split files defining
    train/val/test sets.
    The actual CDL mask product (e.g., "cdl_2021", "cdl") should be specified via the 'product' param.
    """

    IMAGE_ROOT_FORMAT: str = "{sensor}" # Example: data/imagery/desis/
    MASK_ROOT_FORMAT: str = "{product}"  # Example: data/masks/cdl_2021/
    # SPLIT_PATH_FORMAT uses base default: splits/{sensor}_{product}/{split}.txt
    # or falls back to splits/{product}/{split}.txt

    # Define CMAPS and DEFAULT_CLASSES for any product starting with 'cdl'
    # These will be dynamically populated in __init__ if not overridden by a more specific subclass
    CMAPS: Dict[str, Dict[int, Tuple[int, int, int, int]]] = {}
    DEFAULT_CLASSES: Dict[str, List[int]] = {}

    def __init__(
        self,
        root: str = "data",
        sensor_config: Dict[str, Any] = None,
        product: str = "cdl", # e.g., "cdl", "cdl_2020", "cdl_2021"
        split: str = "train",
        split_root: str = "data/splits",
        classes: Optional[List[int]] = None, # User-specified raw foreground classes
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        raw_mask: bool = False,
    ) -> None:
        """
        Args:
            root: Root directory where the dataset is stored.
            sensor_config: Complete sensor configuration from hydra.
            product: Product name, typically "cdl" or "cdl_YYYY". This is used for 
                     locating mask data under MASK_ROOT_FORMAT and for CMAPS/DEFAULT_CLASSES.
            split: One of "train", "val", or "test".
            split_root: Root directory containing split files.
            classes: List of raw CDL foreground classes to include. If None, uses all available
                     CDL classes (excluding 0) from the standard CDL cmap.
            transforms: Optional callable applied to each sample.
            raw_mask: If True, do not remap the mask (e.g., for debugging).
            subset_percent: If set (0.0-1.0), use a random subset of the current split's samples.
        """
        if sensor_config is None:
            raise ValueError("sensor_config is required")
        
        # Populate CMAPS and DEFAULT_CLASSES for CDL-like products if not already set by a subclass
        # This check allows subclasses to define more specific CDL product cmaps if needed.
        if product.startswith("cdl") and product not in self.CMAPS:
            self.CMAPS[product] = TorchGeoCDL.cmap
        if product.startswith("cdl") and product not in self.DEFAULT_CLASSES:
            # Default classes for CDL are all its cmap keys, foreground_classes_raw in base will exclude 0
            self.DEFAULT_CLASSES[product] = list(TorchGeoCDL.cmap.keys())

        super().__init__(
            root=root,
            sensor_config=sensor_config,
            product=product,
            split=split,
            split_root=split_root,
            classes=classes,
            transforms=transforms,
            raw_mask=raw_mask,
        )
       
    # _load_image, _load_mask, __getitem__, __len__, plot are inherited from BaseSegmentationDataset

if __name__ == '__main__':
    print("Testing refactored CDLDataset...")
    
    # Example sensor configs for testing
    enmap_config = {
        "name": "enmap",
        "num_bands": 202,
        "rgb_indices": [43, 28, 10],
        "min_val": 0.0,
        "max_val": 10000.0
    }
    
    desis_config = {
        "name": "desis",
        "num_bands": 235,
        "rgb_indices": [25, 15, 5],
        "min_val": 0.0,
        "max_val": 10000.0
    }
    
    s2_config = {
        "name": "s2",
        "num_bands": 12,
        "rgb_indices": [3, 2, 1],
        "min_val": 0.0,
        "max_val": 10000.0
    }

    print("\nTesting CDLDataset with DESIS sensor...")
    try:
        # DESIS sensor with cdl_2021 product
        desis_cdl_train = CDLDataset(
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=desis_config,
            product="cdl_2021", 
            split="train",
        )
        print(f"DESIS CDL product 'cdl_2021' train dataset initialized. Samples: {len(desis_cdl_train)}")
        
        if len(desis_cdl_train) > 0:
            sample = desis_cdl_train[0]
            print("Sample loaded:")
            print(f"  Image shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
            print(f"  Mask shape: {sample['mask'].shape}, dtype: {sample['mask'].dtype}") 
            print(f"  Mask unique values: {torch.unique(sample['mask'])}")
            print(f"  Foreground classes raw: {desis_cdl_train.foreground_classes_raw}")
            print(f"  Num effective classes: {desis_cdl_train.num_effective_classes}")
            print(f"  Sensor: {desis_cdl_train.sensor}")
            print(f"  RGB indices: {desis_cdl_train.rgb_indices}")
    except Exception as e:
        print(f"Error during CDLDataset example (DESIS): {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting CDLDataset with EnMAP sensor...")
    try:
        enmap_cdl_val = CDLDataset(
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=enmap_config,
            product="cdl", # Standard "cdl" product name
            split="val",
            subset_percent=0.5 # Use 50% of validation samples
        )
        print(f"EnMAP CDL product 'cdl' val dataset initialized. Samples: {len(enmap_cdl_val)}")
        if len(enmap_cdl_val) > 0:
            sample_enmap = enmap_cdl_val[0]
            print("EnMAP Sample loaded:")
            print(f"  Image shape: {sample_enmap['image'].shape}")
            print(f"  Sensor: {enmap_cdl_val.sensor}")
            print(f"  RGB indices: {enmap_cdl_val.rgb_indices}")
    except Exception as e:
        print(f"Error during CDLDataset example (EnMAP): {e}")
        import traceback
        traceback.print_exc()

    # Test with a specific set of CDL classes (crop types)
    print("\nTesting CDLDataset with specific crop classes...")
    try:
        # Example crop class codes for corn, soybeans, wheat
        crop_classes = [1, 5, 24] 
        s2_cdl_train = CDLDataset(
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=s2_config,
            product="cdl_2023",
            split="train",
            classes=crop_classes
        )
        print(f"Sentinel-2 CDL train dataset with specific crops. Samples: {len(s2_cdl_train)}")
        print(f"  Selected crop classes: {crop_classes}")
        print(f"  Foreground classes raw: {s2_cdl_train.foreground_classes_raw}")
        print(f"  Num effective classes: {s2_cdl_train.num_effective_classes}")
        print(f"  Sensor: {s2_cdl_train.sensor}")
    except Exception as e:
        print(f"Error with crop-specific CDLDataset: {e}")
        
    # Test with missing sensor_config
    try:
        print("\nTesting CDLDataset with missing sensor_config (should fail)...")
        CDLDataset(
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=None,
            product="cdl_2019",
            split="train",
        )
    except ValueError as e:
        print(f"Correctly caught ValueError for missing sensor_config: {e}")
    except Exception as e:
        print(f"Unexpected error for missing sensor_config: {e}")
        import traceback
        traceback.print_exc() 
