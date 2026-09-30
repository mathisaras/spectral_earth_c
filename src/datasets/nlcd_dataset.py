# This file will be populated with the NLCDDataset class.
# Original content from enmap_cdl_nlcd.py will be moved and refactored here. 

import os
from typing import Callable, Optional, List, Dict, Tuple, Any
import random

from torch import Tensor
from torchgeo.datasets.nlcd import NLCD as TorchGeoNLCD # For NLCD cmap

from .base_segmentation_dataset import BaseSegmentationDataset
# Note: CDL import is removed as this class will focus on NLCD.
# Users wanting CDL should use the CDLDataset class.

class NLCDDataset(BaseSegmentationDataset):
    """National Land Cover Database (NLCD) dataset.

    This dataset is designed to be generic across different sensors.
    It expects sensor imagery and NLCD masks, with split files defining
    train/val/test sets. The NLCD product year (e.g., "nlcd_2019") 
    should be specified via the 'product' parameter.
    """

    IMAGE_ROOT_FORMAT: str = "{sensor}" # e.g., root/enmap/
    MASK_ROOT_FORMAT: str = "{product}" # e.g., root/nlcd_2019/
    # SPLIT_PATH_FORMAT will use BaseSegmentationDataset default.
    # e.g. splits/{sensor}_{product}/{split}.txt or splits/{product}/{split}.txt

    # RGB_INDICES is now inherited from BaseSegmentationDataset
    # RGB_INDICES: Dict[str, List[int]] = {
    #     "enmap": [43, 28, 10],  # Default for EnMAP, 0-indexed
    #     # "s2": [3,2,1], # Example for Sentinel-2
    # }

    # NLCD-specific colormap and default classes
    # The product name should be like "nlcd_2019", "nlcd_2016", etc.
    # We assume the NLCD.cmap is standard for all NLCD years.
    # If specific years have different cmaps, this needs adjustment or multiple product entries.
    CMAPS: Dict[str, Dict[int, Tuple[int, int, int, int]]] = {} # Populated in __init__
    DEFAULT_CLASSES: Dict[str, List[int]] = {} # Populated in __init__

    def __init__(
        self,
        root: str = "data",
        sensor_config: Dict[str, Any] = None,
        product: str = "nlcd", # e.g., "nlcd", "nlcd_2019" - must start with "nlcd"
        split: str = "train",
        split_root: str = "data/splits",
        classes: Optional[List[int]] = None, # User-specified raw foreground NLCD classes
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        raw_mask: bool = False,
        subset_percent: Optional[float] = None, # If set, use a random subset (0.0 to 1.0)
    ) -> None:
        """
        Args:
            root: Root directory.
            sensor_config: Complete sensor configuration from hydra.
            product: Product name, must start with "nlcd" (e.g., "nlcd", "nlcd_2019").
            split: "train", "val", or "test".
            split_root: Root directory containing split files.
            classes: List of raw NLCD foreground classes. If None, uses all NLCD classes from cmap.
            transforms: Optional transforms.
            raw_mask: If True, do not remap mask.
            subset_percent: If set (0.0-1.0), use a random subset of the current split's samples.
        """
        if not product.startswith("nlcd"):
            raise ValueError(f"Product name for NLCDDataset must start with 'nlcd', got {product}.")

        if sensor_config is None:
            raise ValueError("sensor_config is required")

        # Dynamically populate CMAPS and DEFAULT_CLASSES for the NLCD product
        if product not in self.CMAPS:
            self.CMAPS[product] = TorchGeoNLCD.cmap
        if product not in self.DEFAULT_CLASSES:
            self.DEFAULT_CLASSES[product] = list(TorchGeoNLCD.cmap.keys())
        
        # Removed block for setting RGB_INDICES, as it's now handled by the base class
        # if rgb_indices:
        #     type(self).RGB_INDICES[sensor] = rgb_indices


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
        
        self.subset_percent = subset_percent
        # Apply subset sampling if specified (after samples are loaded by super().__init__)
        if self.subset_percent is not None and self.split in ["train", "val"]: # Typically for training/validation
            if not (0.0 < self.subset_percent <= 1.0):
                raise ValueError("subset_percent must be between 0.0 (exclusive) and 1.0 (inclusive).")

            original_count = len(self.sample_collection)
            if original_count == 0:
                print(f"Warning: Subset percent requested, but no samples in '{self.split}' split to subset.")
                return

            new_count = int(original_count * self.subset_percent)
            
            if new_count == 0 and original_count > 0: # Ensure at least one sample if possible
                new_count = 1 
            
            if new_count < original_count:
                rng = random.Random(42) # Seeded for reproducibility
                self.sample_collection = rng.sample(self.sample_collection, new_count)
                print(f"Subset applied to '{self.split}' split ({self.sensor}/{product}): {new_count}/{original_count} samples selected.")
            # else: No need to subset if new_count is not less than original_count

    # All core methods (__getitem__, __len__, _load_image, _load_mask, plot) are inherited.

if __name__ == '__main__':
    print("Testing refactored NLCDDataset...")

    # Example sensor config for testing
    enmap_config = {
        "name": "enmap",
        "num_bands": 202,
        "rgb_indices": [43, 28, 10],
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

    # Test with NLCD product for EnMAP sensor
    try:
        print("\n--- Testing NLCD with EnMAP sensor ---")
        enmap_nlcd_train = NLCDDataset(
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=enmap_config,
            product="nlcd_2019", # Example NLCD product year
            split="train",
            subset_percent=0.1 # Test subsetting
        )
        print(f"EnMAP NLCD train dataset initialized. Samples: {len(enmap_nlcd_train)}")
        print(f"  Sensor: {enmap_nlcd_train.sensor}")
        print(f"  RGB indices: {enmap_nlcd_train.rgb_indices}")
        if len(enmap_nlcd_train) > 0:
            sample = enmap_nlcd_train[0]
            print(f"  Sample - Image: {sample['image'].shape}, Mask: {sample['mask'].shape}")
            print(f"  Mask unique values: {torch.unique(sample['mask'])}")
    except Exception as e:
        print(f"Error testing NLCD with EnMAP: {e}")
        import traceback
        traceback.print_exc()

    # Example: Test with NLCD product for Sentinel-2 sensor
    try:
        print("\n--- Testing NLCD with Sentinel-2 sensor ---")
        s2_nlcd_val = NLCDDataset(
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=s2_config,
            product="nlcd_2016",
            split="val"
        )
        print(f"S2 NLCD val dataset initialized. Samples: {len(s2_nlcd_val)}")
        print(f"  Sensor: {s2_nlcd_val.sensor}")
        print(f"  RGB indices: {s2_nlcd_val.rgb_indices}")
        if len(s2_nlcd_val) > 0:
            sample_s2 = s2_nlcd_val[0]
            print(f"  S2 Sample - Image: {sample_s2['image'].shape}, Mask: {sample_s2['mask'].shape}")
    except Exception as e:
        print(f"Error testing NLCD with S2: {e}")
        print("  (This might be due to missing data/split files for s2/nlcd_2016 example setup)")

    # Test with invalid product name
    try:
        print("\n--- Testing Invalid Product for NLCDDataset ---")
        invalid_prod_dataset = NLCDDataset(
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=enmap_config,
            product="not_nlcd_anything", # Invalid
            split="train",
        )
    except ValueError as e:
        print(f"Correctly caught error for invalid product: {e}")
    except Exception as e:
        print(f"Unexpected error for invalid product: {e}")
        
    # Test with missing sensor_config
    try:
        print("\n--- Testing NLCDDataset with missing sensor_config (should fail) ---")
        NLCDDataset(
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=None,
            product="nlcd_2019",
            split="train",
        )
    except ValueError as e:
        print(f"Correctly caught ValueError for missing sensor_config: {e}")
    except Exception as e:
        print(f"Unexpected error for missing sensor_config: {e}")
        import traceback
        traceback.print_exc() 
