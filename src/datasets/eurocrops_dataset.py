# Refactored EurocropsDataset will be placed here. 

import os
from typing import Callable, Optional, List, Dict, Tuple, Any

import torch # Keep for type hints if any, though Tensor is used directly
from torch import Tensor
# No need for numpy, rasterio, matplotlib.pyplot, matplotlib.cm here if all handled by base

from .base_segmentation_dataset import BaseSegmentationDataset

class EurocropsDataset(BaseSegmentationDataset):
    """EuroCrops dataset with custom class mapping.

    The user provides a list of raw agricultural classes they are interested in (foreground classes).
    These foreground classes are re-mapped to ordinal labels [0, 1, ..., N-1].
    All other pixel values in the raw mask are mapped to a single background class N.
    The colormap for foreground classes is dynamically generated if not found in CMAPS.
    """

    IMAGE_ROOT_FORMAT: str = "{sensor}" 
    MASK_ROOT_FORMAT: str = "{product}" # Product name will be 'eurocrops'
    # SPLIT_PATH_FORMAT uses BaseSegmentationDataset default.

    # For EuroCrops, CMAPS is usually empty as classes are highly variable and user-defined.
    # The base class will dynamically generate colors for specified foreground classes.
    CMAPS: Dict[str, Dict[int, Tuple[int, int, int, int]]] = {}
    # DEFAULT_CLASSES are not typically used as 'classes' argument is mandatory for EuroCrops.
    DEFAULT_CLASSES: Dict[str, List[int]] = {}

    def __init__(
        self,
        classes: List[int], # Mandatory: list of raw foreground class values from EuroCrops legend
        root: str = "data",
        sensor_config: Dict[str, Any] = None,
        product: str = "eurocrops", # Product name will be 'eurocrops'
        split: str = "train",
        split_root: str = "data/splits",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        raw_mask: bool = False,
    ) -> None:
        """Args:
            classes: Mandatory list of raw foreground class codes from the EuroCrops legend.
                     These will be mapped to [0, ..., N-1]. Others map to background N.
            root: Root directory (e.g., "data").
            sensor_config: Complete sensor configuration from hydra.
            product: Product name (default: "eurocrops").
            split: Which split to use ("train", "val", or "test").
            split_root: Root directory containing split files.
            transforms: Optional transforms to apply.
            raw_mask: If True, do not remap the mask (mask contains raw class values).
        """
        if not classes:
            raise ValueError("The 'classes' argument (list of foreground classes) is mandatory for EurocropsDataset and cannot be empty.")

        if sensor_config is None:
            raise ValueError("sensor_config is required")

        super().__init__(
            root=root,
            sensor_config=sensor_config,
            product=product,
            split=split,
            split_root=split_root,
            classes=classes, # Pass the user's foreground classes directly
            transforms=transforms,
            raw_mask=raw_mask,
        )
        # The base class __init__ now handles: 
        # - Setting self.foreground_classes_raw from 'classes' argument.
        # - Creating self.ordinal_map.
        # - Creating self.ordinal_cmap (dynamically if product not in CMAPS or class not in product's cmap).

    # __getitem__, __len__, _load_image, _load_mask, plot methods are inherited from BaseSegmentationDataset.
    # The custom _remap_mask is no longer needed due to enhancements in BaseSegmentationDataset._load_mask and ordinal_map setup.

if __name__ == '__main__':
    print("Testing refactored EurocropsDataset...")

    example_foreground_classes = [10, 20, 30, 45] # e.g., Wheat, Barley, Corn, Sunflower
    
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

    try:
        print(f"\n--- Testing EurocropsDataset with EnMAP sensor, classes: {example_foreground_classes} ---")
        eurocrops_train_enmap = EurocropsDataset(
            classes=example_foreground_classes,
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=enmap_config,
            split="train",
        )
        print(f"EnMAP Eurocrops train dataset initialized. Samples: {len(eurocrops_train_enmap)}")
        print(f"  Foreground classes raw: {eurocrops_train_enmap.foreground_classes_raw}")
        print(f"  Num effective classes: {eurocrops_train_enmap.num_effective_classes}")
        print(f"  Ordinal cmap shape: {eurocrops_train_enmap.ordinal_cmap.shape}")
        print(f"  Sensor: {eurocrops_train_enmap.sensor}")
        print(f"  RGB indices: {eurocrops_train_enmap.rgb_indices}")

        if len(eurocrops_train_enmap) > 0:
            sample = eurocrops_train_enmap[0]
            print(f"  Sample - Image: {sample['image'].shape}, Mask: {sample['mask'].shape}")
            print(f"  Mask unique values: {torch.unique(sample['mask'])}")

    except Exception as e:
        print(f"Error testing EnMAP Eurocrops: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n--- Testing EurocropsDataset with Sentinel-2 sensor --- ")
        eurocrops_train_s2 = EurocropsDataset(
            classes=example_foreground_classes,
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=s2_config,
            split="train", # Assuming a split file like data/splits/s2_eurocrops/train.txt
        )
        print(f"S2 Eurocrops train dataset initialized. Samples: {len(eurocrops_train_s2)}")
        print(f"  Sensor: {eurocrops_train_s2.sensor}")
        print(f"  RGB indices: {eurocrops_train_s2.rgb_indices}")
    except Exception as e:
        print(f"Error testing S2 Eurocrops: {e}")
        print(" (This might be due to missing data/split files for s2/eurocrops example setup)")


    try:
        print("\n--- Testing EurocropsDataset with empty class list (should fail) ---")
        EurocropsDataset(
            classes=[], # Empty list
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=enmap_config,
            split="train",
        )
    except ValueError as e:
        print(f"Correctly caught ValueError for empty classes: {e}")
    except Exception as e:
        print(f"Unexpected error for empty classes: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        print("\n--- Testing EurocropsDataset with missing sensor_config (should fail) ---")
        EurocropsDataset(
            classes=example_foreground_classes,
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=None,
            split="train",
        )
    except ValueError as e:
        print(f"Correctly caught ValueError for missing sensor_config: {e}")
    except Exception as e:
        print(f"Unexpected error for missing sensor_config: {e}")
        import traceback
        traceback.print_exc() 
