# Refactored TreeMapDataset will be placed here. 

import os
from typing import Callable, Optional, List, Dict, Tuple, Any

import torch # Keep for type hints
from torch import Tensor

from .base_segmentation_dataset import BaseSegmentationDataset

class TreeMapDataset(BaseSegmentationDataset):
    """TreeMap dataset for forest mask segmentation with custom class mapping.

    The user provides a list of raw foreground classes (e.g., specific tree types).
    These are re-mapped to ordinal labels [0, 1, ..., N-1].
    All other pixel values in the raw mask are mapped to a single background class N.
    The colormap for foreground classes is dynamically generated if not specified in CMAPS.
    """

    IMAGE_ROOT_FORMAT: str = "{sensor}"
    MASK_ROOT_FORMAT: str = "{product}" # Product name will be 'treemap'
    # SPLIT_PATH_FORMAT uses BaseSegmentationDataset default.

    # RGB_INDICES is now inherited from BaseSegmentationDataset
    # RGB_INDICES: Dict[str, List[int]] = {
    #     "enmap": [43, 28, 10],  # 0-indexed
    #     # Add other sensors here, e.g., "s2": [3,2,1]
    # }

    CMAPS: Dict[str, Dict[int, Tuple[int, int, int, int]]] = {}
    DEFAULT_CLASSES: Dict[str, List[int]] = {} # Not used as classes are mandatory

    def __init__(
        self,
        classes: List[int], # Mandatory: list of raw foreground class values from TreeMap legend
        root: str = "data",
        sensor_config: Dict[str, Any] = None,
        product: str = "treemap",
        split: str = "train",
        split_root: str = "data/splits",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        raw_mask: bool = False,
    ) -> None:
        """Args:
            classes: List of raw foreground class codes (e.g., from TreeMap legend).
                     These will be mapped to [0, ..., N-1]. Others map to background N.
            root: Root directory (e.g., "data").
            sensor_config: Complete sensor configuration from hydra.
            product: Product name (default: "treemap").
            split: Which split to use ("train", "val", or "test").
            split_root: Root directory containing split files.
            transforms: Optional transforms to apply.
            raw_mask: If True, do not remap the mask.
        """
        if not classes:
            raise ValueError("The 'classes' argument (list of foreground classes) is mandatory for TreeMapDataset and cannot be empty.")

        if sensor_config is None:
            raise ValueError("sensor_config is required")

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

    # Core data handling methods inherited from BaseSegmentationDataset

if __name__ == '__main__':
    print("Testing refactored TreeMapDataset...")

    example_foreground_classes_treemap = [1, 5, 12] # e.g., specific tree type IDs
    
    # Example sensor config for testing
    enmap_config = {
        "name": "enmap",
        "num_bands": 202,
        "rgb_indices": [43, 28, 10],
        "min_val": 0.0,
        "max_val": 10000.0
    }

    try:
        print(f"\n--- Testing TreeMapDataset with EnMAP sensor, classes: {example_foreground_classes_treemap} ---")
        treemap_val_enmap = TreeMapDataset(
            classes=example_foreground_classes_treemap,
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/codes/spectral_earth_mm/data",
            sensor_config=enmap_config,
            split="val",
        )
        print(f"EnMAP TreeMap val dataset initialized. Samples: {len(treemap_val_enmap)}")
        print(f"  Foreground classes raw: {treemap_val_enmap.foreground_classes_raw}")
        print(f"  Num effective classes: {treemap_val_enmap.num_effective_classes}")
        print(f"  Sensor: {treemap_val_enmap.sensor}")
        print(f"  RGB indices: {treemap_val_enmap.rgb_indices}")

        if len(treemap_val_enmap) > 0:
            sample = treemap_val_enmap[0]
            print(f"  Sample - Image: {sample['image'].shape}, Mask: {sample['mask'].shape}")
            print(f"  Mask unique values: {torch.unique(sample['mask'])}")

    except Exception as e:
        print(f"Error testing EnMAP TreeMap: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n--- Testing TreeMapDataset with empty class list (should fail) ---")
        TreeMapDataset(
            classes=[],
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
        print("\n--- Testing TreeMapDataset with missing sensor_config (should fail) ---")
        TreeMapDataset(
            classes=example_foreground_classes_treemap,
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
