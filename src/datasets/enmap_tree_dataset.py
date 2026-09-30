import os
from typing import Callable, Optional, List, Dict, Tuple, Any

import torch # For type hints
from torch import Tensor

from .base_segmentation_dataset import BaseSegmentationDataset

class EnMAPTreeDataset(BaseSegmentationDataset):
    """EnMAP tree species dataset with custom class mapping.

    The user provides a list of raw foreground classes (e.g., specific forest types).
    These are re-mapped to ordinal labels [0, 1, ..., N-1].
    All other pixel values in the raw mask are mapped to a single background class N.
    The colormap for foreground classes is dynamically generated if not specified in CMAPS.
    """

    IMAGE_ROOT_FORMAT: str = "images"
    MASK_ROOT_FORMAT: str = "labels"
    # SPLIT_PATH_FORMAT uses BaseSegmentationDataset default.

    # RGB_INDICES is now inherited from BaseSegmentationDataset
    # RGB_INDICES: Dict[str, List[int]] = {
    #     "enmap": [43, 28, 10],  # 0-indexed
    #     # Add other sensors as needed
    # }

    CMAPS: Dict[str, Dict[int, Tuple[int, int, int, int]]] = {}
    DEFAULT_CLASSES: Dict[str, List[int]] = {} # Not used as classes are mandatory

    def __init__(
        self,
        classes: List[int], # Mandatory: list of raw foreground class values from EnMAP tree species legend
        root: str = "data",
        sensor_config: Dict[str, Any] = None,
        product: str = "enmap_tree",
        split: str = "train",
        split_root: str = "data/splits",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        raw_mask: bool = False,
    ) -> None:
        """Args:
            classes: List of raw foreground class codes (e.g., from EnMAP tree species legend).
                     These will be mapped to [0, ..., N-1]. Others map to background N.
            root: Root directory (e.g., "data").
            sensor_config: Complete sensor configuration from hydra.
            product: Product name (default: "enmap_tree").
            split: Which split to use ("train", "val", or "test").
            split_root: Root directory containing split files.
            transforms: Optional transforms to apply.
            raw_mask: If True, do not remap the mask.
        """
        if not classes:
            raise ValueError("The 'classes' argument (list of foreground classes) is mandatory for EnMAPTreeDataset and cannot be empty.")

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
    print("Testing refactored EnMAPTreeDataset...")

    example_foreground_classes_enmap_tree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 17, 18] # Real classes from EnMAP tree species legend
    
    # Example sensor config for testing
    enmap_config = {
        "name": "enmap",
        "num_bands": 202,
        "rgb_indices": [43, 28, 10],
        "min_val": 0.0,
        "max_val": 10000.0
    }

    try:
        print(f"\n--- Testing EnMAPTreeDataset with EnMAP sensor, classes: {example_foreground_classes_enmap_tree} ---")
        enmap_tree_train_enmap = EnMAPTreeDataset(
            classes=example_foreground_classes_enmap_tree,
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/mathi/final_patches/enmap_tree",
            sensor_config=enmap_config,
            split="train",
        )
        print(f"EnMAP tree species train dataset initialized. Samples: {len(enmap_tree_train_enmap)}")
        print(f"  Foreground classes raw: {enmap_tree_train_enmap.foreground_classes_raw}")
        print(f"  Num effective classes: {enmap_tree_train_enmap.num_effective_classes}")
        print(f"  Sensor: {enmap_tree_train_enmap.sensor}")
        print(f"  RGB indices: {enmap_tree_train_enmap.rgb_indices}")

        if len(enmap_tree_train_enmap) > 0:
            sample = enmap_tree_train_enmap[0]
            print(f"  Sample - Image: {sample['image'].shape}, Mask: {sample['mask'].shape}")
            print(f"  Mask unique values: {torch.unique(sample['mask'])}")

    except Exception as e:
        print(f"Error testing EnMAP tree species dataset: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n--- Testing EnMAPTreeDataset with empty class list (should fail) ---")
        EnMAPTreeDataset(
            classes=[],
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/mathi/final_patches/enmap_tree",
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
        print("\n--- Testing EnMAPTreeDataset with missing sensor_config (should fail) ---")
        EnMAPTreeDataset(
            classes=example_foreground_classes_enmap_tree,
            root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/mathi/final_patches/enmap_tree",
            sensor_config=None,
            split="train",
        )
    except ValueError as e:
        print(f"Correctly caught ValueError for missing sensor_config: {e}")
    except Exception as e:
        print(f"Unexpected error for missing sensor_config: {e}")
        import traceback
        traceback.print_exc()