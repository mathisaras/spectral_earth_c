import os
from typing import Callable, Optional, List, Dict, Tuple, Any

import torch 
from torch import Tensor

from .base_segmentation_dataset import BaseSegmentationDataset

class PrismaTreeDataset(BaseSegmentationDataset):
    """PRISMA tree species dataset with custom class mapping.

    The user provides a list of raw foreground classes (e.g., specific forest types).
    These are re-mapped to ordinal labels [0, 1, ..., N-1].
    All other pixel values in the raw mask are mapped to a single background class N.
    The colormap for foreground classes is dynamically generated if not specified in CMAPS.
    """

    IMAGE_ROOT_FORMAT: str = "images"
    MASK_ROOT_FORMAT: str = "labels"

    CMAPS: Dict[str, Dict[int, Tuple[int, int, int, int]]] = {}
    DEFAULT_CLASSES: Dict[str, List[int]] = {} 

    def __init__(
        self,
        classes: List[int], # Mandatory: list of raw foreground class values from PRISMA tree species legend
        root: str = "data",
        sensor_config: Dict[str, Any] = None,
        product: str = "prisma_tree",
        split: str = "train",
        split_root: str = "data/splits",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        raw_mask: bool = False,
    ) -> None:
        """Args:
            classes: List of raw foreground class codes (e.g., from PRISMA tree species legend).
                     These will be mapped to [0, ..., N-1]. Others map to background N.
            root: Root directory (e.g., "data").
            sensor_config: Complete sensor configuration from hydra.
            product: Product name (default: "prisma_tree").
            split: Which split to use ("train", "val", or "test").
            split_root: Root directory containing split files.
            transforms: Optional transforms to apply.
            raw_mask: If True, do not remap the mask.
        """
        if not classes:
            raise ValueError("The 'classes' argument (list of foreground classes) is mandatory for PrismaTreeDataset and cannot be empty.")

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