from typing import Any, List, Optional, Union

from .base_segmentation_datamodule import BaseSegmentationDataModule
from ..datasets.prisma_tree_dataset import PrismaTreeDataset

class PrismaTreeDataModule(BaseSegmentationDataModule):
    """LightningDataModule for the PRISMA tree species segmentation dataset."""

    def __init__(
        self,
        sensor_config: dict,  # Passed directly from hydra config
        product: str = "prisma_tree",
        classes: Optional[List[int]] = None, # Optional specific tree species classes
        root: str = "data",
        batch_size: int = 64,
        img_size: Union[int, tuple[int, int]] = 128,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            sensor: Sensor configuration from hydra
            product: Product name, typically "prisma_tree"
            classes: List of raw tree species classes. If None, uses all classes defined in DEFAULT_CLASSES
            root: Root directory where dataset can be found
            batch_size: Size of each mini-batch
            img_size: Size of each image
            num_workers: Number of workers for data loading
            subset_percent: If set (0.0-1.0), use a random subset of samples
            **kwargs: Additional keyword arguments passed to the dataset
        """
        dataset_kwargs = {
            "root": root,
            "product": product,
            "classes": classes,
            **kwargs
        }

        super().__init__(
            dataset_class=PrismaTreeDataset,
            sensor_config=sensor_config,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=num_workers,
            **dataset_kwargs
        )