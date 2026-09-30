from typing import Any, List, Union

from .base_segmentation_datamodule import BaseSegmentationDataModule
from ..datasets.treemap_dataset import TreeMapDataset

class TreeMapDataModule(BaseSegmentationDataModule):
    """LightningDataModule for the TreeMap dataset."""

    def __init__(
        self,
        classes: List[int],
        sensor_config: dict,  # Passed directly from hydra config
        root: str = "data",
        batch_size: int = 64,
        img_size: Union[int, tuple[int, int]] = 128,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            classes: List of raw foreground class values (tree types)
            sensor: Sensor configuration from hydra
            root: Root directory where dataset can be found
            batch_size: Size of each mini-batch
            img_size: Size of each image
            num_workers: Number of workers for data loading
            **kwargs: Additional keyword arguments passed to the dataset
        """
        dataset_kwargs = {
            "classes": classes,
            "root": root,
            "product": "treemap",  # Fixed for this dataset
            **kwargs
        }

        super().__init__(
            dataset_class=TreeMapDataset,
            sensor_config=sensor_config,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=num_workers,
            **dataset_kwargs
        ) 