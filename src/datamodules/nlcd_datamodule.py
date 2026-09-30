from typing import Any, List, Optional, Union

from .base_segmentation_datamodule import BaseSegmentationDataModule
from ..datasets.nlcd_dataset import NLCDDataset

class NLCDDataModule(BaseSegmentationDataModule):
    """LightningDataModule for the National Land Cover Database (NLCD) dataset."""

    def __init__(
        self,
        sensor_config: dict,  # Passed directly from hydra config
        product: str = "nlcd", # Product name, e.g., "nlcd_2019", must start with "nlcd"
        classes: Optional[List[int]] = None, # Optional foreground classes, if None uses all NLCD classes
        root: str = "data",
        batch_size: int = 64,
        img_size: Union[int, tuple[int, int]] = 128,
        num_workers: int = 0,
        subset_percent: Optional[float] = None, # For using just a subset of samples
        **kwargs: Any,
    ) -> None:
        """
        Args:
            sensor: Sensor configuration from hydra
            product: Product name, must start with "nlcd" (e.g., "nlcd_2019")
            classes: List of raw foreground class values, if None uses all NLCD classes
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
            "subset_percent": subset_percent,
            **kwargs
        }

        super().__init__(
            dataset_class=NLCDDataset,
            sensor_config=sensor_config,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=num_workers,
            **dataset_kwargs
        ) 