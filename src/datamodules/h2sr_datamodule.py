from typing import Any, List, Union

from .base_segmentation_datamodule import BaseSegmentationDataModule
from ..datasets.h2sr_dataset import H2SRDataset


class H2SRDataModule(BaseSegmentationDataModule):
    """LightningDataModule for the H2SR semantic segmentation dataset."""

    def __init__(
        self,
        classes: List[int],
        sensor_config: dict,  # Passed directly from hydra config
        root: str = "data",
        batch_size: int = 64,
        img_size: Union[int, tuple[int, int], None] = None,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            classes: List of raw foreground class values
            sensor_config: Sensor configuration from hydra
            root: Root directory where dataset can be found
            batch_size: Size of each mini-batch
            img_size: Output spatial size. If None, the base datamodule resolves
                it from the target sensor config when remapping is enabled, or
                from the source sensor config otherwise.
            num_workers: Number of workers for data loading
            **kwargs: Additional keyword arguments passed to the dataset
        """
        dataset_kwargs = {
            "classes": classes,
            "root": root,
            "product": "h2sr",  # Fixed for this dataset
            **kwargs
        }

        super().__init__(
            dataset_class=H2SRDataset,
            sensor_config=sensor_config,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=num_workers,
            **dataset_kwargs
        ) 
