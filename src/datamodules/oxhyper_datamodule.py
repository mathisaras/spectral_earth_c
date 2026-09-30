from __future__ import annotations
 
from typing import Any, List, Union
 
from .base_segmentation_datamodule import BaseSegmentationDataModule
from ..datasets.oxhyper_dataset import OxhyperDataset
 
 
class OxhyperDataModule(BaseSegmentationDataModule):
    """LightningDataModule for the patchified EnMAP+EMIT OxHyper dataset."""
 
    def __init__(
        self,
        classes: List[int],
        class_names: list[str] | None,
        sensor_config: dict,
        root: str,
        batch_size: int = 64,
        img_size: Union[int, tuple[int, int], None] = 128,
        num_workers: int = 0,
        num_classes: int | None = None,
        ignore_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        del num_classes, ignore_index, class_names
        dataset_kwargs = {
            "classes": classes,
            "root": root,
            "product": "oxhyper",
            **kwargs,
        }
        super().__init__(
            dataset_class=OxhyperDataset,
            sensor_config=sensor_config,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=num_workers,
            **dataset_kwargs,
        )