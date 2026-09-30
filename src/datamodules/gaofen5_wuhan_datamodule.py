from __future__ import annotations

from typing import Any, List, Union

from .base_segmentation_datamodule import BaseSegmentationDataModule
from ..datasets.gaofen5_wuhan_dataset import Gaofen5WuhanDataset


class Gaofen5WuhanDataModule(BaseSegmentationDataModule):
    """LightningDataModule for the patchified Wuhan GF-5 dataset."""

    def __init__(
        self,
        classes: List[int],
        sensor_config: dict,
        root: str,
        batch_size: int = 64,
        img_size: Union[int, tuple[int, int], None] = 128,
        num_workers: int = 0,
        num_classes: int | None = None,
        ignore_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        del num_classes, ignore_index
        dataset_kwargs = {
            "classes": classes,
            "root": root,
            "product": "gaofen5_wuhan",
            **kwargs,
        }
        super().__init__(
            dataset_class=Gaofen5WuhanDataset,
            sensor_config=sensor_config,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=num_workers,
            **dataset_kwargs,
        )
