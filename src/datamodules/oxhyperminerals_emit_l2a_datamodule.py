from __future__ import annotations

from typing import Any, List, Union

from .base_segmentation_datamodule import BaseSegmentationDataModule
from ..datasets.oxhyperminerals_emit_l2a_dataset import OxHyperMineralsEMITL2ADataset


class OxHyperMineralsEMITL2ADataModule(BaseSegmentationDataModule):
    """LightningDataModule for the patchified OxHyperMinerals EMIT L2A dataset."""

    def __init__(
        self,
        classes: List[int],
        class_names: list[str] | None,
        sensor_config: dict,
        root: str,
        batch_size: int = 64,
        img_size: Union[int, tuple[int, int], None] = 64,
        num_workers: int = 0,
        num_classes: int | None = None,
        ignore_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        del num_classes, ignore_index, class_names
        dataset_kwargs = {
            "classes": classes,
            "root": root,
            "product": "oxhyperminerals_emit_l2a",
            **kwargs,
        }
        super().__init__(
            dataset_class=OxHyperMineralsEMITL2ADataset,
            sensor_config=sensor_config,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=num_workers,
            **dataset_kwargs,
        )
