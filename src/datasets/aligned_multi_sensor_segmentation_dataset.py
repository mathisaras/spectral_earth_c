"""Aligned multi-sensor wrapper for segmentation datasets."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Mapping

from torch.utils.data import Dataset


def _sensor_key(name: str) -> str:
    return str(name).strip().upper()


class AlignedMultiSensorSegmentationDataset(Dataset):
    """
    Wrap several single-sensor segmentation datasets for the same split.

    The wrapped datasets are expected to use aligned split files, where the same
    sample id appears at the same index for every sensor. The wrapper returns
    ``image_by_sensor`` for the model and keeps one reference ``mask``/``image``
    for logging and target sizing.
    """

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        output_sensor: str | None = None,
        strict_alignment: bool = True,
    ) -> None:
        if not datasets:
            raise ValueError("datasets must not be empty.")

        self.datasets = OrderedDict((_sensor_key(k), v) for k, v in datasets.items())
        self.output_sensor = _sensor_key(output_sensor or next(iter(self.datasets)))
        if self.output_sensor not in self.datasets:
            raise ValueError(
                f"output_sensor={self.output_sensor!r} is not among "
                f"{list(self.datasets.keys())}."
            )

        lengths = {sensor: len(dataset) for sensor, dataset in self.datasets.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Aligned datasets have different lengths: {lengths}")
        self._length = next(iter(lengths.values()))

        if strict_alignment:
            self._validate_alignment()

        ref_dataset = self.datasets[self.output_sensor]
        for attr in (
            "product",
            "split",
            "foreground_classes_raw",
            "num_effective_classes",
            "ignore_index",
            "ordinal_cmap",
        ):
            if hasattr(ref_dataset, attr):
                setattr(self, attr, getattr(ref_dataset, attr))

    def _sample_id_at(self, dataset: Dataset, index: int) -> str:
        sample_collection = getattr(dataset, "sample_collection", None)
        img_dir_path = getattr(dataset, "img_dir_path", None)
        if sample_collection is None or img_dir_path is None:
            raise TypeError(
                "AlignedMultiSensorSegmentationDataset requires wrapped datasets "
                "to expose sample_collection and img_dir_path."
            )
        img_path = sample_collection[index][0]
        return os.path.normpath(os.path.relpath(img_path, img_dir_path))

    def _validate_alignment(self) -> None:
        if self._length == 0:
            return

        ref_sensor = next(iter(self.datasets))
        ref_dataset = self.datasets[ref_sensor]
        check_indices = sorted({0, self._length // 2, self._length - 1})
        for idx in check_indices:
            expected = self._sample_id_at(ref_dataset, idx)
            for sensor, dataset in self.datasets.items():
                observed = self._sample_id_at(dataset, idx)
                if observed != expected:
                    raise ValueError(
                        "Joint split alignment mismatch at index "
                        f"{idx}: {sensor} has {observed!r}, expected {expected!r}."
                    )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        ref = self.datasets[self.output_sensor][index]
        samples = OrderedDict([(self.output_sensor, ref)])

        for sensor, dataset in self.datasets.items():
            if sensor == self.output_sensor:
                continue
            sample_collection = getattr(dataset, "sample_collection", None)
            load_image = getattr(dataset, "_load_image", None)
            transforms = getattr(dataset, "transforms", None)
            if sample_collection is not None and load_image is not None and transforms is None:
                img_path = sample_collection[index][0]
                samples[sensor] = {
                    "image": load_image(img_path),
                    "path": img_path,
                }
            else:
                samples[sensor] = dataset[index]

        image_by_sensor = OrderedDict(
            (sensor, sample["image"]) for sensor, sample in samples.items()
        )
        path_by_sensor = OrderedDict(
            (sensor, sample.get("path", "")) for sensor, sample in samples.items()
        )

        return {
            "image": image_by_sensor[self.output_sensor],
            "image_by_sensor": image_by_sensor,
            "mask": ref["mask"],
            "path": ref.get("path", ""),
            "path_by_sensor": path_by_sensor,
        }
