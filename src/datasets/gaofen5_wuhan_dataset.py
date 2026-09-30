from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset


RAW_CLASS_NAMES = OrderedDict(
    [
        (5, "Surface water"),
        (6, "Street"),
        (11, "Urban Fabric"),
        (12, "Industrial, commercial and transport"),
        (13, "Mine, dump and construction sites"),
        (14, "Artificial, vegetated areas"),
        (21, "Arable Land"),
        (22, "Permanent Crops"),
        (23, "Pastures"),
        (31, "Forests"),
        (32, "Shrub"),
        (33, "Open spaces with no vegetation"),
        (41, "Inland wetlands"),
    ]
)

RAW_CLASS_COLORS = {
    0: (0, 0, 0, 255),
    5: (0, 255, 255, 255),
    6: (255, 255, 255, 255),
    11: (255, 0, 0, 255),
    12: (221, 160, 221, 255),
    13: (148, 0, 211, 255),
    14: (255, 0, 255, 255),
    21: (255, 255, 0, 255),
    22: (205, 133, 63, 255),
    23: (189, 183, 107, 255),
    31: (0, 255, 0, 255),
    32: (154, 205, 50, 255),
    33: (139, 69, 19, 255),
    41: (72, 61, 139, 255),
}


class Gaofen5WuhanDataset(NonGeoDataset):
    """Patchified Wuhan GF-5 semantic-segmentation dataset.

    The patchification script writes:
    - HSI image patches under ``<root>/images``
    - label patches under ``<root>/labels``
    - metadata under ``<root>/metadata``

    Labels on disk keep the original raw Wuhan class ids, except that padded
    pixels use ``background_value`` (default ``0``). In the training mask that
    this dataset returns:
    - foreground classes map to contiguous indices ``0..N-1``
    - padded background maps to ``ignore_index = N``
    """

    DEFAULT_CLASSES = list(RAW_CLASS_NAMES.keys())

    def __init__(
        self,
        root: str,
        split: str,
        classes: Optional[list[int]] = None,
        product: str = "gaofen5_wuhan",
        sensor_config: Optional[Dict[str, Any]] = None,
        split_root: str = "data/splits",
        raw_mask: bool = False,
        background_value: int = 0,
        manifest_name: str = "patch_manifest.csv",
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.product = str(product)
        self.split_root = Path(split_root)
        self.raw_mask = bool(raw_mask)
        self.background_value = int(background_value)
        self.sensor_config = sensor_config or {}
        self.sensor = str(self.sensor_config.get("name", "gaofen5"))
        self.num_bands = int(self.sensor_config.get("num_bands", 116))
        self.rgb_indices = list(self.sensor_config.get("rgb_indices", [23, 13, 5]))

        classes = self.DEFAULT_CLASSES if classes is None else classes
        self.classes = sorted({int(c) for c in classes})
        if not self.classes:
            raise ValueError("Gaofen5WuhanDataset requires a non-empty class list.")
        if self.background_value in self.classes:
            raise ValueError(
                f"background_value={self.background_value} must not appear in classes={self.classes}."
            )

        unexpected_classes = [c for c in self.classes if c not in RAW_CLASS_NAMES]
        if unexpected_classes:
            raise ValueError(
                f"Unsupported raw Wuhan label ids: {unexpected_classes}. "
                f"Expected subset of {list(RAW_CLASS_NAMES.keys())}."
            )

        self.class_to_index = {raw_value: idx for idx, raw_value in enumerate(self.classes)}
        self.index_to_raw_class = {idx: raw_value for raw_value, idx in self.class_to_index.items()}
        self.ignore_index = len(self.classes)
        self.num_classes = len(self.classes)
        self.class_names = {idx: RAW_CLASS_NAMES[raw] for idx, raw in self.index_to_raw_class.items()}

        self.image_dir = self.root / "images"
        self.label_dir = self.root / "labels"
        self.metadata_dir = self.root / "metadata"
        self.manifest_path = self.metadata_dir / manifest_name

        split_file = self.split_root / self.product / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                "Stage a patchified GF-5 Wuhan dataset with matching split files first."
            )
        with split_file.open("r") as handle:
            self.sample_ids = [line.strip() for line in handle.readlines() if line.strip()]
        if not self.sample_ids:
            raise RuntimeError(f"No samples listed in split file {split_file}.")

        self.manifest_by_id: dict[str, dict[str, str]] = {}
        if self.manifest_path.exists():
            with self.manifest_path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    sample_id = row.get("sample_id")
                    if sample_id:
                        self.manifest_by_id[sample_id] = row

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_image(self, path: Path) -> Tensor:
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)
        tensor = torch.from_numpy(image)
        if tensor.shape[0] != self.num_bands:
            raise ValueError(
                f"Expected {self.num_bands} bands for {path}, found {tensor.shape[0]}."
            )
        return tensor

    def _load_mask(self, path: Path) -> Tensor:
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.int64)
        tensor = torch.from_numpy(mask).long()

        if self.raw_mask:
            return tensor

        unique_values = sorted(int(v) for v in torch.unique(tensor).tolist())
        unexpected = [
            raw_value
            for raw_value in unique_values
            if raw_value != self.background_value and raw_value not in self.class_to_index
        ]
        if unexpected:
            raise ValueError(
                f"Unexpected raw Wuhan labels in {path}: {unexpected}. "
                f"Expected {self.classes} plus background_value={self.background_value}."
            )

        remapped = torch.full_like(tensor, fill_value=self.ignore_index)
        for raw_value, mapped_value in self.class_to_index.items():
            remapped[tensor == raw_value] = mapped_value
        return remapped

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range.")

        sample_id = self.sample_ids[index]
        image_path = self.image_dir / sample_id
        label_path = self.label_dir / sample_id
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image patch: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label patch: {label_path}")

        sample = {
            "image": self._load_image(image_path),
            "mask": self._load_mask(label_path),
            "sample_id": sample_id,
            "path": str(image_path),
            "label_path": str(label_path),
        }
        manifest_row = self.manifest_by_id.get(sample_id)
        if manifest_row:
            sample["metadata"] = manifest_row
        return sample

    def _mapped_cmap(self) -> np.ndarray:
        cmap = np.zeros((self.num_classes + 1, 4), dtype=np.uint8)
        for mapped_value, raw_value in self.index_to_raw_class.items():
            cmap[mapped_value] = np.asarray(RAW_CLASS_COLORS[raw_value], dtype=np.uint8)
        cmap[self.ignore_index] = np.asarray(RAW_CLASS_COLORS[self.background_value], dtype=np.uint8)
        return cmap

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        ncols = 2 + int("prediction" in sample)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        if ncols == 1:
            axes = [axes]

        image = sample["image"][self.rgb_indices].detach().cpu().numpy().transpose(1, 2, 0)
        lo = float(np.percentile(image, 2.0))
        hi = float(np.percentile(image, 98.0))
        image = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

        mask = sample["mask"].detach().cpu().numpy()
        cmap = self._mapped_cmap()

        axes[0].imshow(image)
        axes[0].axis("off")
        axes[1].imshow(cmap[mask], interpolation="none")
        axes[1].axis("off")

        if show_titles:
            axes[0].set_title("HSI RGB")
            axes[1].set_title("Mask")

        if "prediction" in sample:
            prediction = sample["prediction"].detach().cpu().numpy()
            if prediction.ndim == 3 and prediction.shape[0] == 1:
                prediction = prediction[0]
            axes[2].imshow(cmap[prediction], interpolation="none")
            axes[2].axis("off")
            if show_titles:
                axes[2].set_title("Prediction")

        if suptitle is not None:
            fig.suptitle(suptitle)
        fig.tight_layout()
        return fig
