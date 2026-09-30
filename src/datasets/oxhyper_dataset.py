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


REPO_ROOT = Path(__file__).resolve().parents[2]


LABEL_PRODUCT_CLASS_NAMES = {
    "minerals9": OrderedDict(
        [
            (0, "Calcite"),
            (1, "Chlorite"),
            (2, "Dolomite"),
            (3, "Goethite"),
            (4, "Gypsum"),
            (5, "Hematite"),
            (6, "Illite+Muscovite"),
            (7, "Kaolinite"),
            (8, "Montmorillonite"),
        ]
    ),
}

LABEL_PRODUCT_CLASS_COLORS = {
    "minerals9": {
        0: (0, 114, 178, 255),
        1: (0, 158, 115, 255),
        2: (204, 121, 167, 255),
        3: (230, 159, 0, 255),
        4: (240, 228, 66, 255),
        5: (213, 94, 0, 255),
        6: (86, 180, 233, 255),
        7: (0, 114, 178, 255),
        8: (148, 103, 189, 255),
    },
}

IGNORE_COLOR = (255, 221, 87, 255)


def _unique_preserve_order(values: list[int]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        value = int(value)
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _resolve_path(path_value: str | Path, *, base_dir: Path = REPO_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


class OxhyperDataset(NonGeoDataset):
    """Patchified EnMAP+EMIT OxHyper multi-label segmentation dataset.
    10th band from original "minerals10" dataset removed, leaving 9 classes.
    - multiple bands can be active at the same pixel, so this is multi-label
      segmentation rather than single-label multiclass segmentation
    """

    DEFAULT_CLASSES = {
        product: list(class_names.keys())
        for product, class_names in LABEL_PRODUCT_CLASS_NAMES.items()
    }

    def __init__(
        self,
        root: str,
        split: str,
        classes: Optional[list[int]] = None,
        product: str = "oxhyper",
        label_product: str = "minerals9",
        sensor_config: Optional[Dict[str, Any]] = None,
        split_root: str = "data/splits",
        split_dir: Optional[str] = None,
        split_name: Optional[str] = None,
        raw_mask: bool = False,
        raw_ignore_value: int = 255,
        ignore_value: float = -1.0,
        invalid_value: float = -9999.0,
        image_fill_value: float = 0.0,
        manifest_name: str = "patch_manifest.csv",
    ) -> None:
        self.root = _resolve_path(root)
        self.split = str(split)
        self.product = str(product)
        self.label_product = str(label_product)
        self.split_root = _resolve_path(split_root)
        self.split_dir = (
            _resolve_path(split_dir)
            if split_dir is not None
            else (self.split_root / self.product).resolve()
        )
        self.split_name = str(split_name or self.split_dir.name)
        self.raw_mask = bool(raw_mask)
        self.raw_ignore_value = int(raw_ignore_value)
        self.ignore_value = float(ignore_value)
        self.invalid_value = float(invalid_value)
        self.image_fill_value = float(image_fill_value)
        if sensor_config is None:
            raise ValueError(
                "sensor_config is required - pass a real sensor config, e.g. "
                "{'name': 'emit', 'num_bands': 244, 'rgb_indices': [45, 30, 15]} "
                "or {'name': 'enmap', 'num_bands': 202, 'rgb_indices': [43, 28, 10]}."
            )
        self.sensor_config = sensor_config
        self.sensor = str(self.sensor_config["name"])
        self.num_bands = int(self.sensor_config["num_bands"])
        self.rgb_indices = list(self.sensor_config["rgb_indices"])

        if self.label_product not in LABEL_PRODUCT_CLASS_NAMES:
            raise ValueError(
                f"Unsupported label_product='{self.label_product}'. "
                f"Expected one of {sorted(LABEL_PRODUCT_CLASS_NAMES)}."
            )

        classes = self.DEFAULT_CLASSES[self.label_product] if classes is None else list(classes)
        self.classes = _unique_preserve_order([int(c) for c in classes])
        if not self.classes:
            raise ValueError("OxhyperDataset requires a non-empty class list.")

        allowed_channels = set(LABEL_PRODUCT_CLASS_NAMES[self.label_product].keys())
        unexpected_classes = [c for c in self.classes if c not in allowed_channels]
        if unexpected_classes:
            raise ValueError(
                f"Unsupported class channel ids for {self.label_product}: {unexpected_classes}. "
                f"Expected subset of {sorted(allowed_channels)}."
            )

        self.num_classes = len(self.classes)
        self.class_names = {
            channel_idx: LABEL_PRODUCT_CLASS_NAMES[self.label_product][channel_idx]
            for channel_idx in self.classes
        }

        self.image_dir = self.root / "images"
        self.label_dir = self.root / "labels"
        self.metadata_dir = self.root / "metadata"
        self.manifest_path = self.metadata_dir / manifest_name

        split_file = self.split_dir / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                "Stage the patchified dataset and matching split directory first."
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
            nodata = src.nodata

        invalid_mask = np.zeros_like(image, dtype=bool)
        if np.isfinite(self.invalid_value):
            invalid_mask |= image == self.invalid_value
        if nodata is not None and np.isfinite(float(nodata)):
            invalid_mask |= image == float(nodata)
        invalid_mask |= ~np.isfinite(image)
        if invalid_mask.any():
            image[invalid_mask] = self.image_fill_value

        tensor = torch.from_numpy(image)
        if tensor.shape[0] != self.num_bands:
            raise ValueError(
                f"Expected {self.num_bands} bands for {path}, found {tensor.shape[0]}."
            )
        return tensor

    def _load_mask(self, path: Path) -> Tensor:
        with rasterio.open(path) as src:
            mask = src.read().astype(np.float32)

        mask = mask[self.classes, :, :]
        tensor = torch.from_numpy(mask)

        if self.raw_mask:
            return tensor.to(torch.uint8)

        ignore_mask = tensor == float(self.raw_ignore_value)
        unexpected = tensor[~ignore_mask]
        if unexpected.numel() > 0:
            unique_values = sorted(float(v) for v in torch.unique(unexpected).tolist())
            if any(v not in (0.0, 1.0) for v in unique_values):
                raise ValueError(
                    f"Unexpected raw labels in {path}: {unique_values}. "
                    f"Expected 0/1 plus raw_ignore_value={self.raw_ignore_value}."
                )

        tensor = tensor.float()
        tensor[ignore_mask] = self.ignore_value
        return tensor

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

        mask = self._load_mask(label_path)
        ignore_mask = (mask < 0).any(dim=0)
        sample = {
            "image": self._load_image(image_path),
            "mask": mask,
            "ignore_mask": ignore_mask,
            "sample_id": sample_id,
            "path": str(image_path),
            "label_path": str(label_path),
            "label_product": self.label_product,
        }
        manifest_row = self.manifest_by_id.get(sample_id)
        if manifest_row:
            sample["metadata"] = manifest_row
        return sample

    def _render_multilabel(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim != 3:
            raise ValueError(f"Expected mask with shape (C,H,W), got {mask.shape}")

        valid_mask = ~(mask < 0).any(axis=0)
        binary_mask = mask > 0.5
        h, w = binary_mask.shape[1:]
        rendered = np.zeros((h, w, 4), dtype=np.float32)

        for local_idx, class_channel in enumerate(self.classes):
            color = np.asarray(
                LABEL_PRODUCT_CLASS_COLORS[self.label_product][class_channel],
                dtype=np.float32,
            ) / 255.0
            rendered[binary_mask[local_idx]] += color

        active_counts = binary_mask.sum(axis=0, keepdims=False).astype(np.float32)
        active_counts = np.maximum(active_counts, 1.0)
        positive_pixels = binary_mask.any(axis=0)
        rendered[positive_pixels] /= active_counts[positive_pixels, None]
        rendered[~positive_pixels, :] = np.asarray((0, 0, 0, 255), dtype=np.float32) / 255.0
        rendered[~valid_mask, :] = np.asarray(IGNORE_COLOR, dtype=np.float32) / 255.0
        return rendered

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

        axes[0].imshow(image)
        axes[0].axis("off")
        axes[1].imshow(self._render_multilabel(mask), interpolation="none")
        axes[1].axis("off")

        if show_titles:
            axes[0].set_title("RGB")
            axes[1].set_title(f"Labels ({self.label_product})")

        if "prediction" in sample:
            prediction = sample["prediction"].detach().cpu().numpy()
            if prediction.ndim == 4 and prediction.shape[0] == 1:
                prediction = prediction[0]
            if prediction.ndim != 3:
                raise ValueError(
                    f"Expected prediction with shape (C,H,W), got {prediction.shape}"
                )
            if np.issubdtype(prediction.dtype, np.floating):
                prediction = (prediction > 0.5).astype(np.float32)
            axes[2].imshow(self._render_multilabel(prediction), interpolation="none")
            axes[2].axis("off")
            if show_titles:
                axes[2].set_title("Prediction")

        if suptitle is not None:
            fig.suptitle(suptitle)
        fig.tight_layout()
        return fig