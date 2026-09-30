from pathlib import Path
from typing import Any, Dict, Optional

import rasterio
import torch
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset


class H2SRDataset(NonGeoDataset):
    """H2SR semantic segmentation dataset (Non-Geo).

    Label convention on disk:
    - Foreground semantic classes are encoded directly as raw ids 0..7.
    - Background is encoded as raw value 15.

    In the processed training mask returned by this dataset:
    - Foreground classes keep contiguous indices 0..7.
    - Background is remapped to ignore_index = len(classes) by default.
    """

    def __init__(
        self,
        root: str,
        split: str,
        classes: list[int],
        product: str = "h2sr",
        sensor_config: Optional[Dict[str, Any]] = None,
        split_root: str = "data/splits",
        raw_mask: bool = False,
        background_value: int = 15,
        patch_size: Optional[int | tuple[int, int]] = None,
        patch_stride: Optional[int | tuple[int, int]] = None,
        patch_source_size: Optional[int | tuple[int, int]] = None,
        drop_incomplete_patches: bool = True,
    ) -> None:
        """
        Args:
            root: Root directory of dataset
            split: One of 'train', 'val', 'test'
            classes: List of valid raw foreground class ids from disk
            product: Product identifier
            sensor_config: Optional sensor configuration dict
            split_root: Root directory containing split files
            raw_mask: If True, return the raw mask without remapping
            background_value: Raw background value in the label rasters
            patch_size: Optional spatial patch size. If provided, every split
                tile is exposed as virtual patches while preserving the split.
            patch_stride: Optional patch stride. Defaults to patch_size for
                non-overlapping patches.
            patch_source_size: Optional known source tile size. Providing this
                avoids opening every raster during dataset initialization.
            drop_incomplete_patches: If True, keep only full patches. If False,
                also include edge-aligned partial-coverage windows.
        """
        self.root = self._resolve_root(Path(root))
        self.split = split
        self.product = product
        self.split_root = split_root
        self.raw_mask = bool(raw_mask)
        self.sensor_config = sensor_config or {}
        self.sensor = self.sensor_config.get("name", "ammis")
        self.num_bands = self.sensor_config.get("num_bands")
        self.rgb_indices = self.sensor_config.get("rgb_indices", [0, 1, 2])
        self.background_value = int(background_value)

        self.classes = sorted({int(c) for c in classes})
        if not self.classes:
            raise ValueError("H2SRDataset requires a non-empty list of foreground classes.")
        if self.background_value in self.classes:
            raise ValueError(
                f"Background value {self.background_value} must not appear in classes={self.classes}."
            )

        self.class_to_index = {raw_class: idx for idx, raw_class in enumerate(self.classes)}
        self.ignore_index = len(self.classes)

        split_file = Path(self.split_root) / self.product / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                "Expected repo-managed split files under "
                f"{Path(self.split_root) / self.product}."
            )

        with split_file.open("r") as f:
            self.files = [line.strip() for line in f.readlines() if line.strip()]

        if len(self.files) == 0:
            raise RuntimeError(f"No files found in split {split}")

        self.patch_size = self._to_hw_pair(patch_size, name="patch_size")
        self.patch_stride = self._to_hw_pair(
            patch_stride if patch_stride is not None else patch_size,
            name="patch_stride",
        )
        self.patch_source_size = self._to_hw_pair(
            patch_source_size,
            name="patch_source_size",
        )
        self.drop_incomplete_patches = bool(drop_incomplete_patches)
        if self.patch_size is not None:
            if self.patch_stride is None:
                raise ValueError("patch_stride could not be resolved.")
            self.patches = self._build_patch_index()
        else:
            self.patches = None

    @staticmethod
    def _resolve_root(root: Path) -> Path:
        """Allow both dataset roots: .../H2SR_dataset and .../H2SR_dataset/h2sr."""
        if (root / "Input").is_dir() and (root / "Labels").is_dir():
            return root

        nested = root / "h2sr"
        if (nested / "Input").is_dir() and (nested / "Labels").is_dir():
            return nested

        raise FileNotFoundError(
            f"Could not resolve H2SR dataset root from {root}. "
            "Expected either <root>/Input + <root>/Labels or <root>/h2sr/Input + <root>/h2sr/Labels."
        )

    @staticmethod
    def _to_hw_pair(
        value: Optional[int | tuple[int, int]],
        *,
        name: str,
    ) -> Optional[tuple[int, int]]:
        if value is None:
            return None
        if isinstance(value, int):
            height = width = int(value)
        else:
            if len(value) != 2:
                raise ValueError(f"{name} must be an int or length-2 tuple, got {value}.")
            height, width = int(value[0]), int(value[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"{name} must be positive, got {(height, width)}.")
        return height, width

    @staticmethod
    def _patch_positions(
        length: int,
        patch: int,
        stride: int,
        *,
        drop_incomplete: bool,
    ) -> list[int]:
        if length < patch:
            if drop_incomplete:
                return []
            return [0]

        positions = list(range(0, length - patch + 1, stride))
        if not positions:
            positions = [0]

        if not drop_incomplete:
            last = length - patch
            if positions[-1] != last:
                positions.append(last)
        return positions

    def _build_patch_index(self) -> list[dict[str, Any]]:
        patch_h, patch_w = self.patch_size
        stride_h, stride_w = self.patch_stride
        patch_index: list[dict[str, Any]] = []

        for rel_path in self.files:
            if self.patch_source_size is not None:
                image_height, image_width = self.patch_source_size
            else:
                image_path, label_path = self._build_paths(rel_path)
                with rasterio.open(image_path) as image_src:
                    image_height, image_width = int(image_src.height), int(image_src.width)
                with rasterio.open(label_path) as label_src:
                    label_height, label_width = int(label_src.height), int(label_src.width)

                if (image_height, image_width) != (label_height, label_width):
                    raise ValueError(
                        f"H2SR image/label size mismatch for {rel_path}: "
                        f"image={(image_height, image_width)} label={(label_height, label_width)}."
                    )

            top_positions = self._patch_positions(
                image_height,
                patch_h,
                stride_h,
                drop_incomplete=self.drop_incomplete_patches,
            )
            left_positions = self._patch_positions(
                image_width,
                patch_w,
                stride_w,
                drop_incomplete=self.drop_incomplete_patches,
            )

            for top in top_positions:
                for left in left_positions:
                    patch_index.append(
                        {
                            "rel_path": rel_path,
                            "top": int(top),
                            "left": int(left),
                            "height": int(min(patch_h, image_height - top)),
                            "width": int(min(patch_w, image_width - left)),
                        }
                    )

        if not patch_index:
            raise RuntimeError(
                f"No patches found for split {self.split} with patch_size={self.patch_size}, "
                f"patch_stride={self.patch_stride}, drop_incomplete_patches={self.drop_incomplete_patches}."
            )
        return patch_index

    def __len__(self) -> int:
        if self.patches is not None:
            return len(self.patches)
        return len(self.files)

    def _build_paths(self, rel_path: str) -> tuple[Path, Path]:
        """
        Convert split entry like:
            H2SR_image_6/137_0_1.tif

        To:
            Input/H2SR_image_6/137_0_1.tif
            Labels/H2SR_label_6/137_0_1.tif
        """
        image_path = self.root / "Input" / rel_path

        # Replace folder name
        label_rel = rel_path.replace("H2SR_image_", "H2SR_label_")
        label_path = self.root / "Labels" / label_rel

        return image_path, label_path

    def _load_image(self, path: Path) -> Tensor:
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)

        img = torch.from_numpy(img).float()
        if self.num_bands is not None and img.shape[0] != int(self.num_bands):
            raise ValueError(
                f"H2SR image {path} has {img.shape[0]} bands, expected {self.num_bands}."
            )

        return img

    def _load_label(self, path: Path) -> Tensor:
        with rasterio.open(path) as src:
            label = src.read(1)  # single channel

        label = torch.from_numpy(label).long()
        if self.raw_mask:
            return label

        unique_values = sorted(torch.unique(label).tolist())
        unexpected_values = [
            int(v)
            for v in unique_values
            if int(v) != self.background_value and int(v) not in self.class_to_index
        ]
        if unexpected_values:
            raise ValueError(
                f"Unexpected raw H2SR labels in {path}: {unexpected_values}. "
                f"Expected foreground classes {self.classes} plus background {self.background_value}."
            )

        remapped = torch.full_like(label, fill_value=self.ignore_index)
        for raw_class, mapped_class in self.class_to_index.items():
            remapped[label == raw_class] = mapped_class
        return remapped

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        patch_info = None
        if self.patches is not None:
            patch_info = self.patches[index]
            rel_path = patch_info["rel_path"]
        else:
            rel_path = self.files[index]
        image_path, label_path = self._build_paths(rel_path)

        image = self._load_image(image_path)
        label = self._load_label(label_path)

        sample_id = rel_path
        if patch_info is not None:
            top = int(patch_info["top"])
            left = int(patch_info["left"])
            height = int(patch_info["height"])
            width = int(patch_info["width"])
            if image.shape[-2] < top + height or image.shape[-1] < left + width:
                raise ValueError(
                    f"H2SR image {image_path} is smaller than configured patch window "
                    f"(top={top}, left={left}, height={height}, width={width}); "
                    f"image shape is {tuple(image.shape[-2:])}."
                )
            if label.shape[-2] < top + height or label.shape[-1] < left + width:
                raise ValueError(
                    f"H2SR label {label_path} is smaller than configured patch window "
                    f"(top={top}, left={left}, height={height}, width={width}); "
                    f"label shape is {tuple(label.shape[-2:])}."
                )
            image = image[..., top : top + height, left : left + width]
            label = label[top : top + height, left : left + width]
            sample_id = f"{rel_path}::patch_t{top}_l{left}_h{height}_w{width}"

        sample = {
            "image": image,
            "mask": label,
            "path": str(image_path),
            "label_path": str(label_path),
            "sample_id": sample_id,
        }
        if patch_info is not None:
            sample["patch"] = patch_info
        return sample
    

# ==========================================================
#                  MANUAL INITIALIZATION
# ==========================================================

if __name__ == "__main__":
    dataset = H2SRDataset(
        root="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0020/H2SR_dataset/h2sr",
        split="train",
        classes=list(range(8)),
        product="h2sr"
    )

    print(dataset)
    print(f"Dataset length: {len(dataset)}")

    sample = dataset[100]

    print("image shape:", sample["image"].shape)
    print("mask shape:", sample["mask"].shape)
    print("image dtype:", sample["image"].dtype)
    print("mask dtype:", sample["mask"].dtype)
    print("Unique mask values:", torch.unique(sample["mask"]))

    # Value range diagnostics
    image = sample["image"]
    print("\n--- Hyperspectral image value ranges ---")
    print(f"  Global min:  {image.min().item():.4f}")
    print(f"  Global max:  {image.max().item():.4f}")
    print(f"  Global mean: {image.mean().item():.4f}")
    print(f"  Global std:  {image.std().item():.4f}")
    print(f"  NaN count:   {torch.isnan(image).sum().item()}")
    print(f"  Inf count:   {torch.isinf(image).sum().item()}")
