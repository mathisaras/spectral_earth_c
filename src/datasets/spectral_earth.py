import os
import random
from typing import Callable, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor

from torchgeo.datasets.geo import NonGeoDataset


class SpectralEarthDataset(NonGeoDataset):
    """
    Spectral Earth Dataset.

    Each patch has the following properties:

    - 128 x 128 pixels
    - Single multispectral GeoTIFF file
    - 202 spectral bands
    - 30 m spatial resolution
    """

    class _Metadata(TypedDict):
        num_bands: int
        rgb_bands: list[int]

    def __init__(
        self,
        root: str,
        sensor_config: dict,
        img_size: int = None,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        return_two_views: bool = False,
        patch_subset_path: str = None,
    ) -> None:
        """
        Initialize a new SpectralEarthDataset instance.

        Args:
            root: Root directory where the dataset can be found.
            sensor_config: Dictionary with sensor configuration (must contain 'name', 'num_bands', 'rgb_indices', and optionally 'img_size').
            img_size: Size of the patches (overrides sensor_config['img_size'] if provided).
            transforms: Optional transforms to apply to the samples.
            return_two_views: If True, returns two random views of the same patch.
        """
        self.sensor_config = sensor_config
        self.sensor = sensor_config["name"]
        self.root = root
        self.num_bands = sensor_config.get("num_bands", -1)
        self.rgb_indices = sensor_config.get("rgb_indices", [0, 1, 2])
        if img_size is not None:
            self.patch_size = img_size
        else:
            self.patch_size = sensor_config.get("img_size", 128)
        self.transforms = transforms
        self.return_two_views = return_two_views
        self.patch_paths = {}
        self.patch_subset_path = patch_subset_path

        # Build the patch_paths dictionary if no patch_subset_path is provided
        sensor_dir = os.path.join(self.root, self.sensor)
        if not os.path.exists(sensor_dir):
            raise FileNotFoundError(f"Sensor directory not found: {sensor_dir}")
        if self.patch_subset_path is None:
            patch_dirs = os.listdir(sensor_dir)
            for patch_dir in patch_dirs:
                patch_path = os.path.join(sensor_dir, patch_dir)
                if os.path.isdir(patch_path):
                    image_files = [
                        os.path.join(patch_path, f)
                        for f in os.listdir(patch_path)
                        if f.endswith(".tif")
                    ]
                    if image_files:
                        self.patch_paths[patch_dir] = image_files
        else:
            # We have a txt file with {patch_id}/{tile_id}.tif for every row
            with open(self.patch_subset_path, "r") as f:
                for line in f:
                    patch_id, tile_id = line.strip().split("/")
                    patch_path = os.path.join(sensor_dir, patch_id, f"{tile_id}.tif")
                    if patch_id not in self.patch_paths:
                        self.patch_paths[patch_id] = []
                    self.patch_paths[patch_id].append(patch_path)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        Return a data sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A dictionary containing the image(s).
        """
        patch_id = list(self.patch_paths.keys())[index]
        image_paths = self.patch_paths[patch_id]

        if self.return_two_views:
            if len(image_paths) >= 2:
                chosen_paths = random.sample(image_paths, 2)
            else:
                # If less than 2 images are available, duplicate the same image
                chosen_paths = [random.choice(image_paths)] * 2

            images = []
            for path in chosen_paths:
                with rasterio.open(path) as f:
                    image = torch.from_numpy(f.read().astype(np.int16))
                images.append(image)

            sample = {"image1": images[0], "image2": images[1]}
        else:
            chosen_path = random.choice(image_paths)
            with rasterio.open(chosen_path) as f:
                image = torch.from_numpy(f.read().astype(np.int16))
            sample = {"image": image}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of patches in the dataset."""
        return len(self.patch_paths)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot a sample from the dataset.

        Args:
            sample: A sample returned by `__getitem__`.
            show_titles: Whether to show titles above each panel.
            suptitle: Optional string for the figure's suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        if self.return_two_views:
            image = sample["image1"][self.rgb_indices].numpy()
        else:
            image = sample["image"][self.rgb_indices].numpy()

        image = image.transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
