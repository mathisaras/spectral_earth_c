from collections import defaultdict
import glob
import os
import random
from typing import Callable, Optional, List, Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset


class ConusminDataset(NonGeoDataset):  
    """EMIT CONUSMin Mineral Dataset for multi-label classification.  

    Dataset intended for evaluating SSL techniques with images
    and corresponding multi-label mineral classifications (which minerals
    are present anywhere in the patch, one multi-hot vector per patch).

    Supports 18 mineral classes.  
    """

    class_names: List[str] = [
    "Eolian sediment",
    "Alluvial sediment, undifferentiated",
    "Young alluvial sediment",
    "Felsic volcanic rocks",
    "Playa sediment",
    "Granitic rocks",
    "Clastic sediments, undifferentiated",
    "Extrusive igneous rocks",
    "Clastic sediments, undifferentiated",
    "Old alluvial sediment",
    "Undifferentiated Precambrian rocks",
    "Sedimentary rocks",
    "Sedimentary rocks",
    "Sedimentary rocks",
    "Sedimentary rocks",
    "Sedimentary rocks",
    "Sedimentary and metasedimentary rocks",
    "Sedimentary rocks",
    ]

    code_to_ordinal_dict = {
        8: 0, 9: 1, 10: 2, 45: 3, 49: 4, 50: 5, 59: 6, 60: 7, 63: 8,
        64: 9, 81: 10, 82: 11, 91: 12, 93: 13, 102: 14, 110: 15, 113: 16, 115: 17,
    }

   
    code_to_ordinal = defaultdict(lambda: 18, code_to_ordinal_dict)  
    # Define path formats, similar to BaseSegmentationDataset for consistency
    IMAGE_ROOT_FORMAT: str = "images"  # e.g. data/emit/
    MASK_ROOT_FORMAT: str = "labels"  # e.g. data/emit_conusmin/ (masks used to derive labels)
    SPLIT_PATH_FORMAT: str = "data/splits/{product}/{sensor}/{split}.txt"

    
    RGB_INDICES: Dict[str, List[int]] = {
        "emit": [39, 24, 14],
    }

    

    def __init__(
        self,
        root: str = "data",
        sensor_config: Dict[str, Any] = None,
        product: str = "conusmin",  
        split: str = "train",
        num_classes: int = 18,  
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        return_mask: bool = False,  # If true, returns the spatial mask used for label derivation
        subset_percent: Optional[float] = None,
    ) -> None:
        super().__init__()

        if sensor_config is None:
            raise ValueError("sensor_config is required")

        self.root = root
        self.sensor_config = sensor_config
        self.sensor = sensor_config["name"]
        self.num_bands = sensor_config.get("num_bands", -1)
        self.rgb_indices = sensor_config.get("rgb_indices", [0, 1, 2])

        self.product = product
        self.split = split
        self.num_classes = num_classes
        self.transforms = transforms
        self.return_mask = return_mask
        self.subset_percent = subset_percent

        
        if self.num_classes != 18:
            raise ValueError("Number of classes must be 18.")  
       
        # list above, not from a class_sets[self.num_classes] lookup.

        self.img_dir_path = os.path.join(self.root, self.IMAGE_ROOT_FORMAT.format(sensor=self.sensor))
        self.mask_dir_path = os.path.join(self.root, self.MASK_ROOT_FORMAT.format(product=self.product))

        # Construct split file path using the new format
        current_split_file = self.SPLIT_PATH_FORMAT.format(sensor=self.sensor, product=self.product, split=self.split)

        # Fallback if sensor_product path doesn't exist (e.g. more generic split)
        if not os.path.exists(current_split_file):
            alt_split_format = "splits/{product}/{split}.txt"
            alt_split_file = os.path.join(self.root, alt_split_format.format(product=self.product, split=self.split))
            if os.path.exists(alt_split_file):
                current_split_file = alt_split_file
            else:
                raise FileNotFoundError(f"Split file not found. Checked: {current_split_file} and {alt_split_file}")
        self.split_file = current_split_file

        self.sample_collection = self._read_split_file()

        if self.subset_percent is not None:
            if not (0.0 < self.subset_percent <= 1.0):
                raise ValueError("subset_percent must be between 0.0 and 1.0.")
            num_samples = int(len(self.sample_collection) * self.subset_percent)
            if num_samples == 0 and len(self.sample_collection) > 0:
                num_samples = 1  # ensure at least one sample
            self.sample_collection = random.sample(self.sample_collection, num_samples)

    def _read_split_file(self) -> List[Tuple[str, str]]:
        """Reads a split file containing image identifiers (one per line).
        Returns a list of (image_path, mask_path) tuples.
        """
        with open(self.split_file, "r") as f:
            sample_ids = [line.strip() for line in f.readlines() if line.strip()]

        sample_list = []
        for sample_id in sample_ids:
            img_path = os.path.join(self.img_dir_path, sample_id)
            mask_path = os.path.join(self.mask_dir_path, sample_id)
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found {img_path} from split file {self.split_file}")
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file not found {mask_path} from split file {self.split_file}")
            sample_list.append((img_path, mask_path))

        if not sample_list:
            print(f"Warning: No samples loaded from split file {self.split_file}. Check paths and content.")
        return sample_list

    def __len__(self) -> int:
        return len(self.sample_collection)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        img_path, mask_path = self.sample_collection[index]

        image = self._load_image(img_path)
        label = self._load_label(mask_path)  # This derives multi-label from mask

        sample = {"image": image, "label": label, "path": img_path}

        if self.return_mask:
            # Optionally return the spatial mask as well
            sample["mask"] = self._load_mask(mask_path)

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def _load_image(self, path: str) -> Tensor:
        with rasterio.open(path) as src:
            image_data = src.read()
        if image_data.ndim == 3:
            if self.num_bands != -1 and image_data.shape[0] != self.num_bands:
                if image_data.shape[2] == self.num_bands:  # (H, W, C)
                    image_data = np.transpose(image_data, (2, 0, 1))
                else:
                    print(f"Warning: Expected {self.num_bands} bands, image {path} has {image_data.shape[0]}. Using as is or first {self.num_bands} if more.")
                    if image_data.shape[0] > self.num_bands:
                        image_data = image_data[:self.num_bands, :, :]
            elif self.num_bands == -1:
                self.num_bands = image_data.shape[0]  # Auto-detect from first image

        return torch.from_numpy(image_data.astype(np.float32)).float()

    def _load_mask(self, path: str) -> Tensor:
        """Loads the raw CONUSMin mask (spatial, raw mineral codes)."""  
        with rasterio.open(path) as src:
            mask_data = src.read(1)
        return torch.from_numpy(mask_data.astype(np.int64)).long()

    def _load_label(self, path: str) -> Tensor:
        """Derives a multi-hot label vector from the CONUSMin mask."""  
        mask = self._load_mask(path)  # Load the spatial mask
        unique_values = torch.unique(mask)  

        
        final_labels = np.array([
            self.code_to_ordinal[val.item()]
            for val in unique_values
            if val.item() != 0 and val.item() in self.code_to_ordinal_dict  
        ])

        # Create multi-hot encoded label
        multihot_label = torch.zeros(self.num_classes, dtype=torch.int)
        # Filter out any potential out-of-bound labels (e.g. default 18 from defaultdict)
        valid_indices = final_labels[final_labels < self.num_classes]
        if len(valid_indices) > 0:
            multihot_label[torch.from_numpy(valid_indices).long()] = 1

        return multihot_label

    def _onehot_labels_to_names(self, label_vector: Tensor) -> List[str]:
        """Converts a one-hot label vector to a list of class names."""
        if not isinstance(label_vector, Tensor):
            label_vector = torch.tensor(label_vector)
        indices = torch.where(label_vector == 1)[0]
        return [self.class_names[i] for i in indices if i < len(self.class_names)]

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        image = sample["image"]
        label = sample["label"]

        rgb_bands_to_use = self.RGB_INDICES.get(self.sensor, [0, 1, 2] if image.shape[0] >= 3 else [0, 0, 0])
        if image.shape[0] == 1 and rgb_bands_to_use == [0, 1, 2]:  # Adjust for grayscale
            rgb_bands_to_use = [0, 0, 0]

        img_display = image[rgb_bands_to_use].cpu().numpy()
        img_display = np.transpose(img_display, (1, 2, 0))
        min_val, max_val = np.percentile(img_display, 2), np.percentile(img_display, 98)
        img_display = np.clip((img_display - min_val) / (max_val - min_val + 1e-8), 0, 1)

        num_cols = 1
        if self.return_mask and "mask" in sample:
            num_cols = 2

        fig, ax = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
        current_ax = ax[0] if num_cols > 1 else ax

        current_ax.imshow(img_display)
        current_ax.axis("off")
        if show_titles:
            current_ax.set_title("Image")

        label_names = self._onehot_labels_to_names(label)
        plot_title = f"Labels: {', '.join(label_names)}" if label_names else "No Labels"

        if num_cols == 2 and "mask" in sample:
            mask_display = sample["mask"].squeeze().cpu().numpy()
            ax[1].imshow(mask_display, cmap='viridis', interpolation='none')
            ax[1].axis("off")
            if show_titles:
                ax[1].set_title("Source Mask")
            # Adjust image title to make space for label text below figure
            if show_titles:
                current_ax.set_title("Image")  # Reset title without labels
        elif show_titles:
            current_ax.set_title(plot_title)  # Put labels in title if no mask

        if suptitle:
            fig.suptitle(suptitle)
        elif num_cols == 1 and not show_titles:  # No title anywhere yet, use labels
            fig.suptitle(plot_title)
        elif num_cols == 2 and show_titles:  # Labels as suptitle if mask is shown
            fig.suptitle(plot_title)

        plt.tight_layout()
        return fig

    # The old plot_with_mask is effectively merged into plot method with return_mask=True

# Example Usage:
if __name__ == '__main__':
    print("Testing ConusminDataset...")  
    try:
        print("\n--- ConusminDataset (EMIT sensor, 18 classes) ---")  
        conusmin_train = ConusminDataset(  
            root="data/splits/{product}/{sensor}/{split}.txt",  
            sensor_config={
                "name": "emit",  
                "num_bands": 244,  
                "rgb_indices": [39, 24, 14],  
                "min_val": 0.0,
                "max_val": 32767.0  
            },
            product="conusmin",  
            split="train",
            num_classes=18,  
            return_mask=True  # Also test returning the mask
        )
        print(f"CONUSMin train (EMIT, 18 cls) initialized. Samples: {len(conusmin_train)}")  
        if len(conusmin_train) > 0:
            sample = conusmin_train[0]
            print(f"  Image shape: {sample['image'].shape}, Label shape: {sample['label'].shape}")
            if "mask" in sample: print(f"  Mask shape: {sample['mask'].shape}")
            print(f"  Labels: {conusmin_train._onehot_labels_to_names(sample['label'])}")
            # fig = conusmin_train.plot(sample)
            # plt.show()

    except Exception as e:
        print(f"Error testing ConusminDataset (EMIT, 18 cls): {e}")  
        import traceback
        traceback.print_exc()

  