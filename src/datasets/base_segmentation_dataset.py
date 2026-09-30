import os
from typing import Callable, Optional, List, Dict, Tuple, Any

import torch
from torch import Tensor
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Added for dynamic colormap generation
from torchgeo.datasets.geo import NonGeoDataset

class BaseSegmentationDataset(NonGeoDataset):
    """Base class for segmentation datasets."""

    # To be defined by subclasses
    # Example: IMAGE_ROOT_FORMAT = "imagery/{sensor}"
    # Example: MASK_ROOT_FORMAT = "masks/{product}"
    # Example: SPLIT_PATH_FORMAT = "{split_root}/{sensor}_{product}/{split}.txt"
    # Example: CMAPS = {"product_name": {0: (0,0,0,0), 1: (255,0,0,255), ...}}
    # Example: DEFAULT_CLASSES = {"product_name": [0, 1, 2, ...]} # Default classes if not provided

    IMAGE_ROOT_FORMAT: str = "{sensor}"  # Default format, e.g., root/enmap/
    MASK_ROOT_FORMAT: str = "{product}"   # Default format, e.g., root/nlcd_2019/
    SPLIT_PATH_FORMAT: str = "{split_root}/{product}/{sensor}/{split}.txt"  # Default format for repo-managed split files
    CMAPS: Dict[str, Dict[int, Tuple[int, int, int, int]]] = {}
    DEFAULT_CLASSES: Optional[Dict[str, List[int]]] = None

    def __init__(
        self,
        root: str,
        sensor_config: Dict[str, Any],
        product: str,
        split: str,
        split_root: str = "data/splits",
        classes: Optional[List[int]] = None, # User-specified raw foreground classes
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        raw_mask: bool = False,
    ) -> None:
        """Initialize the base segmentation dataset.
        Args:
            root: Root directory where dataset can be found.
            sensor_config: Complete sensor configuration from hydra.
            product: Mask target product (e.g., "cdl", "nlcd", "eurocrops").
            split: Dataset split, one of ["train", "val", "test"].
            split_root: Root directory containing repo-managed split files.
            classes: List of raw *foreground* class values to include. These will be mapped
                     to ordinal indices [0, ..., N-1]. If None, uses DEFAULT_CLASSES
                     for the product or keys from CMAPS (excluding 0).
                     All other values in the mask (including 0, unless 0 is part of
                     `classes`) will be mapped to a background index N.
            transforms: A function/transform that takes input sample and its target.
            raw_mask: If True, returns the mask as is without remapping.
        """
        super().__init__() # NonGeoDataset doesn't take arguments here

        self.root = root
        self.sensor_config = sensor_config
        self.sensor = sensor_config["name"]
        self.sensor_path_name = sensor_config.get("path_name", self.sensor)
        self.num_bands = sensor_config.get("num_bands", -1)
        self.load_num_bands = self._resolve_load_num_bands(sensor_config)
        self.rgb_indices = sensor_config.get("rgb_indices", [0, 1, 2])
        
        self.product = product
        self.split = split
        self.split_root = split_root
        self.transforms = transforms
        self.raw_mask = raw_mask

        product_cmap_source = self.CMAPS.get(self.product)

        # Determine foreground classes
        if classes is None:
            if self.DEFAULT_CLASSES and self.product in self.DEFAULT_CLASSES:
                # Use default foreground classes for the product (these should not contain 0)
                self.foreground_classes_raw = sorted(list(set(self.DEFAULT_CLASSES[self.product])))
            elif product_cmap_source:
                # Infer foreground classes from CMAPS keys, excluding 0
                self.foreground_classes_raw = sorted(list(set(k for k in product_cmap_source.keys() if k != 0)))
            else:
                # As per user diff, raise ValueError if classes cannot be determined
                raise ValueError(
                    f"No 'classes' provided and no DEFAULT_CLASSES or CMAPS found for product '{self.product}'. "
                    f"Please specify foreground classes for {self.__class__.__name__}."
                )
        else:
            # User-provided foreground classes (filter out 0, as it's handled as BG by default)
            self.foreground_classes_raw = sorted(list(set(c for c in classes if c != 0)))

        if not self.foreground_classes_raw and classes is not None and 0 in classes and len(classes) == 1:
            # Special case: user explicitly asked for only class 0. Treat it as a single foreground.
             self.foreground_classes_raw = [0]


        self.num_foreground_classes = len(self.foreground_classes_raw)
        self.background_class_ordinal = self.num_foreground_classes  # Background mapped to this index

        self.num_effective_classes = self.num_foreground_classes + 1  # FG classes + 1 BG class
        self.ordinal_cmap = torch.zeros((self.num_effective_classes, 4), dtype=torch.uint8)


        # Map foreground classes to their new ordinal indices for colormap definition
        for new_idx, raw_fg_val in enumerate(self.foreground_classes_raw):
            # Ordinal map population removed. This loop now only sets up ordinal_cmap.
            if product_cmap_source and raw_fg_val in product_cmap_source:
                self.ordinal_cmap[new_idx] = torch.tensor(product_cmap_source[raw_fg_val])
            else:
                # Dynamically generate color for this foreground class (e.g., using tab20)
                num_colors_for_cmap = self.num_foreground_classes if self.num_foreground_classes > 0 else 1
                try:
                    dynamic_cmap_func = cm.get_cmap("tab20", num_colors_for_cmap)
                except ValueError: # Fallback
                    dynamic_cmap_func = cm.get_cmap("tab20")
                
                color = dynamic_cmap_func(new_idx % dynamic_cmap_func.N) # Cycle through cmap colors
                self.ordinal_cmap[new_idx] = torch.tensor([int(255 * c) for c in color[:3]] + [255]) # RGBA

        # Define background color
        # User wants background to ALWAYS be black, regardless of product_cmap_source[0]
        self.ordinal_cmap[self.background_class_ordinal] = torch.tensor([0, 0, 0, 255]) # Opaque Black


        # File paths
        # Use self.__class__ to explicitly refer to class attributes for format strings
        self.img_dir_path = os.path.join(self.root, self.__class__.IMAGE_ROOT_FORMAT.format(sensor=self.sensor_path_name, product=self.product))
        self.mask_dir_path = os.path.join(self.root, self.__class__.MASK_ROOT_FORMAT.format(sensor=self.sensor, product=self.product))
        
        _split_file_template = self.__class__.SPLIT_PATH_FORMAT
        self.split_file = _split_file_template.format(
            split_root=self.split_root,
            sensor=self.sensor_path_name,
            product=self.product,
            split=self.split,
        )

        if not os.path.exists(self.split_file):
            # Removed fallback logic. The SPLIT_PATH_FORMAT defined by the subclass must now directly lead to an existing file.
            raise FileNotFoundError(
                f"Split file not found at path: {self.split_file}. "
                f"This path was constructed using the format string: '{_split_file_template}' "
                f"from class {self.__class__.__name__} with root='{self.root}', sensor='{self.sensor}', "
                f"sensor_path_name='{self.sensor_path_name}', product='{self.product}', split='{self.split}', "
                f"and split_root='{self.split_root}'."
            )
        
        self.sample_collection = self._read_split_file()

        # Store the raw class value that maps to the background ordinal index for reference
        # This is a bit tricky because multiple raw values map to background.
        # We know self.background_class_ordinal is the target index.
        # self.ignore_index is effectively self.background_class_ordinal
        self.ignore_index = self.background_class_ordinal

    @staticmethod
    def _resolve_load_num_bands(sensor_config: Dict[str, Any]) -> int:
        """Return the expected on-disk band count for loader sanity checks."""
        num_bands = sensor_config.get("num_bands", -1)
        spectral_config = sensor_config.get("spectral", {}) or {}
        channel_view = str(spectral_config.get("channel_view", "")).strip().lower()

        if channel_view in {"stored_prefiltered", "stored_processed"}:
            return num_bands

        # Sensors such as DESIS/EO1 are stored raw and trimmed later in the
        # datamodule pipeline, so the loader should still expect raw bands.
        return sensor_config.get("num_bands_raw", num_bands)


    def _read_split_file(self) -> List[Tuple[str, str]]:
        """Reads a split file containing image identifiers (one per line).
        Returns a list of (image_path, mask_path) tuples.
        Assumes image and mask share the same identifier (filename).
        """
        with open(self.split_file, "r") as f:
            sample_ids = [line.strip() for line in f.readlines() if line.strip()]
        
        sample_collection = []
        for sample_id in sample_ids:
            img_path = os.path.join(self.img_dir_path, sample_id)
            mask_path = os.path.join(self.mask_dir_path, sample_id)
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found {img_path} from split file {self.split_file}")
            if not os.path.exists(mask_path):
                 print(f"Warning: Mask file not found {mask_path} from split file {self.split_file}")
            sample_collection.append((img_path, mask_path))
            
        if not sample_collection:
            print(f"Warning: No samples loaded from split file: {self.split_file}. Check paths and file content.")
            
        return sample_collection

    def _load_image(self, path: str) -> Tensor:
        """Load a single image from path."""
        with rasterio.open(path) as src:
            image_data = src.read() # Reads as (bands, height, width) or (height, width, bands)
        
        # Ensure image_data is (bands, height, width)
        if image_data.ndim == 3:
            # Heuristic: if first dim is largest, it might be (bands, H, W) if many bands
            # Or if third dim is small (e.g. 3, 4 for RGB, RGBA), it's likely (H, W, C)
            if image_data.shape[0] != self.load_num_bands and image_data.shape[2] == self.load_num_bands and self.load_num_bands != -1 : # (H, W, C) and num_bands is C
                 image_data = np.transpose(image_data, (2, 0, 1))
            elif image_data.shape[0] > image_data.shape[1] and image_data.shape[0] > image_data.shape[2] and image_data.shape[0] != self.load_num_bands and self.load_num_bands == -1:
                # This case is ambiguous without num_bands. Assuming (bands, H, W) if first dim is largest and not H or W.
                # If num_bands is known, it's easier.
                pass # Assume (bands, H, W) if first dim is largest and not clearly H or W
            elif image_data.shape[2] < image_data.shape[0] and image_data.shape[2] < image_data.shape[1] and image_data.shape[2] not in [self.load_num_bands, -1]: # Likely (H,W,C)
                 image_data = np.transpose(image_data, (2, 0, 1))


        tensor_image = torch.from_numpy(image_data.astype(np.float32)).float()
        
        # If num_bands is specified and doesn't match, print warning and attempt to select/pad
        if self.load_num_bands != -1 and tensor_image.shape[0] != self.load_num_bands:
            print(f"Warning: Expected {self.load_num_bands} bands, but image {path} has {tensor_image.shape[0]} bands.")
            if tensor_image.shape[0] > self.load_num_bands:
                tensor_image = tensor_image[:self.load_num_bands, :, :]
            # else: # Fewer bands than expected
            #     # Padding could be an option, or raise error. For now, just warn.
            #     print(f"Error: Image {path} has {tensor_image.shape[0]} bands, expected {self.num_bands}. Padding not implemented.")
            #     # raise ValueError(f"Image {path} has {tensor_image.shape[0]} bands, expected {self.num_bands}.")
        elif self.load_num_bands == -1:
            self.load_num_bands = tensor_image.shape[0] # Auto-detect num_bands from first image

        return tensor_image


    def _load_mask(self, path: str) -> Tensor:
        """Load a single mask from path."""
        with rasterio.open(path) as src:
            # As per user diff: use src.read(). For single band, rasterio typically returns (1, H, W)
            mask_data = src.read()
        # As per user diff: direct conversion. If mask_data is (1,H,W) from rasterio, mask will be (1,H,W) torch.long.
        # If mask_data was (H,W), mask would be (H,W) torch.long.
        mask = torch.from_numpy(mask_data.astype(np.int64)).long()

        if self.raw_mask:
            # As per user diff: return mask as is (e.g. (1,H,W) or (H,W))
            return mask 
        
        # Remap mask values using a loop-based approach
        # Initialize remapped_mask with the background value
        # torch.full_like will preserve the shape of mask (e.g. (1,H,W) or (H,W))
        remapped_mask = torch.full_like(mask,
                                        fill_value=self.background_class_ordinal,
                                        dtype=torch.long)

        # Iterate over defined foreground classes and map them
        # This works if mask is (1,H,W) or (H,W) due to broadcasting of raw_fg_val
        for new_idx, raw_fg_val in enumerate(self.foreground_classes_raw):
            remapped_mask[mask == raw_fg_val] = new_idx
        
        # As per user diff: return remapped_mask as is
        return remapped_mask


    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            A dict containing "image", "mask", and "path".
        """
        if index < 0 or index >= len(self.sample_collection):
            raise IndexError(f"Index {index} out of bounds for dataset with length {len(self.sample_collection)}")
            
        img_path, mask_path = self.sample_collection[index]

        try:
            image = self._load_image(img_path)
            mask = self._load_mask(mask_path)
        except Exception as e:
            print(f"Error loading sample for img: {img_path}, mask: {mask_path}. Error: {e}")
            # Option 1: Raise the error
            # raise e
            # Option 2: Return a dummy sample or skip (could lead to issues in dataloader)
            # For now, let's make it more robust by allowing __getitem__ to signal an issue
            # A common way is to return None and have collate_fn handle it, but that adds complexity.
            # Re-raising is safer for now.
            raise RuntimeError(f"Failed to load sample at index {index} ({img_path}, {mask_path})") from e


        sample = {
            "image": image,
            "mask": mask,
            "path": img_path  # Useful for debugging
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sample_collection)

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        rgb_bands_override: Optional[List[int]] = None, # Allow overriding RGB bands for this plot call
    ) -> plt.Figure:
        """Plots a sample.
        Args:
            sample: a sample returned by __getitem__
            show_titles: whether to display titles
            suptitle: optional suptitle for the plot
            rgb_bands_override: Optional list of 3 indices for R, G, B channels for this plot.
        Returns:
            A matplotlib Figure object.
        """
        image = sample["image"]
        # As per user diff: mask_to_plot is sample["mask"]. Shape could be (1,H,W) or (H,W).
        mask_to_plot = sample["mask"]

        # Determine RGB bands for plotting
        current_rgb_bands: List[int]
        if rgb_bands_override is not None:
            current_rgb_bands = rgb_bands_override
        elif hasattr(self, 'rgb_indices'):
            current_rgb_bands = self.rgb_indices
        elif image.shape[0] >= 3:
            current_rgb_bands = [0, 1, 2] # Default to first three bands
            print(f"Warning: rgb_indices not defined for sensor '{self.sensor}'. Defaulting to bands [0, 1, 2] for plotting.")
        elif image.shape[0] == 1: # Grayscale
            current_rgb_bands = [0,0,0] # plot grayscale by repeating the band
            print(f"Info: Image has only 1 band. Plotting as grayscale.")
        else: # Not enough bands for a 3-channel RGB plot
            print(f"Warning: Cannot determine 3 RGB bands for plotting image with shape {image.shape} for sensor {self.sensor}. Plotting first band as grayscale if possible.")
            if image.shape[0] > 0:
                current_rgb_bands = [0,0,0] # Try plotting first band as grayscale
            else:
                raise ValueError(f"Cannot plot image with shape {image.shape}. No bands available.")


        # Select and prepare image for plotting
        # Ensure selected bands are within image dimensions
        if not all(b < image.shape[0] for b in current_rgb_bands):
             # This can happen if default [0,1,2] is used for an image with <3 bands but not 1
             if image.shape[0] == 1 and current_rgb_bands == [0,1,2]: # common case of default for grayscale
                 current_rgb_bands = [0,0,0]
             else:
                raise ValueError(f"RGB band indices {current_rgb_bands} are out of bounds for image with {image.shape[0]} bands.")
        
        img_display_np = image[current_rgb_bands, :, :].cpu().numpy()
        img_display_np = np.transpose(img_display_np, (1, 2, 0)) # C, H, W -> H, W, C

        # Normalize image for display (robust percentile normalization)
        min_val = np.percentile(img_display_np, 2)
        max_val = np.percentile(img_display_np, 98)
        img_display_np = (img_display_np - min_val) / (max_val - min_val + 1e-8)
        img_display_np = np.clip(img_display_np, 0, 1)


        ncols = 2
        fig, ax = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))

        ax[0].imshow(img_display_np)
        ax[0].axis("off")
        if show_titles:
            ax[0].set_title("Image")

        # Plotting the mask
        mask_numpy = mask_to_plot.cpu().numpy() # dtype will be int64
        
        if self.raw_mask:
            # Plot raw mask directly. If mask_numpy is (1,H,W), imshow handles it by taking the first plane.
            ax[1].imshow(mask_numpy.squeeze(), cmap='viridis', interpolation="none") # Squeeze for imshow
            title_mask = f"Raw Mask (Product: {self.product})"
        elif self.ordinal_cmap is not None:
            # Ensure mask_numpy is 2D (H,W) for colormap indexing
            mask_numpy_2d = mask_numpy
            if mask_numpy.ndim == 3 and mask_numpy.shape[0] == 1:
                mask_numpy_2d = mask_numpy.squeeze(0)
            elif mask_numpy.ndim != 2:
                ax[1].text(0.5, 0.5, f'Cannot display mask\nUnexpected shape: {mask_numpy.shape}', 
                             horizontalalignment='center', verticalalignment='center')
                if show_titles: ax[1].set_title("Mask (Shape Error)")
                ax[1].axis("off")
                if suptitle is not None: plt.suptitle(suptitle)
                return fig

            # Clip values to be valid indices for ordinal_cmap. ordinal_cmap is a torch.Tensor.
            clamped_indices_np = np.clip(mask_numpy_2d, 0, self.ordinal_cmap.shape[0] - 1)
            
            # Index ordinal_cmap (torch tensor) with clamped_indices_np (numpy array of np.int64).
            # PyTorch handles numpy array indexing automatically and correctly with int64 type.
            colored_mask = self.ordinal_cmap[clamped_indices_np].cpu().numpy()
            ax[1].imshow(colored_mask, interpolation="none")
            title_mask = f"Mask (Product: {self.product})"
        else:
            ax[1].text(0.5, 0.5, 'Cannot display mask', horizontalalignment='center', verticalalignment='center')
            title_mask = "Mask"
        
        if show_titles:
            ax[1].set_title(title_mask)
        ax[1].axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)
        
        return fig 
