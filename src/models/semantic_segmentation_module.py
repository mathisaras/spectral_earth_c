"""Refactored semantic segmentation module using Hydra for backbone/decoder instantiation."""

import os
from typing import Any, Optional, Dict, List, Union
import torch
import torch.nn as nn
from torch import Tensor
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from omegaconf import DictConfig, OmegaConf
import numpy as np

# Temporarily import wandb and matplotlib to log images
import matplotlib.pyplot as plt
import warnings
from torchgeo.datasets.utils import unbind_samples
import wandb # Explicitly import wandb for wandb.Image

from .model_utils import (
    instantiate_component,
    create_model,
    load_pretrained_weights,
    infer_spatial_decoder_patch_size,
    infer_multitap_decoder_info,
)


class SemanticSegmentationModule(LightningModule):
    def __init__(
        self,
        backbone_config: DictConfig,
        decoder_config: DictConfig,
        num_classes: int,
        pretrained_weights: Optional[str] = None,
        model_type: str = "vit_seg",
        tap_indices: Optional[List[int]] = None,
        multi_tap_is_vit: Optional[bool] = None,
        probe_in_chans: Optional[int] = None,
        probe_img_size: Optional[Union[int, List[int]]] = None,
        freeze_backbone: bool = False,
        finetune_adapter: bool = False,
        finetune_first_n_layers: int = 0,
        class_weights: Optional[Tensor] = None,
        ignore_index: Optional[int] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        t_max: int = 10,
        eta_min: float = 1e-5,
        log_validation_images: bool = True,
        validation_image_log_every_n_epochs: int = 20,
        num_validation_images_to_log: int = 8,
        rgb_indices: Optional[List[int]] = None,
    ) -> None:
        """Initialize the LightningModule with Hydra-based backbone and decoder.

        Args:
            backbone_config: Hydra config for backbone instantiation
            decoder_config: Hydra config for decoder instantiation
            num_classes: Number of semantic classes to predict
            in_channels: Number of input channels
            img_size: Input image size
            pretrained_weights: Path to pretrained weights
            token_patch_size: Patch size for tokens
            model_type: Type of model architecture ("vit_seg" or "fcn_seg")
            freeze_backbone: Whether to freeze the backbone
            finetune_adapter: Whether to finetune the adapter
            finetune_first_n_layers: Number of layers to finetune
            class_weights: Optional rescaling weights for classes
            ignore_index: Class index to ignore in loss and metrics
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            t_max: T_max parameter for learning rate scheduler
            eta_min: Minimum learning rate for scheduler

        """
        super().__init__()

        print("<<<<< SEMANTIC SEGMENTATION MODULE __INIT__ CALLED >>>>>", flush=True) # Add flush=True

        print("backbone_config", backbone_config)
        print("decoder_config", decoder_config)

        # Store original DictConfig for reference if ever needed, but work with a mutable copy for modifications.
        self.backbone_config_original = backbone_config 
        # Convert to a Python dict for easier, potentially safer modification before instantiation.
        # resolve=True ensures any OmegaConf interpolations in the original config are resolved to their primitive values.
        self.backbone_config = OmegaConf.to_container(backbone_config, resolve=True)
        # Decoder config can often remain DictConfig if instantiate_component handles it directly
        # or if it doesn't need pre-instantiation modification by this module.
        self.decoder_config = decoder_config 

        self.num_classes = num_classes
        self.pretrained_weights = pretrained_weights
        self.model_type = model_type
        self.tap_indices = [int(idx) for idx in tap_indices] if tap_indices is not None else None
        self.multi_tap_is_vit = multi_tap_is_vit
        self.probe_in_chans = int(probe_in_chans) if probe_in_chans is not None else None
        self.probe_img_size = probe_img_size
        self.freeze_backbone = freeze_backbone
        self.finetune_adapter = finetune_adapter
        self.finetune_first_n_layers = finetune_first_n_layers
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.t_max = t_max
        self.eta_min = eta_min
        self.log_validation_images = bool(log_validation_images)
        self.validation_image_log_every_n_epochs = int(validation_image_log_every_n_epochs)
        self.num_validation_images_to_log = int(num_validation_images_to_log)
        self.rgb_indices = rgb_indices
        
        # Attempt to get token_patch_size from backbone_config if present.
        # This is primarily for ViT models where create_model might need it.
        # If not present, it will be None. create_model should handle None if model_type is not ViT-based.
        self.vit_patch_size = self.backbone_config.get("patch_size") 

        # Validate ignore_index (moved after attribute assignment)
        if not isinstance(self.ignore_index, (int, type(None))):
            raise ValueError("ignore_index must be an int or None")

        # Configure the task (initialize model, loss, etc.)
        self._build_model()
        self._configure_task()

        # Initialize metrics
        self.train_metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                    average="micro",
                ),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        
        self.max_val_metrics = {}
        for key in self.val_metrics.keys():
            self.max_val_metrics['max_' + key] = 0.0

    def _build_model(self) -> None:
        """Build the model using Hydra instantiation.
        Modifies self.backbone_config (Python dict) for ResNets for segmentation.
        """
        
        if not isinstance(self.backbone_config, dict):
            raise TypeError(f"self.backbone_config is not a dict, but {type(self.backbone_config)}. Cannot modify.")

        # --- Adapt backbone_config for ResNets for segmentation task ---
        is_resnet = False
        target_path = self.backbone_config.get("_target_")
        if isinstance(target_path, str) and "resnet" in target_path.lower():
            is_resnet = True
        
        model_name_from_config = self.backbone_config.get("model_name")
        if not is_resnet and isinstance(model_name_from_config, str) and "resnet" in model_name_from_config.lower():
            is_resnet = True

        if is_resnet:
            print(f"SemanticSegmentationModule: Simplifying ResNet config for segmentation: {model_name_from_config or target_path}")
            
            # Set replace_stride_with_dilation to [True, True, True, True]
            new_dilation_config = [True, True, True, True]
            if self.backbone_config.get("replace_stride_with_dilation") != new_dilation_config:
                self.backbone_config["replace_stride_with_dilation"] = new_dilation_config
                print(f"  Setting replace_stride_with_dilation: {new_dilation_config}")

            # Set return_features to True
            # This assumes the ResNet wrapper/model understands this argument.
            if self.backbone_config.get("return_features") is not True:
                self.backbone_config["return_features"] = True
                print(f"  Setting return_features: True")
            
            # Remove other ResNet-specific adaptations from previous versions if any were missed
            # For this simplified version, we are NOT setting num_classes, features_only, or out_indices here.
            # The user will revisit this. The backbone config or wrapper must handle feature extraction.
            # Example: if "num_classes" in self.backbone_config: del self.backbone_config["num_classes"]
            # Example: if "features_only" in self.backbone_config: del self.backbone_config["features_only"]
            # Example: if "out_indices" in self.backbone_config: del self.backbone_config["out_indices"]

        # Instantiate backbone using the (potentially modified) Python dict config
        self.backbone = instantiate_component(config=self.backbone_config)

        # Load pretrained weights for the backbone if path is provided
        if self.pretrained_weights:
            load_pretrained_weights(self.backbone, self.pretrained_weights, strict=False)
        
        # Determine backbone output features
        backbone_output_channels: Union[int, List[int]]
        if hasattr(self.backbone, 'num_features'):
            backbone_output_channels = self.backbone.num_features
        elif hasattr(self.backbone, 'feature_channels'):
            backbone_output_channels = self.backbone.feature_channels
        elif hasattr(self.backbone, 'encoder_channels'):
            backbone_output_channels = self.backbone.encoder_channels
        else:
            raise AttributeError(
                f"Backbone {self.backbone.__class__.__name__} (from {self.backbone_config.get('_target_')}) "
                f"must have a 'num_features', 'feature_channels', or 'encoder_channels' attribute."
            )
        
        if not backbone_output_channels:
             raise ValueError(
                f"Backbone {self.backbone.__class__.__name__} reported empty/None feature channels: {backbone_output_channels}."
            )


        # Check if this is a UperNet decoder or multi scale conv decoder
        is_upernet = False
        if hasattr(self.decoder_config, '_target_'):
            target = self.decoder_config._target_
            if isinstance(target, str) and 'upernet' in target.lower():
                is_upernet = True
        elif isinstance(self.decoder_config, dict) and '_target_' in self.decoder_config:
            target = self.decoder_config['_target_']
            if isinstance(target, str) and 'upernet' in target.lower():
                is_upernet = True

        is_multiscale_conv = False
        is_lightweight_multitap = False
        if hasattr(self.decoder_config, '_target_'):
            target = self.decoder_config._target_
            if isinstance(target, str) and 'multiscale_conv' in target.lower():
                is_multiscale_conv = True
            if isinstance(target, str) and 'lightweight_multitap' in target.lower():
                is_lightweight_multitap = True
        elif isinstance(self.decoder_config, dict) and '_target_' in self.decoder_config:
            target = self.decoder_config['_target_']
            if isinstance(target, str) and 'multiscale_conv' in target.lower():
                is_multiscale_conv = True
            if isinstance(target, str) and 'lightweight_multitap' in target.lower():
                is_lightweight_multitap = True
        
        if is_lightweight_multitap:
            multitap_info = infer_multitap_decoder_info(
                self.backbone,
                tap_indices=self.tap_indices,
                is_vit=self.multi_tap_is_vit,
                patch_size=self.vit_patch_size,
                probe_in_chans=self.probe_in_chans,
                probe_img_size=self.probe_img_size,
            )
            self.tap_indices = multitap_info["tap_indices"]
            self.multi_tap_is_vit = multitap_info["is_vit"]
            if multitap_info["patch_size"] is not None:
                self.vit_patch_size = int(multitap_info["patch_size"])
            decoder_override_params = {
                "in_channels_list": multitap_info["in_channels_list"],
                "num_classes": self.num_classes,
            }
            print(
                "LightweightMultiTapSegHead: "
                f"is_vit={self.multi_tap_is_vit}, taps={self.tap_indices}, "
                f"in_channels={multitap_info['in_channels_list']}, "
                f"probe_shape={multitap_info['probe_input_shape']}."
            )
        elif is_upernet or is_multiscale_conv:
            # UperNet doesn't need num_input_features, it will determine this from the features
            decoder_override_params = {
                "num_classes": self.num_classes,
            }
            if is_multiscale_conv:
                decoder_patch_size = infer_spatial_decoder_patch_size(
                    self.backbone,
                    fallback=self.vit_patch_size,
                )
                if decoder_patch_size is not None:
                    decoder_override_params["patch_size"] = int(decoder_patch_size)
                    print(
                        "MultiScaleConvHead: using patch_size="
                        f"{decoder_patch_size} from active backbone spatial stride."
                    )
        else:
            # Other decoders need num_input_features
            decoder_override_params = {
                "num_input_features": backbone_output_channels,
                "num_classes": self.num_classes,
            }

        self.decoder = instantiate_component(
            config=self.decoder_config,
            **decoder_override_params
        )
        
        self.model = create_model(
            backbone=self.backbone,
            decoder=self.decoder,
            model_type=self.model_type,
            patch_size=self.vit_patch_size,
            tap_indices=self.tap_indices,
            multi_tap_is_vit=self.multi_tap_is_vit,
        )

    def _configure_task(self) -> None:
        """Configure loss function and handle freezing/unfreezing."""
        
        # Create loss function
        if self.class_weights is not None:
            self.loss = nn.CrossEntropyLoss(
                weight=self.class_weights, ignore_index=self.ignore_index
            )
        else:
            self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # Handle freezing/unfreezing
        if self.freeze_backbone:
            self._freeze_encoder()

        if self.finetune_adapter:
            self._unfreeze_adapter_layers()

        if self.finetune_first_n_layers > 0:
            self._unfreeze_first_n_layers()

    def _unfreeze_first_n_layers(self) -> None:
        """Unfreezes the first n layers of the backbone."""
        if hasattr(self.model.encoder, 'blocks'):
            # ViT-style models
            for i, block in enumerate(self.model.encoder.blocks):
                if i < self.finetune_first_n_layers:
                    for param in block.parameters():
                        param.requires_grad = True
        else:
            print("Warning: Unable to unfreeze first n layers - model structure not recognized")

    def _unfreeze_adapter_layers(self) -> None:
        """Unfreezes the adapter layers."""
        try:
            for param in self.model.encoder.spectral_adapter.parameters():
                param.requires_grad = True
        except AttributeError:
            print("Warning: The backbone does not have 'spectral_adapter' attributes.")

    def _freeze_encoder(self) -> None:
        """Freezes the encoder parameters."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self) -> None:
        """Unfreezes the encoder parameters."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    @staticmethod
    def _select_model_input(batch: Dict[str, Any]) -> Any:
        return batch.get("image_by_sensor", batch["image"])

    @staticmethod
    def _input_batch_size(x: Any) -> int:
        if isinstance(x, Tensor):
            return int(x.size(0))
        if isinstance(x, dict):
            for value in x.values():
                if isinstance(value, Tensor):
                    return int(value.size(0))
        raise ValueError("Could not infer batch size from model input.")

    def forward(
        self,
        x: Any,
        output_size: Optional[Union[int, List[int], tuple[int, int]]] = None,
    ) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Model output.
        """
        if output_size is not None:
            return self.model(x, output_size=output_size)
        return self.model(x)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            Training loss.
        """
        x = self._select_model_input(batch)
        y = batch["mask"]
        batch_size = self._input_batch_size(x)
        y_hat = self(x, output_size=y.shape[-2:])
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True, batch_size=batch_size)
        self.train_metrics(y_hat_hard, y)

        return loss

    def on_train_epoch_end(self) -> None:
        """Logs epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """Compute validation loss and log metrics.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.
        """
        x = self._select_model_input(batch)
        y = batch["mask"]
        batch_size = self._input_batch_size(x)
        y_hat = self(x, output_size=y.shape[-2:])
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.val_metrics(y_hat_hard, y)
        
        # Only log every 20 epochs for image visualization
        current_epoch = self.trainer.current_epoch
        if (
            self.log_validation_images
            and
            batch_idx < 1
            and hasattr(self.trainer, "datamodule")
            and self.logger  
            and self.validation_image_log_every_n_epochs > 0
            and current_epoch % self.validation_image_log_every_n_epochs == 0
            and self.trainer.is_global_zero
        ):
            self._log_validation_images(batch["image"], y, y_hat_hard, batch_idx)

    def _log_validation_images(self, x: Tensor, y: Tensor, y_hat: Tensor, batch_idx: int) -> None:
        """Log a matplotlib figure of image, true mask, and prediction to wandb for multiple samples."""
        
        num_samples_to_log = min(x.size(0), self.num_validation_images_to_log)

        for sample_idx in range(num_samples_to_log):
            try:
                # Move tensors to CPU and convert to NumPy for plotting
                img_tensor = x[sample_idx].cpu()
                true_mask_tensor = y[sample_idx].cpu().numpy()
                pred_mask_tensor = y_hat[sample_idx].cpu().numpy()

                # Handle image tensor for plotting
                if img_tensor.ndim == 3:
                    img_np = img_tensor.permute(1, 2, 0).numpy()
                elif img_tensor.ndim == 2:
                    img_np = img_tensor.numpy()
                else:
                    warnings.warn(f"Sample {sample_idx}: Image tensor has unexpected dimensions: {img_tensor.shape}. Skipping plot for this sample.")
                    continue # Skip to next sample

                # Select bands for RGB display or handle grayscale
                if img_np.ndim == 3 and img_np.shape[2] > 3:
                    img_display = img_np[:, :, self.rgb_indices] if self.rgb_indices else img_np[:, :, :3]
                    min_vals = img_display.min(axis=(0, 1), keepdims=True)
                    max_vals = img_display.max(axis=(0, 1), keepdims=True)
                    range_vals = max_vals - min_vals
                    range_vals[range_vals == 0] = 1 
                    img_display = (img_display - min_vals) / range_vals
                    img_display = np.clip(img_display, 0, 1)
                elif img_np.ndim == 3 and img_np.shape[2] == 1:
                    img_display = img_np[:, :, 0]
                else: # Already 2D or 3-channel RGB
                    img_display = img_np
                    if img_display.ndim == 3:
                        min_vals = img_display.min(axis=(0, 1), keepdims=True)
                        max_vals = img_display.max(axis=(0, 1), keepdims=True)
                        range_vals = max_vals - min_vals
                        range_vals[range_vals == 0] = 1 
                        img_display = (img_display - min_vals) / range_vals
                        img_display = np.clip(img_display, 0, 1)
                    elif img_display.ndim == 2:
                        min_val = img_display.min()
                        max_val = img_display.max()
                        if max_val > min_val:
                            img_display = (img_display - min_val) / (max_val - min_val)
                        img_display = np.clip(img_display, 0, 1)

                # Create matplotlib figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f"Sample {sample_idx + 1}/B{batch_idx}/E{self.current_epoch}", fontsize=10) # Add a super title for context
                
                axes[0].imshow(img_display)
                axes[0].set_title("Input Image")
                axes[0].axis('off')

                cmap = plt.cm.get_cmap("tab20", self.num_classes).copy()
                cmap.set_bad(color="black")

                if self.ignore_index is not None:
                    background_mask = (true_mask_tensor == self.ignore_index)
                    true_mask_display = np.ma.masked_equal(true_mask_tensor, self.ignore_index)
                    pred_mask_display = np.ma.masked_array(pred_mask_tensor, mask=background_mask)
                else:
                    true_mask_display = true_mask_tensor
                    pred_mask_display = pred_mask_tensor

                axes[1].imshow(true_mask_display, cmap=cmap, vmin=0, vmax=self.num_classes - 1)
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis('off')

                axes[2].imshow(pred_mask_display, cmap=cmap, vmin=0, vmax=self.num_classes - 1)
                axes[2].set_title("Predicted Mask")
                axes[2].axis('off')

                plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle

                if self.logger and hasattr(self.logger, 'experiment') and hasattr(wandb, 'Image'):
                    log_key = f"validation/epoch{self.current_epoch}_batch{batch_idx}_sample{sample_idx}"
                    try:
                        self.logger.experiment.log({log_key: wandb.Image(fig)})
                    except Exception as e:
                        warnings.warn(f"Could not log matplotlib figure for sample {sample_idx} to wandb: {e}")
                else:
                    warnings.warn("Wandb logger or wandb.Image not available. Skipping plot logging for this sample.")
            
            except Exception as e:
                warnings.warn(f"Failed to create or log validation plot for sample {sample_idx} (epoch {self.current_epoch}, batch {batch_idx}): {e}")
            finally:
                if 'fig' in locals() and fig is not None:
                    plt.close(fig)
                    fig = None # Ensure fig is reset for the next iteration

    def _create_colormap(self, num_classes: int) -> Dict[int, str]:
        """Create a simple colormap for visualization."""
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
        return {i: colors[i % len(colors)] for i in range(num_classes)}

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """Compute test loss and log metrics.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.
        """
        x = self._select_model_input(batch)
        y = batch["mask"]
        batch_size = self._input_batch_size(x)
        y_hat = self(x, output_size=y.shape[-2:])
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # Log test loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.test_metrics(y_hat_hard, y)

    def on_validation_epoch_end(self) -> None:
        """Logs epoch-level validation metrics and tracks best metrics."""
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        
        # Track maximum metrics
        for key, value in metrics.items():
            max_key = 'max_' + key
            if value > self.max_val_metrics[max_key]:
                self.max_val_metrics[max_key] = value.item()
                self.log(max_key, self.max_val_metrics[max_key], sync_dist=True)
        
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Logs epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=self.eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        } 
