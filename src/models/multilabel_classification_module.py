import os
from typing import Any, Dict, Optional, List, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAveragePrecision,
    MultilabelFBetaScore,
)
from lightning.pytorch import LightningModule
from omegaconf import DictConfig

from .model_utils import instantiate_component, load_pretrained_weights

class MultiLabelClassificationModule(LightningModule):
    """LightningModule for multi-label image classification."""

    def __init__(
        self,
        backbone_config: DictConfig,
        num_classes: int,
        sensor_config: Optional[DictConfig] = None,
        pretrained_weights: Optional[str] = None,
        freeze_backbone: bool = False,
        finetune_adapter: bool = False,
        finetune_first_n_layers: int = 0,
        loss: str = "bce",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        t_max: int = 100,
        eta_min: float = 0,
    ) -> None:
        """Initialize the LightningModule with a Hydra-based backbone and loss function.

        Args:
            backbone_config: Hydra config for backbone instantiation.
            num_classes: Number of prediction classes.
            sensor_config: Sensor configuration (e.g. from configs/sensor/*) used
                for dataset/image metadata such as img_size, num_bands, etc.
            pretrained_weights: Path to pretrained weights for the backbone.
                                Note: backbone_config might also handle its own pretrained setup.
            freeze_backbone: If True, freezes the backbone network for linear probing.
            finetune_adapter: If True, finetunes the spectral adapter of the backbone.
            finetune_first_n_layers: Number of initial layers/blocks to finetune in the backbone.
            loss: Name of the loss function ('bce' supported).
            learning_rate: Learning rate for the optimizer.
            weight_decay: Weight decay for the optimizer.
            t_max: Maximum number of iterations for the CosineAnnealingLR scheduler.
            eta_min: Minimum learning rate for the CosineAnnealingLR scheduler.
        """
        super().__init__()

        # Store configurations and parameters
        self.backbone_config = backbone_config
        self.num_classes = num_classes
        self.sensor_config = sensor_config
        # Derive image size from sensor_config if available (for logging or
        # potential future behavior that depends on the effective patch size).
        self.img_size: Optional[int] = None
        if self.sensor_config is not None:
            img_size_val = self.sensor_config.get("img_size", None)
            if img_size_val is not None:
                # img_size may be int or tuple; for classification we typically
                # care about the scalar patch size, so coerce sensibly.
                if isinstance(img_size_val, (list, tuple)):
                    self.img_size = int(img_size_val[0])
                else:
                    self.img_size = int(img_size_val)
        self.pretrained_weights = pretrained_weights
        self.freeze_backbone = freeze_backbone
        self.finetune_adapter = finetune_adapter
        self.finetune_first_n_layers = finetune_first_n_layers
        self.loss_name = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.t_max = t_max
        self.eta_min = eta_min

        # Build model (backbone + classifier head)
        self._build_model()

        # Configure loss function
        self._configure_loss()

        # Configure metrics
        self._configure_metrics()

        # Apply freezing/finetuning logic after model is built and before optimizer is configured
        if self.freeze_backbone:
            self._freeze_backbone()  # Unfreezes classifier head by default
        
        if self.finetune_adapter:
            self._unfreeze_adapter_layers()
        
        if self.finetune_first_n_layers > 0:
            self._unfreeze_first_n_layers()

    def _build_model(self) -> None:
        """Build the model: instantiate backbone from config and add a classifier head."""
        # Instantiate backbone using Hydra config
        self.backbone = instantiate_component(config=self.backbone_config)

        # Load pretrained weights for the backbone if a path is provided
        if self.pretrained_weights:
            if not os.path.exists(self.pretrained_weights):
                print(f"Warning: Pretrained weights path not found: {self.pretrained_weights}")
            else:
                load_pretrained_weights(self.backbone, self.pretrained_weights, strict=False)

        declared_features = self._get_declared_backbone_output_dim()
        inferred_features = self._infer_backbone_output_dim()

        if inferred_features is not None:
            if declared_features is not None and int(declared_features) != int(inferred_features):
                print(
                    "Warning: backbone metadata reports output dim "
                    f"{declared_features}, but a dummy forward returned {inferred_features}. "
                    "Using the observed forward output dimension for the classifier head."
                )
            num_backbone_features = int(inferred_features)
        else:
            num_backbone_features = declared_features

        if num_backbone_features is None or int(num_backbone_features) == 0:
            raise AttributeError(
                f"Could not determine output feature dimension for backbone {self.backbone.__class__.__name__} "
                f"(from config {self.backbone_config.get('_target_')}). "
                f"Ensure it has 'num_features', 'feature_info', or 'embed_dim'."
            )

        # Add the classification head
        self.classifier = nn.Linear(int(num_backbone_features), self.num_classes)

    def _get_declared_backbone_output_dim(self) -> Optional[int]:
        """Read the backbone's advertised output dimension from common metadata fields."""
        if hasattr(self.backbone, "num_features"):
            return int(self.backbone.num_features)
        if hasattr(self.backbone, "feature_info") and isinstance(self.backbone.feature_info, list) and self.backbone.feature_info:
            return int(self.backbone.feature_info[-1]["num_ftrs"])
        if hasattr(self.backbone, "embed_dim"):
            return int(self.backbone.embed_dim)
        return None

    def _infer_backbone_output_dim(self) -> Optional[int]:
        """Infer the backbone output dimension with a lightweight dummy forward pass."""
        if self.sensor_config is None or self.img_size is None:
            return None

        num_bands = self.sensor_config.get("num_bands", None)
        if num_bands is None:
            return None

        was_training = self.backbone.training
        dummy = torch.zeros(
            1,
            int(num_bands),
            int(self.img_size),
            int(self.img_size),
            dtype=torch.float32,
        )

        try:
            self.backbone.eval()
            with torch.no_grad():
                features = self._pool_backbone_features(self.backbone(dummy))
        except Exception as exc:
            print(
                "Warning: dummy forward-based backbone output-dim inference failed for "
                f"{self.backbone.__class__.__name__}: {exc}"
            )
            return None
        finally:
            self.backbone.train(was_training)

        if features.dim() != 2:
            print(
                "Warning: dummy forward-based backbone output-dim inference expected pooled "
                f"features of shape (batch_size, num_features), got {tuple(features.shape)}."
            )
            return None
        return int(features.shape[1])

    @staticmethod
    def _pool_backbone_features(
        features: Union[Tensor, List[Tensor], tuple[Tensor, ...]],
    ) -> Tensor:
        """Convert common backbone feature outputs to pooled classification vectors."""
        if isinstance(features, (list, tuple)):
            if not features:
                raise ValueError("Backbone returned an empty feature list.")
            features = features[-1]

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                "Backbone is expected to return a Tensor or non-empty list/tuple "
                f"of Tensors, got {type(features)}."
            )

        if features.dim() == 4:
            return features.mean(dim=(2, 3))
        if features.dim() == 3:
            return features.mean(dim=1)
        return features

    def _configure_loss(self) -> None:
        """Configure the loss function."""
        if self.loss_name == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Loss type '{self.loss_name}' is not valid.")

    def _freeze_backbone(self) -> None:
        """Freeze the backbone parameters. The classifier head remains unfrozen."""
        print(f"Freezing backbone: {self.backbone.__class__.__name__}")
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Ensure classifier is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def _unfreeze_adapter_layers(self) -> None:
        """Unfreeze spectral adapter layers if they exist in the backbone."""
        if hasattr(self.backbone, 'spectral_adapter') and isinstance(self.backbone.spectral_adapter, nn.Module):
            print(f"Unfreezing spectral_adapter in {self.backbone.__class__.__name__}")
            for param in self.backbone.spectral_adapter.parameters():
                param.requires_grad = True
        else:
            if self.finetune_adapter: # Only warn if user explicitly asked to finetune adapter
                 print(f"Warning: 'finetune_adapter' is True, but backbone {self.backbone.__class__.__name__} "
                       f"does not have a 'spectral_adapter' attribute or it's not an nn.Module.")

    def _unfreeze_first_n_layers(self) -> None:
        """Unfreeze the first N layers/blocks of the backbone."""
        n = self.finetune_first_n_layers
        if n <= 0:
            return

        unfrozen_param_count = 0

        # Standard ViT-style backbone (e.g., timm ViTs, custom ViTs with 'blocks')
        if hasattr(self.backbone, "blocks") and isinstance(self.backbone.blocks, nn.ModuleList):
            print(f"Attempting to unfreeze first {n} blocks of ViT-style backbone {self.backbone.__class__.__name__}")
            for i, block in enumerate(self.backbone.blocks):
                if i < n:
                    for param in block.parameters():
                        param.requires_grad = True
                        unfrozen_param_count += 1
                else:
                    break 
        # ResNet-style backbone (e.g., timm ResNets with 'layer1', 'layer2', etc.)
        elif hasattr(self.backbone, "layer1"): # Check for layer1 as a proxy
            print(f"Attempting to unfreeze first {n} stages of ResNet-style backbone {self.backbone.__class__.__name__}")
            stages_to_unfreeze = []
            for i in range(1, 5): # ResNets typically have up to 4 stages (layer1 to layer4)
                layer_attr = f"layer{i}"
                if hasattr(self.backbone, layer_attr):
                    if i <=n:
                        stages_to_unfreeze.append(getattr(self.backbone, layer_attr))
                    else:
                        break # Stop if we've collected enough stages
            
            for stage in stages_to_unfreeze:
                for param in stage.parameters():
                    param.requires_grad = True
                    unfrozen_param_count +=1
        # DOFA/HyperSigma style if they have an 'encoder' with 'blocks'
        elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "blocks") and isinstance(self.backbone.encoder.blocks, nn.ModuleList):
            print(f"Attempting to unfreeze first {n} blocks of encoder in {self.backbone.__class__.__name__}")
            for i, block in enumerate(self.backbone.encoder.blocks):
                if i < n:
                    for param in block.parameters():
                        param.requires_grad = True
                        unfrozen_param_count += 1
                else:
                    break
        else:
             print(f"Warning: Could not determine how to unfreeze first {n} layers for backbone {self.backbone.__class__.__name__}. "
                   f"Structure not recognized (ViT blocks, ResNet layers, or encoder.blocks).")

        if unfrozen_param_count > 0:
            print(f"Successfully unfroze {unfrozen_param_count} parameters in the first {n} layers/blocks of {self.backbone.__class__.__name__}.")
        elif n > 0 :
             print(f"Warning: Requested to unfreeze first {n} layers, but no parameters were unfrozen in {self.backbone.__class__.__name__}.")

    def _configure_metrics(self) -> None:
        """Configure metrics for training, validation, and testing."""
        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": MultilabelAccuracy(
                    num_labels=self.num_classes, average="micro"
                ),
                "AverageAccuracy": MultilabelAccuracy(
                    num_labels=self.num_classes, average="macro"
                ),
                "F1Score": MultilabelFBetaScore(
                    num_labels=self.num_classes,
                    beta=1.0,
                    average="micro",
                ),
                "AveragePrecision": MultilabelAveragePrecision(
                    num_labels=self.num_classes, average="macro"
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        
        self.max_val_metrics = {}
        for key in self.val_metrics.keys():
            self.max_val_metrics['max_' + key] = 0.0

    def forward(self, x: Any) -> Tensor:
        """Forward pass of the model. Backbone -> Classifier Head.

        We expect the backbone to return already pooled features of shape
        (batch_size, num_features). Any deviation is treated as an error so
        misconfigured backbones are caught early.
        """
        features = self._pool_backbone_features(self.backbone(x))

        if features.dim() != 2:
            raise ValueError(
                f"Backbone {self.backbone.__class__.__name__} is expected to return pooled features "
                f"of shape (batch_size, num_features), but got tensor with shape {tuple(features.shape)}."
            )

        if features.shape[1] != self.classifier.in_features:
            raise ValueError(
                f"Classifier in_features ({self.classifier.in_features}) does not match backbone output "
                f"feature dimension ({features.shape[1]}). Check backbone_config and num_features."
            )

        logits = self.classifier(features)
        return logits

    @staticmethod
    def _select_model_input(batch: Dict[str, Any]) -> Any:
        return batch.get("image_by_sensor", batch["image"])

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """Compute and return the training loss."""
        x, y = self._select_model_input(batch), batch["label"] # Assuming 'label' for multi-label
        y_hat = self(x) # Raw logits

        loss = self.loss(y_hat, y.float()) # Loss function expects logits

        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        
        # Convert logits to probabilities for metrics
        y_hat_probs = torch.sigmoid(y_hat)
        self.train_metrics(y_hat_probs, y.int()) # Metrics expect probabilities and int targets
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """Compute validation loss and log metrics."""
        x, y = self._select_model_input(batch), batch["label"]
        y_hat = self(x) # Raw logits

        loss = self.loss(y_hat, y.float()) # Loss function expects logits
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # Convert logits to probabilities for metrics
        y_hat_probs = torch.sigmoid(y_hat)
        self.val_metrics(y_hat_probs, y.int()) # Metrics expect probabilities and int targets

    def on_validation_epoch_end(self) -> None:
        val_metrics_computed = self.val_metrics.compute()
        self.log_dict(val_metrics_computed)
        
        # Update max validation metrics
        for key, value in val_metrics_computed.items():
            max_key = 'max_' + key
            if value > self.max_val_metrics.get(max_key, -float('inf')): # Initialize with -inf if not exists
                self.max_val_metrics[max_key] = value
        self.log_dict(self.max_val_metrics, on_step=False, on_epoch=True) # Log all max metrics

        self.val_metrics.reset()

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """Compute test loss and log metrics."""
        x, y = self._select_model_input(batch), batch["label"]
        y_hat = self(x) # Raw logits

        loss = self.loss(y_hat, y.float()) # Loss function expects logits
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # Convert logits to probabilities for metrics
        y_hat_probs = torch.sigmoid(y_hat)
        self.test_metrics(y_hat_probs, y.int()) 

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(
        self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Perform a prediction step."""
        x = self._select_model_input(batch)
        y_hat = self(x)
        # Apply sigmoid to get probabilities for multi-label
        return torch.sigmoid(y_hat)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizer and learning rate scheduler."""
        # Collect parameters that require gradients.
        # This is important if parts of the model are frozen.
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        optimizer = torch.optim.AdamW(
            trainable_params, # Pass only trainable parameters
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=self.eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
