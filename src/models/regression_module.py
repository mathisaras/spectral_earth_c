import os
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from lightning.pytorch import LightningModule
from torchmetrics import MeanSquaredError, MetricCollection
from omegaconf import DictConfig

from ..backbones.registry import BACKBONE_REGISTRY

from ..backbones.dofa_encoder import DofaBasePatch16, DofaLargePatch16
from ..backbones.hypersigma import hypersigma_b, hypersigma_l
from .model_utils import instantiate_component, load_pretrained_weights

# =============================================================================
# CUSTOM MSE LOSS
# =============================================================================
def custom_mse_loss(
    y_pred: Tensor,
    y_true: Tensor,
    baseline_outputs: Tensor = torch.tensor([0.0, 0.0, 0.0, 0.0])
) -> Tensor:
    """
    Computes the custom MSE loss as the ratio between the model's MSE and the MSE 
    of a naive baseline (which returns the empirical mean, here assumed to be 0).

    Args:
        y_pred: Tensor of shape (batch_size, 4), predictions from the model.
        y_true: Tensor of shape (batch_size, 4), ground truth values.
        baseline_outputs: Tensor of shape (4,), baseline predictions (default zeros).

    Returns:
        A scalar tensor representing the normalized MSE loss.
    """
    # Ensure baseline_outputs is on the same device as y_true.
    baseline_outputs = baseline_outputs.to(y_true.device)
    
    # Compute MSE for the model's predictions (per target).
    mse_model = F.mse_loss(y_pred, y_true, reduction="none").mean(dim=0)
    
    # Compute MSE for the naive baseline (mean predictor).
    baseline_tensor = baseline_outputs.unsqueeze(0).expand_as(y_true)
    mse_baseline = F.mse_loss(baseline_tensor, y_true, reduction="none").mean(dim=0)
    
    # Compute normalized MSE (per target) and average them.
    normalized_mse = mse_model / mse_baseline
    loss = normalized_mse.mean()
    return loss

# =============================================================================
# REGRESSION MODULE
# =============================================================================
class RegressionModule(LightningModule):
    """
    LightningModule for regression on hyperspectral data using a backbone model
    from timm. The module uses a custom MSE loss that computes the ratio between the
    model's MSE and the MSE of a naive baseline (which returns the normalized target mean, 0).
    """
    def __init__(
        self,
        backbone_config: DictConfig,
        num_outputs: int,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_dropout: float = 0.2,
        pretrained_weights_path: Optional[str] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        t_max: int = 100,
        freeze_backbone: bool = False,
        finetune_adapter: bool = False,
        custom_loss_baseline: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self._build_model()

        if self.hparams.custom_loss_baseline is not None:
            self.baseline_for_loss = torch.tensor(self.hparams.custom_loss_baseline, dtype=torch.float32)
            self.loss_fn = custom_mse_loss
        else:
            self.loss_fn = nn.MSELoss()

        self.train_metrics = MetricCollection({"RMSE": MeanSquaredError(squared=False)}, prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.best_val_loss = float("inf")

        if self.hparams.freeze_backbone:
            self._freeze_backbone()
        if self.hparams.finetune_adapter:
            self._unfreeze_adapter_layers()

    def _build_model(self) -> None:
        self.backbone = instantiate_component(config=self.hparams.backbone_config)

        if self.hparams.pretrained_weights_path:
            load_pretrained_weights(self.backbone, self.hparams.pretrained_weights_path, strict=False)

        if hasattr(self.backbone, 'num_features'):
            in_features_for_mlp = self.backbone.num_features
        elif hasattr(self.backbone, 'feature_info') and callable(getattr(self.backbone, 'feature_info', None)):
            try:
                raw_feature_info = self.backbone.feature_info.get_dicts()
                if raw_feature_info: in_features_for_mlp = raw_feature_info[-1]['num_ftrs']
                else: raise AttributeError("feature_info.get_dicts() returned empty.")
            except Exception:
                 raise AttributeError(f"Backbone {self.backbone.__class__.__name__} lacks 'num_features' and couldn't infer from 'feature_info'.")
        else:
            raise AttributeError(f"Backbone {self.backbone.__class__.__name__} must have 'num_features' or a compatible 'feature_info' method.")
        
        if not isinstance(in_features_for_mlp, int) or in_features_for_mlp <= 0:
            raise ValueError(f"Could not determine a valid positive integer for backbone output features (got {in_features_for_mlp}).")

        hidden_dims = self.hparams.mlp_hidden_dims if self.hparams.mlp_hidden_dims is not None else []
        all_dims = [in_features_for_mlp] + hidden_dims + [self.hparams.num_outputs]
        
        layers = []
        layers.append(nn.BatchNorm1d(in_features_for_mlp))
        
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
            if i < len(all_dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if self.hparams.mlp_dropout > 0:
                    layers.append(nn.Dropout(self.hparams.mlp_dropout))
        self.mlp_head_layers = nn.Sequential(*layers)

        self.model = nn.Sequential(self.backbone, self.mlp_head_layers)
        
        if hasattr(self, '_configure_backbone'):
            delattr(self, '_configure_backbone')
        if hasattr(self, '_load_pretrained_weights') and callable(getattr(self, '_load_pretrained_weights')):
             delattr(self, '_load_pretrained_weights')

    def _freeze_backbone(self) -> None:
        print(f"Freezing backbone: {self.backbone.__class__.__name__}")
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Ensuring MLP head ({self.mlp_head_layers.__class__.__name__}) is trainable.")
        for param in self.mlp_head_layers.parameters():
            param.requires_grad = True

    def _unfreeze_adapter_layers(self) -> None:
        if hasattr(self.backbone, 'spectral_adapter') and isinstance(self.backbone.spectral_adapter, nn.Module):
            print(f"Unfreezing spectral_adapter in {self.backbone.__class__.__name__}")
            for param in self.backbone.spectral_adapter.parameters():
                param.requires_grad = True
        else:
            if self.hparams.finetune_adapter:
                 print(f"Warning: 'finetune_adapter' is True, but backbone {self.backbone.__class__.__name__} has no 'spectral_adapter' or it's not an nn.Module.")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        
        if hasattr(self, 'baseline_for_loss'):
            loss = self.loss_fn(y_hat, y, self.baseline_for_loss.to(self.device))
        else:
            loss = self.loss_fn(y_hat, y)
            
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_metrics(y_hat, y)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        if hasattr(self, 'baseline_for_loss'):
            loss = self.loss_fn(y_hat, y, self.baseline_for_loss.to(self.device))
        else:
            loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics(y_hat, y)

    def on_validation_epoch_end(self) -> None:
        avg_loss = self.trainer.logged_metrics.get("val_loss_epoch", None)
        if avg_loss is not None and avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        self.log("best_val_loss", self.best_val_loss, prog_bar=True, sync_dist=True)
        
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        if hasattr(self, 'baseline_for_loss'):
            loss = self.loss_fn(y_hat, y, self.baseline_for_loss.to(self.device))
        else:
            loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_metrics(y_hat, y)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.test_metrics.reset()

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        x = batch["image"]
        return self(x)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.t_max, eta_min=self.hparams.learning_rate / 100
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }