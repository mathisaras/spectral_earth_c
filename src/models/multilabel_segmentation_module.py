"""LightningModule for multi-label semantic segmentation."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelFBetaScore,
    MultilabelJaccardIndex,
    MultilabelPrecision,
    MultilabelRecall,
)

from .model_utils import (
    create_model,
    infer_spatial_decoder_patch_size,
    infer_multitap_decoder_info,
    instantiate_component,
    load_pretrained_weights,
)


class MultiLabelSegmentationModule(LightningModule):
    """Segmentation module for per-pixel multi-label targets.

    Targets are expected to have shape `(B, C, H, W)` with values:
    - `0/1` for valid binary supervision
    - `-1` for ignored pixels (applied to every channel at those locations)
    """

    def __init__(
        self,
        backbone_config: DictConfig,
        decoder_config: DictConfig,
        num_classes: int,
        pretrained_weights: Optional[str] = None,
        model_type: str = "vit_seg",
        tap_indices: Optional[list[int]] = None,
        multi_tap_is_vit: Optional[bool] = None,
        probe_in_chans: Optional[int] = None,
        probe_img_size: Optional[Union[int, list[int]]] = None,
        class_names: Optional[list[str]] = None,
        log_per_class_metrics: bool = True,
        log_validation_images: bool = True,
        validation_image_log_interval: int = 20,
        validation_image_max_samples: int = 4,
        validation_image_batch_idx: int = 0,
        validation_prediction_threshold: float = 0.5,
        rgb_indices: Optional[list[int]] = None,
        freeze_backbone: bool = False,
        finetune_adapter: bool = False,
        finetune_first_n_layers: int = 0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        t_max: int = 10,
        eta_min: float = 1.0e-5,
    ) -> None:
        super().__init__()

        self.backbone_config_original = backbone_config
        self.backbone_config = OmegaConf.to_container(backbone_config, resolve=True)
        self.decoder_config = decoder_config

        self.num_classes = int(num_classes)
        self.pretrained_weights = pretrained_weights
        self.model_type = str(model_type)
        self.tap_indices = [int(idx) for idx in tap_indices] if tap_indices is not None else None
        self.multi_tap_is_vit = multi_tap_is_vit
        self.probe_in_chans = int(probe_in_chans) if probe_in_chans is not None else None
        self.probe_img_size = probe_img_size
        self.log_per_class_metrics = bool(log_per_class_metrics)
        self.log_validation_images = bool(log_validation_images)
        self.validation_image_log_interval = int(validation_image_log_interval)
        self.validation_image_max_samples = int(validation_image_max_samples)
        self.validation_image_batch_idx = int(validation_image_batch_idx)
        self.validation_prediction_threshold = float(validation_prediction_threshold)
        self.rgb_indices = [int(idx) for idx in (rgb_indices or [45, 30, 15])]
        self.freeze_backbone = bool(freeze_backbone)
        self.finetune_adapter = bool(finetune_adapter)
        self.finetune_first_n_layers = int(finetune_first_n_layers)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.t_max = int(t_max)
        self.eta_min = float(eta_min)
        self.class_names = self._resolve_class_names(class_names)
        self.class_metric_suffixes = [
            self._metric_safe_name(class_name) for class_name in self.class_names
        ]
        self._class_colors = self._build_class_colors(self.num_classes)

        self.vit_patch_size = self.backbone_config.get("patch_size")

        self._build_model()
        self._configure_metrics()

        if self.freeze_backbone:
            self._freeze_encoder()
        if self.finetune_adapter:
            self._unfreeze_adapter_layers()
        if self.finetune_first_n_layers > 0:
            self._unfreeze_first_n_layers()

    def _build_model(self) -> None:
        if not isinstance(self.backbone_config, dict):
            raise TypeError(
                f"self.backbone_config is not a dict, but {type(self.backbone_config)}."
            )

        target_path = self.backbone_config.get("_target_")
        model_name_from_config = self.backbone_config.get("model_name")
        is_resnet = False
        if isinstance(target_path, str) and "resnet" in target_path.lower():
            is_resnet = True
        if not is_resnet and isinstance(model_name_from_config, str) and "resnet" in model_name_from_config.lower():
            is_resnet = True
        if is_resnet:
            self.backbone_config["replace_stride_with_dilation"] = [True, True, True, True]
            self.backbone_config["return_features"] = True

        self.backbone = instantiate_component(config=self.backbone_config)
        if self.pretrained_weights:
            load_pretrained_weights(self.backbone, self.pretrained_weights, strict=False)

        if hasattr(self.backbone, "num_features"):
            backbone_output_channels: Union[int, list[int]] = self.backbone.num_features
        elif hasattr(self.backbone, "feature_channels"):
            backbone_output_channels = self.backbone.feature_channels
        elif hasattr(self.backbone, "encoder_channels"):
            backbone_output_channels = self.backbone.encoder_channels
        else:
            raise AttributeError(
                f"Backbone {self.backbone.__class__.__name__} must expose "
                f"'num_features', 'feature_channels', or 'encoder_channels'."
            )

        is_upernet = False
        is_multiscale_conv = False
        is_lightweight_multitap = False
        target = None
        if hasattr(self.decoder_config, "_target_"):
            target = self.decoder_config._target_
        elif isinstance(self.decoder_config, dict):
            target = self.decoder_config.get("_target_")
        if isinstance(target, str):
            is_upernet = "upernet" in target.lower()
            is_multiscale_conv = "multiscale_conv" in target.lower()
            is_lightweight_multitap = "lightweight_multitap" in target.lower()

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
            decoder_override_params = {"num_classes": self.num_classes}
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
            decoder_override_params = {
                "num_input_features": backbone_output_channels,
                "num_classes": self.num_classes,
            }

        self.decoder = instantiate_component(
            config=self.decoder_config,
            **decoder_override_params,
        )
        self.model = create_model(
            backbone=self.backbone,
            decoder=self.decoder,
            model_type=self.model_type,
            patch_size=self.vit_patch_size,
            tap_indices=self.tap_indices,
            multi_tap_is_vit=self.multi_tap_is_vit,
        )

    def _resolve_class_names(self, class_names: Optional[list[str]]) -> list[str]:
        if class_names is None:
            return [f"class_{idx}" for idx in range(self.num_classes)]
        resolved = [str(name) for name in class_names]
        if len(resolved) != self.num_classes:
            raise ValueError(
                f"class_names length {len(resolved)} does not match "
                f"num_classes={self.num_classes}."
            )
        return resolved

    @staticmethod
    def _metric_safe_name(name: str) -> str:
        normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
        return normalized or "class"

    def _build_per_class_metrics(self) -> nn.ModuleDict:
        metric_kwargs = {"num_labels": self.num_classes}
        return nn.ModuleDict(
            {
                "PerClassPrecision": MultilabelPrecision(average=None, **metric_kwargs),
                "PerClassRecall": MultilabelRecall(average=None, **metric_kwargs),
                "PerClassF1": MultilabelFBetaScore(beta=1.0, average=None, **metric_kwargs),
                "PerClassIoU": MultilabelJaccardIndex(average=None, **metric_kwargs),
            }
        )

    def _configure_metrics(self) -> None:
        metric_kwargs = {"num_labels": self.num_classes}
        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": MultilabelAccuracy(average="micro", **metric_kwargs),
                "MeanAccuracy": MultilabelAccuracy(average="macro", **metric_kwargs),
                "MicroF1": MultilabelFBetaScore(beta=1.0, average="micro", **metric_kwargs),
                "MacroF1": MultilabelFBetaScore(beta=1.0, average="macro", **metric_kwargs),
                "MicroIoU": MultilabelJaccardIndex(average="micro", **metric_kwargs),
                "MeanIoU": MultilabelJaccardIndex(average="macro", **metric_kwargs),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        self.train_per_class_metrics = self._build_per_class_metrics()
        self.val_per_class_metrics = self._build_per_class_metrics()
        self.test_per_class_metrics = self._build_per_class_metrics()

        self.max_val_metrics = {}
        for key in self.val_metrics.keys():
            self.max_val_metrics[f"max_{key}"] = 0.0

    def _freeze_encoder(self) -> None:
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_adapter_layers(self) -> None:
        try:
            for param in self.model.encoder.spectral_adapter.parameters():
                param.requires_grad = True
        except AttributeError:
            print("Warning: The backbone does not have 'spectral_adapter' attributes.")

    def _unfreeze_first_n_layers(self) -> None:
        if hasattr(self.model.encoder, "blocks"):
            for i, block in enumerate(self.model.encoder.blocks):
                if i < self.finetune_first_n_layers:
                    for param in block.parameters():
                        param.requires_grad = True
        else:
            print("Warning: Unable to unfreeze first n layers - model structure not recognized")

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

    def forward(self, x: Any, output_size: Optional[Union[int, list[int], tuple[int, int]]] = None) -> Tensor:
        if output_size is not None:
            return self.model(x, output_size=output_size)
        return self.model(x)

    def _compute_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        valid_mask = target >= 0
        target_valid = target.clamp_min(0.0)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target_valid,
            reduction="none",
        )
        loss = loss * valid_mask.float()
        denom = valid_mask.float().sum().clamp_min(1.0)
        return loss.sum() / denom

    def _valid_pixel_targets(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        # Invalid pixels are marked across every channel at the same location.
        valid_pixels = (target >= 0).all(dim=1)
        if int(valid_pixels.sum()) == 0:
            return None, None

        probs = torch.sigmoid(logits).permute(0, 2, 3, 1)[valid_pixels]
        target_int = target.clamp_min(0).permute(0, 2, 3, 1)[valid_pixels].int()
        return probs, target_int

    def _update_per_class_metrics(self, metric_dict: nn.ModuleDict, probs: Tensor, target: Tensor) -> None:
        if not self.log_per_class_metrics:
            return
        for metric in metric_dict.values():
            metric.update(probs, target)

    def _log_per_class_metrics(self, metric_dict: nn.ModuleDict, stage: str) -> None:
        if not self.log_per_class_metrics:
            return

        for metric_name, metric in metric_dict.items():
            values = metric.compute()
            if values.ndim == 0:
                values = values.unsqueeze(0)
            for class_idx, class_value in enumerate(values):
                class_suffix = self.class_metric_suffixes[class_idx]
                self.log(
                    f"{stage}_{metric_name}_{class_suffix}",
                    class_value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    @staticmethod
    def _reset_per_class_metrics(metric_dict: nn.ModuleDict) -> None:
        for metric in metric_dict.values():
            metric.reset()

    @staticmethod
    def _build_class_colors(num_classes: int) -> Tensor:
        # Color-blind friendly palette. Extra classes cycle through the palette.
        base_colors = torch.tensor(
            [
                [230, 159, 0],
                [213, 94, 0],
                [86, 180, 233],
                [0, 158, 115],
                [240, 228, 66],
                [0, 114, 178],
                [204, 121, 167],
                [148, 103, 189],
                [89, 89, 89],
                [17, 119, 51],
            ],
            dtype=torch.float32,
        ) / 255.0
        repeats = int(np.ceil(max(num_classes, 1) / len(base_colors)))
        return base_colors.repeat((repeats, 1))[:num_classes]

    def _should_log_validation_images(self, batch_idx: int) -> bool:
        if not self.log_validation_images:
            return False
        if not getattr(self.trainer, "is_global_zero", False):
            return False
        if self.validation_image_max_samples <= 0:
            return False
        if batch_idx != self.validation_image_batch_idx:
            return False
        interval = max(1, self.validation_image_log_interval)
        return self.current_epoch == 0 or (self.current_epoch + 1) % interval == 0

    def _image_to_rgb(self, image: Tensor) -> np.ndarray:
        image = image.detach().float().cpu()
        num_bands = image.shape[0]
        indices = [idx for idx in self.rgb_indices if 0 <= idx < num_bands]
        if len(indices) < 3:
            if num_bands >= 3:
                indices = [0, min(1, num_bands - 1), min(2, num_bands - 1)]
            else:
                indices = [0, 0, 0]
        rgb = image[indices[:3]].permute(1, 2, 0).numpy()
        if np.nanmin(rgb) >= 0.0 and np.nanmax(rgb) <= 1.0:
            return np.clip(rgb, 0.0, 1.0)
        lo = float(np.nanpercentile(rgb, 2.0))
        hi = float(np.nanpercentile(rgb, 98.0))
        return np.clip((rgb - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)

    def _multilabel_to_rgb(self, mask: Tensor) -> np.ndarray:
        mask = mask.detach().float().cpu()
        valid_pixels = (mask >= 0).all(dim=0)
        binary = mask > 0.5
        colors = self._class_colors.to(mask.device)
        composite = torch.zeros((*binary.shape[1:], 3), dtype=torch.float32)
        active_counts = binary.sum(dim=0).clamp_min(1).float()
        for class_idx in range(self.num_classes):
            composite += binary[class_idx].unsqueeze(-1).float() * colors[class_idx]
        positive_pixels = binary.any(dim=0)
        composite[positive_pixels] = composite[positive_pixels] / active_counts[positive_pixels].unsqueeze(-1)
        composite[~valid_pixels] = torch.tensor([1.0, 0.87, 0.34], dtype=torch.float32)
        return composite.numpy()

    def _error_to_rgb(self, prediction: Tensor, target: Tensor) -> np.ndarray:
        prediction = prediction.detach().bool().cpu()
        target = target.detach().float().cpu()
        valid_pixels = (target >= 0).all(dim=0)
        target_binary = target > 0.5
        false_positive = (prediction & ~target_binary).any(dim=0)
        false_negative = (~prediction & target_binary).any(dim=0)
        true_positive = (prediction & target_binary).any(dim=0)
        image = torch.zeros((*target.shape[1:], 3), dtype=torch.float32)
        image[true_positive] = torch.tensor([0.15, 0.68, 0.38])
        image[false_positive] = torch.tensor([0.90, 0.36, 0.20])
        image[false_negative] = torch.tensor([0.20, 0.45, 0.85])
        image[false_positive & false_negative] = torch.tensor([0.78, 0.32, 0.76])
        image[~valid_pixels] = torch.tensor([1.0, 0.87, 0.34])
        return image.numpy()

    @staticmethod
    def _make_visual_row(images: list[np.ndarray]) -> np.ndarray:
        separator = np.ones((images[0].shape[0], 2, 3), dtype=np.float32)
        row = images[0]
        for image in images[1:]:
            row = np.concatenate([row, separator, image], axis=1)
        return np.clip(row, 0.0, 1.0)

    def _log_validation_images(self, batch: Dict[str, Tensor], logits: Tensor) -> None:
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return
        experiment = self.logger.experiment
        if not hasattr(experiment, "log"):
            return

        try:
            import wandb
        except ImportError:
            return

        images = batch["image"].detach()
        target = batch["mask"].detach()
        probabilities = torch.sigmoid(logits.detach())
        prediction = probabilities >= self.validation_prediction_threshold
        max_samples = min(self.validation_image_max_samples, images.shape[0])
        logged_images = []

        for sample_idx in range(max_samples):
            row = self._make_visual_row(
                [
                    self._image_to_rgb(images[sample_idx]),
                    self._multilabel_to_rgb(target[sample_idx]),
                    self._multilabel_to_rgb(prediction[sample_idx].float()),
                    self._error_to_rgb(prediction[sample_idx], target[sample_idx]),
                ]
            )
            sample_id = batch.get("sample_id")
            if isinstance(sample_id, (list, tuple)) and sample_idx < len(sample_id):
                caption_id = str(sample_id[sample_idx])
            else:
                caption_id = f"sample_{sample_idx}"
            caption = (
                f"{caption_id} | input | gt | pred | error "
                f"(green=TP, red=FP, blue=FN, yellow=ignore)"
            )
            logged_images.append(wandb.Image(row, caption=caption))

        if logged_images:
            experiment.log(
                {"val/multilabel_examples": logged_images},
                step=int(self.global_step),
            )

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x = self._select_model_input(batch)
        y = batch["mask"].float()
        batch_size = self._input_batch_size(x)
        logits = self(x, output_size=y.shape[-2:])
        loss = self._compute_loss(logits, y)

        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True, batch_size=batch_size)
        probs, target_int = self._valid_pixel_targets(logits, y)
        if probs is not None and target_int is not None:
            self.train_metrics(probs, target_int)
            self._update_per_class_metrics(self.train_per_class_metrics, probs, target_int)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self._log_per_class_metrics(self.train_per_class_metrics, stage="train")
        self.train_metrics.reset()
        self._reset_per_class_metrics(self.train_per_class_metrics)

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        x = self._select_model_input(batch)
        y = batch["mask"].float()
        batch_size = self._input_batch_size(x)
        logits = self(x, output_size=y.shape[-2:])
        loss = self._compute_loss(logits, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        probs, target_int = self._valid_pixel_targets(logits, y)
        if probs is not None and target_int is not None:
            self.val_metrics(probs, target_int)
            self._update_per_class_metrics(self.val_per_class_metrics, probs, target_int)
        if self._should_log_validation_images(batch_idx):
            self._log_validation_images(batch, logits)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self._log_per_class_metrics(self.val_per_class_metrics, stage="val")
        for key, value in metrics.items():
            max_key = f"max_{key}"
            if value > self.max_val_metrics[max_key]:
                self.max_val_metrics[max_key] = value.item()
                self.log(max_key, self.max_val_metrics[max_key], sync_dist=True)
        self.val_metrics.reset()
        self._reset_per_class_metrics(self.val_per_class_metrics)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        x = self._select_model_input(batch)
        y = batch["mask"].float()
        batch_size = self._input_batch_size(x)
        logits = self(x, output_size=y.shape[-2:])
        loss = self._compute_loss(logits, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        probs, target_int = self._valid_pixel_targets(logits, y)
        if probs is not None and target_int is not None:
            self.test_metrics(probs, target_int)
            self._update_per_class_metrics(self.test_per_class_metrics, probs, target_int)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self._log_per_class_metrics(self.test_per_class_metrics, stage="test")
        self.test_metrics.reset()
        self._reset_per_class_metrics(self.test_per_class_metrics)

    def predict_step(
        self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        x = self._select_model_input(batch)
        output_size = batch["mask"].shape[-2:] if "mask" in batch else None
        return torch.sigmoid(self(x, output_size=output_size))

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
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
