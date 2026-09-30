import copy
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import kornia.augmentation as K
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightly.models.utils import deactivate_requires_grad
from lightly.utils.scheduler import cosine_schedule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.transforms.augmentations import SpatialAugmentation
from src.transforms.normalize_mm import SingleSensorNormalizer
from src.transforms.standardize_mm import SingleSensorStandardizer
from src.utils.sensor_registry import SensorRegistry

from .model_utils import instantiate_component


def _normalize_sensor_name(name: str) -> str:
    return str(name).upper()


def _odd_kernel_for_size(size_hw: Tuple[int, int]) -> int:
    min_size = int(min(size_hw))
    kernel_size = max(3, min_size // 10)
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


class MultimodalViewBase(LightningModule):
    """Shared view construction and optimization code for multimodal pretraining."""

    def __init__(
        self,
        backbone_config: Dict[str, Any],
        modality_channels: Optional[Dict[str, int]] = None,
        global_views: int = 2,
        global_modalities_per_view: int = 4,
        n_local_views: int = 4,
        local_modalities_per_view: int = 1,
        local_modalities_from_global_only: bool = False,
        local_excluded_modalities: Optional[List[str]] = None,
        apply_input_normalization: bool = True,
        standardize_modalities: Optional[List[str]] = None,
        standardization_mode: str = "bandwise",
        standardization_stats_path: Optional[str] = None,
        standardization_eps: float = 1.0e-6,
        global_crop_scale: Tuple[float, float] = (0.4, 1.0),
        local_crop_scale: Tuple[float, float] = (0.1, 0.4),
        radiometric_aug_p: float = 0.5,
        radiometric_brightness_range: Tuple[float, float] = (0.8, 1.2),
        radiometric_bias_range: Tuple[float, float] = (-0.1, 0.1),
        lr: float = 1.0e-4,
        weight_decay: float = 1.0e-6,
        warmup_epochs: int = 20,
        max_epochs: int = 100,
        teacher_momentum_start: float = 0.996,
        teacher_momentum_end: float = 1.0,
        create_teacher_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone_config"])

        if int(global_views) < 1:
            raise ValueError(f"global_views must be >= 1, got {global_views}.")
        if int(n_local_views) < 0:
            raise ValueError(f"n_local_views must be >= 0, got {n_local_views}.")
        if int(global_modalities_per_view) < 1:
            raise ValueError("global_modalities_per_view must be >= 1.")
        if int(local_modalities_per_view) < 1:
            raise ValueError("local_modalities_per_view must be >= 1.")

        self.student_backbone = instantiate_component(config=backbone_config)
        self.teacher_backbone: Optional[nn.Module] = None
        if bool(create_teacher_backbone):
            self.teacher_backbone = copy.deepcopy(self.student_backbone)
            deactivate_requires_grad(self.teacher_backbone)

        if hasattr(self.student_backbone, "embed_dim"):
            self.embed_dim = int(getattr(self.student_backbone, "embed_dim"))
        elif hasattr(self.student_backbone, "num_features"):
            self.embed_dim = int(getattr(self.student_backbone, "num_features"))
        else:
            raise ValueError("Backbone must expose embed_dim or num_features.")

        if modality_channels is not None:
            self.modality_channels = {
                _normalize_sensor_name(k): int(v) for k, v in modality_channels.items()
            }
        elif hasattr(self.student_backbone, "sensor_in_chans"):
            self.modality_channels = {
                _normalize_sensor_name(k): int(v)
                for k, v in self.student_backbone.sensor_in_chans.items()
            }
        else:
            raise ValueError("modality_channels was not provided and backbone has no sensor_in_chans.")
        self.modality_order = sorted(self.modality_channels.keys())
        if not self.modality_order:
            raise ValueError("At least one modality must be configured.")

        self.local_excluded_modalities = self._normalize_modality_list(local_excluded_modalities)
        unknown_local = sorted(m for m in self.local_excluded_modalities if m not in self.modality_order)
        if unknown_local:
            raise ValueError(f"Unknown local_excluded_modalities: {unknown_local}.")

        default_input_size: Optional[Tuple[int, int]] = None
        if hasattr(self.student_backbone, "input_size"):
            default_input_size = tuple(int(v) for v in self.student_backbone.input_size)
        sensor_input_size_cfg = getattr(self.student_backbone, "sensor_input_size", {}) or {}
        self.sensor_input_sizes: Dict[str, Tuple[int, int]] = {}
        for modality in self.modality_order:
            size = sensor_input_size_cfg.get(modality, default_input_size)
            if size is None:
                raise ValueError(f"Backbone is missing input size for modality '{modality}'.")
            self.sensor_input_sizes[modality] = tuple(int(v) for v in size)

        self.normalizers = nn.ModuleDict()
        self.standardizers = nn.ModuleDict()
        self.global_blur_aug = nn.ModuleDict()
        self.local_blur_aug = nn.ModuleDict()

        standardize_modalities_norm = self._normalize_modality_list(standardize_modalities)
        unknown_standardize = sorted(m for m in standardize_modalities_norm if m not in self.modality_order)
        if unknown_standardize:
            raise ValueError(f"Unknown standardize_modalities: {unknown_standardize}.")
        if standardize_modalities_norm and not standardization_stats_path:
            raise ValueError("standardize_modalities requires standardization_stats_path.")
        standardization_mode_norm = str(standardization_mode).lower()
        if standardization_mode_norm not in {"bandwise", "global"}:
            raise ValueError("standardization_mode must be 'bandwise' or 'global'.")
        self.standardize_modalities = standardize_modalities_norm

        self.global_spatial_aug = SpatialAugmentation(
            p=1.0,
            horizontal_flip=True,
            vertical_flip=True,
            random_crop=True,
            crop_scale=global_crop_scale,
            output_size=None,
            consistent=True,
        )
        self.local_spatial_aug = SpatialAugmentation(
            p=1.0,
            horizontal_flip=True,
            vertical_flip=True,
            random_crop=True,
            crop_scale=local_crop_scale,
            output_size=None,
            consistent=True,
        )
        for modality in self.modality_order:
            if bool(apply_input_normalization):
                self.normalizers[modality] = SingleSensorNormalizer(
                    sensor_config=SensorRegistry.get(modality)
                )
            if modality in self.standardize_modalities:
                self.standardizers[modality] = SingleSensorStandardizer.from_stats_file(
                    stats_path=str(standardization_stats_path),
                    sensor_name=modality,
                    mode=standardization_mode_norm,
                    eps=float(standardization_eps),
                )
            kernel_size = _odd_kernel_for_size(self.sensor_input_sizes[modality])
            self.global_blur_aug[modality] = K.AugmentationSequential(
                K.RandomGaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.1, 2.0), p=0.5),
                data_keys=["input"],
            )
            self.local_blur_aug[modality] = K.AugmentationSequential(
                K.RandomGaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.1, 2.0), p=0.5),
                data_keys=["input"],
            )

        self.radiometric_aug_p = float(radiometric_aug_p)
        self.radiometric_brightness_range = (
            float(radiometric_brightness_range[0]),
            float(radiometric_brightness_range[1]),
        )
        self.radiometric_bias_range = (
            float(radiometric_bias_range[0]),
            float(radiometric_bias_range[1]),
        )
        self.avg_output_std = 0.0

    @staticmethod
    def _normalize_modality_list(values: Optional[Sequence[str]]) -> List[str]:
        out: List[str] = []
        if values is None:
            return out
        for value in values:
            normalized = _normalize_sensor_name(value)
            if normalized not in out:
                out.append(normalized)
        return out

    def _get_available_modalities(self, batch: Dict[str, Any]) -> List[str]:
        available: List[str] = []
        for modality in self.modality_order:
            lower = modality.lower()
            if modality in batch and isinstance(batch[modality], Tensor) and batch[modality].ndim == 4:
                available.append(modality)
            elif lower in batch and isinstance(batch[lower], Tensor) and batch[lower].ndim == 4:
                available.append(modality)
        if not available:
            raise KeyError(f"No modality tensor found in batch. Keys={list(batch.keys())}")
        return available

    def _get_modality_tensor(self, batch: Dict[str, Any], modality: str) -> Tensor:
        return batch[modality] if modality in batch else batch[modality.lower()]

    def _prepare_modality_tensor(self, x: Tensor, modality: str) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
        x = x.float()
        expected_size = self.sensor_input_sizes[modality]
        current_size = tuple(int(v) for v in x.shape[-2:])
        if current_size != expected_size:
            raise ValueError(f"Modality '{modality}' must have size {expected_size}, got {current_size}.")
        return x

    @staticmethod
    def _sample_modalities(available_modalities: Sequence[str], k: int) -> List[str]:
        available = list(available_modalities)
        if not available:
            return []
        count = max(1, min(int(k), len(available)))
        perm = torch.randperm(len(available)).tolist()
        return [available[idx] for idx in perm[:count]]

    def _apply_radiometric_augmentation(self, x: Tensor) -> Tensor:
        if self.radiometric_aug_p <= 0.0 or random.random() > self.radiometric_aug_p:
            return x
        batch_size = x.shape[0]
        brightness = torch.empty(batch_size, 1, 1, 1, device=x.device).uniform_(
            self.radiometric_brightness_range[0],
            self.radiometric_brightness_range[1],
        )
        bias = torch.empty(batch_size, 1, 1, 1, device=x.device).uniform_(
            self.radiometric_bias_range[0],
            self.radiometric_bias_range[1],
        )
        return x * brightness + bias

    def _augment_view(self, tensors: Dict[str, Tensor], local: bool) -> Dict[str, Tensor]:
        view: Dict[str, Tensor] = {}
        for modality, x in tensors.items():
            y = self.normalizers[modality](x) if modality in self.normalizers else x
            y = self.standardizers[modality](y) if modality in self.standardizers else y
            view[modality] = y
        spatial_aug = self.local_spatial_aug if local else self.global_spatial_aug
        view = spatial_aug(view)

        out: Dict[str, Tensor] = {}
        for modality, x in view.items():
            blur_aug = self.local_blur_aug[modality] if local else self.global_blur_aug[modality]
            out[modality] = self._apply_radiometric_augmentation(blur_aug(x))
        return out

    def _build_views(
        self,
        modality_tensors: Dict[str, Tensor],
    ) -> Tuple[List[Dict[str, Tensor]], List[Dict[str, Tensor]], List[List[str]], List[List[str]]]:
        available_modalities = list(modality_tensors.keys())
        global_views: List[Dict[str, Tensor]] = []
        global_modalities: List[List[str]] = []
        local_views: List[Dict[str, Tensor]] = []
        local_modalities: List[List[str]] = []

        with torch.no_grad():
            for _ in range(int(self.hparams.global_views)):
                selected = self._sample_modalities(
                    available_modalities,
                    self.hparams.global_modalities_per_view,
                )
                global_views.append(self._augment_view({m: modality_tensors[m] for m in selected}, local=False))
                global_modalities.append(selected)

            local_source_modalities = list(available_modalities)
            if bool(self.hparams.local_modalities_from_global_only):
                local_source_modalities = sorted({m for mods in global_modalities for m in mods})
                if not local_source_modalities:
                    local_source_modalities = list(available_modalities)
            if self.local_excluded_modalities:
                local_source_modalities = [
                    m for m in local_source_modalities if _normalize_sensor_name(m) not in self.local_excluded_modalities
                ]
                if not local_source_modalities:
                    raise RuntimeError("No modalities remain for local views.")

            for _ in range(int(self.hparams.n_local_views)):
                selected = self._sample_modalities(
                    local_source_modalities,
                    self.hparams.local_modalities_per_view,
                )
                local_views.append(self._augment_view({m: modality_tensors[m] for m in selected}, local=True))
                local_modalities.append(list(selected))

        return global_views, local_views, global_modalities, local_modalities

    def _forward_backbone_student(self, modalities: Dict[str, Tensor]) -> Tensor:
        feat = self.student_backbone(modalities)
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)
        return feat.flatten(start_dim=1)

    def _forward_backbone_teacher(self, modalities: Dict[str, Tensor]) -> Tensor:
        if self.teacher_backbone is None:
            raise RuntimeError("teacher_backbone is not initialized.")
        feat = self.teacher_backbone(modalities)
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)
        return feat.flatten(start_dim=1)

    def _compute_momentum(self) -> float:
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return float(self.hparams.teacher_momentum_start)
        max_steps = int(getattr(trainer, "estimated_stepping_batches", 0))
        if max_steps <= 0:
            return float(self.hparams.teacher_momentum_start)
        return float(
            cosine_schedule(
                step=trainer.global_step,
                max_steps=max_steps,
                start_value=self.hparams.teacher_momentum_start,
                end_value=self.hparams.teacher_momentum_end,
            )
        )

    def _log_optimizer_state_metrics(self, sync_dist: bool = False) -> None:
        trainer = getattr(self, "_trainer", None)
        optimizers = getattr(trainer, "optimizers", None) if trainer is not None else None
        if not optimizers or not getattr(optimizers[0], "param_groups", None):
            return
        group0 = optimizers[0].param_groups[0]
        self.log("train_lr", float(group0.get("lr", 0.0)), on_step=True, on_epoch=False, sync_dist=sync_dist)
        if "weight_decay" in group0:
            self.log(
                "train_weight_decay",
                float(group0.get("weight_decay", 0.0)),
                on_step=True,
                on_epoch=False,
                sync_dist=sync_dist,
            )

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[SequentialLR]]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        warmup_epochs = int(self.hparams.warmup_epochs)
        max_epochs = int(self.hparams.max_epochs)
        if warmup_epochs <= 0:
            scheduler = SequentialLR(
                optimizer,
                schedulers=[CosineAnnealingLR(optimizer, T_max=max_epochs)],
                milestones=[],
            )
            return [optimizer], [scheduler]

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=max(1.0 / max(warmup_epochs, 1), 1e-4),
                    total_iters=warmup_epochs,
                ),
                CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs)),
            ],
            milestones=[warmup_epochs],
        )
        return [optimizer], [scheduler]
