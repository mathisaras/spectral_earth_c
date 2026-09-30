from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.models.utils import deactivate_requires_grad, update_momentum
from torch import Tensor

from src.losses.sigreg import SIGReg

from .lejepa_components import (
    LEJEPA_INVARIANCE_MODES,
    LeJEPAProjectionHead,
    compute_lejepa_invariance_loss,
)
from .multimodal_view_base import MultimodalViewBase


class MultimodalLeJEPAModule(MultimodalViewBase):
    """Multimodal LeJEPA pretraining module."""

    def __init__(
        self,
        backbone_config: Dict[str, Any],
        modality_channels: Optional[Dict[str, int]] = None,
        output_dim: int = 128,
        projector_hidden_dim: int = 2048,
        projector_num_layers: int = 3,
        projector_use_bn: bool = True,
        lamb: float = 0.05,
        invariance_mode: str = "coupled_center",
        ema_teacher_enabled: bool = False,
        teacher_momentum_start: float = 0.996,
        teacher_momentum_end: float = 1.0,
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
        global_crop_scale: Tuple[float, float] = (0.3, 1.0),
        local_crop_scale: Tuple[float, float] = (0.05, 0.3),
        radiometric_aug_p: float = 0.5,
        radiometric_brightness_range: Tuple[float, float] = (0.8, 1.2),
        radiometric_bias_range: Tuple[float, float] = (-0.1, 0.1),
        lr: float = 1.0e-4,
        weight_decay: float = 1.0e-5,
        warmup_epochs: int = 20,
        max_epochs: int = 100,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_num_vectors: int = 256,
        sigreg_gather_distributed: bool = False,
    ) -> None:
        if not (0.0 <= float(lamb) <= 1.0):
            raise ValueError(f"`lamb` must be in [0, 1], got {lamb}.")
        invariance_mode = str(invariance_mode).lower()
        if invariance_mode not in LEJEPA_INVARIANCE_MODES:
            raise ValueError(
                f"`invariance_mode` must be one of {LEJEPA_INVARIANCE_MODES}, got {invariance_mode}."
            )
        ema_teacher_enabled = bool(ema_teacher_enabled)

        super().__init__(
            backbone_config=backbone_config,
            modality_channels=modality_channels,
            global_views=global_views,
            global_modalities_per_view=global_modalities_per_view,
            n_local_views=n_local_views,
            local_modalities_per_view=local_modalities_per_view,
            local_modalities_from_global_only=local_modalities_from_global_only,
            local_excluded_modalities=local_excluded_modalities,
            apply_input_normalization=apply_input_normalization,
            standardize_modalities=standardize_modalities,
            standardization_mode=standardization_mode,
            standardization_stats_path=standardization_stats_path,
            standardization_eps=standardization_eps,
            global_crop_scale=global_crop_scale,
            local_crop_scale=local_crop_scale,
            radiometric_aug_p=radiometric_aug_p,
            radiometric_brightness_range=radiometric_brightness_range,
            radiometric_bias_range=radiometric_bias_range,
            lr=lr,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            teacher_momentum_start=teacher_momentum_start,
            teacher_momentum_end=teacher_momentum_end,
            create_teacher_backbone=ema_teacher_enabled,
        )
        self.save_hyperparameters(
            ignore=[
                "backbone_config",
                "radiometric_aug_p",
                "radiometric_brightness_range",
                "radiometric_bias_range",
                "standardization_mode",
                "standardization_stats_path",
                "standardization_eps",
            ]
        )

        self.projection_head = LeJEPAProjectionHead(
            input_dim=self.embed_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=output_dim,
            num_layers=projector_num_layers,
            use_bn=projector_use_bn,
        )
        self.sigreg = SIGReg(
            knots=sigreg_knots,
            t_max=sigreg_t_max,
            num_vectors=sigreg_num_vectors,
            gather_distributed=sigreg_gather_distributed,
        )
        self.lamb = float(lamb)
        self.invariance_mode = invariance_mode
        self.ema_teacher_enabled = ema_teacher_enabled
        self.teacher_projection_head: Optional[nn.Module] = None
        if self.ema_teacher_enabled:
            self.teacher_projection_head = copy.deepcopy(self.projection_head)
            deactivate_requires_grad(self.teacher_projection_head)
        self._pending_teacher_momentum: Optional[float] = None

    def forward(self, modalities: Dict[str, Tensor]) -> Tensor:
        feat = self._forward_backbone_student(modalities)
        return self.projection_head(feat)

    def _compute_teacher_center_loss(
        self,
        student_proj: Tensor,
        teacher_global_proj: Tensor,
        n_global: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        target_center = teacher_global_proj.mean(dim=0, keepdim=True).detach()
        inv_total = (target_center - student_proj).square().mean()
        inv_global = (target_center - student_proj[:n_global]).square().mean()
        if int(student_proj.shape[0]) > n_global:
            inv_local = (target_center - student_proj[n_global:]).square().mean()
        else:
            inv_local = inv_total.new_zeros(())
        return inv_total, inv_global, inv_local

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer
        if not self.ema_teacher_enabled:
            return
        if self.teacher_backbone is None or self.teacher_projection_head is None:
            raise RuntimeError(
                "EMA teacher is enabled but teacher backbone/projection head is missing."
            )
        momentum = (
            float(self._pending_teacher_momentum)
            if self._pending_teacher_momentum is not None
            else self._compute_momentum()
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.projection_head, self.teacher_projection_head, m=momentum)
        self._pending_teacher_momentum = None

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        del batch_idx

        available_modalities = self._get_available_modalities(batch)
        modality_tensors = {
            m: self._prepare_modality_tensor(self._get_modality_tensor(batch, m), modality=m)
            for m in available_modalities
        }
        global_views, local_views, global_modalities, local_modalities = self._build_views(
            modality_tensors=modality_tensors
        )
        all_views = global_views + local_views
        if not global_views:
            raise RuntimeError("MultimodalLeJEPAModule requires at least one global view.")

        features_per_view: List[Tensor] = []
        for view in all_views:
            features_per_view.append(self._forward_backbone_student(view))

        projections = [self.projection_head(feat) for feat in features_per_view]
        proj = torch.stack(projections, dim=0)  # [V, B, D]
        n_global = len(global_views)
        teacher_momentum: Optional[float] = None
        if self.ema_teacher_enabled:
            if self.teacher_backbone is None or self.teacher_projection_head is None:
                raise RuntimeError(
                    "EMA teacher is enabled but teacher backbone/projection head is missing."
                )
            with torch.no_grad():
                teacher_global_proj = torch.stack(
                    [
                        self.teacher_projection_head(self._forward_backbone_teacher(view))
                        for view in global_views
                    ],
                    dim=0,
                )
            inv_loss, inv_loss_global, inv_loss_local = self._compute_teacher_center_loss(
                student_proj=proj,
                teacher_global_proj=teacher_global_proj,
                n_global=n_global,
            )
            teacher_momentum = self._compute_momentum()
            self._pending_teacher_momentum = teacher_momentum
        else:
            inv_loss, inv_loss_global, inv_loss_local = compute_lejepa_invariance_loss(
                proj,
                n_global=n_global,
                invariance_mode=self.invariance_mode,
            )
        sigreg_loss = self.sigreg(proj)

        loss = (self.lamb * sigreg_loss) + ((1.0 - self.lamb) * inv_loss)
        weighted_sigreg_loss = self.lamb * sigreg_loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss_inv", inv_loss, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            "train_loss_inv_global",
            inv_loss_global,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "train_loss_inv_local",
            inv_loss_local,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "train_loss_sigreg",
            sigreg_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "train_loss_sigreg_weighted",
            weighted_sigreg_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "train_global_modalities_mean",
            torch.tensor(
                [len(v) for v in global_modalities],
                dtype=torch.float32,
                device=loss.device,
            ).mean(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        if local_modalities:
            self.log(
                "train_local_modalities_mean",
                torch.tensor(
                    [len(v) for v in local_modalities],
                    dtype=torch.float32,
                    device=loss.device,
                ).mean(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        self.log("train_num_views", float(len(all_views)), on_step=True, on_epoch=False, sync_dist=True)
        self.log("train_num_global_views", float(n_global), on_step=True, on_epoch=False, sync_dist=True)
        if teacher_momentum is not None:
            self.log("teacher_momentum", teacher_momentum, on_step=True, on_epoch=False, sync_dist=True)
        self._log_optimizer_state_metrics(sync_dist=True)

        with torch.no_grad():
            ref_features = features_per_view[0]
            norm_features = F.normalize(ref_features, dim=1)
            output_std = torch.std(norm_features, dim=0).mean().item()
            self.avg_output_std = 0.9 * self.avg_output_std + 0.1 * output_std
            self.log("train_ssl_std", self.avg_output_std, on_step=True, on_epoch=False, sync_dist=True)

        return loss
