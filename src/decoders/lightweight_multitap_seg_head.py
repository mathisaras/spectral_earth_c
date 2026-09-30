"""Lightweight shared multi-tap segmentation decoder."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _valid_group_count(channels: int, requested_groups: int) -> int:
    """Return the largest valid GroupNorm divisor not exceeding the request."""

    channels = int(channels)
    requested_groups = int(requested_groups)
    if channels <= 0:
        raise ValueError(f"channels must be > 0, got {channels}")
    if requested_groups <= 0:
        return 1

    for groups in range(min(requested_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class LightweightMultiTapSegHead(nn.Module):
    """SegFormer/Lite-FPN style decoder shared by ViTs and Hiera-like models.

    Fairness principle: Hiera-style backbones are allowed to expose native
    pyramid stages because hierarchy is part of the backbone.  Vanilla ViTs get
    multiple intermediate block taps instead of only the final block.  This head
    deliberately stays lightweight so the benchmark remains backbone-dominated,
    rather than becoming a comparison of bespoke heavy decoders.
    """

    def __init__(
        self,
        in_channels_list: Sequence[int],
        decoder_dim: int = 128,
        num_classes: int = 1,
        fuse: str = "concat",
        num_fusion_convs: int = 1,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()

        if not in_channels_list:
            raise ValueError("in_channels_list must contain at least one tap.")
        if int(decoder_dim) <= 0:
            raise ValueError(f"decoder_dim must be > 0, got {decoder_dim}")
        if int(num_classes) <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")
        if fuse not in {"concat", "sum"}:
            raise ValueError(f"fuse must be 'concat' or 'sum', got {fuse!r}")
        if int(num_fusion_convs) not in {1, 2}:
            raise ValueError(
                f"num_fusion_convs must be 1 or 2, got {num_fusion_convs}"
            )

        self.in_channels_list = [int(c) for c in in_channels_list]
        if any(c <= 0 for c in self.in_channels_list):
            raise ValueError(
                f"All in_channels_list entries must be > 0, got {self.in_channels_list}"
            )
        self.decoder_dim = int(decoder_dim)
        self.num_classes = int(num_classes)
        self.fuse = str(fuse)
        self.num_fusion_convs = int(num_fusion_convs)
        self.norm_groups = int(norm_groups)

        self.projections = nn.ModuleList(
            [
                nn.Conv2d(in_channels, self.decoder_dim, kernel_size=1)
                for in_channels in self.in_channels_list
            ]
        )

        fusion_in_channels = (
            self.decoder_dim * len(self.in_channels_list)
            if self.fuse == "concat"
            else self.decoder_dim
        )

        fusion_layers: list[nn.Module] = []
        for block_idx in range(self.num_fusion_convs):
            in_channels = fusion_in_channels if block_idx == 0 else self.decoder_dim
            fusion_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        self.decoder_dim,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(
                        _valid_group_count(self.decoder_dim, self.norm_groups),
                        self.decoder_dim,
                    ),
                    nn.ReLU(inplace=True),
                ]
            )
        self.fusion = nn.Sequential(*fusion_layers)
        self.classifier = nn.Conv2d(self.decoder_dim, self.num_classes, kernel_size=1)

    def forward(
        self,
        features: Sequence[Tensor],
        output_size: tuple[int, int] | torch.Size | None = None,
    ) -> Tensor:
        if len(features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} feature taps, got {len(features)}."
            )

        projected: list[Tensor] = []
        target_h = max(int(feature.shape[-2]) for feature in features)
        target_w = max(int(feature.shape[-1]) for feature in features)
        target_size = (target_h, target_w)

        for feature, projection, expected_channels in zip(
            features, self.projections, self.in_channels_list
        ):
            if feature.ndim != 4:
                raise ValueError(
                    "LightweightMultiTapSegHead expects 4D feature maps "
                    f"[B, C, H, W], got shape {tuple(feature.shape)}."
                )
            if int(feature.shape[1]) != expected_channels:
                raise ValueError(
                    f"Feature channel mismatch: expected {expected_channels}, "
                    f"got {int(feature.shape[1])}."
                )

            y = projection(feature)
            if y.shape[-2:] != target_size:
                y = F.interpolate(
                    y, size=target_size, mode="bilinear", align_corners=False
                )
            projected.append(y)

        if self.fuse == "concat":
            fused = torch.cat(projected, dim=1)
        else:
            fused = torch.stack(projected, dim=0).sum(dim=0)

        logits = self.classifier(self.fusion(fused))
        if output_size is not None:
            logits = F.interpolate(
                logits,
                size=tuple(int(v) for v in output_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits

