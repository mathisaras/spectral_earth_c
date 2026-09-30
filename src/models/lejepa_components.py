from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class LeJEPAProjectionHead(nn.Module):
    """MLP projection head used by the LeJEPA objective."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 3,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        if num_layers == 1:
            self.net = nn.Linear(input_dim, output_dim)
            return

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for idx in range(len(dims) - 1):
            is_last = idx == len(dims) - 2
            layers.append(nn.Linear(dims[idx], dims[idx + 1], bias=(not use_bn or is_last)))
            if not is_last:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dims[idx + 1]))
                layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


LEJEPA_INVARIANCE_MODES = ("coupled_center", "detached_global_center")


def compute_lejepa_invariance_loss(
    proj: Tensor,
    *,
    n_global: int,
    invariance_mode: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    if proj.ndim != 3:
        raise ValueError(f"Expected proj [V, B, D], got {tuple(proj.shape)}")
    if not (1 <= int(n_global) <= int(proj.shape[0])):
        raise ValueError(
            f"n_global must be in [1, {int(proj.shape[0])}], got {n_global}."
        )

    mode = str(invariance_mode).lower()
    if mode not in LEJEPA_INVARIANCE_MODES:
        raise ValueError(
            f"invariance_mode must be one of {LEJEPA_INVARIANCE_MODES}, got {invariance_mode}."
        )

    global_proj = proj[:n_global]
    local_proj = proj[n_global:]
    center = global_proj.mean(dim=0, keepdim=True)
    zero = global_proj.new_zeros(())

    if mode == "coupled_center":
        inv_total = (center - proj).square().mean()
        inv_global = (center - global_proj).square().mean()
        inv_local = (center - local_proj).square().mean() if local_proj.numel() > 0 else zero
        return inv_total, inv_global, inv_local

    detached_center = center.detach()
    inv_global = (detached_center - global_proj).square().mean()
    inv_local = (detached_center - local_proj).square().mean() if local_proj.numel() > 0 else zero
    inv_total = inv_global + inv_local
    return inv_total, inv_global, inv_local
