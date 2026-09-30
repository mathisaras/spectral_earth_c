#!/usr/bin/env python3
"""
Forward/backward check for LeJEPAModule.

Validates:
1) Hydra-style dynamic backbone instantiation
2) LeJEPA forward/loss/backward
3) Batch-key compatibility:
   - sensor-key batch (e.g. {"ENMAP": tensor})
   - legacy {"image": tensor}
   - temporal-view {"image1": tensor, "image2": tensor}
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.lejepa_module import LeJEPAModule


def build_module(device: torch.device) -> LeJEPAModule:
    backbone_config = OmegaConf.create(
        {
            "_target_": "timm.create_model",
            "model_name": "vit_tiny_patch16_224",
            "pretrained": False,
            "num_classes": 0,
            "img_size": 64,
            "in_chans": 8,
            "dynamic_img_size": True,
        }
    )

    sensor_config = OmegaConf.create(
        {
            "name": "enmap",
            "num_bands": 8,
            "img_size": 64,
            "rgb_indices": [0, 1, 2],
            "normalization": {"type": "scale", "scale": 10000.0},
        }
    )

    module = LeJEPAModule(
        backbone_config=backbone_config,
        sensor_config=sensor_config,
        output_dim=64,
        projector_hidden_dim=256,
        projector_num_layers=2,
        lamb=0.5,
        lr=1.0e-4,
        warmup_epochs=2,
        max_epochs=5,
        weight_decay=1.0e-5,
        multicrop=True,
        n_views=2,
        local_resize_to_global=True,
        sigreg_num_vectors=64,
    ).to(device)
    module.train()

    # Lightning's self.log expects a Trainer context in this small standalone check.
    def _noop_log(self, *args, **kwargs):
        return None

    def _no_view_logging(self, batch_idx: int) -> bool:
        return False

    module.log = MethodType(_noop_log, module)
    module._should_log_view_viz = MethodType(_no_view_logging, module)
    module._trainer = SimpleNamespace(max_epochs=5)
    return module


def run_case(module: LeJEPAModule, batch: dict, case_name: str) -> None:
    optimizer = torch.optim.AdamW(module.parameters(), lr=1.0e-4)
    optimizer.zero_grad(set_to_none=True)

    loss = module.training_step(batch, batch_idx=0)
    assert torch.is_tensor(loss), f"{case_name}: loss is not a tensor."
    assert loss.ndim == 0, f"{case_name}: expected scalar loss, got shape {tuple(loss.shape)}."
    assert torch.isfinite(loss), f"{case_name}: loss is not finite: {loss.item()}."

    loss.backward()

    grad_norm = 0.0
    for p in module.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.detach().abs().mean().item())
    assert grad_norm > 0.0, f"{case_name}: no gradients were produced."

    optimizer.step()
    print(f"[OK] {case_name}: loss={loss.item():.6f}, grad_norm={grad_norm:.6f}")


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running LeJEPA forward/backward check on device: {device}")

    module = build_module(device)

    B, C, H, W = 2, 8, 64, 64
    x = torch.rand(B, C, H, W, device=device) * 10000.0
    x2 = torch.rand(B, C, H, W, device=device) * 10000.0

    batch_sensor_key = {"ENMAP": x.clone()}
    batch_image_key = {"image": x.clone()}
    batch_temporal_views = {"image1": x.clone(), "image2": x2.clone()}

    run_case(module, batch_sensor_key, "sensor-key batch")
    run_case(module, batch_image_key, "image-key batch")
    run_case(module, batch_temporal_views, "image1/image2 batch")

    opts, scheds = module.configure_optimizers()
    assert len(opts) == 1, "configure_optimizers should return one optimizer."
    assert len(scheds) == 1, "configure_optimizers should return one scheduler."
    print("[OK] configure_optimizers wiring")

    print("LeJEPA forward/backward check passed.")


if __name__ == "__main__":
    main()
