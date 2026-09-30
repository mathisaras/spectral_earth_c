"""Panopticon wrapper for comparison experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn

from src.utils.sensor_registry import SensorRegistry
from src.utils.spectral_metadata import load_spectral_metadata


class PanopticonBackbone(nn.Module):
    """Thin adapter around the Panopticon torch-hub model.

    HyBiomass evaluated Panopticon on ENMAP with channel ids in nanometers.  This
    wrapper keeps that convention while allowing other processed sensor configs
    to supply their own wavelength metadata.
    """

    def __init__(
        self,
        sensor: str = "enmap",
        sensor_config_name: Optional[str] = None,
        hub_repo: str = "Panopticon-FM/panopticon",
        hub_model: str = "panopticon_vitb14",
        hub_arch_model: str = "_panopticon_vitb14",
        checkpoint_path: Optional[Union[str, Path]] = None,
        img_size: int = 224,
        patch_size: int = 14,
        num_features: int = 768,
        output_layers: Optional[list[int]] = None,
        channel_id_mode: str = "floor_nm",
        trust_repo: Optional[bool] = None,
        **hub_kwargs: Any,
    ) -> None:
        super().__init__()
        self.sensor = str(sensor_config_name or sensor).lower()
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.num_features = int(num_features)
        self.output_layers = output_layers or [3, 5, 7, 11]
        self.channel_id_mode = str(channel_id_mode).lower()

        sensor_cfg = SensorRegistry.get(self.sensor)
        metadata = load_spectral_metadata(sensor_cfg, view="processed")
        self.in_chans = metadata.num_bands
        channel_ids = torch.tensor(metadata.band_centers_nm, dtype=torch.float32)
        if self.channel_id_mode in {"floor_nm", "int_nm", "official"}:
            # Panopticon's released dataset helpers cast gaussian.mu values to
            # int16, and the README examples use integer nm channel ids.
            channel_ids = torch.floor(channel_ids).to(torch.long)
        elif self.channel_id_mode == "round_nm":
            channel_ids = torch.round(channel_ids).to(torch.long)
        elif self.channel_id_mode == "float_nm":
            pass
        else:
            raise ValueError(
                "channel_id_mode must be one of: floor_nm, round_nm, float_nm"
            )
        self.register_buffer(
            "chn_ids_nm",
            channel_ids,
            persistent=False,
        )

        load_kwargs = dict(hub_kwargs)
        if trust_repo is not None:
            load_kwargs["trust_repo"] = trust_repo
        self.model = self._load_panopticon_model(
            hub_repo=hub_repo,
            hub_model=hub_model,
            hub_arch_model=hub_arch_model,
            checkpoint_path=checkpoint_path,
            load_kwargs=load_kwargs,
        )

    @staticmethod
    def _torch_load(path: Union[str, Path]):
        try:
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(str(path), map_location="cpu")

    def _load_panopticon_model(
        self,
        hub_repo: str,
        hub_model: str,
        hub_arch_model: str,
        checkpoint_path: Optional[Union[str, Path]],
        load_kwargs: dict[str, Any],
    ) -> nn.Module:
        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        hub_repo_path = Path(hub_repo).expanduser()
        hub_load_kwargs = dict(load_kwargs)
        if hub_repo_path.exists():
            hub_load_kwargs.pop("trust_repo", None)
            if ckpt_path is not None and ckpt_path.exists():
                model = torch.hub.load(
                    str(hub_repo_path),
                    hub_arch_model,
                    source="local",
                    **hub_load_kwargs,
                )
                state_dict = self._torch_load(ckpt_path)
                if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
                    state_dict = state_dict["model"]
                model.load_state_dict(state_dict, strict=True)
                return model

            return torch.hub.load(
                str(hub_repo_path),
                hub_model,
                source="local",
                **hub_load_kwargs,
            )

        if ckpt_path is not None and ckpt_path.exists():
            model = torch.hub.load(hub_repo, hub_arch_model, **load_kwargs)
            state_dict = self._torch_load(ckpt_path)
            if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=True)
            return model

        download_kwargs = dict(load_kwargs)
        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            download_kwargs["dir_to_save_ckpt_in"] = str(ckpt_path.parent)
        return torch.hub.load(hub_repo, hub_model, **download_kwargs)

    def _payload(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        chn_ids = self.chn_ids_nm.to(device=x.device)
        return {
            "imgs": x,
            "chn_ids": chn_ids.unsqueeze(0).repeat(x.shape[0], 1),
        }

    def _reshape_patch_tokens(self, tokens: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape Panopticon patch tokens to a dense feature map.

        Upstream Panopticon returns patch tokens with CLS/register tokens already
        removed. We keep the reshape here instead of asking upstream to reshape
        because the official model receives a dict input, while the inherited
        DINOv2 reshape path assumes a raw image tensor.
        """

        token_h = int(x.shape[-2]) // self.patch_size
        token_w = int(x.shape[-1]) // self.patch_size
        expected_tokens = token_h * token_w
        actual_tokens = int(tokens.shape[1])
        if actual_tokens != expected_tokens:
            raise ValueError(
                "Cannot reshape Panopticon patch tokens: expected "
                f"{expected_tokens} tokens from input size {tuple(x.shape[-2:])} "
                f"and patch_size={self.patch_size}, got {actual_tokens}."
            )
        return (
            tokens.reshape(tokens.shape[0], token_h, token_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        payload = self._payload(x)
        try:
            return self.model(payload)
        except TypeError:
            return self.model(x)

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = True,
    ):
        layers = self.output_layers if n is None else n
        payload = self._payload(x)
        try:
            outputs = self.model.get_intermediate_layers(
                payload,
                n=layers,
                reshape=False,
                return_class_token=return_prefix_tokens,
                norm=norm,
            )
        except TypeError:
            outputs = self.model.get_intermediate_layers(
                x,
                n=layers,
                reshape=False,
                return_class_token=return_prefix_tokens,
                norm=norm,
            )

        normalized = []
        for out in outputs:
            if isinstance(out, tuple):
                tokens, prefix = out
            else:
                tokens, prefix = out, None

            if reshape:
                tokens = self._reshape_patch_tokens(tokens, x)

            normalized.append((tokens, prefix) if return_prefix_tokens else tokens)

        return tuple(normalized)
