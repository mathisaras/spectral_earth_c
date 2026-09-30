"""AMMIS/H2SR visible-bias correction transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.transforms.standardize_mm import SingleSensorStandardizer
from src.utils.sensor_registry import SensorRegistry
from src.utils.spectral_metadata import load_spectral_metadata, resolve_repo_relative_path


def _to_plain_dict(cfg: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if cfg is None:
        return None
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _resolve_existing_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.exists() or path.is_absolute():
        return path
    repo_path = resolve_repo_relative_path(path)
    if repo_path.exists():
        return repo_path
    return path


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=True)
    if window % 2 == 0:
        window += 1
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window <= 1:
        return values.astype(np.float64, copy=True)

    pad = window // 2
    padded = np.pad(values.astype(np.float64), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


class H2SRAmmisToDesisVisibleCorrection(nn.Module):
    """
    Correct the AMMIS/H2SR blue-green excess after projection to DESIS.

    The expected input is a raw reflectance-like tensor already normalized by the
    AMMIS physical normalizer and spectrally resampled to DESIS bands. The
    correction is estimated in DESIS z-space, converted back to raw reflectance
    with DESIS bandwise std, and applied before any optional DESIS
    standardization.

    ``mode="aggregate_mean_visible_smooth_alpha_1"`` implements the correction
    discussed in the H2SR inspection notebook:

    1. estimate AMMIS->DESIS aggregate mean z-shift vs DESIS pretraining stats,
    2. keep only positive shifts up to 650 nm,
    3. smooth the curve along wavelength,
    4. subtract ``alpha * bias_z * desis_std`` from the projected cube,
    5. optionally set DESIS edge bands outside [430, 950] nm to the DESIS mean.
    """

    def __init__(
        self,
        *,
        target_sensor_config: Optional[Mapping[str, Any]] = None,
        sensor_name: str = "DESIS",
        stats_path: str = "data/stats/mm_full_flat_normalized_random_single",
        stats_mode: str = "bandwise",
        stats_eps: float = 1.0e-6,
        reference_npz_path: str = "tmp/h2sr_diagnostics_64/sampled_curves.npz",
        mode: str = "aggregate_mean_visible_smooth_alpha_1",
        alpha: float = 1.0,
        smooth_window: int = 11,
        visible_max_nm: float = 650.0,
        clip_max_z: float = 1.35,
        positive_only: bool = True,
        neutralize_edges: bool = True,
        neutralize_min_nm: float = 430.0,
        neutralize_max_nm: float = 950.0,
        clip_output: bool = True,
        output_min: float = 0.0,
        output_max: float = 1.0,
        bias_z: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if mode != "aggregate_mean_visible_smooth_alpha_1":
            raise ValueError(
                "H2SRAmmisToDesisVisibleCorrection currently implements only "
                f"mode='aggregate_mean_visible_smooth_alpha_1', got {mode!r}."
            )
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")

        target_sensor_config = _to_plain_dict(target_sensor_config)
        if target_sensor_config is None:
            target_sensor_config = SensorRegistry.get(sensor_name)
        sensor_name = str(target_sensor_config.get("name", sensor_name)).upper()

        stats_path_resolved = _resolve_existing_path(stats_path)
        standardizer = SingleSensorStandardizer.from_stats_file(
            stats_path=str(stats_path_resolved),
            sensor_name=sensor_name,
            mode=stats_mode,
            eps=stats_eps,
            sensor_config=target_sensor_config,
        )
        target_mean = standardizer.mean.detach().float()
        target_std = torch.clamp(standardizer.std.detach().float(), min=float(stats_eps))
        if target_mean.ndim != 1 or target_std.ndim != 1:
            raise ValueError("H2SR visible correction requires bandwise target stats.")

        metadata = load_spectral_metadata(target_sensor_config, view="processed")
        wavelengths_nm = np.asarray(metadata.band_centers_nm, dtype=np.float64)
        if wavelengths_nm.shape[0] != target_mean.numel():
            raise ValueError(
                f"DESIS metadata/stat mismatch: wavelengths={wavelengths_nm.shape[0]} "
                f"stats={target_mean.numel()}."
            )

        if bias_z is None:
            bias_source = self._load_aggregate_z_shift(
                reference_npz_path=reference_npz_path,
                target_mean=target_mean.numpy(),
                target_std=target_std.numpy(),
            )
        else:
            bias_source = np.asarray(list(bias_z), dtype=np.float64)

        if bias_source.shape[0] != target_mean.numel():
            raise ValueError(
                f"Correction bias has {bias_source.shape[0]} bands, "
                f"expected {target_mean.numel()} for {sensor_name}."
            )

        visible = wavelengths_nm <= float(visible_max_nm)
        bias = np.where(visible, bias_source, 0.0)
        if positive_only:
            bias = np.maximum(bias, 0.0)
        bias = _smooth_1d(bias, int(smooth_window))
        bias = np.clip(bias, 0.0, float(clip_max_z))

        edge_mask = (
            (wavelengths_nm < float(neutralize_min_nm))
            | (wavelengths_nm > float(neutralize_max_nm))
        )

        self.alpha = float(alpha)
        self.neutralize_edges = bool(neutralize_edges)
        self.clip_output = bool(clip_output)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.register_buffer("bias_z", torch.tensor(bias, dtype=torch.float32))
        self.register_buffer("edge_mask", torch.tensor(edge_mask, dtype=torch.bool))
        self.register_buffer(
            "wavelengths_nm", torch.tensor(wavelengths_nm, dtype=torch.float32)
        )

    @staticmethod
    def _load_aggregate_z_shift(
        *,
        reference_npz_path: str,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> np.ndarray:
        npz_path = _resolve_existing_path(reference_npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                "H2SR solution-6 correction needs a compatible sampled_curves.npz. "
                f"Set reference_npz_path to an existing file; got '{reference_npz_path}'."
            )

        payload = np.load(npz_path)
        if "projected_z_shift" in payload:
            return np.asarray(payload["projected_z_shift"], dtype=np.float64)
        if "h2sr_projected_desis_band_mean" in payload:
            projected_mean = np.asarray(
                payload["h2sr_projected_desis_band_mean"], dtype=np.float64
            )
            return (projected_mean - target_mean.astype(np.float64)) / np.maximum(
                target_std.astype(np.float64), 1.0e-8
            )
        raise KeyError(
            f"{npz_path} must contain either 'projected_z_shift' or "
            "'h2sr_projected_desis_band_mean'."
        )

    def _reshape_band_vector(self, vector: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        vector = vector.to(device=x.device, dtype=x.dtype)
        if x.ndim == 4:
            if x.shape[1] != vector.numel():
                raise ValueError(
                    f"H2SR correction expected {vector.numel()} channels, got {x.shape[1]}."
                )
            return vector.view(1, -1, 1, 1)
        if x.ndim == 5:
            if x.shape[2] != vector.numel():
                raise ValueError(
                    f"H2SR correction expected {vector.numel()} channels, got {x.shape[2]}."
                )
            return vector.view(1, 1, -1, 1, 1)
        raise ValueError(
            f"H2SR correction expects [B,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.alpha * self._reshape_band_vector(self.bias_z * self.target_std, x)
        corrected = x - delta

        if self.neutralize_edges and bool(self.edge_mask.any()):
            mean = self._reshape_band_vector(self.target_mean, x)
            mask = self._reshape_band_vector(self.edge_mask, x).bool()
            corrected = torch.where(mask, mean, corrected)

        if self.clip_output:
            corrected = torch.clamp(corrected, min=self.output_min, max=self.output_max)
        return corrected
