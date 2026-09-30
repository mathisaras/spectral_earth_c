"""
Standardization transforms for Spectral Earth MM.

These transforms are intended to be applied on top of existing physical
normalization (MMNormalizer / SingleSensorNormalizer).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf


def _to_plain_dict(path: Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    out = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(out, dict):
        raise ValueError(f"Expected YAML object in {path}, got {type(out).__name__}")
    return out


def _resolve_sensor_stats_file(stats_path: str, sensor_name: str) -> Path:
    """
    Resolve a sensor stats file from:
    1) legacy directory containing `mu.npy` and `sigma.npy`
    2) sensor yaml file directly
    3) index.yaml with `sensors: {SENSOR: /path/to/file.yaml}`
    4) directory containing `<sensor>.yaml`
    """
    sensor_upper = str(sensor_name).upper()
    sensor_lower = sensor_upper.lower()
    path = Path(stats_path)

    if path.is_dir() and (path / "mu.npy").exists() and (path / "sigma.npy").exists():
        return path

    if path.is_dir():
        candidate = path / f"{sensor_lower}.yaml"
        if not candidate.exists():
            raise FileNotFoundError(f"Stats file not found for sensor '{sensor_upper}' in directory '{path}'")
        return candidate

    if not path.exists():
        raise FileNotFoundError(f"Stats path does not exist: {path}")

    payload = _to_plain_dict(path)

    # Direct sensor file.
    if "stats" in payload and ("sensor" in payload or "bandwise" in payload.get("stats", {})):
        return path

    # Index file.
    sensors_map = payload.get("sensors")
    if isinstance(sensors_map, dict):
        raw_ref = sensors_map.get(sensor_upper) or sensors_map.get(sensor_lower)
        if raw_ref is None:
            raise KeyError(
                f"Sensor '{sensor_upper}' not found in stats index '{path}'. "
                f"Available: {sorted(list(sensors_map.keys()))}"
            )
        ref_path = Path(str(raw_ref))
        if not ref_path.is_absolute():
            ref_path = path.parent / ref_path
        if ref_path.exists():
            return ref_path

        # Fallback to sibling sensor YAML if index stores stale absolute paths.
        sibling = path.parent / f"{sensor_lower}.yaml"
        if sibling.exists():
            return sibling
        raise FileNotFoundError(
            f"Stats reference for '{sensor_upper}' in '{path}' points to missing file '{ref_path}'. "
            f"Tried fallback '{sibling}' but not found."
        )

    raise ValueError(
        f"Could not interpret stats path '{path}'. Expected sensor stats yaml, index yaml, or directory."
    )


def _extract_stats(stats_file: Path, sensor_name: str, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if stats_file.is_dir():
        mu_path = stats_file / "mu.npy"
        sigma_path = stats_file / "sigma.npy"
        if not mu_path.exists() or not sigma_path.exists():
            raise FileNotFoundError(
                f"Legacy stats directory '{stats_file}' must contain mu.npy and sigma.npy."
            )
        mean = torch.from_numpy(np.load(mu_path)).float()
        std = torch.from_numpy(np.load(sigma_path)).float()
        return mean, std

    payload = _to_plain_dict(stats_file)
    sensor_upper = str(sensor_name).upper()
    file_sensor = str(payload.get("sensor", sensor_upper)).upper()
    if file_sensor != sensor_upper:
        warnings.warn(
            f"Stats file '{stats_file}' is for sensor '{file_sensor}', "
            f"but '{sensor_upper}' was requested. Proceeding anyway."
        )

    stats = payload.get("stats", {})
    if not isinstance(stats, dict):
        raise ValueError(f"Invalid stats format in {stats_file}: missing 'stats' map.")

    mode_l = str(mode).lower()
    if mode_l == "global":
        g = stats.get("global", {})
        mean = torch.tensor(float(g["mean"]), dtype=torch.float32)
        std = torch.tensor(float(g["std"]), dtype=torch.float32)
        return mean, std

    if mode_l == "bandwise":
        bw = stats.get("bandwise", {})
        mean = torch.tensor(bw["mean"], dtype=torch.float32)
        std = torch.tensor(bw["std"], dtype=torch.float32)
        return mean, std

    raise ValueError(f"Unsupported standardization mode '{mode}'. Use 'global' or 'bandwise'.")


def _maybe_slice_raw_stats_to_processed_view(
    mean: torch.Tensor,
    std: torch.Tensor,
    sensor_config: Optional[Dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt legacy raw-band stats to the processed band view used downstream."""

    if sensor_config is None or mean.ndim != 1 or std.ndim != 1:
        return mean, std

    num_bands = int(sensor_config.get("num_bands", -1))
    num_bands_raw = int(sensor_config.get("num_bands_raw", num_bands))
    if num_bands <= 0 or num_bands_raw <= 0 or num_bands == num_bands_raw:
        return mean, std

    if mean.numel() == num_bands:
        return mean, std

    if mean.numel() != num_bands_raw or std.numel() != num_bands_raw:
        return mean, std

    from src.utils.spectral_metadata import derive_processed_raw_indices

    keep_indices = torch.from_numpy(derive_processed_raw_indices(sensor_config)).long()
    if keep_indices.numel() != num_bands:
        return mean, std
    return mean[keep_indices], std[keep_indices]


class SingleSensorStandardizer(nn.Module):
    """
    Standardize a single-sensor tensor using precomputed mean/std statistics.

    Supports:
    - global stats: scalar mean/std
    - bandwise stats: per-channel mean/std
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        eps: float = 1e-6,
        sensor_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.eps = float(eps)
        mean, std = _maybe_slice_raw_stats_to_processed_view(
            mean=mean,
            std=std,
            sensor_config=sensor_config,
        )
        self.register_buffer("mean", mean.clone().detach().float())
        self.register_buffer("std", std.clone().detach().float())

    @classmethod
    def from_stats_file(
        cls,
        stats_path: str,
        sensor_name: str,
        mode: str = "bandwise",
        eps: float = 1e-6,
        sensor_config: Optional[Dict[str, Any]] = None,
    ) -> "SingleSensorStandardizer":
        stats_file = _resolve_sensor_stats_file(stats_path=stats_path, sensor_name=sensor_name)
        mean, std = _extract_stats(stats_file=stats_file, sensor_name=sensor_name, mode=mode)
        return cls(mean=mean, std=std, eps=eps, sensor_config=sensor_config)

    def _reshape_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        std = torch.clamp(std, min=self.eps)

        if x.ndim == 4:
            # [B, C, H, W]
            if mean.ndim == 0:
                return mean, std
            if mean.ndim != 1 or mean.numel() != x.shape[1]:
                raise ValueError(
                    f"Bandwise stats channel mismatch: stats C={mean.numel()} vs tensor C={x.shape[1]}"
                )
            return mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)

        if x.ndim == 5:
            # [B, T, C, H, W]
            if mean.ndim == 0:
                return mean, std
            if mean.ndim != 1 or mean.numel() != x.shape[2]:
                raise ValueError(
                    f"Bandwise stats channel mismatch: stats C={mean.numel()} vs tensor C={x.shape[2]}"
                )
            return mean.view(1, 1, -1, 1, 1), std.view(1, 1, -1, 1, 1)

        raise ValueError(f"SingleSensorStandardizer expects 4D/5D tensor, got shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            x = x.float()
        mean, std = self._reshape_stats(x)
        return (x - mean) / std


class MMBatchStandardizer(nn.Module):
    """
    Standardize sensor tensors inside a batch dict.

    The standardizer for a sensor is lazily loaded from stats YAML on first use.
    """

    METADATA_KEYS = {"patch_id", "_group_name", "_available_sensors"}
    IMAGE_ALIAS_KEYS = {"image", "image1", "image2"}
    IMAGE_ALIAS_KEYS_UPPER = {"IMAGE", "IMAGE1", "IMAGE2"}

    def __init__(
        self,
        stats_path: str,
        mode: str = "bandwise",
        eps: float = 1e-6,
        default_sensor_name: Optional[str] = None,
        include_sensors: Optional[list[str]] = None,
    ):
        super().__init__()
        self.stats_path = str(stats_path)
        self.mode = str(mode).lower()
        self.eps = float(eps)
        self.default_sensor_name = default_sensor_name.upper() if default_sensor_name else None
        include_sensors_norm = (
            {str(sensor).upper() for sensor in include_sensors}
            if include_sensors is not None
            else None
        )
        self.include_sensors = include_sensors_norm if include_sensors_norm else None
        self.standardizers = nn.ModuleDict()
        self._warned_missing = set()

        if self.mode not in {"global", "bandwise"}:
            raise ValueError(f"Unsupported standardization mode '{mode}'. Use 'global' or 'bandwise'.")

    def _resolve_sensor_name(self, key: str) -> Optional[str]:
        k = str(key)
        if k.upper() in self.IMAGE_ALIAS_KEYS_UPPER:
            return self.default_sensor_name
        return k.upper()

    def _get_standardizer(self, sensor_name: str) -> SingleSensorStandardizer:
        if sensor_name not in self.standardizers:
            self.standardizers[sensor_name] = SingleSensorStandardizer.from_stats_file(
                stats_path=self.stats_path,
                sensor_name=sensor_name,
                mode=self.mode,
                eps=self.eps,
            )
        return self.standardizers[sensor_name]

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        for key in list(batch_dict.keys()):
            if key in self.METADATA_KEYS or "_mask" in key or key.startswith("_"):
                continue
            value = batch_dict[key]
            if not isinstance(value, torch.Tensor):
                continue

            sensor_name = self._resolve_sensor_name(key)
            if sensor_name is None:
                if key not in self._warned_missing:
                    self._warned_missing.add(key)
                    warnings.warn(
                        f"MMBatchStandardizer: key '{key}' has no sensor mapping. "
                        "Set default_sensor_name to apply standardization to image/image1/image2 keys."
                    )
                continue

            if (
                self.include_sensors is not None
                and sensor_name not in self.include_sensors
            ):
                continue

            try:
                standardizer = self._get_standardizer(sensor_name)
            except Exception as exc:
                if sensor_name not in self._warned_missing:
                    self._warned_missing.add(sensor_name)
                    warnings.warn(
                        f"MMBatchStandardizer: failed to load stats for sensor '{sensor_name}' "
                        f"from '{self.stats_path}'. Passing through unstandardized. Error: {exc}"
                    )
                continue

            batch_dict[key] = standardizer(value)

        return batch_dict
