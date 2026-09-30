"""Project one source sensor image into one or more backbone sensor branches."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transforms.normalize_mm import SingleSensorNormalizer
from src.transforms.spectral_resample import SensorSpectralMapper
from src.transforms.standardize_mm import SingleSensorStandardizer


def _sensor_name(sensor_config: Mapping[str, Any]) -> str:
    name = str(sensor_config.get("name", "")).strip().upper()
    if not name:
        raise ValueError("Sensor config is missing required field `name`.")
    return name


def _sensor_num_bands(sensor_config: Mapping[str, Any]) -> int:
    num_bands = int(sensor_config.get("num_bands", -1))
    if num_bands <= 0:
        raise ValueError(
            f"Sensor '{_sensor_name(sensor_config)}' has invalid num_bands={num_bands}."
        )
    return num_bands


def _resolve_size(value: Any) -> tuple[int, int]:
    if value is None:
        raise ValueError("Cannot resolve a spatial size from None.")
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("Cannot resolve a spatial size from an empty sequence.")
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _normalize_sensor_list(values: Optional[Iterable[str]]) -> Optional[set[str]]:
    if values is None:
        return None
    return {str(value).strip().upper() for value in values if str(value).strip()}


class SensorBranchProjector(nn.Module):
    """
    Project a source image into several model sensor branches.

    The contract is intentionally aligned with the pretraining/downstream
    pipeline:

    raw/processed source values -> source physical normalization -> spectral
    mapping in reflectance-like space -> target branch spatial size -> optional
    target standardization.
    """

    def __init__(
        self,
        source_sensor_config: Mapping[str, Any],
        target_sensor_configs: Mapping[str, Mapping[str, Any]],
        *,
        apply_input_normalization: bool = True,
        standardize: bool = False,
        standardize_sensors: Optional[Iterable[str]] = None,
        standardization_stats_path: Optional[str] = None,
        standardization_mode: str = "bandwise",
        standardization_eps: float = 1.0e-6,
        spectral_mapping_method: str = "auto",
        spectral_max_gap_nm: float = 80.0,
        min_coverage: float = 1.0e-6,
        uncovered_policy: str = "target_mean",
        interpolation_mode: str = "bilinear",
        align_corners: Optional[bool] = False,
    ) -> None:
        super().__init__()
        if min_coverage < 0.0 or min_coverage > 1.0:
            raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage}.")

        self.source_sensor_config = dict(source_sensor_config)
        self.source_sensor = _sensor_name(self.source_sensor_config)
        self.target_sensor_configs = OrderedDict(
            (str(name).upper(), dict(cfg))
            for name, cfg in target_sensor_configs.items()
        )
        if not self.target_sensor_configs:
            raise ValueError("target_sensor_configs must not be empty.")

        self.apply_input_normalization = bool(apply_input_normalization)
        self.standardize = bool(standardize)
        self.standardization_stats_path = standardization_stats_path
        self.standardization_mode = str(standardization_mode).lower()
        self.standardization_eps = float(standardization_eps)
        self.spectral_mapping_method = str(spectral_mapping_method).lower()
        self.spectral_max_gap_nm = float(spectral_max_gap_nm)
        self.min_coverage = float(min_coverage)
        self.uncovered_policy = str(uncovered_policy).lower()
        self.interpolation_mode = str(interpolation_mode)
        self.align_corners = align_corners

        if self.uncovered_policy not in {"target_mean", "zero", "nan", "keep"}:
            raise ValueError(
                "uncovered_policy must be one of "
                "{'target_mean', 'zero', 'nan', 'keep'}."
            )

        self.normalizer = (
            SingleSensorNormalizer(sensor_config=self.source_sensor_config)
            if self.apply_input_normalization
            else nn.Identity()
        )

        self.mappers = nn.ModuleDict()
        self.standardizers = nn.ModuleDict()
        self.fill_stats = nn.ModuleDict()
        self._target_sizes: dict[str, tuple[int, int]] = {}
        self._target_num_bands: dict[str, int] = {}
        self._identity_targets: set[str] = set()

        standardize_include = _normalize_sensor_list(standardize_sensors)
        if self.standardize and not self.standardization_stats_path:
            raise ValueError(
                "standardize=True requires `standardization_stats_path` for "
                "multi-target sensor projection."
            )

        for target_name, target_cfg in self.target_sensor_configs.items():
            resolved_target_name = _sensor_name(target_cfg)
            if resolved_target_name != target_name:
                raise ValueError(
                    f"Target config key '{target_name}' does not match config name "
                    f"'{resolved_target_name}'."
                )

            self._target_sizes[target_name] = _resolve_size(target_cfg.get("img_size"))
            self._target_num_bands[target_name] = _sensor_num_bands(target_cfg)

            if (
                target_name == self.source_sensor
                and _sensor_num_bands(self.source_sensor_config) == self._target_num_bands[target_name]
            ):
                self._identity_targets.add(target_name)
            else:
                self.mappers[target_name] = SensorSpectralMapper.from_sensor_configs(
                    source_sensor_config=self.source_sensor_config,
                    target_sensor_config=target_cfg,
                    method=self.spectral_mapping_method,
                    max_gap_nm=self.spectral_max_gap_nm,
                )

            should_standardize = self.standardize and (
                standardize_include is None or target_name in standardize_include
            )
            if should_standardize:
                self.standardizers[target_name] = SingleSensorStandardizer.from_stats_file(
                    stats_path=str(self.standardization_stats_path),
                    sensor_name=target_name,
                    mode=self.standardization_mode,
                    eps=self.standardization_eps,
                    sensor_config=target_cfg,
                )
            elif (
                self.uncovered_policy == "target_mean"
                and self.standardization_stats_path
            ):
                try:
                    self.fill_stats[target_name] = SingleSensorStandardizer.from_stats_file(
                        stats_path=str(self.standardization_stats_path),
                        sensor_name=target_name,
                        mode=self.standardization_mode,
                        eps=self.standardization_eps,
                        sensor_config=target_cfg,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"SensorBranchProjector: could not load fill stats for "
                        f"{target_name} from '{self.standardization_stats_path}'. "
                        f"Uncovered target bands will fall back to 0. Error: {exc}"
                    )

    @property
    def target_sensors(self) -> list[str]:
        return list(self.target_sensor_configs.keys())

    def coverage_by_sensor(self) -> dict[str, torch.Tensor]:
        coverage: dict[str, torch.Tensor] = {}
        for target_name in self.target_sensors:
            if target_name in self._identity_targets:
                coverage[target_name] = torch.ones(self._target_num_bands[target_name])
            else:
                coverage[target_name] = self.mappers[target_name].coverage.detach().cpu()
        return coverage

    def _resize_to_target(self, x: torch.Tensor, target_name: str) -> torch.Tensor:
        target_size = self._target_sizes[target_name]
        if tuple(x.shape[-2:]) == target_size:
            return x
        return F.interpolate(
            x,
            size=target_size,
            mode=self.interpolation_mode,
            align_corners=self.align_corners if self.interpolation_mode != "nearest" else None,
        )

    def _target_fill_values(
        self,
        target_name: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_bands = self._target_num_bands[target_name]
        if self.uncovered_policy == "zero":
            return torch.zeros(num_bands, device=device, dtype=dtype)
        if self.uncovered_policy == "nan":
            return torch.full((num_bands,), float("nan"), device=device, dtype=dtype)
        if self.uncovered_policy == "target_mean":
            standardizer = (
                self.standardizers[target_name]
                if target_name in self.standardizers
                else self.fill_stats[target_name]
                if target_name in self.fill_stats
                else None
            )
            if standardizer is not None:
                mean = standardizer.mean.to(device=device, dtype=dtype)
                if mean.ndim == 0:
                    return mean.expand(num_bands)
                if mean.numel() == num_bands:
                    return mean
            warnings.warn(
                f"SensorBranchProjector: target_mean fill requested for {target_name}, "
                "but no compatible target standardizer is available. Filling with 0."
            )
        return torch.zeros(num_bands, device=device, dtype=dtype)

    def _fill_low_coverage(
        self,
        x: torch.Tensor,
        target_name: str,
    ) -> torch.Tensor:
        if self.uncovered_policy == "keep" or target_name in self._identity_targets:
            return x

        mapper = self.mappers[target_name] if target_name in self.mappers else None
        if mapper is None:
            return x
        low_coverage = mapper.low_coverage_mask(self.min_coverage).to(
            device=x.device,
            dtype=torch.bool,
        )
        if not bool(low_coverage.any()):
            return x

        fill_values = self._target_fill_values(
            target_name,
            device=x.device,
            dtype=x.dtype,
        )[low_coverage]
        y = x.clone()
        if y.ndim == 3:
            y[low_coverage, :, :] = fill_values.view(-1, 1, 1)
        elif y.ndim == 4:
            y[:, low_coverage, :, :] = fill_values.view(1, -1, 1, 1)
        elif y.ndim == 5:
            y[:, :, low_coverage, :, :] = fill_values.view(1, 1, -1, 1, 1)
        else:
            raise ValueError(
                f"Expected projected tensor with 3D/4D/5D shape, got {tuple(y.shape)}."
            )
        return y

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if not torch.is_floating_point(x):
            x = x.float()
        source_reflectance = self.normalizer(x)

        projected: OrderedDict[str, torch.Tensor] = OrderedDict()
        for target_name in self.target_sensors:
            if target_name in self._identity_targets:
                y = source_reflectance
            else:
                y = self.mappers[target_name](source_reflectance)
                y = self._fill_low_coverage(y, target_name)

            y = self._resize_to_target(y, target_name)
            standardizer = (
                self.standardizers[target_name]
                if target_name in self.standardizers
                else None
            )
            if standardizer is not None:
                y = standardizer(y)
            projected[target_name] = y

        return dict(projected)
