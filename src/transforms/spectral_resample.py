"""Deterministic wavelength-aware sensor-to-sensor spectral resampling."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.spectral_metadata import load_spectral_metadata


DEFAULT_LINEAR_GAP_FILL_MAX_GAP_NM = 80.0


def build_linear_resampling_matrix(
    source_wavelengths_nm: np.ndarray,
    target_wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """
    Build a linear-interpolation matrix from source wavelengths to target wavelengths.

    Each output band is expressed as a convex combination of at most two adjacent
    source bands. Targets outside the source support are clamped to the nearest
    edge band, which keeps the transform stable for partially overlapping sensors.
    """

    src = np.asarray(source_wavelengths_nm, dtype=np.float64)
    tgt = np.asarray(target_wavelengths_nm, dtype=np.float64)

    if src.ndim != 1 or tgt.ndim != 1:
        raise ValueError("Source and target wavelength arrays must be 1D.")
    if src.size == 0 or tgt.size == 0:
        raise ValueError("Source and target wavelength arrays must be non-empty.")
    if np.unique(src).shape[0] != src.shape[0]:
        raise ValueError("Source wavelengths must be unique.")
    if np.unique(tgt).shape[0] != tgt.shape[0]:
        raise ValueError("Target wavelengths must be unique.")

    source_sort = np.argsort(src)
    target_sort = np.argsort(tgt)
    src_sorted = src[source_sort]
    tgt_sorted = tgt[target_sort]

    if np.any(np.diff(src_sorted) <= 0):
        raise ValueError("Sorted source wavelengths must be strictly increasing.")
    if np.any(np.diff(tgt_sorted) <= 0):
        raise ValueError("Sorted target wavelengths must be strictly increasing.")

    weights_sorted = np.zeros((tgt_sorted.size, src_sorted.size), dtype=np.float64)
    insertion_indices = np.searchsorted(src_sorted, tgt_sorted, side="left")

    for row_idx, insert_idx in enumerate(insertion_indices.tolist()):
        target_wavelength = tgt_sorted[row_idx]

        if insert_idx <= 0:
            weights_sorted[row_idx, 0] = 1.0
            continue

        if insert_idx >= src_sorted.size:
            weights_sorted[row_idx, -1] = 1.0
            continue

        left_idx = insert_idx - 1
        right_idx = insert_idx
        left_wavelength = src_sorted[left_idx]
        right_wavelength = src_sorted[right_idx]

        if np.isclose(target_wavelength, right_wavelength):
            weights_sorted[row_idx, right_idx] = 1.0
            continue

        span = right_wavelength - left_wavelength
        if span <= 0:
            raise ValueError("Sorted source wavelengths must be strictly increasing.")

        alpha = (target_wavelength - left_wavelength) / span
        weights_sorted[row_idx, left_idx] = 1.0 - alpha
        weights_sorted[row_idx, right_idx] = alpha

    row_sums = weights_sorted.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise RuntimeError("Resampling rows must sum to 1.")

    weights = np.zeros((tgt.size, src.size), dtype=np.float64)
    weights[np.ix_(target_sort, source_sort)] = weights_sorted
    return weights


def build_linear_no_extrapolation_resampling_matrix(
    source_wavelengths_nm: np.ndarray,
    target_wavelengths_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a center-wavelength linear interpolation matrix without extrapolation.

    Targets outside the source wavelength support receive an all-zero row and
    coverage 0.0. Targets inside the support are expressed as a convex
    combination of the two adjacent source band centers and receive coverage 1.0.
    This is useful for MSI -> HSI diagnostics, where edge clamping would create
    fake spectral information outside the MSI support.
    """

    src = np.asarray(source_wavelengths_nm, dtype=np.float64)
    tgt = np.asarray(target_wavelengths_nm, dtype=np.float64)

    if src.ndim != 1 or tgt.ndim != 1:
        raise ValueError("Source and target wavelength arrays must be 1D.")
    if src.size == 0 or tgt.size == 0:
        raise ValueError("Source and target wavelength arrays must be non-empty.")
    if np.unique(src).shape[0] != src.shape[0]:
        raise ValueError("Source wavelengths must be unique.")
    if np.unique(tgt).shape[0] != tgt.shape[0]:
        raise ValueError("Target wavelengths must be unique.")

    source_sort = np.argsort(src)
    target_sort = np.argsort(tgt)
    src_sorted = src[source_sort]
    tgt_sorted = tgt[target_sort]

    if np.any(np.diff(src_sorted) <= 0):
        raise ValueError("Sorted source wavelengths must be strictly increasing.")
    if np.any(np.diff(tgt_sorted) <= 0):
        raise ValueError("Sorted target wavelengths must be strictly increasing.")

    weights_sorted = np.zeros((tgt_sorted.size, src_sorted.size), dtype=np.float64)
    coverage_sorted = np.zeros(tgt_sorted.size, dtype=np.float64)
    insertion_indices = np.searchsorted(src_sorted, tgt_sorted, side="left")

    for row_idx, insert_idx in enumerate(insertion_indices.tolist()):
        target_wavelength = tgt_sorted[row_idx]

        if insert_idx <= 0:
            if np.isclose(target_wavelength, src_sorted[0]):
                weights_sorted[row_idx, 0] = 1.0
                coverage_sorted[row_idx] = 1.0
            continue

        if insert_idx >= src_sorted.size:
            if np.isclose(target_wavelength, src_sorted[-1]):
                weights_sorted[row_idx, -1] = 1.0
                coverage_sorted[row_idx] = 1.0
            continue

        left_idx = insert_idx - 1
        right_idx = insert_idx
        left_wavelength = src_sorted[left_idx]
        right_wavelength = src_sorted[right_idx]

        if np.isclose(target_wavelength, right_wavelength):
            weights_sorted[row_idx, right_idx] = 1.0
            coverage_sorted[row_idx] = 1.0
            continue

        span = right_wavelength - left_wavelength
        if span <= 0:
            raise ValueError("Sorted source wavelengths must be strictly increasing.")

        alpha = (target_wavelength - left_wavelength) / span
        weights_sorted[row_idx, left_idx] = 1.0 - alpha
        weights_sorted[row_idx, right_idx] = alpha
        coverage_sorted[row_idx] = 1.0

    covered = coverage_sorted > 0.0
    if covered.any():
        row_sums = weights_sorted[covered].sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise RuntimeError("Covered resampling rows must sum to 1.")

    weights = np.zeros((tgt.size, src.size), dtype=np.float64)
    coverage = np.zeros(tgt.size, dtype=np.float64)
    weights[np.ix_(target_sort, source_sort)] = weights_sorted
    coverage[target_sort] = coverage_sorted
    return weights, coverage


def build_linear_gap_fill_resampling_matrix(
    source_wavelengths_nm: np.ndarray,
    target_wavelengths_nm: np.ndarray,
    source_lower_edges_nm: np.ndarray,
    source_upper_edges_nm: np.ndarray,
    *,
    max_gap_nm: float = DEFAULT_LINEAR_GAP_FILL_MAX_GAP_NM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a center-wavelength interpolation matrix with a gap-aware coverage mask.

    This is intended for HSI -> HSI and MSI -> HSI projection.  The weights use
    the same stable center-wavelength interpolation as ``method='linear'`` for
    covered regions.  Small gaps are interpolated; targets just inside a large
    support gap are continued from the nearest edge band; targets deeper inside
    large gaps are marked uncovered.  This avoids sharp artificial outliers at
    tiny metadata gaps without inventing spectra through real absorption windows.
    """

    if max_gap_nm < 0.0:
        raise ValueError(f"max_gap_nm must be non-negative, got {max_gap_nm}.")

    weights = build_linear_resampling_matrix(
        source_wavelengths_nm=source_wavelengths_nm,
        target_wavelengths_nm=target_wavelengths_nm,
    )

    src_centers, src_lower, src_upper = _validate_band_edges(
        source_wavelengths_nm,
        source_lower_edges_nm,
        source_upper_edges_nm,
        label="source sensor",
    )
    tgt_centers = np.asarray(target_wavelengths_nm, dtype=np.float64)
    if tgt_centers.ndim != 1:
        raise ValueError("Target wavelengths must be 1D.")
    if tgt_centers.size == 0:
        raise ValueError("Target wavelengths must be non-empty.")

    order = np.argsort(src_centers)
    centers = src_centers[order]
    lower = src_lower[order]
    upper = src_upper[order]
    coverage = np.zeros(tgt_centers.size, dtype=np.float64)

    for idx, wavelength in enumerate(tgt_centers.tolist()):
        inside_source_band = np.any((lower <= wavelength) & (wavelength <= upper))
        if inside_source_band:
            coverage[idx] = 1.0
            continue

        insert = int(np.searchsorted(centers, wavelength, side="left"))
        if insert <= 0:
            distance_to_support = lower[0] - wavelength
            if distance_to_support <= max_gap_nm:
                coverage[idx] = 1.0
            continue
        if insert >= centers.size:
            distance_to_support = wavelength - upper[-1]
            if distance_to_support <= max_gap_nm:
                coverage[idx] = 1.0
            continue

        source_support_gap = lower[insert] - upper[insert - 1]
        if source_support_gap <= max_gap_nm:
            coverage[idx] = 1.0
            continue

        distance_to_left_support = wavelength - upper[insert - 1]
        distance_to_right_support = lower[insert] - wavelength
        if distance_to_left_support <= max_gap_nm:
            coverage[idx] = 1.0
            weights[idx, :] = 0.0
            weights[idx, order[insert - 1]] = 1.0
        elif distance_to_right_support <= max_gap_nm:
            coverage[idx] = 1.0
            weights[idx, :] = 0.0
            weights[idx, order[insert]] = 1.0

    return weights, coverage


def _validate_band_edges(
    centers_nm: np.ndarray,
    lower_edges_nm: np.ndarray,
    upper_edges_nm: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(centers_nm, dtype=np.float64)
    lower = np.asarray(lower_edges_nm, dtype=np.float64)
    upper = np.asarray(upper_edges_nm, dtype=np.float64)
    if centers.ndim != 1 or lower.ndim != 1 or upper.ndim != 1:
        raise ValueError(f"{label} centers/lower/upper arrays must be 1D.")
    if not (centers.shape == lower.shape == upper.shape):
        raise ValueError(
            f"{label} centers/lower/upper arrays must have matching shapes."
        )
    if centers.size == 0:
        raise ValueError(f"{label} band metadata must be non-empty.")
    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError(f"{label} band metadata contains non-finite values.")
    if np.any(upper <= lower):
        raise ValueError(f"{label} upper band edges must be greater than lower edges.")
    return centers, lower, upper


def build_bandpass_overlap_resampling_matrix(
    source_lower_edges_nm: np.ndarray,
    source_upper_edges_nm: np.ndarray,
    target_lower_edges_nm: np.ndarray,
    target_upper_edges_nm: np.ndarray,
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a bandpass-overlap matrix from source bands to target bands.

    Each target band is expressed as a weighted average of source bands whose
    wavelength supports overlap the target support.  The returned coverage is
    the fraction of each target band support covered by at least one source
    band; it is useful for detecting spectral gaps instead of silently clamping
    to an edge band.
    """

    src_lower = np.asarray(source_lower_edges_nm, dtype=np.float64)
    src_upper = np.asarray(source_upper_edges_nm, dtype=np.float64)
    tgt_lower = np.asarray(target_lower_edges_nm, dtype=np.float64)
    tgt_upper = np.asarray(target_upper_edges_nm, dtype=np.float64)

    if src_lower.ndim != 1 or src_upper.ndim != 1 or tgt_lower.ndim != 1 or tgt_upper.ndim != 1:
        raise ValueError("Source and target band edge arrays must be 1D.")
    if src_lower.shape != src_upper.shape:
        raise ValueError("Source lower/upper edge arrays must have matching shapes.")
    if tgt_lower.shape != tgt_upper.shape:
        raise ValueError("Target lower/upper edge arrays must have matching shapes.")
    if src_lower.size == 0 or tgt_lower.size == 0:
        raise ValueError("Source and target edge arrays must be non-empty.")
    if np.any(src_upper <= src_lower):
        raise ValueError("Source upper band edges must be greater than lower edges.")
    if np.any(tgt_upper <= tgt_lower):
        raise ValueError("Target upper band edges must be greater than lower edges.")

    lower = np.maximum(tgt_lower[:, None], src_lower[None, :])
    upper = np.minimum(tgt_upper[:, None], src_upper[None, :])
    overlap = np.maximum(upper - lower, 0.0)

    target_width = tgt_upper - tgt_lower
    overlap_sum = overlap.sum(axis=1)
    coverage = overlap_sum / target_width
    weights = overlap.astype(np.float64, copy=True)

    if normalize:
        covered = overlap_sum > 0.0
        weights[covered] = weights[covered] / overlap_sum[covered, None]

    return weights, coverage


def _apply_weight_matrix(x: torch.Tensor, weight_matrix: torch.Tensor) -> torch.Tensor:
    source_num_bands = int(weight_matrix.shape[1])
    target_num_bands = int(weight_matrix.shape[0])

    if x.ndim == 3:
        channels, height, width = x.shape
        if channels != source_num_bands:
            raise ValueError(
                f"Expected {source_num_bands} source bands, got {channels}."
            )
        flat = x.reshape(channels, height * width)
        out = torch.matmul(weight_matrix, flat)
        return out.reshape(target_num_bands, height, width)

    if x.ndim == 4:
        batch, channels, height, width = x.shape
        if channels != source_num_bands:
            raise ValueError(
                f"Expected {source_num_bands} source bands, got {channels}."
            )
        flat = x.permute(0, 2, 3, 1).reshape(-1, channels)
        out = torch.matmul(flat, weight_matrix.t())
        return out.reshape(batch, height, width, target_num_bands).permute(0, 3, 1, 2)

    if x.ndim == 5:
        batch, time, channels, height, width = x.shape
        if channels != source_num_bands:
            raise ValueError(
                f"Expected {source_num_bands} source bands, got {channels}."
            )
        flat = x.permute(0, 1, 3, 4, 2).reshape(-1, channels)
        out = torch.matmul(flat, weight_matrix.t())
        return out.reshape(batch, time, height, width, target_num_bands).permute(0, 1, 4, 2, 3)

    raise ValueError(
        f"Spectral resampling expects 3D/4D/5D tensors, got shape {tuple(x.shape)}."
    )


def _sensor_domain(sensor_config: Dict[str, Any]) -> str:
    spectral = sensor_config.get("spectral", {})
    if not isinstance(spectral, dict):
        return ""
    return str(spectral.get("domain", "")).lower()


def _is_hyperspectral(sensor_config: Dict[str, Any]) -> bool:
    domain = _sensor_domain(sensor_config)
    try:
        num_bands = int(sensor_config.get("num_bands", 0) or 0)
    except (TypeError, ValueError):
        num_bands = 0
    return "hyperspectral" in domain or num_bands >= 32


def _resolve_auto_mapping_method(
    source_sensor_config: Dict[str, Any],
    target_sensor_config: Dict[str, Any],
) -> str:
    # Target HSI branches are best synthesized by local spectral interpolation:
    # tiny support mismatches should be continued, but real gaps remain uncovered
    # through the coverage mask.  Broad MSI/OLI targets keep bandpass aggregation.
    if _is_hyperspectral(target_sensor_config):
        return "linear_gap_fill"
    return "bandpass"


class SpectralResampler(nn.Module):
    """Project spectra from one processed sensor wavelength grid onto another."""

    def __init__(
        self,
        source_wavelengths_nm: np.ndarray,
        target_wavelengths_nm: np.ndarray,
    ) -> None:
        super().__init__()
        weight_matrix = build_linear_resampling_matrix(
            source_wavelengths_nm=source_wavelengths_nm,
            target_wavelengths_nm=target_wavelengths_nm,
        )
        self.register_buffer(
            "weight_matrix",
            torch.as_tensor(weight_matrix, dtype=torch.float32),
            persistent=True,
        )

    @classmethod
    def from_sensor_configs(
        cls,
        source_sensor_config: Dict[str, Any],
        target_sensor_config: Dict[str, Any],
    ) -> "SpectralResampler":
        source_metadata = load_spectral_metadata(source_sensor_config, view="processed")
        target_metadata = load_spectral_metadata(target_sensor_config, view="processed")
        return cls(
            source_wavelengths_nm=source_metadata.band_centers_nm,
            target_wavelengths_nm=target_metadata.band_centers_nm,
        )

    @property
    def source_num_bands(self) -> int:
        return int(self.weight_matrix.shape[1])

    @property
    def target_num_bands(self) -> int:
        return int(self.weight_matrix.shape[0])

    def _weight_matrix_for(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight_matrix.to(device=x.device, dtype=x.dtype)

    def _apply_3d(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_weight_matrix(x, self._weight_matrix_for(x))

    def _apply_4d(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_weight_matrix(x, self._weight_matrix_for(x))

    def _apply_5d(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_weight_matrix(x, self._weight_matrix_for(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return self._apply_3d(x)
        if x.ndim == 4:
            return self._apply_4d(x)
        if x.ndim == 5:
            return self._apply_5d(x)
        raise ValueError(
            f"SpectralResampler expects 3D/4D/5D tensors, got shape {tuple(x.shape)}."
        )


class SensorSpectralMapper(nn.Module):
    """
    General sensor-to-sensor spectral mapper.

    ``method='linear'`` preserves the legacy center-wavelength interpolation.
    ``method='linear_gap_fill'`` uses center interpolation while marking target
    bands in large source support gaps as uncovered.
    ``method='bandpass'`` integrates source bands over target band supports and
    is preferred for HSI -> MSI synthesis when both sensors expose lower/upper
    band edges. ``method='auto'`` uses gap-aware interpolation for HSI targets
    and bandpass aggregation otherwise.
    """

    SUPPORTED_METHODS = {
        "auto",
        "linear",
        "linear_no_extrapolate",
        "linear_gap_fill",
        "bandpass",
    }

    def __init__(
        self,
        weight_matrix: np.ndarray,
        coverage: Optional[np.ndarray] = None,
        *,
        method: str,
        source_sensor: str = "",
        target_sensor: str = "",
    ) -> None:
        super().__init__()
        method_l = str(method).lower()
        if method_l not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported spectral mapping method '{method}'. "
                f"Use one of {sorted(self.SUPPORTED_METHODS)}."
            )
        weights = np.asarray(weight_matrix, dtype=np.float64)
        if weights.ndim != 2:
            raise ValueError("weight_matrix must be 2D [target_bands, source_bands].")
        if weights.shape[0] == 0 or weights.shape[1] == 0:
            raise ValueError("weight_matrix must be non-empty.")
        if coverage is None:
            coverage = np.ones(weights.shape[0], dtype=np.float64)
        coverage_arr = np.asarray(coverage, dtype=np.float64)
        if coverage_arr.shape != (weights.shape[0],):
            raise ValueError(
                "coverage must have one value per target band: "
                f"expected {(weights.shape[0],)}, got {coverage_arr.shape}."
            )

        self.method = method_l
        self.source_sensor = str(source_sensor).upper()
        self.target_sensor = str(target_sensor).upper()
        self.register_buffer(
            "weight_matrix",
            torch.as_tensor(weights, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "coverage",
            torch.as_tensor(coverage_arr, dtype=torch.float32),
            persistent=True,
        )

    @classmethod
    def from_sensor_configs(
        cls,
        source_sensor_config: Dict[str, Any],
        target_sensor_config: Dict[str, Any],
        *,
        method: str = "auto",
        max_gap_nm: float = DEFAULT_LINEAR_GAP_FILL_MAX_GAP_NM,
    ) -> "SensorSpectralMapper":
        method_l = str(method).lower()
        if method_l not in cls.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported spectral mapping method '{method}'. "
                f"Use one of {sorted(cls.SUPPORTED_METHODS)}."
            )
        if method_l == "auto":
            method_l = _resolve_auto_mapping_method(
                source_sensor_config,
                target_sensor_config,
            )

        source_metadata = load_spectral_metadata(source_sensor_config, view="processed")
        target_metadata = load_spectral_metadata(target_sensor_config, view="processed")

        if method_l == "linear":
            weights = build_linear_resampling_matrix(
                source_wavelengths_nm=source_metadata.band_centers_nm,
                target_wavelengths_nm=target_metadata.band_centers_nm,
            )
            coverage = np.ones(weights.shape[0], dtype=np.float64)
            resolved_method = "linear"
        elif method_l == "linear_no_extrapolate":
            weights, coverage = build_linear_no_extrapolation_resampling_matrix(
                source_wavelengths_nm=source_metadata.band_centers_nm,
                target_wavelengths_nm=target_metadata.band_centers_nm,
            )
            resolved_method = "linear_no_extrapolate"
        elif method_l == "linear_gap_fill":
            weights, coverage = build_linear_gap_fill_resampling_matrix(
                source_wavelengths_nm=source_metadata.band_centers_nm,
                target_wavelengths_nm=target_metadata.band_centers_nm,
                source_lower_edges_nm=source_metadata.lower_edges_nm,
                source_upper_edges_nm=source_metadata.upper_edges_nm,
                max_gap_nm=max_gap_nm,
            )
            resolved_method = "linear_gap_fill"
        else:
            _validate_band_edges(
                source_metadata.band_centers_nm,
                source_metadata.lower_edges_nm,
                source_metadata.upper_edges_nm,
                label=f"source sensor {source_metadata.sensor_name}",
            )
            _validate_band_edges(
                target_metadata.band_centers_nm,
                target_metadata.lower_edges_nm,
                target_metadata.upper_edges_nm,
                label=f"target sensor {target_metadata.sensor_name}",
            )
            weights, coverage = build_bandpass_overlap_resampling_matrix(
                source_lower_edges_nm=source_metadata.lower_edges_nm,
                source_upper_edges_nm=source_metadata.upper_edges_nm,
                target_lower_edges_nm=target_metadata.lower_edges_nm,
                target_upper_edges_nm=target_metadata.upper_edges_nm,
            )
            resolved_method = "bandpass"

        return cls(
            weight_matrix=weights,
            coverage=coverage,
            method=resolved_method,
            source_sensor=str(source_sensor_config.get("name", "")),
            target_sensor=str(target_sensor_config.get("name", "")),
        )

    @property
    def source_num_bands(self) -> int:
        return int(self.weight_matrix.shape[1])

    @property
    def target_num_bands(self) -> int:
        return int(self.weight_matrix.shape[0])

    def low_coverage_mask(self, min_coverage: float) -> torch.Tensor:
        return self.coverage < float(min_coverage)

    def _weight_matrix_for(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight_matrix.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_weight_matrix(x, self._weight_matrix_for(x))
