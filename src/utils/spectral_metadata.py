"""Utilities for sensor spectral metadata and band-space bookkeeping."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SpectralBandMetadata:
    """Resolved spectral band metadata for a concrete channel view."""

    sensor_name: str
    view: str
    metadata_path: Path
    band_centers_nm: np.ndarray
    band_widths_nm: np.ndarray
    lower_edges_nm: np.ndarray
    upper_edges_nm: np.ndarray
    raw_indices_0based: np.ndarray

    @property
    def num_bands(self) -> int:
        return int(self.band_centers_nm.shape[0])


def resolve_repo_relative_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _require_spectral_config(sensor_config: Mapping[str, Any]) -> Mapping[str, Any]:
    spectral_cfg = sensor_config.get("spectral")
    if not isinstance(spectral_cfg, Mapping):
        sensor_name = str(sensor_config.get("name", "unknown"))
        raise ValueError(
            f"Sensor '{sensor_name}' is missing a 'spectral' config block."
        )
    return spectral_cfg


def _load_float_column(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(column)
        if value is None or value == "":
            raise ValueError(f"Missing required spectral metadata column '{column}'.")
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _load_int_column(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(column)
        if value is None or value == "":
            raise ValueError(f"Missing required spectral metadata column '{column}'.")
        values.append(int(float(value)))
    return np.asarray(values, dtype=np.int64)


def _load_band_metadata_csv(
    sensor_name: str,
    view: str,
    metadata_path: Path,
) -> SpectralBandMetadata:
    with metadata_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Spectral metadata file is empty: {metadata_path}")

    band_centers_nm = _load_float_column(rows, "wavelength_nm")
    band_widths_nm = _load_float_column(rows, "width_nm")
    lower_edges_nm = _load_float_column(rows, "lower_edge_nm")
    upper_edges_nm = _load_float_column(rows, "upper_edge_nm")
    raw_indices_0based = _load_int_column(rows, "raw_index_0based")

    if np.unique(band_centers_nm).shape[0] != band_centers_nm.shape[0]:
        raise ValueError(
            f"Spectral metadata wavelengths must be unique: {metadata_path}"
        )

    return SpectralBandMetadata(
        sensor_name=sensor_name,
        view=view,
        metadata_path=metadata_path,
        band_centers_nm=band_centers_nm,
        band_widths_nm=band_widths_nm,
        lower_edges_nm=lower_edges_nm,
        upper_edges_nm=upper_edges_nm,
        raw_indices_0based=raw_indices_0based,
    )


def load_spectral_metadata(
    sensor_config: Mapping[str, Any],
    view: str = "processed",
) -> SpectralBandMetadata:
    """
    Load wavelength metadata for the configured sensor.

    The spectral metadata path always refers to the same channel view used by the
    downstream model. For sensors that are trimmed on the fly, that means the
    processed metadata describes the post-trim channel layout.
    """

    spectral_cfg = _require_spectral_config(sensor_config)
    sensor_name = str(sensor_config.get("name", "unknown"))

    if view == "processed":
        path_key = "processed_band_metadata_path"
        expected_bands = int(sensor_config.get("num_bands", -1))
    elif view == "raw":
        path_key = "raw_band_metadata_path"
        expected_bands = int(
            sensor_config.get("num_bands_raw", sensor_config.get("num_bands", -1))
        )
    else:
        raise ValueError(f"Unsupported spectral metadata view '{view}'.")

    metadata_value = spectral_cfg.get(path_key)
    if not metadata_value:
        raise ValueError(
            f"Sensor '{sensor_name}' does not define spectral metadata path '{path_key}'."
        )

    metadata_path = resolve_repo_relative_path(metadata_value)
    metadata = _load_band_metadata_csv(
        sensor_name=sensor_name,
        view=view,
        metadata_path=metadata_path,
    )

    if expected_bands > 0 and metadata.num_bands != expected_bands:
        raise ValueError(
            f"Sensor '{sensor_name}' {view} metadata has {metadata.num_bands} bands, "
            f"expected {expected_bands}."
        )

    return metadata


def build_band_trim_transform(
    sensor_config: Mapping[str, Any],
    sensor_key: str = "image",
) -> Optional[object]:
    """Build the configured raw->processed band selection transform, if any."""
    from src.transforms.desis_band_trim import DESISBandTrim
    from src.transforms.eo1_band_trim import EO1BandTrim

    band_trim_cfg = sensor_config.get("band_trim")
    if not isinstance(band_trim_cfg, Mapping):
        return None

    if (
        "drop_ranges_1based" in band_trim_cfg
        or "optional_drop_ranges_1based" in band_trim_cfg
    ):
        return EO1BandTrim(
            drop_ranges_1based=band_trim_cfg.get("drop_ranges_1based"),
            optional_drop_ranges_1based=band_trim_cfg.get("optional_drop_ranges_1based"),
            include_optional=bool(band_trim_cfg.get("include_optional", False)),
            sensor_key=sensor_key,
        )

    trim_start = int(band_trim_cfg.get("trim_start", 0))
    trim_end = int(band_trim_cfg.get("trim_end", 0))
    if trim_start > 0 or trim_end > 0:
        return DESISBandTrim(
            trim_start=trim_start,
            trim_end=trim_end,
            sensor_key=sensor_key,
        )

    return None


def derive_processed_raw_indices(sensor_config: Mapping[str, Any]) -> np.ndarray:
    """Return raw band indices that survive the configured raw->processed step."""
    raw_count = int(sensor_config.get("num_bands_raw", sensor_config.get("num_bands", -1)))
    if raw_count <= 0:
        raise ValueError(
            f"Sensor '{sensor_config.get('name', 'unknown')}' has invalid raw band count {raw_count}."
        )

    band_trim = build_band_trim_transform(sensor_config=sensor_config, sensor_key="image")
    if band_trim is None:
        return np.arange(raw_count, dtype=np.int64)

    from src.transforms.desis_band_trim import DESISBandTrim
    from src.transforms.eo1_band_trim import EO1BandTrim

    if isinstance(band_trim, EO1BandTrim):
        return band_trim.get_keep_indices(raw_count).cpu().numpy().astype(np.int64)

    if isinstance(band_trim, DESISBandTrim):
        start = int(band_trim.trim_start)
        end = raw_count - int(band_trim.trim_end)
        return np.arange(start, end, dtype=np.int64)

    raise TypeError(f"Unsupported band trim transform type: {type(band_trim)}")


def validate_spectral_metadata_consistency(sensor_config: Mapping[str, Any]) -> None:
    """
    Validate raw/processed spectral metadata against the config band semantics.

    This is intentionally strict because subtle off-by-one errors here would
    silently break cross-sensor remapping.
    """

    processed_metadata = load_spectral_metadata(sensor_config, view="processed")
    raw_metadata = load_spectral_metadata(sensor_config, view="raw")

    if processed_metadata.num_bands != int(sensor_config.get("num_bands", -1)):
        raise ValueError(
            f"Processed band count mismatch for sensor '{sensor_config.get('name')}'."
        )

    raw_expected = int(
        sensor_config.get("num_bands_raw", sensor_config.get("num_bands", -1))
    )
    if raw_metadata.num_bands != raw_expected:
        raise ValueError(
            f"Raw band count mismatch for sensor '{sensor_config.get('name')}'."
        )

    spectral_cfg = _require_spectral_config(sensor_config)
    processed_raw_indices = processed_metadata.raw_indices_0based

    if np.any(np.diff(processed_raw_indices) <= 0):
        raise ValueError(
            f"Processed raw-index mapping must be strictly increasing for sensor "
            f"'{sensor_config.get('name')}'."
        )
    if processed_raw_indices.min() < 0 or processed_raw_indices.max() >= raw_expected:
        raise ValueError(
            f"Processed raw-index mapping falls outside raw band range for sensor "
            f"'{sensor_config.get('name')}'."
        )

    if sensor_config.get("band_trim") is not None:
        expected_raw_indices = derive_processed_raw_indices(sensor_config)
        if not np.array_equal(processed_raw_indices, expected_raw_indices):
            raise ValueError(
                f"Processed raw-index mapping does not match trim config for sensor "
                f"'{sensor_config.get('name')}'."
            )
        return

    channel_view = str(spectral_cfg.get("channel_view", "")).strip().lower()
    if channel_view not in {"stored_prefiltered", "stored_processed"}:
        raise ValueError(
            f"Sensor '{sensor_config.get('name')}' has no band_trim and unknown "
            f"channel_view '{spectral_cfg.get('channel_view')}'."
        )
