"""
EO-1 Hyperion band trimming transform.

Drops known low-signal / absorption-window band ranges from EO-1 hyperspectral
data. The default ranges are based on empirical inspection of the downstream
EO1 CDL split plus wavelength mapping metadata.
"""

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch


class EO1BandTrim:
    """
    Remove problematic EO-1 bands by dropping explicit 1-based inclusive ranges.

    Default drop ranges:
    - 94-106
    - 139-156
    - 195-198

    Optional additional drop range:
    - 159-161

    These ranges are expressed in 1-based indexing to match the sensor band
    mapping spreadsheets typically used during analysis. Internally they are
    converted to 0-based tensor indices.

    Args:
        drop_ranges_1based: Main 1-based inclusive band ranges to remove.
        optional_drop_ranges_1based: Additional ranges that can be enabled.
        include_optional: If True, also remove optional ranges.
        sensor_key: Key in the batch dict for EO1 data. Default is "EO1".
    """

    DEFAULT_DROP_RANGES_1BASED: Tuple[Tuple[int, int], ...] = (
        (94, 106),
        (139, 156),
        (195, 198),
    )
    OPTIONAL_DROP_RANGES_1BASED: Tuple[Tuple[int, int], ...] = (
        (159, 161),
    )

    def __init__(
        self,
        drop_ranges_1based: Optional[Sequence[Sequence[int]]] = None,
        optional_drop_ranges_1based: Optional[Sequence[Sequence[int]]] = None,
        include_optional: bool = False,
        sensor_key: str = "EO1",
    ) -> None:
        self.drop_ranges_1based = self._normalize_ranges(
            drop_ranges_1based
            if drop_ranges_1based is not None
            else self.DEFAULT_DROP_RANGES_1BASED
        )
        self.optional_drop_ranges_1based = self._normalize_ranges(
            optional_drop_ranges_1based
            if optional_drop_ranges_1based is not None
            else self.OPTIONAL_DROP_RANGES_1BASED
        )
        self.include_optional = bool(include_optional)
        self.sensor_key = sensor_key
        self._keep_indices_cache: Dict[int, torch.Tensor] = {}

    @staticmethod
    def _normalize_ranges(
        ranges_1based: Sequence[Sequence[int]],
    ) -> Tuple[Tuple[int, int], ...]:
        normalized = []
        for item in ranges_1based:
            if len(item) != 2:
                raise ValueError(
                    f"Each EO1 trim range must have exactly 2 values, got {item}."
                )
            start, end = int(item[0]), int(item[1])
            if start < 1 or end < 1:
                raise ValueError(
                    f"EO1 trim ranges are 1-based and must be >= 1, got {item}."
                )
            if end < start:
                raise ValueError(
                    f"EO1 trim range end must be >= start, got {item}."
                )
            normalized.append((start, end))
        return tuple(normalized)

    def _resolve_sensor_key(self, batch: Dict[str, Any]) -> Optional[str]:
        """Resolve sensor key case-insensitively to support mixed key conventions."""
        if self.sensor_key in batch:
            return self.sensor_key

        target = str(self.sensor_key).upper()
        for key in batch.keys():
            if isinstance(key, str) and key.upper() == target:
                return key
        return None

    def get_active_drop_ranges_1based(self) -> Tuple[Tuple[int, int], ...]:
        """Return the active 1-based inclusive drop ranges."""
        if self.include_optional:
            return self.drop_ranges_1based + self.optional_drop_ranges_1based
        return self.drop_ranges_1based

    def get_keep_indices(self, num_bands: int) -> torch.Tensor:
        """Return cached 0-based keep indices for a given band count."""
        num_bands = int(num_bands)
        cached = self._keep_indices_cache.get(num_bands)
        if cached is not None:
            return cached

        keep_mask = torch.ones(num_bands, dtype=torch.bool)
        for start_1based, end_1based in self.get_active_drop_ranges_1based():
            start = start_1based - 1
            end = end_1based - 1
            if end >= num_bands:
                raise ValueError(
                    f"EO1 trim range {(start_1based, end_1based)} exceeds "
                    f"available band count {num_bands}."
                )
            keep_mask[start : end + 1] = False

        keep_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
        if keep_indices.numel() == 0:
            raise ValueError(
                f"EO1BandTrim would remove all {num_bands} bands; check the ranges."
            )

        self._keep_indices_cache[num_bands] = keep_indices
        return keep_indices

    def trim_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Trim a tensor with shape (C,H,W), (B,C,H,W), or (B,T,C,H,W)."""
        if not isinstance(tensor, torch.Tensor):
            return tensor

        if tensor.ndim == 3:
            band_dim = 0
        elif tensor.ndim == 4:
            band_dim = 1
        elif tensor.ndim == 5:
            band_dim = 2
        else:
            return tensor

        num_bands = int(tensor.shape[band_dim])
        keep_indices = self.get_keep_indices(num_bands).to(tensor.device)
        return torch.index_select(tensor, dim=band_dim, index=keep_indices)

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply EO1 band trimming to the configured sensor key in a batch."""
        sensor_key = self._resolve_sensor_key(batch)
        if sensor_key is None:
            return batch

        batch[sensor_key] = self.trim_tensor(batch[sensor_key])
        return batch

    def __repr__(self) -> str:
        return (
            "EO1BandTrim("
            f"drop_ranges_1based={self.drop_ranges_1based}, "
            f"optional_drop_ranges_1based={self.optional_drop_ranges_1based}, "
            f"include_optional={self.include_optional}, "
            f"sensor_key='{self.sensor_key}'"
            ")"
        )
