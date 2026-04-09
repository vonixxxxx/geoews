"""Alerting helpers for indicator thresholding."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["percentile_threshold", "first_crossing"]

def percentile_threshold(
    indicator: NDArray[np.float64],
    baseline_end: int,
    percentile: float = 95.0,
) -> float:
    """Compute percentile threshold from a baseline segment."""
    x = np.asarray(indicator, dtype=float)
    baseline_end = int(np.clip(baseline_end, 1, len(x)))
    return float(np.percentile(x[:baseline_end], percentile))


def first_crossing(indicator: NDArray[np.float64], threshold: float) -> int | None:
    """Return first index where indicator exceeds threshold."""
    x = np.asarray(indicator, dtype=float)
    idx = np.where(x > float(threshold))[0]
    return int(idx[0]) if idx.size else None

