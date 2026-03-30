"""Alerting helpers for indicator thresholding."""

from __future__ import annotations

import numpy as np


def percentile_threshold(indicator: np.ndarray, baseline_end: int, percentile: float = 95.0) -> float:
    x = np.asarray(indicator, dtype=float)
    baseline_end = int(np.clip(baseline_end, 1, len(x)))
    return float(np.percentile(x[:baseline_end], percentile))


def first_crossing(indicator: np.ndarray, threshold: float) -> int | None:
    x = np.asarray(indicator, dtype=float)
    idx = np.where(x > float(threshold))[0]
    return int(idx[0]) if idx.size else None

