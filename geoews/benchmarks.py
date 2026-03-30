"""Simple benchmark indicators."""

from __future__ import annotations

import numpy as np


def rolling_variance(x: np.ndarray, window_size: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(x, dtype=float).ravel()
    n = len(s)
    n_windows = (n - window_size) // step
    if n_windows <= 0:
        raise ValueError("Not enough data for at least one sliding window.")
    times = np.zeros(n_windows, dtype=float)
    out = np.zeros(n_windows, dtype=float)
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        w = s[start:end]
        times[i] = 0.5 * (start + end)
        out[i] = float(np.var(w, ddof=1))
    return times, out

