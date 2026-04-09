"""Simple benchmark indicators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "rolling_variance",
    "rolling_lag1_autocorrelation",
    "variance_ews",
    "acf_ews",
]

def rolling_variance(
    x: NDArray[np.float64],
    window_size: int,
    step: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute rolling variance as a classical early-warning signal."""
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


def rolling_lag1_autocorrelation(
    x: NDArray[np.float64],
    window_size: int,
    step: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rolling lag-1 autocorrelation (Dakos-style early warning benchmark)."""
    s = np.asarray(x, dtype=float).ravel()
    n = len(s)
    n_windows = (n - window_size) // step
    if n_windows <= 0:
        raise ValueError("Not enough data for at least one sliding window.")
    times = np.zeros(n_windows, dtype=float)
    acf = np.zeros(n_windows, dtype=float)
    for i in range(n_windows):
        t_start = i * step
        window = s[t_start : t_start + window_size]
        times[i] = 0.5 * (t_start + t_start + window_size)
        if window_size < 2:
            acf[i] = np.nan
            continue
        a = window[:-1]
        b = window[1:]
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            acf[i] = np.nan
        else:
            acf[i] = float(np.corrcoef(a, b)[0, 1])
    return times, acf


def variance_ews(
    x: NDArray[np.float64], window: int, step: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rolling variance series for classical EWS comparison."""
    return rolling_variance(x, window_size=window, step=step)


def acf_ews(
    x: NDArray[np.float64], window: int, step: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rolling lag-1 ACF series for classical EWS comparison."""
    return rolling_lag1_autocorrelation(x, window_size=window, step=step)

