"""Sliding-window Gaussian parameter estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Canonical value extracted from config.py in source repository.
COVARIANCE_REGULARIZATION: float = 1e-6

__all__ = ["COVARIANCE_REGULARIZATION", "estimate_gaussian_params"]


def estimate_gaussian_params(
    x: NDArray[np.float64],
    window_size: int,
    step: int = 1,
    regularization: float = COVARIANCE_REGULARIZATION,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimate Gaussian parameters (mean, covariance) in sliding windows.

    Parameters
    ----------
    x : ndarray of shape (T,) or (T, d)
        Time series. Shape (T,) for univariate or (T, d) for multivariate.
    window_size : int
        Number of data points per window.
    step : int, default=1
        Step size between consecutive windows.
    regularization : float, default=1e-6
        Diagonal ridge added to variance/covariance for numerical stability.

    Returns
    -------
    times : ndarray of shape (n_windows,)
        Center time index of each window, shape (n_windows,).
    mus : ndarray
        Estimated mean at each window. Shape (n_windows,) for univariate
        and (n_windows, d) for multivariate.
    sigmas : ndarray
        Estimated variance/covariance. For univariate: shape (n_windows,).
        For multivariate: shape (n_windows, d, d).
    """
    if window_size <= 1:
        raise ValueError("window_size must be > 1 for ddof=1 variance/covariance.")
    if step <= 0:
        raise ValueError("step must be a positive integer.")

    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)

    t_len, d = x_arr.shape
    n_windows = (t_len - window_size) // step
    if n_windows <= 0:
        raise ValueError("Not enough data for at least one sliding window.")

    times = np.zeros(n_windows, dtype=float)
    mus = np.zeros((n_windows, d), dtype=float)

    if d == 1:
        sigmas: np.ndarray = np.zeros(n_windows, dtype=float)
    else:
        sigmas = np.zeros((n_windows, d, d), dtype=float)

    reg = float(regularization)

    for i in range(n_windows):
        t_start = i * step
        t_end = t_start + window_size
        window = x_arr[t_start:t_end]

        times[i] = 0.5 * (t_start + t_end)
        mus[i] = np.mean(window, axis=0)

        if d == 1:
            sigmas[i] = float(np.var(window, ddof=1)) + reg
        else:
            cov = np.cov(window, rowvar=False, ddof=1)
            cov += reg * np.eye(d)
            sigmas[i] = cov

    if d == 1:
        mus = mus.flatten()

    return times, mus, sigmas
