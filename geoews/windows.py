"""Sliding-window Gaussian parameter estimation."""

from __future__ import annotations

import numpy as np

# Canonical value extracted from config.py in source repository.
COVARIANCE_REGULARIZATION = 1e-6


def estimate_gaussian_params(
    x: np.ndarray, window_size: int, step: int = 1, regularization: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate Gaussian parameters (mean, covariance) in sliding windows.

    Parameters
    ----------
    x:
        Time series. Shape (T,) for univariate or (T, d) for multivariate.
    window_size:
        Number of data points per window.
    step:
        Step size between consecutive windows.

    Returns
    -------
    times:
        Center time index of each window, shape (n_windows,).
    mus:
        Estimated mean at each window. Shape (n_windows,) for univariate
        and (n_windows, d) for multivariate.
    sigmas:
        Estimated variance/covariance. For univariate: shape (n_windows,).
        For multivariate: shape (n_windows, d, d).
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    t_len, d = x.shape
    n_windows = (t_len - window_size) // step
    if n_windows <= 0:
        raise ValueError("Not enough data for at least one sliding window.")

    times = np.zeros(n_windows, dtype=float)
    mus = np.zeros((n_windows, d), dtype=float)

    if d == 1:
        sigmas: np.ndarray = np.zeros(n_windows, dtype=float)
    else:
        sigmas = np.zeros((n_windows, d, d), dtype=float)

    reg = float(COVARIANCE_REGULARIZATION if regularization is None else regularization)

    for i in range(n_windows):
        t_start = i * step
        t_end = t_start + window_size
        window = x[t_start:t_end]

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
