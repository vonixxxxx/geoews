"""Canonical geoews indicators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d

from .manifold import _step_distances

__all__ = [
    "fisher_rao_distance",
    "kl_divergence_rate",
    "kl_rate",
    "geodesic_acceleration",
]


def kl_divergence_rate(
    mus: NDArray[np.float64],
    sigmas: NDArray[np.float64],
) -> NDArray[np.float64]:
    """KL divergence rate between consecutive sliding-window Gaussians.

    Computes ``D_KL(p_t || p_{t-1})`` for each pair of consecutive windows.
    For univariate Gaussian windows:

    ``KL = 0.5 * (v_t/v_{t-1} + dmu^2/v_{t-1} - 1 + log(v_{t-1}/v_t))``

    For multivariate Gaussian windows:

    ``KL = 0.5 * (tr(S_{t-1}^{-1} S_t) + dmu^T S_{t-1}^{-1} dmu - d + logdet(S_{t-1}) - logdet(S_t))``

    Parameters
    ----------
    mus : ndarray
        Sliding-window means. Shape ``(T,)`` for univariate or ``(T, d)``
        for multivariate.
    sigmas : ndarray
        Sliding-window variance/covariance series. Shape ``(T,)`` for
        univariate or ``(T, d, d)`` for multivariate.

    Returns
    -------
    ndarray of shape (T,)
        KL rate series. Index 0 is zero. Values are clipped to be non-negative.
    """
    mus = np.asarray(mus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    t_len = len(mus)
    univariate = sigmas.ndim == 1

    kl = np.zeros(t_len, dtype=float)

    for t in range(1, t_len):
        if univariate:
            v_prev = max(float(sigmas[t - 1]), 1e-15)
            v_curr = max(float(sigmas[t]), 1e-15)
            dmu = float(mus[t] - mus[t - 1])
            kl[t] = 0.5 * (v_curr / v_prev + dmu**2 / v_prev - 1.0 + np.log(v_prev / v_curr))
        else:
            d = mus.shape[1]
            dmu = mus[t] - mus[t - 1]
            try:
                s_prev_inv = np.linalg.inv(sigmas[t - 1])
                _, ld_prev = np.linalg.slogdet(sigmas[t - 1])
                _, ld_curr = np.linalg.slogdet(sigmas[t])
            except np.linalg.LinAlgError:
                kl[t] = 0.0
                continue

            trace_term = float(np.trace(s_prev_inv @ sigmas[t]))
            mu_term = float(dmu @ s_prev_inv @ dmu)
            kl[t] = 0.5 * (trace_term + mu_term - d + ld_prev - ld_curr)

    return np.maximum(kl, 0.0)


kl_rate = kl_divergence_rate


def fisher_rao_distance(
    mus: NDArray[np.float64],
    sigmas: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fisher-Rao step distances between consecutive windows.

    Parameters
    ----------
    mus : ndarray
        Sliding-window means. Shape (T,) or (T, d).
    sigmas : ndarray
        Sliding-window variance/covariance sequence. Shape (T,) or (T, d, d).

    Returns
    -------
    ndarray of shape (T,)
        Fisher-Rao step-distance series. Index 0 equals zero.
    """
    return _step_distances(np.asarray(mus, dtype=float), np.asarray(sigmas, dtype=float))


def _rolling_sum(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Efficient rolling sum using cumulative sum."""
    t_len = len(arr)
    cs = np.cumsum(arr)
    result = np.zeros(t_len, dtype=float)
    for t in range(t_len):
        lo = max(0, t - window + 1)
        result[t] = cs[t] - (cs[lo - 1] if lo > 0 else 0.0)
    return result


def geodesic_acceleration(
    mus: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    cumul_window: int = 30,
) -> NDArray[np.float64]:
    """Cumulative geodesic acceleration indicator.

    Defines Fisher-Rao step distances as the manifold velocity and uses
    first differences to approximate acceleration. The acceleration is
    smoothed and accumulated in a rolling window.

    Parameters
    ----------
    mus : ndarray
        Sliding-window means. Shape ``(T,)`` or ``(T, d)``.
    sigmas : ndarray
        Sliding-window variance/covariance sequence. Shape ``(T,)`` or
        ``(T, d, d)``.
    cumul_window : int, default=30
        Rolling accumulation window for the acceleration signal.

    Returns
    -------
    ndarray of shape (T,)
        Geodesic acceleration series.
    """
    if cumul_window <= 0:
        raise ValueError("cumul_window must be positive.")
    velocity = _step_distances(np.asarray(mus, dtype=float), np.asarray(sigmas, dtype=float))
    accel = np.zeros_like(velocity)
    accel[1:] = velocity[1:] - velocity[:-1]
    smooth_accel = uniform_filter1d(accel, size=5, mode="nearest")
    pos_accel = np.where(smooth_accel > 0.0, smooth_accel, 0.01 * smooth_accel)
    return _rolling_sum(pos_accel, cumul_window)

