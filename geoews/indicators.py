"""Canonical geoews indicators."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .manifold import _step_distances


def kl_divergence_rate(mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """
    KL(p_t || p_{t-1}) between consecutive sliding-window Gaussians.

    Univariate:
        KL = 0.5 * (s2_t/s2_{t-1} + dmu^2/s2_{t-1} - 1 + ln(s2_{t-1}/s2_t))

    Multivariate:
        KL = 0.5 * (tr(S_{t-1}^{-1} S_t) + dmu^T S_{t-1}^{-1} dmu - d + ln det(S_{t-1}/S_t))
    """
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
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


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Efficient rolling sum using cumulative sum."""
    t_len = len(arr)
    cs = np.cumsum(arr)
    result = np.zeros(t_len, dtype=float)
    for t in range(t_len):
        lo = max(0, t - window + 1)
        result[t] = cs[t] - (cs[lo - 1] if lo > 0 else 0.0)
    return result


def geodesic_acceleration(mus: np.ndarray, sigmas: np.ndarray, cumul_window: int = 30) -> np.ndarray:
    """
    Cumulative positive acceleration along the geodesic.

    velocity(t) = d_FR(p_{t-1}, p_t)
    acceleration(t) = velocity(t) - velocity(t-1)
    indicator(t) = rolling sum over max(acceleration, 0), with slight penalty on negatives.
    """
    velocity = _step_distances(np.asarray(mus), np.asarray(sigmas))
    accel = np.zeros_like(velocity)
    accel[1:] = velocity[1:] - velocity[:-1]
    smooth_accel = uniform_filter1d(accel, size=5, mode="nearest")
    pos_accel = np.where(smooth_accel > 0.0, smooth_accel, 0.01 * smooth_accel)
    return _rolling_sum(pos_accel, cumul_window)

