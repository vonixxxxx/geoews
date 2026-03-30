"""Fisher-Rao distances used by geoews indicators."""

from __future__ import annotations

import numpy as np


def _fisher_rao_distance_univariate(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """
    Exact Fisher-Rao distance between N(mu1, sigma1^2) and N(mu2, sigma2^2).

    Parameters: mu1, mu2 are means; sigma1, sigma2 are STANDARD DEVIATIONS.
    """
    s1 = max(abs(sigma1), 1e-15)
    s2 = max(abs(sigma2), 1e-15)
    du = (mu1 - mu2) / np.sqrt(2.0)
    dv = s1 - s2
    arg = 1.0 + (du**2 + dv**2) / (2.0 * s1 * s2)
    return float(np.sqrt(2.0) * np.arccosh(max(arg, 1.0)))


def _fisher_rao_distance_multivariate(
    mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray
) -> float:
    """
    Approximate Fisher-Rao distance between multivariate Gaussians
    using the midpoint infinitesimal metric.

    ds^2 = dmu^T Sigma_avg^{-1} dmu + 0.5 tr(Sigma_avg^{-1} dSigma Sigma_avg^{-1} dSigma)
    """
    dmu = np.asarray(mu2, dtype=float) - np.asarray(mu1, dtype=float)
    sigma_avg = 0.5 * (np.asarray(cov1) + np.asarray(cov2))
    try:
        sigma_avg_inv = np.linalg.inv(sigma_avg)
    except np.linalg.LinAlgError:
        return 0.0
    dsigma = np.asarray(cov2) - np.asarray(cov1)

    mu_term = float(dmu @ sigma_avg_inv @ dmu)
    a_mat = sigma_avg_inv @ dsigma
    sigma_term = 0.5 * float(np.trace(a_mat @ a_mat))
    return float(np.sqrt(max(mu_term + sigma_term, 0.0)))


def _step_distances(mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Consecutive Fisher-Rao distances."""
    t_len = len(mus)
    univariate = sigmas.ndim == 1
    dists = np.zeros(t_len, dtype=float)
    for t in range(1, t_len):
        if univariate:
            s1 = np.sqrt(max(float(sigmas[t - 1]), 1e-15))
            s2 = np.sqrt(max(float(sigmas[t]), 1e-15))
            dists[t] = _fisher_rao_distance_univariate(float(mus[t - 1]), s1, float(mus[t]), s2)
        else:
            dists[t] = _fisher_rao_distance_multivariate(mus[t - 1], sigmas[t - 1], mus[t], sigmas[t])
    return dists


fisher_rao_distance_univariate = _fisher_rao_distance_univariate
fisher_rao_distance_multivariate = _fisher_rao_distance_multivariate
step_distances = _step_distances

