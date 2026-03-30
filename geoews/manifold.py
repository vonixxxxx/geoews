"""Fisher-Rao distances and high-level ManifoldEWS API."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class EWSResult:
    """Detection output from :meth:`ManifoldEWS.detect`."""

    warning_index: int | None
    threshold: float
    times: np.ndarray
    kl_rate: np.ndarray
    geodesic_acceleration: np.ndarray
    triggered: bool


class ManifoldEWS:
    """
    Sliding-window Gaussian fit plus KL-rate / geodesic-acceleration indicators.

    Uses :func:`geoews.windows.estimate_gaussian_params` and canonical formulas
    from :mod:`geoews.indicators`.
    """

    def __init__(
        self,
        window: int = 50,
        *,
        step: int = 1,
        cumul_window: int = 30,
        baseline_fraction: float = 0.3,
        threshold_percentile: float = 95.0,
        regularization: float | None = None,
    ) -> None:
        self.window = int(window)
        self.step = int(step)
        self.cumul_window = int(cumul_window)
        self.baseline_fraction = float(baseline_fraction)
        self.threshold_percentile = float(threshold_percentile)
        self.regularization = regularization

        self._times: np.ndarray | None = None
        self._mus: np.ndarray | None = None
        self._sigmas: np.ndarray | None = None

    def fit(self, x: np.ndarray, /) -> ManifoldEWS:
        from .windows import estimate_gaussian_params

        self._times, self._mus, self._sigmas = estimate_gaussian_params(
            np.asarray(x, dtype=float),
            window_size=self.window,
            step=self.step,
            regularization=self.regularization,
        )
        return self

    def detect(self) -> EWSResult:
        from .alerts import first_crossing, percentile_threshold
        from .indicators import geodesic_acceleration, kl_divergence_rate

        if self._times is None or self._mus is None or self._sigmas is None:
            raise RuntimeError("Call fit(x) before detect().")

        times = self._times
        mus = self._mus
        sigmas = self._sigmas

        kl = kl_divergence_rate(mus, sigmas)
        ga = geodesic_acceleration(mus, sigmas, cumul_window=self.cumul_window)

        n = len(kl)
        bl_end = max(5, int(self.baseline_fraction * n))
        threshold = percentile_threshold(kl, bl_end, self.threshold_percentile)
        warning_index = first_crossing(kl, threshold)
        triggered = warning_index is not None

        return EWSResult(
            warning_index=warning_index,
            threshold=float(threshold),
            times=np.asarray(times, dtype=float),
            kl_rate=np.asarray(kl, dtype=float),
            geodesic_acceleration=np.asarray(ga, dtype=float),
            triggered=triggered,
        )
