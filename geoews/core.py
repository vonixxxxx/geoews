"""High-level pipeline and result container for geoews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .alerts import first_crossing
from .indicators import fisher_rao_distance, geodesic_acceleration, kl_rate
from .windows import COVARIANCE_REGULARIZATION, estimate_gaussian_params

__all__ = ["EWSResult", "ManifoldEWS"]


@dataclass(frozen=True)
class EWSResult:
    """Container for geometric early-warning signal results.

    Attributes
    ----------
    times : ndarray of shape (T,)
        Center indices of each sliding window.
    kl_rate : ndarray of shape (T,)
        KL divergence rate series.
    fisher_rao : ndarray of shape (T,)
        Fisher-Rao step-distance series.
    geodesic_acceleration : ndarray of shape (T,)
        Cumulative geodesic acceleration indicator.
    alert_index : int or None
        First index where ``kl_rate`` exceeds ``threshold``.
    threshold : float
        Detection threshold value used by ``detect``.
    """

    times: NDArray[np.float64]
    kl_rate: NDArray[np.float64]
    fisher_rao: NDArray[np.float64]
    geodesic_acceleration: NDArray[np.float64]
    alert_index: int | None
    threshold: float

    @property
    def triggered(self) -> bool:
        """Whether an alert was triggered."""
        return self.alert_index is not None

    @property
    def warning_index(self) -> int | None:
        """Backward-compatible alias for ``alert_index``."""
        return self.alert_index

    def to_dataframe(self) -> pd.DataFrame:
        """Return the result series as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``time``, ``kl_rate``, ``fisher_rao``,
            and ``geodesic_acceleration``.
        """
        return pd.DataFrame(
            {
                "time": self.times,
                "kl_rate": self.kl_rate,
                "fisher_rao": self.fisher_rao,
                "geodesic_acceleration": self.geodesic_acceleration,
            }
        )

    def plot(self, figsize: tuple[int, int] = (10, 8), **kwargs: Any) -> plt.Figure:
        """Plot indicator trajectories with optional alert line.

        Parameters
        ----------
        figsize : tuple of int, default=(10, 8)
            Figure size passed to matplotlib.
        **kwargs : dict
            Additional keyword arguments forwarded to ``Axes.plot``.

        Returns
        -------
        matplotlib.figure.Figure
            Created matplotlib figure.
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        axes[0].plot(self.times, self.kl_rate, **kwargs)
        axes[0].axhline(self.threshold, color="red", linestyle="--", linewidth=1.2)
        axes[0].set_ylabel("KL rate")
        axes[0].grid(alpha=0.3)

        axes[1].plot(self.times, self.fisher_rao, **kwargs)
        axes[1].set_ylabel("Fisher-Rao")
        axes[1].grid(alpha=0.3)

        axes[2].plot(self.times, self.geodesic_acceleration, **kwargs)
        axes[2].set_ylabel("Geo. accel.")
        axes[2].set_xlabel("Window index")
        axes[2].grid(alpha=0.3)

        if self.alert_index is not None:
            x_alert = self.times[self.alert_index]
            for ax in axes:
                ax.axvline(x_alert, color="red", linestyle=":", linewidth=1.2)

        fig.tight_layout()
        return fig


class ManifoldEWS:
    """High-level pipeline for information-geometric early warning signals.

    Follows a scikit-learn-style ``fit`` / ``detect`` interface.

    Parameters
    ----------
    window : int
        Sliding-window length for Gaussian fitting.
    step : int, default=1
        Step size between consecutive windows.
    cumul_window : int, default=30
        Rolling accumulation window for geodesic acceleration.
    threshold_sigma : float, default=2.0
        Number of baseline standard deviations above baseline mean used
        for the detection threshold.
    baseline_fraction : float, default=0.3
        Fraction of the series used as baseline for threshold estimation.
    regularization : float, default=1e-6
        Diagonal ridge regularization for covariance estimation.
    """

    def __init__(
        self,
        window: int = 50,
        *,
        step: int = 1,
        cumul_window: int = 30,
        threshold_sigma: float = 2.0,
        baseline_fraction: float = 0.3,
        regularization: float = COVARIANCE_REGULARIZATION,
    ) -> None:
        self.window = int(window)
        self.step = int(step)
        self.cumul_window = int(cumul_window)
        self.threshold_sigma = float(threshold_sigma)
        self.baseline_fraction = float(baseline_fraction)
        self.regularization = float(regularization)

        self._times: NDArray[np.float64] | None = None
        self._mus: NDArray[np.float64] | None = None
        self._sigmas: NDArray[np.float64] | None = None
        self._result: EWSResult | None = None

    def fit(self, x: NDArray[np.float64]) -> ManifoldEWS:
        """Fit sliding-window Gaussian parameters from input series.

        Parameters
        ----------
        x : ndarray of shape (T,) or (T, d)
            Input time series.

        Returns
        -------
        ManifoldEWS
            Fitted estimator (for fluent chaining).
        """
        times, mus, sigmas = estimate_gaussian_params(
            np.asarray(x, dtype=float),
            window_size=self.window,
            step=self.step,
            regularization=self.regularization,
        )
        self._times = times
        self._mus = mus
        self._sigmas = sigmas
        self._result = None
        return self

    def detect(self, threshold_sigma: float | None = None) -> EWSResult:
        """Run threshold detection and return structured results.

        Parameters
        ----------
        threshold_sigma : float or None, optional
            Override for ``self.threshold_sigma``.

        Returns
        -------
        EWSResult
            Result object with indicator time series and alert index.
        """
        if self._times is None or self._mus is None or self._sigmas is None:
            raise RuntimeError("Call fit(x) before detect().")

        sigma_mult = self.threshold_sigma if threshold_sigma is None else float(threshold_sigma)
        if sigma_mult < 0:
            raise ValueError("threshold_sigma must be non-negative.")

        kl = kl_rate(self._mus, self._sigmas)
        fr = fisher_rao_distance(self._mus, self._sigmas)
        ga = geodesic_acceleration(self._mus, self._sigmas, cumul_window=self.cumul_window)

        baseline_end = max(5, int(self.baseline_fraction * len(kl)))
        baseline = kl[:baseline_end]
        threshold = float(np.mean(baseline) + sigma_mult * np.std(baseline))
        alert_index = first_crossing(kl, threshold)

        result = EWSResult(
            times=np.asarray(self._times, dtype=float),
            kl_rate=np.asarray(kl, dtype=float),
            fisher_rao=np.asarray(fr, dtype=float),
            geodesic_acceleration=np.asarray(ga, dtype=float),
            alert_index=alert_index,
            threshold=threshold,
        )
        self._result = result
        return result

    def plot(self, figsize: tuple[int, int] = (10, 8), **kwargs: Any) -> plt.Figure:
        """Plot indicators for the most recent detection result.

        Returns
        -------
        matplotlib.figure.Figure
            Figure returned by ``EWSResult.plot``.
        """
        if self._result is None:
            self.detect()
        assert self._result is not None
        return self._result.plot(figsize=figsize, **kwargs)

