from __future__ import annotations

import numpy as np
import pytest

from geoews import ManifoldEWS, kl_rate
from geoews.core import EWSResult
from geoews.indicators import fisher_rao_distance, geodesic_acceleration, kl_divergence_rate
from geoews.manifold import fisher_rao_distance_univariate, step_distances
from geoews.windows import COVARIANCE_REGULARIZATION, estimate_gaussian_params


def test_regularization_constant():
    assert COVARIANCE_REGULARIZATION == 1e-6


def test_sliding_window_shapes_univariate():
    x = np.linspace(0.0, 1.0, 128)
    times, mus, sigmas = estimate_gaussian_params(x, window_size=32, step=1)
    assert times.ndim == 1
    assert mus.ndim == 1
    assert sigmas.ndim == 1
    assert len(times) == len(mus) == len(sigmas)


def test_kl_rate_nonnegative():
    x = np.sin(np.linspace(0, 6, 200)) + 0.05 * np.random.default_rng(0).normal(size=200)
    _, mus, sigmas = estimate_gaussian_params(x, window_size=40, step=1)
    kl = kl_rate(mus, sigmas)
    assert np.all(kl >= 0.0)
    assert np.allclose(kl, kl_divergence_rate(mus, sigmas))


def test_step_and_accel_lengths_match():
    x = np.sin(np.linspace(0, 8, 250))
    _, mus, sigmas = estimate_gaussian_params(x, window_size=50, step=1)
    vel = step_distances(mus, sigmas)
    acc = geodesic_acceleration(mus, sigmas, cumul_window=30)
    assert len(vel) == len(acc) == len(mus)


def test_manifold_ews_fit_detect():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    r = ManifoldEWS(window=30).fit(x).detect()
    assert isinstance(r, EWSResult)
    assert r.times.shape == r.kl_rate.shape == r.geodesic_acceleration.shape == r.fisher_rao.shape
    assert r.threshold == r.threshold  # finite float
    assert r.alert_index is None or isinstance(r.alert_index, int)
    assert {"time", "kl_rate", "fisher_rao", "geodesic_acceleration"} == set(r.to_dataframe().columns)


def test_short_series_raises():
    with pytest.raises(ValueError):
        estimate_gaussian_params(np.array([1.0, 2.0, 3.0]), window_size=10)


def test_constant_input_safe():
    x = np.ones(128, dtype=float)
    _, mus, sigmas = estimate_gaussian_params(x, window_size=32)
    kl = kl_rate(mus, sigmas)
    fr = fisher_rao_distance(mus, sigmas)
    assert np.all(np.isfinite(kl))
    assert np.all(np.isfinite(fr))
    assert np.all(kl >= 0.0)


def test_fisher_rao_univariate_symmetry():
    d12 = fisher_rao_distance_univariate(0.3, 1.2, -0.1, 0.8)
    d21 = fisher_rao_distance_univariate(-0.1, 0.8, 0.3, 1.2)
    assert np.isclose(d12, d21)
