from __future__ import annotations

import numpy as np

from geoews import ManifoldEWS, kl_rate
from geoews.indicators import geodesic_acceleration, kl_divergence_rate
from geoews.manifold import EWSResult, step_distances
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
    assert r.times.shape == r.kl_rate.shape == r.geodesic_acceleration.shape
    assert r.threshold == r.threshold  # finite float
    assert isinstance(r.triggered, bool)
