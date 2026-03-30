"""Dataset utilities for examples and quick tests."""

from __future__ import annotations

import numpy as np


def synthetic_fold_series(n: int = 2000, seed: int = 42) -> np.ndarray:
    """Generate a simple synthetic univariate series for demos."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=float)
    drift = np.linspace(-0.5, 0.8, n)
    for t in range(1, n):
        x[t] = 0.98 * x[t - 1] + drift[t] + 0.2 * rng.normal()
    return x

