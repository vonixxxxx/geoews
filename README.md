# geoews

`geoews` is a pip-installable Python package for information-geometric early warning signals.

This package extracts the canonical implementations from the source research repository:
- sliding-window Gaussian fitting
- KL divergence rate between consecutive Gaussian windows
- Fisher-Rao step distance (univariate exact, multivariate midpoint approximation)
- geodesic acceleration indicator

## Install

```bash
pip install .
```

## Quick start

```python
import numpy as np
from geoews.windows import estimate_gaussian_params
from geoews.indicators import kl_divergence_rate, geodesic_acceleration

x = np.sin(np.linspace(0, 12, 2000))
times, mus, sigmas = estimate_gaussian_params(x, window_size=50, step=1)

kl = kl_divergence_rate(mus, sigmas)
acc = geodesic_acceleration(mus, sigmas, cumul_window=30)
```

## Canonical constants

- `COVARIANCE_REGULARIZATION = 1e-6`

## Development

```bash
pip install -e ".[dev]"
pytest
```
