# geoews

[![PyPI version](https://img.shields.io/pypi/v/geoews.svg)](https://pypi.org/project/geoews/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/geoews.svg)](https://pypi.org/project/geoews/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Information-geometric early warning signals** on sliding-window Gaussian models: **KL divergence rate**, **Fisher–Rao step distances**, and **geodesic acceleration**, plus classical benchmarks and optional data helpers.

Math matches the validated research implementation (sliding-window Gaussian fit, same KL and FR formulas, same regularization).

- **Repository:** [github.com/vonixxxxx/geoews](https://github.com/vonixxxxx/geoews)
- **PyPI:** [pypi.org/project/geoews](https://pypi.org/project/geoews/)

## Install

From PyPI (recommended):

```bash
pip install geoews
```

Upgrade:

```bash
pip install geoews --upgrade
```

## Quick start — high-level API

`ManifoldEWS` fits sliding-window Gaussians, computes indicators, and runs a simple baseline-threshold detection:

```python
import numpy as np
from geoews import ManifoldEWS

x = np.random.default_rng(0).standard_normal(500)
result = ManifoldEWS(window=40, cumul_window=30).fit(x).detect()
print(result)  # EWSResult with kl_rate, geodesic_acceleration, threshold, etc.
```

## Lower-level API (same math, full control)

```python
import numpy as np
from geoews.windows import estimate_gaussian_params, COVARIANCE_REGULARIZATION
from geoews.indicators import kl_rate, geodesic_acceleration

x = np.sin(np.linspace(0, 12, 2000))
times, mus, sigmas = estimate_gaussian_params(x, window_size=50, step=1)

kl = kl_rate(mus, sigmas)
acc = geodesic_acceleration(mus, sigmas, cumul_window=30)
```

`kl_rate` is the KL divergence rate between consecutive window Gaussians (alias of `kl_divergence_rate`).

## Classical benchmarks

```python
from geoews import variance_ews, acf_ews

times_v, var_series = variance_ews(x, window=50, step=1)
times_a, acf_series = acf_ews(x, window=50, step=1)
```

## Optional datasets (local files)

`load_ngrip` and `load_peter_lake` expect paths to your own copies of the public datasets (Excel/CSV). See the `examples/` notebooks in the repo for typical usage.

## Constants

- **`COVARIANCE_REGULARIZATION`** defaults to **`1e-6`** (diagonal ridge on covariance / variance).

## Development (from a git clone)

```bash
git clone https://github.com/vonixxxxx/geoews.git
cd geoews
pip install -e ".[dev]"
pytest
```

## Citation

See `CITATION.cff` in the repository for citation metadata.

## License

MIT — see `LICENSE`.
