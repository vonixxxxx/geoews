<p align="center">
  <h1 align="center">geoews</h1>
  <p align="center">
    <strong>Information-geometric early warning signals for critical transitions</strong>
  </p>
  <p align="center">
    Detect tipping points earlier using the geometry of statistical manifolds — KL divergence rates, Fisher–Rao distances, and geodesic acceleration on sliding-window Gaussian models.
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/geoews/"><img src="https://img.shields.io/pypi/v/geoews.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/geoews/"><img src="https://img.shields.io/pypi/pyversions/geoews.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/vonixxxxx/geoews/actions"><img src="https://github.com/vonixxxxx/geoews/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <!-- <a href="https://geoews.readthedocs.io"><img src="https://readthedocs.org/projects/geoews/badge/?version=latest" alt="Documentation"></a> -->
  <!-- <a href="https://doi.org/10.5281/zenodo.XXXXXXX"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg" alt="DOI"></a> -->
</p>

---

## Why geoews?

Classical early warning signals (rising variance, lag-1 autocorrelation, Kendall trend tests) detect critical slowing down - but they measure *individual* summary statistics of a time series window. **geoews** takes a fundamentally different approach: it tracks how the *entire probability distribution* of the data evolves over time, using the natural geometry of statistical manifolds.

| Feature | Classical EWS (`ewstools`, etc.) | **geoews** |
|---|---|---|
| **What is tracked** | Single statistics (variance, ACF) | Full distribution geometry |
| **Theoretical scaling** | Variance ~ \|r\|^-1/2 near bifurcation | KL rate ~ \|r\|^-2 (provably faster divergence) |
| **Geometric acceleration** | - | Geodesic acceleration on the Fisher-Rao manifold |
| **Multi-moment sensitivity** | Separate indicators for each moment | Intrinsically captures mean, variance, and higher-order shifts simultaneously |
| **Classical benchmarks** | yes | yes (built-in for direct comparison) |

The theoretical advantage is not just asymptotic: on empirical data (paleoclimate, ecology, clinical medicine), the information-geometric indicators provide earlier and more robust warnings. See our [upcoming paper](#scientific-background) for proofs and validation.

## Installation

**From PyPI** (recommended):

```bash
pip install geoews
```

**Upgrade to latest:**

```bash
pip install geoews --upgrade
```

**Development install** (from source):

```bash
git clone https://github.com/vonixxxxx/geoews.git
cd geoews
pip install -e ".[dev]"
pytest  # run test suite
```

Requires Python >= 3.10. Dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`.

## Quick start

### High-level API: `ManifoldEWS`

The `ManifoldEWS` class provides a scikit-learn-style interface - fit sliding-window Gaussians, compute all geometric indicators, and run baseline-threshold detection in three lines:

```python
import numpy as np
from geoews import ManifoldEWS

# Simulate a time series approaching a bifurcation
rng = np.random.default_rng(42)
n = 1000
r = np.linspace(1.0, 0.01, n)  # control parameter approaching zero
x = np.cumsum(rng.normal(scale=np.sqrt(1 / (2 * r))))  # OU process with diverging variance

# Fit and detect
result = ManifoldEWS(window=50, cumul_window=30).fit(x).detect()

# Access results
print(result.kl_rate)               # KL divergence rate between consecutive windows
print(result.geodesic_acceleration)  # acceleration on the Fisher-Rao manifold
print(result.alert_index)            # index where threshold is first exceeded
```

### Lower-level API (full control)

For custom pipelines or when you need direct access to the underlying computations:

```python
import numpy as np
from geoews.windows import estimate_gaussian_params
from geoews.indicators import kl_rate, fisher_rao_distance, geodesic_acceleration

# Your time series data
x = np.loadtxt("my_data.csv")

# Step 1: Fit sliding-window Gaussians
times, mus, sigmas = estimate_gaussian_params(x, window_size=50, step=1)

# Step 2: Compute geometric indicators
kl = kl_rate(mus, sigmas)                                   # KL divergence rate
fr = fisher_rao_distance(mus, sigmas)                        # Fisher-Rao step distances
acc = geodesic_acceleration(mus, sigmas, cumul_window=30)    # geodesic acceleration

# Step 3: Compare with classical EWS
from geoews import variance_ews, acf_ews
times_v, var_series = variance_ews(x, window=50, step=1)
times_a, acf_series = acf_ews(x, window=50, step=1)
```

### Built-in classical benchmarks

geoews includes classical EWS for direct head-to-head comparisons:

```python
from geoews import variance_ews, acf_ews

times_v, var = variance_ews(x, window=50, step=1)
times_a, acf = acf_ews(x, window=50, step=1)
```

## Examples

The [`examples/`](https://github.com/vonixxxxx/geoews/tree/main/examples) directory contains Jupyter notebooks demonstrating geoews on real-world data:

- **Peter Lake** - detecting regime shifts in a whole-lake manipulation experiment (ecology)
- **PhysioNet Sepsis** - early prediction of sepsis onset from clinical vital signs (medicine)

<!-- Additional notebooks planned for NGRIP ice core (paleoclimate) and EEG seizure prediction. -->

## Scientific background

geoews implements the theoretical framework developed in:

> **Information-geometric early warning signals for critical transitions**
> Alexander Sokol (2026). *In preparation.*
> <!-- [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) -->

**Core idea.** Given a time series, geoews fits a Gaussian distribution N(mu_t, sigma_t^2) to each sliding window. The sequence of fitted distributions traces a curve on the 2D Gaussian statistical manifold, equipped with the Fisher information metric. As the system approaches a bifurcation:

1. **KL divergence rate** between consecutive windows diverges as |r|^-2, provably faster than the classical variance scaling of |r|^-1/2.
2. **Fisher-Rao distance** (the geodesic distance on the statistical manifold) captures simultaneous changes in both mean and variance in a single, geometrically natural scalar.
3. **Geodesic acceleration** detects *changes in the rate of change* - a second-order signal that can flag an approaching tipping point even before first-order indicators rise appreciably.

Regularization: all covariance estimates use a diagonal ridge of epsilon = 10^-6 (`geoews.windows.COVARIANCE_REGULARIZATION`) for numerical stability.

## API reference

### Core classes

| Class / Function | Module | Description |
|---|---|---|
| `ManifoldEWS` | `geoews` | High-level fit -> detect pipeline |
| `EWSResult` | `geoews` | Structured result container |

### Geometric indicators

| Function | Module | Description |
|---|---|---|
| `kl_rate` / `kl_divergence_rate` | `geoews.indicators` | KL divergence rate D(p_t || p_t-1) |
| `fisher_rao_distance` | `geoews.indicators` | Geodesic distance on the Gaussian manifold |
| `geodesic_acceleration` | `geoews.indicators` | Second derivative of the manifold trajectory |

### Classical EWS

| Function | Module | Description |
|---|---|---|
| `variance_ews` | `geoews` | Rolling variance |
| `acf_ews` | `geoews` | Lag-1 autocorrelation |

### Windowing

| Function / Constant | Module | Description |
|---|---|---|
| `estimate_gaussian_params` | `geoews.windows` | Sliding-window Gaussian MLE |
| `COVARIANCE_REGULARIZATION` | `geoews.windows` | Ridge constant (default 1e-6) |

### Data loaders

| Function | Module | Description |
|---|---|---|
| `load_peter_lake` | `geoews.data` | Load Peter Lake dataset (requires local file) |
| `load_ngrip` | `geoews.data` | Load NGRIP ice core dataset (requires local file) |

<!-- Full API documentation: [geoews.readthedocs.io](https://geoews.readthedocs.io) -->

## Comparison with ewstools

geoews is designed to complement, not replace, [ewstools](https://github.com/ThomasMBury/ewstools). The two packages address different layers of the EWS stack:

- **ewstools** provides a comprehensive classical EWS toolbox with detrending, spectral EWS, deep learning classifiers, and visualization - a mature, JOSS-published package.
- **geoews** introduces a new *class* of indicators based on information geometry, with a theoretical basis for earlier detection. It includes classical benchmarks so you can compare directly.

A typical workflow might use both: run `ewstools` for classical analysis and deep learning classifiers, then run `geoews` for geometric indicators that may detect the transition earlier.

## Citation

If you use geoews in your research, please cite:

```bibtex
@software{sokol2026geoews,
  author    = {Sokol, Alexander},
  title     = {geoews: Information-geometric early warning signals},
  year      = {2026},
  url       = {https://github.com/vonixxxxx/geoews},
  version   = {0.2.0},
  license   = {MIT}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata. When citing the underlying theory, please also cite the accompanying paper (reference to be added upon publication).

## Roadmap

- [ ] ReadTheDocs documentation with full API reference and tutorials
- [ ] Publication-quality example notebooks with ewstools side-by-side comparisons
- [ ] Multivariate extension (matrix Fisher-Rao geometry)
- [ ] Spectral EWS on the manifold (power spectrum curvature)
- [ ] Zenodo DOI and archival release
- [ ] arXiv preprint link
- [ ] JOSS submission

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

```bash
git clone https://github.com/vonixxxxx/geoews.git
cd geoews
pip install -e ".[dev]"
pytest                # run tests
```

## License

MIT - see [`LICENSE`](LICENSE).
