# Changelog

## 0.2.0 - 2026-03-30

- Refactor API into `core`/`classical` modules with explicit top-level exports.
- Add frozen `EWSResult` dataclass with `to_dataframe()` and plotting helper.
- Add full type hints and upgraded NumPy-style docstrings in key API modules.
- Add dynamic versioning via `geoews._version`.
- Add docs scaffolding (`docs/`, `.readthedocs.yaml`) and CI test workflow (`tests.yml`).
- Expand tests for edge cases and geometric properties.

## 0.1.1 - 2026-03-30

- README: PyPI-first install (`pip install geoews`), high-level `ManifoldEWS` example, clearer structure and links.
- Package metadata: short description and author field aligned with the project.

## 0.1.0 - 2026-03-30

- Initial package scaffold for `geoews`.
- Added canonical implementations for:
  - `estimate_gaussian_params`
  - `kl_divergence_rate`
  - Fisher-Rao step distances
  - `geodesic_acceleration`
- Added tests, example notebooks, and publishing workflow scaffold.
