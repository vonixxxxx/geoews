#!/usr/bin/env python3
"""
Smoke-test geoews on real files (paths below point into the research repo by default).

Usage:
  python examples/test_real_data.py

Requires: pandas, numpy, geoews (pip install -e . from package root).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Default: research repo data/ (sibling layout: geometry_of_collapse/data/...)
_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
LAKE_CSV = _DEFAULT_ROOT / "carpenter_chlorophyll_temperature_2008_2019.csv"
# Long enough record with SepsisLabel turning on (for qualitative timing checks).
SEPSIS_PSV = _DEFAULT_ROOT / "physionet_sepsis" / "training_setB" / "p100016.psv"

VITALS = ("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP")


def test_peter_lake() -> None:
    from geoews import ManifoldEWS
    from geoews.datasets import load_peter_lake

    if not LAKE_CSV.is_file():
        print(f"SKIP Peter Lake: missing file {LAKE_CSV}")
        return

    data = load_peter_lake(str(LAKE_CSV))
    x = np.asarray(data["chlorophyll"], dtype=float)
    ti = data.get("transition_index")

    r = ManifoldEWS(window=100, step=1, cumul_window=30).fit(x).detect()
    i_max = int(np.argmax(r.kl_rate))
    print("\n=== Peter Lake (chlorophyll) ===")
    print(f"  n={len(x)}, transition_index={ti}")
    print(f"  ManifoldEWS: triggered={r.triggered}, threshold={r.threshold:.6g}")
    print(f"  max KL at window index {i_max} (time coord {r.times[i_max]:.1f})")
    if ti is not None:
        print(f"  |argmax KL - transition| in window indices: {abs(i_max - ti)}")


def _load_sepsis_vitals(path: Path) -> tuple[np.ndarray, int | None]:
    df = pd.read_csv(path, sep="|")
    for c in VITALS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in VITALS:
        df[c] = df[c].ffill(limit=2)
    df = df.dropna(subset=list(VITALS))
    if len(df) < 60:
        raise ValueError("Too few complete rows after ffill.")
    vit = df[list(VITALS)].to_numpy(dtype=float)
    lab = pd.to_numeric(df.get("SepsisLabel", 0), errors="coerce").fillna(0).astype(int).to_numpy()
    ti = None
    for i in range(1, len(lab)):
        if lab[i] == 1 and lab[i - 1] == 0:
            ti = i
            break
    if ti is None and (lab == 1).any():
        ti = int(np.where(lab == 1)[0][0])
    return vit, ti


def test_sepsis() -> None:
    from geoews import ManifoldEWS

    if not SEPSIS_PSV.is_file():
        print(f"SKIP Sepsis: missing file {SEPSIS_PSV}")
        return

    vit, ti = _load_sepsis_vitals(SEPSIS_PSV)
    r = ManifoldEWS(window=48, step=1, cumul_window=30).fit(vit).detect()
    i_max = int(np.argmax(r.kl_rate))
    print("\n=== PhysioNet sepsis (6 vitals, multivariate) ===")
    print(f"  file={SEPSIS_PSV.name}, T={len(vit)}, transition_index={ti}")
    print(f"  ManifoldEWS: triggered={r.triggered}, threshold={r.threshold:.6g}")
    print(f"  max KL at window index {i_max} (time coord {r.times[i_max]:.1f})")
    if ti is not None:
        w_trans = max(0, min(len(r.kl_rate) - 1, ti - 48))
        print(f"  approximate window index for onset (ti - window): {w_trans}")


def main() -> None:
    try:
        import geoews  # noqa: F401
    except ImportError:
        print("Install geoews first: pip install -e .", file=sys.stderr)
        sys.exit(1)

    print("geoews real-data smoke test")
    print(f"  default data root (if used): {_DEFAULT_ROOT}")
    test_peter_lake()
    test_sepsis()
    print("\nDone.")


if __name__ == "__main__":
    main()
