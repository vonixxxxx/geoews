from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from geoews import acf_ews, variance_ews
from geoews.alerts import first_crossing, percentile_threshold
from geoews.data import load_ngrip as load_ngrip_alias
from geoews.datasets import load_ngrip, load_peter_lake
from geoews.plot import plot_ews


def _write_ngrip_csv(path: Path) -> None:
    age = np.linspace(30_000, 50_000, 300)
    d18 = np.sin(np.linspace(0, 8, 300))
    pd.DataFrame({"age": age, "d18o": d18}).to_csv(path, index=False)


def _write_lake_csv(path: Path) -> None:
    dates = pd.date_range("2008-01-01", periods=420, freq="D")
    peter = pd.DataFrame(
        {
            "year": dates.year,
            "lake": "Peter",
            "lakeid": "P",
            "date": dates,
            "doy": dates.dayofyear,
            "chlorophyll": np.linspace(1.0, 5.0, len(dates)),
            "temperature": np.linspace(10.0, 20.0, len(dates)),
        }
    )
    paul = peter.copy()
    paul["lake"] = "Paul"
    pd.concat([peter, paul], ignore_index=True).to_csv(path, index=False)


def test_alert_helpers():
    x = np.array([0.1, 0.2, 0.3, 0.8, 1.2], dtype=float)
    thr = percentile_threshold(x, baseline_end=3, percentile=95.0)
    idx = first_crossing(x, thr)
    assert thr >= 0.2
    assert idx is not None


def test_classical_benchmarks_and_plot():
    x = np.sin(np.linspace(0, 12, 200))
    t_var, var = variance_ews(x, window=40, step=1)
    t_acf, acf = acf_ews(x, window=40, step=1)
    assert len(t_var) == len(var)
    assert len(t_acf) == len(acf)
    fig = plot_ews(t_var, var, title="variance")
    assert fig is not None


def test_load_ngrip_and_alias(tmp_path: Path):
    csv_path = tmp_path / "ngrip.csv"
    _write_ngrip_csv(csv_path)

    out = load_ngrip(str(csv_path))
    out_alias = load_ngrip_alias(str(csv_path))
    assert "d18o" in out
    assert len(out["time"]) > 10
    assert np.allclose(out["d18o"], out_alias["d18o"])


def test_load_peter_lake(tmp_path: Path):
    csv_path = tmp_path / "lake.csv"
    _write_lake_csv(csv_path)
    out = load_peter_lake(str(csv_path), transition_date="2009-01-15")
    assert "chlorophyll" in out
    assert len(out["chlorophyll"]) > 100
    assert out["transition_index"] is not None

