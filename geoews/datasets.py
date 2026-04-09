"""Dataset loaders for examples (require ``pandas``)."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["load_ngrip", "load_peter_lake"]

# Defaults aligned with validated research ``config.py`` / NGRIP pipeline.
NGRIP_AGE_MIN_B2K = 30_000.0
NGRIP_AGE_MAX_B2K = 50_000.0
NGRIP_GRID_STEP_YEARS = 20
NGRIP_LOWESS_FRAC = 0.1
NGRIP_DO8_AGE_B2K = 38_200.0
NGRIP_DO9_AGE_B2K = 40_200.0
NGRIP_DO10_AGE_B2K = 41_500.0


def _require_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("load_ngrip / load_peter_lake require pandas. pip install pandas") from e
    return pd


def _find_first_matching_column(df, candidates: tuple[str, ...]) -> str:
    cols_lc = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cols_lc:
            return str(cols_lc[key])
    for cand in candidates:
        sub = str(cand).strip().lower()
        for k, orig in cols_lc.items():
            if sub in k:
                return str(orig)
    raise KeyError(f"No column matching {candidates} in {list(df.columns)}")


def _read_ngrip_table(filepath: str) -> tuple[Any, str | None]:
    pd = _require_pandas()
    path_lower = filepath.lower()
    if path_lower.endswith((".xls", ".xlsx")):
        xl = pd.ExcelFile(filepath)
        preferred = "NGRIP-2 d18O and Dust"
        if preferred in xl.sheet_names:
            sheet = preferred
        elif "NGRIP-1 d18O" in xl.sheet_names:
            sheet = "NGRIP-1 d18O"
        else:
            sheet = xl.sheet_names[-1]
        df = pd.read_excel(filepath, sheet_name=sheet, header=0)
        return df, sheet
    df = pd.read_csv(filepath, sep=None, engine="python")
    return df, None


def load_ngrip(
    filepath: str,
    *,
    age_min_b2k: float = NGRIP_AGE_MIN_B2K,
    age_max_b2k: float = NGRIP_AGE_MAX_B2K,
    grid_step_years: int = NGRIP_GRID_STEP_YEARS,
    lowess_frac: float = NGRIP_LOWESS_FRAC,
    do8_age: float = NGRIP_DO8_AGE_B2K,
    do9_age: float = NGRIP_DO9_AGE_B2K,
    do10_age: float = NGRIP_DO10_AGE_B2K,
    transition_ages_b2k: list[float] | None = None,
    transition_labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load NGRIP δ¹⁸O: filter b2k window, interpolate to uniform grid, LOWESS detrend.

    Same structure as validated research ``src.data.ngrip_ice_core.load_ngrip``.
    """
    pd = _require_pandas()
    df, sheet = _read_ngrip_table(filepath)
    cols_lc = {str(c).strip().lower(): c for c in df.columns}

    def pick_age_col() -> str:
        for key, orig in cols_lc.items():
            if "gicc05" in key and "age" in key:
                return str(orig)
        return _find_first_matching_column(
            df,
            ("gicc05 age (yr b2k)", "gicc05 age", "age (b2k)", "age_b2k", "age"),
        )

    def pick_d18_col() -> str:
        for key, orig in cols_lc.items():
            if "delta" in key and "18" in key:
                return str(orig)
        return _find_first_matching_column(
            df,
            ("delta o18 (permil)", "delta18o", "d18o", "δ18o", "delta18o", "d18o"),
        )

    age_col = pick_age_col()
    d_col = pick_d18_col()
    age = pd.to_numeric(df[age_col], errors="coerce").to_numpy(dtype=float)
    d18 = pd.to_numeric(df[d_col], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(age) & np.isfinite(d18)
    age, d18 = age[m], d18[m]

    wm = (age >= age_min_b2k) & (age <= age_max_b2k)
    age, d18 = age[wm], d18[wm]
    if len(age) < 10:
        raise ValueError(f"Too few points after filtering to [{age_min_b2k}, {age_max_b2k}] b2k.")

    order = np.argsort(age)
    age, d18 = age[order], d18[order]

    df_sub = pd.DataFrame({"age": age, "d18": d18}).groupby("age", as_index=False)["d18"].mean()
    age_u = df_sub["age"].to_numpy(dtype=float)
    d18_u = df_sub["d18"].to_numpy(dtype=float)

    grid = np.arange(age_min_b2k, age_max_b2k + grid_step_years, grid_step_years, dtype=float)
    d18_raw = np.interp(grid, age_u, d18_u)

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        trend = np.asarray(lowess(d18_raw, grid, frac=lowess_frac, return_sorted=False), dtype=float)
    except ModuleNotFoundError:
        w = max(5, int(len(grid) * lowess_frac) | 1)
        pad = w // 2
        padded = np.pad(d18_raw, (pad, pad), mode="edge")
        kernel = np.ones(w, dtype=float) / w
        trend = np.convolve(padded, kernel, mode="valid")

    d18o_detrended = d18_raw - trend

    def nearest_idx(a: float) -> int:
        return int(np.argmin(np.abs(grid - float(a))))

    if transition_ages_b2k is not None:
        trans_ages = [float(a) for a in transition_ages_b2k]
        if transition_labels is not None and len(transition_labels) != len(trans_ages):
            raise ValueError("transition_labels must match transition_ages_b2k length.")
        tlabels = (
            list(transition_labels)
            if transition_labels is not None
            else [f"marker {a:.0f} b2k" for a in trans_ages]
        )
    else:
        trans_ages = [do8_age, do9_age, do10_age]
        tlabels = ["D-O 8", "D-O 9", "D-O 10"]
    trans_idx = [nearest_idx(a) for a in trans_ages]

    return {
        "time": grid,
        "d18o": d18o_detrended.astype(float),
        "d18o_raw": d18_raw.astype(float),
        "trend": trend.astype(float),
        "transition_indices": trans_idx,
        "transition_ages_b2k": trans_ages,
        "transition_labels": tlabels,
        "sheet_used": sheet,
    }


def _resample_daily_linear(series: Any) -> Any:
    _require_pandas()
    s = series.sort_index()
    daily = s.asfreq("D")
    return daily.interpolate(method="time").astype(float)


def load_peter_lake(filepath: str, *, transition_date: str = "2009-07-15") -> dict[str, Any]:
    """
    Load Peter Lake chlorophyll time series from LTER-style CSV.

    Returns keys: ``time``, ``chlorophyll``, ``dates``, ``transition_index``, ``dataset_note``.
    """
    pd = _require_pandas()
    csv_kw: dict[str, Any] = {"na_values": ["NA", "NaN", ""]}
    peek = pd.read_csv(filepath, nrows=5, **csv_kw)
    nc = {str(c).strip().lower().replace(" ", "_"): str(c) for c in peek.columns}

    if "sampledate" in nc and "abundance_pct" in nc and "lake" not in nc:
        raise NotImplementedError(
            "NTL-272 macrophyte CSV is not supported by load_peter_lake; use a Peter/Paul chlorophyll CSV."
        )

    df = pd.read_csv(filepath, **csv_kw)
    date_col = _find_first_matching_column(df, ("date", "datetime", "time", "sampledate"))
    lake_col = _find_first_matching_column(df, ("lake",))
    chl_col = _find_first_matching_column(df, ("chlorophyll", "chl", "chlorophyll_a"))

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[lake_col] = df[lake_col].astype(str)
    df[chl_col] = pd.to_numeric(df[chl_col], errors="coerce")
    df = df.dropna(subset=[date_col, lake_col, chl_col])

    transition_ts = pd.to_datetime(transition_date, utc=False)
    sub = df[df[lake_col].str.lower() == "peter"].copy()
    if sub.empty:
        raise ValueError(f"Could not find lake='Peter' in column '{lake_col}'.")

    s = sub.set_index(date_col)[chl_col]
    if isinstance(s.index, pd.DatetimeIndex) and len(s) > 1:
        delta = (s.index.sort_values()[-1] - s.index.sort_values()[0]).total_seconds() / max(len(s) - 1, 1)
        if delta < 23 * 3600:
            s = s.groupby(pd.Grouper(freq="D")).mean()

    chl_daily = _resample_daily_linear(s)
    dates = chl_daily.index
    idx = np.where(dates >= transition_ts)[0]
    return {
        "time": np.arange(len(chl_daily), dtype=int),
        "chlorophyll": chl_daily.values.astype(float),
        "dates": dates,
        "transition_index": int(idx[0]) if len(idx) > 0 else None,
        "dataset_note": "Peter Lake chlorophyll (LTER-style lake CSV).",
    }
