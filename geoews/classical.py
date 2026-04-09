"""Classical early-warning signal benchmarks."""

from __future__ import annotations

from .benchmarks import (
    acf_ews,
    rolling_lag1_autocorrelation,
    rolling_variance,
    variance_ews,
)

__all__ = [
    "rolling_variance",
    "rolling_lag1_autocorrelation",
    "variance_ews",
    "acf_ews",
]

