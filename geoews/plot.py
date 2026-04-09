"""Plotting helpers for geoews outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

__all__ = ["plot_indicator", "plot_ews"]

def plot_indicator(
    time: NDArray[np.float64],
    indicator: NDArray[np.float64],
    title: str = "Indicator",
) -> plt.Figure:
    """Plot a single indicator time series.

    Parameters
    ----------
    time : ndarray of shape (T,)
        Time or index values.
    indicator : ndarray of shape (T,)
        Indicator values.
    title : str, default="Indicator"
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    t = np.asarray(time, dtype=float)
    y = np.asarray(indicator, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, y, lw=1.8)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ews(
    time: NDArray[np.float64],
    indicator: NDArray[np.float64],
    title: str = "Early warning signal",
) -> plt.Figure:
    """Plot an EWS time series (notebook-friendly alias)."""
    return plot_indicator(time, indicator, title=title)

