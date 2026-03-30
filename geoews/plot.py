"""Plotting helpers for geoews outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_indicator(time: np.ndarray, indicator: np.ndarray, title: str = "Indicator") -> plt.Figure:
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

