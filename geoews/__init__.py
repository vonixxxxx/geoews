from .manifold import ManifoldEWS, EWSResult
from .indicators import kl_rate, geodesic_acceleration
from .benchmarks import variance_ews, acf_ews
from .datasets import load_ngrip, load_peter_lake
from .plot import plot_ews

__version__ = "0.1.0"
__author__ = "Alexander Sokol"

__all__ = [
    "ManifoldEWS",
    "EWSResult",
    "kl_rate",
    "geodesic_acceleration",
    "variance_ews",
    "acf_ews",
    "load_ngrip",
    "load_peter_lake",
    "plot_ews",
]
