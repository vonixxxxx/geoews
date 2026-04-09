"""geoews: Information-geometric early warning signals for critical transitions."""

from geoews._version import __version__
from geoews.classical import acf_ews, variance_ews
from geoews.core import EWSResult, ManifoldEWS
from geoews.indicators import (
    fisher_rao_distance,
    geodesic_acceleration,
    kl_divergence_rate,
    kl_rate,
)
from geoews.windows import COVARIANCE_REGULARIZATION, estimate_gaussian_params

__all__ = [
    "ManifoldEWS",
    "EWSResult",
    "variance_ews",
    "acf_ews",
    "kl_rate",
    "kl_divergence_rate",
    "fisher_rao_distance",
    "geodesic_acceleration",
    "estimate_gaussian_params",
    "COVARIANCE_REGULARIZATION",
    "__version__",
]
