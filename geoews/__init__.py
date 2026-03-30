"""geoews: information-geometric early warning signals."""

from .indicators import geodesic_acceleration, kl_divergence_rate
from .manifold import (
    fisher_rao_distance_multivariate,
    fisher_rao_distance_univariate,
    step_distances,
)
from .windows import COVARIANCE_REGULARIZATION, estimate_gaussian_params

__all__ = [
    "COVARIANCE_REGULARIZATION",
    "estimate_gaussian_params",
    "fisher_rao_distance_univariate",
    "fisher_rao_distance_multivariate",
    "step_distances",
    "kl_divergence_rate",
    "geodesic_acceleration",
]
