"""Float32 PyTorch acceleration for the project's supported PyCOL metrics."""

from .api import SUPPORTED_METRICS, PyCOLMetricsResult, compute_metrics
from .geometry import DistanceResult, GeometryResult, build_geometry, synchronize_device
from .knn import KNN_TIE_POLICY, KNNGraph, build_knn_graph
from .metrics import MetricResult, compute_f1, compute_neighbor_metrics
from .n1 import (
    PYCOL_EDGE_CUTOFF,
    TIE_POLICY,
    N1Result,
    N1Score,
    compute_n1,
    fixed_range_scale,
    minimum_spanning_forest,
    pairwise_distances,
    resolve_device,
    score_n1,
)
from .reference import compute_n1_reference

__version__ = "0.2.0"

__all__ = [
    "DistanceResult",
    "GeometryResult",
    "KNNGraph",
    "KNN_TIE_POLICY",
    "MetricResult",
    "PYCOL_EDGE_CUTOFF",
    "PyCOLMetricsResult",
    "SUPPORTED_METRICS",
    "TIE_POLICY",
    "N1Result",
    "N1Score",
    "__version__",
    "build_geometry",
    "build_knn_graph",
    "compute_f1",
    "compute_metrics",
    "compute_n1",
    "compute_n1_reference",
    "compute_neighbor_metrics",
    "fixed_range_scale",
    "minimum_spanning_forest",
    "pairwise_distances",
    "resolve_device",
    "score_n1",
    "synchronize_device",
]
