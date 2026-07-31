"""Deterministic device-side k-nearest-neighbour graph construction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from .geometry import DistanceResult, GeometryResult, synchronize_device

DistanceSource: TypeAlias = GeometryResult | DistanceResult | torch.Tensor | NDArray[Any]

KNN_TIE_POLICY = "distance_then_smallest_sample_index"


@dataclass(frozen=True, slots=True)
class KNNGraph:
    """Ordered nearest neighbours stored on the distance matrix's device."""

    indices: torch.Tensor
    distances: torch.Tensor | None
    diagnostics: dict[str, Any]

    @property
    def device(self) -> torch.device:
        """Return the device on which the graph tensors reside."""

        return self.indices.device

    @property
    def max_k(self) -> int:
        """Return the number of ordered neighbours retained per sample."""

        return int(self.indices.shape[1])


def _distance_tensor(source: DistanceSource) -> torch.Tensor:
    if isinstance(source, (GeometryResult, DistanceResult)):
        tensor = source.distances_device
    elif isinstance(source, torch.Tensor):
        tensor = source
    else:
        matrix = np.asarray(source)
        if not np.issubdtype(matrix.dtype, np.number):
            raise TypeError("distances must contain numerical values")
        tensor = torch.as_tensor(np.ascontiguousarray(matrix, dtype=np.float32))

    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("distances must be a square matrix")
    if tensor.shape[0] < 2:
        raise ValueError("kNN requires at least two samples")
    if tensor.is_complex():
        raise TypeError("distances must be real-valued")
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    tensor = tensor.detach()

    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("distances must contain only finite values")
    if bool((tensor < 0.0).any().item()):
        raise ValueError("distances cannot contain negative values")
    return tensor


def build_knn_graph(
    source: DistanceSource,
    max_k: int,
    *,
    return_distances: bool = True,
) -> KNNGraph:
    """Build an ordered dense kNN graph with PyCOL-compatible tie handling.

    The diagonal alone is excluded. Off-diagonal zero-distance duplicates are
    valid neighbours. Repeated row-wise ``argmin`` returns the first occurrence
    of an exact minimum, so sample index is the deterministic secondary key.
    """

    if isinstance(max_k, bool) or not isinstance(max_k, int):
        raise TypeError("max_k must be an integer")

    distances = _distance_tensor(source)
    n_samples = int(distances.shape[0])
    if max_k < 1 or max_k >= n_samples:
        raise ValueError("max_k must satisfy 1 <= max_k < number of samples")

    device = distances.device
    synchronize_device(device)
    start = time.perf_counter()

    work = distances.clone()
    work.fill_diagonal_(torch.inf)
    indices = torch.empty((n_samples, max_k), dtype=torch.long, device=device)

    for rank in range(max_k):
        nearest = torch.argmin(work, dim=1)
        indices[:, rank] = nearest
        work.scatter_(1, nearest.unsqueeze(1), torch.inf)

    selected_distances = torch.gather(distances, 1, indices) if return_distances else None
    synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1_000.0

    return KNNGraph(
        indices=indices,
        distances=selected_distances,
        diagnostics={
            "algorithm": "repeated_rowwise_argmin_scatter",
            "tie_policy": KNN_TIE_POLICY,
            "self_exclusion": "diagonal_only",
            "zero_distance_duplicates_retained": True,
            "device": str(device),
            "dtype": "float32",
            "n_samples": n_samples,
            "max_k": max_k,
            "returned_distances": return_distances,
            "timings_ms": {"knn_graph": elapsed_ms},
        },
    )


def build_knn_graph_blockwise(
    scaled_vectors: NDArray[Any] | torch.Tensor,
    max_k: int,
    *,
    block_size: int,
    device: str | torch.device = "auto",
) -> KNNGraph:
    """Reserved exact blockwise implementation for cohorts exceeding dense memory."""

    del scaled_vectors, max_k, block_size, device
    raise NotImplementedError(
        "Exact blockwise kNN is a later phase; use build_geometry and build_knn_graph "
        "for the validated dense implementation"
    )
