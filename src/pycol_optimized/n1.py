"""Float32 PyTorch acceleration for PyCOL's N1 data-complexity measure.

The implementation intentionally keeps the two parts of N1 separate:

* PyTorch computes the dense pairwise Euclidean distance matrix on CPU, MPS,
  or CUDA.
* A deterministic NumPy Prim routine computes the minimum spanning forest on
  CPU using the same total edge order as PyCOL's upper-triangle SciPy/Kruskal
  reference.

This hybrid is substantially faster than launching one small GPU kernel for
each of Prim's sequential iterations at the cohort sizes used by this project.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from .geometry import (
    fixed_range_scale,
    resolve_device,
)
from .geometry import (
    synchronize_device as _synchronize,
)

Float32Array: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.integer[Any]]
BoundaryCount: TypeAlias = int | dict[str, int]

PYCOL_EDGE_CUTOFF = 1.0e-8
TIE_POLICY = "distance_then_upper_triangle_edge_id"


@dataclass(frozen=True, slots=True)
class N1Score:
    """Label-dependent N1 score for a previously constructed forest."""

    value: float
    boundary_count: BoundaryCount
    boundary_mask: NDArray[np.bool_]
    per_label: dict[str, float]
    label_weighted_value: float | None
    valid_label_names: tuple[str, ...]
    skipped_label_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ForestResult:
    """Deterministic minimum-spanning-forest output."""

    edges: NDArray[np.int64]
    component_count: int
    selection_tie_count: int
    update_tie_count: int
    absent_edge_count: int


@dataclass(frozen=True, slots=True)
class N1Result:
    """Complete optimized N1 result and reproducibility metadata."""

    value: float
    boundary_count: BoundaryCount
    boundary_mask: NDArray[np.bool_]
    edges: NDArray[np.int64]
    per_label: dict[str, float]
    label_weighted_value: float | None
    diagnostics: dict[str, Any]


def _duplicate_groups(matrix: Float32Array) -> tuple[list[NDArray[np.int64]], int]:
    """Return exact float32-identical row groups without an O(n²d) broadcast."""

    row_dtype = np.dtype((np.void, matrix.dtype.itemsize * matrix.shape[1]))
    row_keys = np.ascontiguousarray(matrix).view(row_dtype).reshape(-1)
    _, inverse, counts = np.unique(row_keys, return_inverse=True, return_counts=True)
    duplicate_ids = np.flatnonzero(counts > 1)
    groups = [
        np.flatnonzero(inverse == group_id).astype(np.int64, copy=False)
        for group_id in duplicate_ids
    ]
    pair_count = int(sum(len(group) * (len(group) - 1) // 2 for group in groups))
    return groups, pair_count


def _pairwise_distances_with_metadata(
    scaled_vectors: Float32Array,
    *,
    device: str | torch.device,
) -> tuple[Float32Array, dict[str, Any]]:
    resolved = resolve_device(device)
    # torch.cdist selects its fast matrix-multiplication implementation when
    # either point count exceeds 25. Squared-distance expansion around a large
    # common centroid is cancellation-prone in float32. Centering is exactly
    # translation invariant and keeps the fast path numerically stable.
    centered_for_mm = scaled_vectors.shape[0] > 25
    if centered_for_mm:
        distance_input = scaled_vectors - scaled_vectors.mean(
            axis=0,
            dtype=np.float32,
            keepdims=True,
        )
        distance_input = np.ascontiguousarray(distance_input, dtype=np.float32)
    else:
        distance_input = scaled_vectors

    transfer_start = time.perf_counter()
    tensor = torch.as_tensor(distance_input, dtype=torch.float32, device=resolved)
    _synchronize(resolved)
    input_transfer_ms = (time.perf_counter() - transfer_start) * 1_000.0

    distance_start = time.perf_counter()
    distances_tensor = torch.cdist(
        tensor,
        tensor,
        p=2.0,
        compute_mode="use_mm_for_euclid_dist_if_necessary",
    )
    distances_tensor.fill_diagonal_(0.0)
    _synchronize(resolved)
    distance_ms = (time.perf_counter() - distance_start) * 1_000.0

    output_start = time.perf_counter()
    distances = np.ascontiguousarray(distances_tensor.detach().cpu().numpy(), dtype=np.float32)
    groups, duplicate_pair_count = _duplicate_groups(scaled_vectors)
    for group in groups:
        distances[np.ix_(group, group)] = 0.0
    np.fill_diagonal(distances, 0.0)
    output_transfer_and_duplicate_fix_ms = (time.perf_counter() - output_start) * 1_000.0

    return distances, {
        "device": resolved.type,
        "input_transfer_ms": input_transfer_ms,
        "distance_ms": distance_ms,
        "output_transfer_and_duplicate_fix_ms": output_transfer_and_duplicate_fix_ms,
        "mean_centered_for_mm_cdist": centered_for_mm,
        "exact_duplicate_group_count": len(groups),
        "exact_duplicate_pair_count": duplicate_pair_count,
    }


def pairwise_distances(
    scaled_vectors: NDArray[Any],
    *,
    device: str | torch.device = "auto",
) -> Float32Array:
    """Compute a float32 pairwise Euclidean matrix with exact-duplicate repair."""

    matrix = np.ascontiguousarray(scaled_vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("scaled_vectors must be two-dimensional")
    if matrix.shape[0] < 2:
        raise ValueError("pairwise distances require at least two samples")
    if matrix.shape[1] < 1:
        raise ValueError("scaled_vectors must contain at least one feature")
    if not np.isfinite(matrix).all():
        raise ValueError("scaled_vectors must contain only finite values")
    distances, _ = _pairwise_distances_with_metadata(matrix, device=device)
    return distances


def _minimum_spanning_forest_with_metadata(
    distances: NDArray[Any],
    *,
    edge_cutoff: float,
) -> ForestResult:
    matrix = np.asarray(distances)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distances must be a square matrix")
    if matrix.shape[0] < 2:
        raise ValueError("N1 requires at least two samples")
    if not np.issubdtype(matrix.dtype, np.floating):
        matrix = matrix.astype(np.float64)
    if np.any(matrix[np.isfinite(matrix)] < 0.0):
        raise ValueError("distances cannot contain negative finite values")
    if edge_cutoff < 0.0 or not np.isfinite(edge_cutoff):
        raise ValueError("edge_cutoff must be finite and non-negative")

    n_samples = matrix.shape[0]
    selected = np.zeros(n_samples, dtype=bool)
    best_distance = np.full(n_samples, np.inf, dtype=matrix.dtype)
    best_edge_id = np.full(n_samples, np.iinfo(np.int64).max, dtype=np.int64)
    parent = np.full(n_samples, -1, dtype=np.int64)
    edges: list[tuple[int, int]] = []
    component_count = 0
    selection_tie_count = 0
    update_tie_count = 0
    absent_edge_count = 0

    for _ in range(n_samples):
        remaining = np.flatnonzero(~selected)
        remaining_weights = best_distance[remaining]
        minimum_weight = remaining_weights.min()

        if not np.isfinite(minimum_weight):
            # The smallest remaining vertex is the deterministic root of a new
            # forest component.
            selected_vertex = int(remaining[0])
            component_count += 1
        else:
            weight_candidates = remaining[remaining_weights == minimum_weight]
            if len(weight_candidates) > 1:
                selection_tie_count += 1
            candidate_edge_ids = best_edge_id[weight_candidates]
            minimum_edge_id = candidate_edge_ids.min()
            edge_candidates = weight_candidates[candidate_edge_ids == minimum_edge_id]
            selected_vertex = int(edge_candidates[0])

        selected[selected_vertex] = True
        selected_parent = int(parent[selected_vertex])
        if selected_parent >= 0:
            edges.append(
                (
                    min(selected_parent, selected_vertex),
                    max(selected_parent, selected_vertex),
                )
            )

        candidates = np.flatnonzero(~selected)
        if len(candidates) == 0:
            continue
        rows = np.minimum(selected_vertex, candidates)
        columns = np.maximum(selected_vertex, candidates)
        weights = matrix[rows, columns]
        edge_ids = rows.astype(np.int64) * n_samples + columns.astype(np.int64)
        valid = np.isfinite(weights) & (weights > edge_cutoff)
        # Every unordered pair is visited exactly once: when its first
        # endpoint enters the forest and the other endpoint is unselected.
        absent_edge_count += int(len(valid) - np.count_nonzero(valid))
        equal_distance_better_edge = (
            valid & (weights == best_distance[candidates]) & (edge_ids < best_edge_id[candidates])
        )
        update_tie_count += int(np.count_nonzero(equal_distance_better_edge))
        improves = valid & ((weights < best_distance[candidates]) | equal_distance_better_edge)
        improved_vertices = candidates[improves]
        best_distance[improved_vertices] = weights[improves]
        best_edge_id[improved_vertices] = edge_ids[improves]
        parent[improved_vertices] = selected_vertex

    edge_array = np.asarray(edges, dtype=np.int64)
    if edge_array.size == 0:
        edge_array = np.empty((0, 2), dtype=np.int64)
    else:
        edge_array = edge_array.reshape(-1, 2)
    return ForestResult(
        edges=edge_array,
        component_count=component_count,
        selection_tie_count=selection_tie_count,
        update_tie_count=update_tie_count,
        absent_edge_count=absent_edge_count,
    )


def minimum_spanning_forest(
    distances: NDArray[Any],
    *,
    edge_cutoff: float = PYCOL_EDGE_CUTOFF,
) -> NDArray[np.int64]:
    """Construct the PyCOL-compatible deterministic minimum spanning forest."""

    return _minimum_spanning_forest_with_metadata(
        distances,
        edge_cutoff=edge_cutoff,
    ).edges


def score_n1(
    edges: NDArray[Any],
    labels: NDArray[Any],
    *,
    multilabel: bool | None = None,
    label_names: list[str] | tuple[str, ...] | None = None,
) -> N1Score:
    """Score one label vector or all valid multilabel columns on one forest."""

    target = np.asarray(labels)
    edge_array = np.asarray(edges, dtype=np.int64)
    if edge_array.size == 0:
        edge_array = np.empty((0, 2), dtype=np.int64)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edges must have shape [edge_count, 2]")
    if target.ndim not in {1, 2}:
        raise ValueError("labels must be one- or two-dimensional")
    if target.shape[0] < 2:
        raise ValueError("N1 requires at least two samples")
    if len(edge_array) and (edge_array.min() < 0 or edge_array.max() >= target.shape[0]):
        raise ValueError("edges contain a sample index outside labels")

    inferred_multilabel = target.ndim == 2
    if multilabel is None:
        multilabel = inferred_multilabel
    if multilabel != inferred_multilabel:
        expected = "two-dimensional" if multilabel else "one-dimensional"
        raise ValueError(f"labels must be {expected} for multilabel={multilabel}")

    n_samples = target.shape[0]
    if not multilabel:
        if np.unique(target).size < 2:
            raise ValueError("N1 requires at least two target classes")
        boundary = np.zeros(n_samples, dtype=bool)
        if len(edge_array):
            cross_label = target[edge_array[:, 0]] != target[edge_array[:, 1]]
            cross_edges = edge_array[cross_label]
            boundary[cross_edges.reshape(-1)] = True
        count = int(boundary.sum())
        return N1Score(
            value=count / n_samples,
            boundary_count=count,
            boundary_mask=boundary,
            per_label={},
            label_weighted_value=None,
            valid_label_names=(),
            skipped_label_names=(),
        )

    n_labels = target.shape[1]
    if label_names is None:
        names = tuple(f"label_{index}" for index in range(n_labels))
    else:
        names = tuple(str(name) for name in label_names)
        if len(names) != n_labels:
            raise ValueError("label_names must match the number of label columns")

    valid_indices = [index for index in range(n_labels) if np.unique(target[:, index]).size >= 2]
    if not valid_indices:
        raise ValueError("No multilabel target column contains at least two classes")
    valid_names = tuple(names[index] for index in valid_indices)
    skipped_names = tuple(names[index] for index in range(n_labels) if index not in valid_indices)
    valid_target = target[:, valid_indices]
    boundary = np.zeros((n_samples, len(valid_indices)), dtype=bool)

    if len(edge_array):
        different = valid_target[edge_array[:, 0], :] != valid_target[edge_array[:, 1], :]
        edge_indices, label_indices = np.nonzero(different)
        boundary[edge_array[edge_indices, 0], label_indices] = True
        boundary[edge_array[edge_indices, 1], label_indices] = True

    counts = boundary.sum(axis=0).astype(np.int64)
    values = counts.astype(np.float64) / n_samples
    per_label = {name: float(value) for name, value in zip(valid_names, values, strict=True)}
    count_mapping = {name: int(count) for name, count in zip(valid_names, counts, strict=True)}
    positive_counts = np.count_nonzero(valid_target, axis=0).astype(np.float64)
    positive_total = float(positive_counts.sum())
    weighted = (
        float(np.average(values, weights=positive_counts))
        if positive_total > 0.0
        else float(values.mean())
    )
    return N1Score(
        value=float(values.mean()),
        boundary_count=count_mapping,
        boundary_mask=boundary,
        per_label=per_label,
        label_weighted_value=weighted,
        valid_label_names=valid_names,
        skipped_label_names=skipped_names,
    )


def compute_n1(
    vectors: NDArray[Any],
    labels: NDArray[Any],
    *,
    device: str | torch.device = "auto",
    backend: str = "torch_numpy",
    edge_cutoff: float = PYCOL_EDGE_CUTOFF,
    multilabel: bool | None = None,
    label_names: list[str] | tuple[str, ...] | None = None,
    reference_min: NDArray[Any] | None = None,
    reference_max: NDArray[Any] | None = None,
) -> N1Result:
    """Compute float32 PyCOL-compatible N1 using PyTorch plus NumPy.

    Compatibility is exact with PyCOL for the *same float32 distance matrix*.
    It is not guaranteed to equal PyCOL's float64 result when casting changes
    distance order or collapses distinct observations.
    """

    if backend != "torch_numpy":
        raise ValueError("The float32 package currently supports backend='torch_numpy' only")
    total_start = time.perf_counter()
    requested_device = str(device)
    vector_array = np.asarray(vectors)
    target_array = np.asarray(labels)
    if target_array.ndim not in {1, 2}:
        raise ValueError("labels must be one- or two-dimensional")
    if vector_array.ndim == 2 and target_array.shape[0] != vector_array.shape[0]:
        raise ValueError("vectors and labels must contain the same number of samples")

    scaling_start = time.perf_counter()
    scaled = fixed_range_scale(
        vectors,
        reference_min=reference_min,
        reference_max=reference_max,
    )
    scaling_ms = (time.perf_counter() - scaling_start) * 1_000.0

    distances, distance_metadata = _pairwise_distances_with_metadata(
        scaled,
        device=device,
    )

    forest_start = time.perf_counter()
    forest = _minimum_spanning_forest_with_metadata(
        distances,
        edge_cutoff=edge_cutoff,
    )
    forest_ms = (time.perf_counter() - forest_start) * 1_000.0

    scoring_start = time.perf_counter()
    score = score_n1(
        forest.edges,
        target_array,
        multilabel=multilabel,
        label_names=label_names,
    )
    scoring_ms = (time.perf_counter() - scoring_start) * 1_000.0

    diagnostics: dict[str, Any] = {
        "backend": backend,
        "algorithm": "torch_cdist_float32_plus_numpy_lexicographic_prim",
        "device_requested": requested_device,
        "device": distance_metadata["device"],
        "dtype": "float32",
        "n_samples": int(scaled.shape[0]),
        "n_features": int(scaled.shape[1]),
        "edge_cutoff": float(edge_cutoff),
        "tie_policy": TIE_POLICY,
        "component_count": forest.component_count,
        "edge_count": int(len(forest.edges)),
        "selection_tie_count": forest.selection_tie_count,
        "update_tie_count": forest.update_tie_count,
        "absent_or_zero_upper_edge_count": forest.absent_edge_count,
        "mean_centered_for_mm_cdist": distance_metadata["mean_centered_for_mm_cdist"],
        "exact_duplicate_group_count": distance_metadata["exact_duplicate_group_count"],
        "exact_duplicate_pair_count": distance_metadata["exact_duplicate_pair_count"],
        "range_source": "input" if reference_min is None else "provided",
        "valid_label_names": list(score.valid_label_names),
        "skipped_label_names": list(score.skipped_label_names),
        "mst_build_count": 1,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "timings_ms": {
            "scaling": scaling_ms,
            "input_transfer": distance_metadata["input_transfer_ms"],
            "distance": distance_metadata["distance_ms"],
            "output_transfer_and_duplicate_fix": distance_metadata[
                "output_transfer_and_duplicate_fix_ms"
            ],
            "minimum_spanning_forest": forest_ms,
            "scoring": scoring_ms,
            "total": 0.0,
        },
    }
    diagnostics["timings_ms"]["total"] = (time.perf_counter() - total_start) * 1_000.0
    return N1Result(
        value=score.value,
        boundary_count=score.boundary_count,
        boundary_mask=score.boundary_mask,
        edges=forest.edges,
        per_label=score.per_label,
        label_weighted_value=score.label_weighted_value,
        diagnostics=diagnostics,
    )
