"""Pinned scientific-reference adapter for PyCOL 1.0.4 N1."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .n1 import PYCOL_EDGE_CUTOFF, N1Result, score_n1


def _fixed_range_scale_float64(
    vectors: NDArray[Any],
    *,
    reference_min: NDArray[Any] | None,
    reference_max: NDArray[Any] | None,
) -> NDArray[np.float64]:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("vectors must have shape [at least 2 samples, at least 1 feature]")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must contain only finite values")
    if (reference_min is None) != (reference_max is None):
        raise ValueError("reference_min and reference_max must be supplied together")
    lower = (
        matrix.min(axis=0) if reference_min is None else np.asarray(reference_min, dtype=np.float64)
    )
    upper = (
        matrix.max(axis=0) if reference_max is None else np.asarray(reference_max, dtype=np.float64)
    )
    if lower.shape != (matrix.shape[1],) or upper.shape != (matrix.shape[1],):
        raise ValueError("reference bounds must match the embedding width")
    widths = upper - lower
    safe_widths = np.where(widths > 0.0, widths, 1.0)
    scaled = (matrix - lower) / safe_widths
    scaled[:, widths <= 0.0] = 0.0
    return scaled


def _official_scalar_n1(distances: NDArray[np.float64], labels: NDArray[Any]) -> float:
    try:
        from pycol_complexity.complexity import Complexity
    except ImportError as error:  # pragma: no cover - explicit dependency failure
        raise ImportError(
            "The scientific reference requires the 'reference' extra: "
            "pip install 'pycol-optimized[reference]'"
        ) from error

    target = np.asarray(labels)
    engine = Complexity.__new__(Complexity)
    engine.y = target
    engine.classes, counts = np.unique(target, return_counts=True)
    engine.class_count = counts.astype(float)
    engine.dist_matrix = distances
    engine.metrics = {"feature": {}, "struct": {}, "instance": {}, "multi": {}}
    return float(engine.N1())


def compute_n1_reference(
    vectors: NDArray[Any],
    labels: NDArray[Any],
    *,
    edge_cutoff: float = PYCOL_EDGE_CUTOFF,
    multilabel: bool | None = None,
    label_names: list[str] | tuple[str, ...] | None = None,
    reference_min: NDArray[Any] | None = None,
    reference_max: NDArray[Any] | None = None,
) -> N1Result:
    """Run the official float64 PyCOL/SciPy N1 scientific reference."""

    try:
        from scipy.sparse.csgraph import (
            minimum_spanning_tree as scipy_minimum_spanning_tree,
        )
        from scipy.spatial.distance import cdist
    except ImportError as error:  # pragma: no cover - explicit dependency failure
        raise ImportError(
            "The scientific reference requires the 'reference' extra: "
            "pip install 'pycol-optimized[reference]'"
        ) from error

    if edge_cutoff != PYCOL_EDGE_CUTOFF:
        raise ValueError("The official PyCOL reference has a fixed effective cutoff of 1e-8")
    start = time.perf_counter()
    scaled = _fixed_range_scale_float64(
        vectors,
        reference_min=reference_min,
        reference_max=reference_max,
    )
    target = np.asarray(labels)
    if target.ndim not in {1, 2}:
        raise ValueError("labels must be one- or two-dimensional")
    if target.shape[0] != scaled.shape[0]:
        raise ValueError("vectors and labels must contain the same number of samples")
    distance_start = time.perf_counter()
    distances = cdist(scaled, scaled, metric="euclidean")
    distance_ms = (time.perf_counter() - distance_start) * 1_000.0
    tree_start = time.perf_counter()
    tree = scipy_minimum_spanning_tree(
        np.triu(distances, k=1),
        overwrite=True,
    )
    rows, columns = tree.nonzero()
    edges = np.column_stack((rows, columns)).astype(np.int64, copy=False)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    tree_ms = (time.perf_counter() - tree_start) * 1_000.0

    inferred_multilabel = target.ndim == 2
    use_multilabel = inferred_multilabel if multilabel is None else multilabel
    score = score_n1(
        edges,
        target,
        multilabel=use_multilabel,
        label_names=label_names,
    )

    official_values: dict[str, float] = {}
    official_mst_build_count = 0
    if not use_multilabel:
        official = _official_scalar_n1(distances, target)
        official_mst_build_count = 1
        if official != score.value:
            raise RuntimeError("Independent boundary scoring disagrees with official PyCOL N1")
        official_values["scalar"] = official
    else:
        names = (
            tuple(f"label_{index}" for index in range(target.shape[1]))
            if label_names is None
            else tuple(label_names)
        )
        for index, name in enumerate(names):
            if np.unique(target[:, index]).size < 2:
                continue
            official = _official_scalar_n1(distances, target[:, index])
            official_mst_build_count += 1
            if official != score.per_label[str(name)]:
                raise RuntimeError(
                    f"Independent boundary scoring disagrees with official PyCOL N1 for {name}"
                )
            official_values[str(name)] = official

    total_ms = (time.perf_counter() - start) * 1_000.0
    component_count = int(len(scaled) - len(edges))
    return N1Result(
        value=score.value,
        boundary_count=score.boundary_count,
        boundary_mask=score.boundary_mask,
        edges=edges,
        per_label=score.per_label,
        label_weighted_value=score.label_weighted_value,
        diagnostics={
            "backend": "pycol_reference",
            "algorithm": "pycol_1.0.4_scipy_float64_kruskal",
            "device": "cpu",
            "dtype": "float64",
            "n_samples": int(scaled.shape[0]),
            "n_features": int(scaled.shape[1]),
            "edge_cutoff": PYCOL_EDGE_CUTOFF,
            "component_count": component_count,
            "edge_count": int(len(edges)),
            "range_source": "input" if reference_min is None else "provided",
            "official_pycol_values": official_values,
            # One structural extraction above, plus one official PyCOL MST per
            # scalar/valid label used to verify the independent score.
            "mst_build_count": 1 + official_mst_build_count,
            "timings_ms": {
                "distance": distance_ms,
                "minimum_spanning_forest": tree_ms,
                "total": total_ms,
            },
        },
    )
