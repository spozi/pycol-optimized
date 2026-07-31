"""Combined, shared-computation API for optimized PyCOL metrics."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .geometry import build_geometry, fixed_range_scale
from .knn import build_knn_graph
from .metrics import MetricResult, compute_f1, compute_neighbor_metrics
from .n1 import minimum_spanning_forest, score_n1

SUPPORTED_METRICS = ("F1", "N1", "N3", "kDN", "CM", "C1")
NEIGHBOR_METRICS = frozenset({"N3", "kDN", "CM", "C1"})


@dataclass(frozen=True, slots=True)
class PyCOLMetricsResult:
    """Project-compatible aggregate plus optimized-backend diagnostics."""

    metrics: dict[str, float]
    project_composite: float
    per_label: dict[str, dict[str, float]]
    label_weighted_metrics: dict[str, float]
    diagnostics: dict[str, Any]


def _merge_metric_result(
    result: MetricResult,
    *,
    requested: set[str],
    metrics: dict[str, float],
    per_label: dict[str, dict[str, float]],
    weighted: dict[str, float],
) -> None:
    for name, value in result.metrics.items():
        if name in requested:
            metrics[name] = float(value)
    for label, values in result.per_label.items():
        target = per_label.setdefault(label, {})
        for name, value in values.items():
            if name in requested or name.startswith("N3_"):
                target[name] = float(value)
    for name, value in result.label_weighted_metrics.items():
        if name in requested:
            weighted[name] = float(value)


def compute_metrics(
    vectors: NDArray[Any],
    labels: NDArray[Any],
    *,
    metrics: Sequence[str] = SUPPORTED_METRICS,
    neighbors: int = 5,
    device: str | torch.device = "auto",
    multilabel: bool | None = None,
    label_names: Sequence[str] | None = None,
    reference_min: NDArray[Any] | None = None,
    reference_max: NDArray[Any] | None = None,
) -> PyCOLMetricsResult:
    """Compute requested PyCOL metrics with shared float32 Torch primitives.

    F1 uses class statistics and therefore bypasses pairwise distances when it
    is the only requested metric. N3, kDN, CM, and C1 share one canonical
    neighbor graph. N1 reuses the same distance matrix when requested with the
    neighbor family.
    """

    started = time.perf_counter()
    requested_order = tuple(str(name) for name in metrics)
    if not requested_order:
        raise ValueError("metrics cannot be empty")
    if len(set(requested_order)) != len(requested_order):
        raise ValueError("metrics cannot contain duplicates")
    unsupported = set(requested_order) - set(SUPPORTED_METRICS)
    if unsupported:
        raise ValueError(
            f"Unsupported metrics {sorted(unsupported)}; choose from {SUPPORTED_METRICS}"
        )
    requested = set(requested_order)

    matrix = np.asarray(vectors)
    target = np.asarray(labels)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")
    if target.ndim not in {1, 2}:
        raise ValueError("labels must be one- or two-dimensional")
    if len(matrix) != len(target):
        raise ValueError("vectors and labels must contain the same number of samples")
    inferred_multilabel = target.ndim == 2
    use_multilabel = inferred_multilabel if multilabel is None else bool(multilabel)
    if use_multilabel != inferred_multilabel:
        expected = "two-dimensional" if use_multilabel else "one-dimensional"
        raise ValueError(f"labels must be {expected} for multilabel={use_multilabel}")
    names = None if label_names is None else tuple(str(name) for name in label_names)
    if names is not None and use_multilabel and len(names) != target.shape[1]:
        raise ValueError("label_names must match the number of multilabel columns")

    needs_neighbors = bool(requested & NEIGHBOR_METRICS)
    if needs_neighbors and (neighbors < 1 or neighbors >= len(matrix)):
        raise ValueError("neighbors must satisfy 1 <= neighbors < number of samples")

    geometry = None
    geometry_diagnostics: dict[str, Any] = {}
    if needs_neighbors or "N1" in requested:
        geometry = build_geometry(
            matrix,
            reference_min=reference_min,
            reference_max=reference_max,
            device=device,
            return_cpu_distances="N1" in requested,
        )
        scaled = geometry.scaled_device
        geometry_diagnostics = dict(geometry.diagnostics)
    else:
        scaled = fixed_range_scale(
            matrix,
            reference_min=reference_min,
            reference_max=reference_max,
        )

    values: dict[str, float] = {}
    per_label: dict[str, dict[str, float]] = {}
    weighted: dict[str, float] = {}
    component_diagnostics: dict[str, Any] = {}
    top_level_scientific_diagnostics: dict[str, Any] = {}

    if "F1" in requested:
        f1_result = compute_f1(
            scaled,
            target,
            device=device if geometry is None else str(geometry.device),
            multilabel=use_multilabel,
            label_names=names,
        )
        _merge_metric_result(
            f1_result,
            requested=requested,
            metrics=values,
            per_label=per_label,
            weighted=weighted,
        )
        component_diagnostics["F1"] = f1_result.diagnostics

    if needs_neighbors:
        assert geometry is not None
        maximum_k = neighbors if requested & {"kDN", "CM", "C1"} else 1
        graph = build_knn_graph(geometry, max_k=maximum_k, return_distances=False)
        neighbor_names = tuple(name for name in requested_order if name in NEIGHBOR_METRICS)
        neighbor_result = compute_neighbor_metrics(
            graph.indices,
            target,
            metrics=neighbor_names,
            neighbors=maximum_k,
            device=str(geometry.device),
            multilabel=use_multilabel,
            label_names=names,
        )
        _merge_metric_result(
            neighbor_result,
            requested=requested,
            metrics=values,
            per_label=per_label,
            weighted=weighted,
        )
        component_diagnostics["neighbors"] = {
            "graph": graph.diagnostics,
            "metrics": neighbor_result.diagnostics,
        }
        for key in ("N3_micro", "N3_micro_label_macro"):
            if key in neighbor_result.diagnostics:
                top_level_scientific_diagnostics[key] = neighbor_result.diagnostics[key]

    if "N1" in requested:
        assert geometry is not None
        distances = geometry.distances_cpu
        if distances is None:
            raise RuntimeError("N1 requested without a CPU distance matrix")
        edges = minimum_spanning_forest(distances)
        n1_score = score_n1(
            edges,
            target,
            multilabel=use_multilabel,
            label_names=None if names is None else list(names),
        )
        values["N1"] = float(n1_score.value)
        if use_multilabel:
            for label, value in n1_score.per_label.items():
                per_label.setdefault(label, {})["N1"] = float(value)
            if n1_score.label_weighted_value is not None:
                weighted["N1"] = float(n1_score.label_weighted_value)
        component_diagnostics["N1"] = {
            "edge_count": int(len(edges)),
            "component_count": int(len(matrix) - len(edges)),
            "mst_build_count": 1,
        }

    missing = requested - set(values)
    if missing:
        raise RuntimeError(f"Optimized backend did not compute requested metrics {sorted(missing)}")
    ordered_values = {name: values[name] for name in requested_order}
    total_ms = (time.perf_counter() - started) * 1_000.0
    resolved_device = (
        str(geometry.device)
        if geometry is not None
        else str(component_diagnostics.get("F1", {}).get("device", device))
    )
    diagnostics: dict[str, Any] = {
        **top_level_scientific_diagnostics,
        "backend": "pycol_optimized_float32",
        "device_requested": str(device),
        "device": resolved_device,
        "dtype": "float32",
        "geometry_build_count": int(geometry is not None),
        "knn_build_count": int(needs_neighbors),
        "metrics_requested": list(requested_order),
        "geometry": geometry_diagnostics,
        "components": component_diagnostics,
        "timings_ms": {"total": total_ms},
    }
    return PyCOLMetricsResult(
        metrics=ordered_values,
        project_composite=float(np.mean(list(ordered_values.values()))),
        per_label=per_label,
        label_weighted_metrics=weighted,
        diagnostics=diagnostics,
    )
