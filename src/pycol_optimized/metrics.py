"""Float32 Torch implementations of PyCOL feature and neighbour metrics.

The functions in this module score already prepared geometry.  They do not
scale embeddings or construct a k-nearest-neighbour graph, which lets the
package-level API share those expensive operations across N1, N3, kDN, CM,
and C1.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .geometry import resolve_device, synchronize_device

NEIGHBOR_METRICS = ("N3", "kDN", "CM", "C1")
_F1_PAIR_CHUNK_SIZE = 4_096
_TORCH_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Project-compatible metric values plus JSON-serializable diagnostics."""

    metrics: dict[str, float]
    per_label: dict[str, dict[str, float]]
    label_weighted_metrics: dict[str, float]
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _MultilabelPlan:
    target: NDArray[np.int64]
    valid_indices: tuple[int, ...]
    valid_names: tuple[str, ...]
    skipped_names: tuple[str, ...]
    positive_counts: NDArray[np.int64]


def _as_numpy_labels(labels: NDArray[Any] | torch.Tensor) -> NDArray[Any]:
    if isinstance(labels, torch.Tensor):
        return np.asarray(labels.detach().cpu().numpy())
    return np.asarray(labels)


def _infer_multilabel(target: NDArray[Any], multilabel: bool | None) -> bool:
    if target.ndim not in {1, 2}:
        raise ValueError("labels must be one- or two-dimensional")
    inferred = target.ndim == 2
    selected = inferred if multilabel is None else bool(multilabel)
    if inferred != selected:
        expected = "two-dimensional" if selected else "one-dimensional"
        raise ValueError(f"labels must be {expected} for multilabel={selected}")
    return selected


def _resolved_label_names(
    label_names: Sequence[str] | None,
    count: int,
) -> tuple[str, ...]:
    supplied = () if label_names is None else tuple(str(name) for name in label_names)
    return tuple(
        supplied[index] if index < len(supplied) else f"label_{index}" for index in range(count)
    )


def _prepare_multilabel(
    target: NDArray[Any],
    label_names: Sequence[str] | None,
) -> _MultilabelPlan:
    if target.ndim != 2:
        raise ValueError("multilabel targets must be a two-dimensional indicator matrix")
    if target.shape[1] < 1:
        raise ValueError("multilabel targets must contain at least one label")
    if not np.issubdtype(target.dtype, np.number) and target.dtype != np.bool_:
        raise TypeError("multilabel targets must contain binary numerical values")
    if np.issubdtype(target.dtype, np.number) and not np.isfinite(target).all():
        raise ValueError("multilabel targets must contain only finite values")
    if not np.logical_or(target == 0, target == 1).all():
        raise ValueError("multilabel targets must contain only 0/1 indicators")

    binary = np.ascontiguousarray(target, dtype=np.int64)
    names = _resolved_label_names(label_names, binary.shape[1])
    positive_counts = binary.sum(axis=0, dtype=np.int64)
    valid = tuple(
        int(index)
        for index in np.flatnonzero(
            np.logical_and(positive_counts > 0, positive_counts < len(binary))
        )
    )
    if not valid:
        raise ValueError("No multilabel target contains both positive and negative examples")
    valid_set = set(valid)
    return _MultilabelPlan(
        target=binary,
        valid_indices=valid,
        valid_names=tuple(names[index] for index in valid),
        skipped_names=tuple(
            names[index] for index in range(binary.shape[1]) if index not in valid_set
        ),
        positive_counts=positive_counts[np.asarray(valid, dtype=np.int64)],
    )


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _class_name(value: Any, label_names: Sequence[str] | None) -> str:
    source_value = _python_scalar(value)
    if isinstance(source_value, (int, np.integer)) and label_names is not None:
        class_id = int(source_value)
        if 0 <= class_id < len(label_names):
            return str(label_names[class_id])
    return f"class_{source_value}"


def _prepare_single_label(
    target: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[np.int64], NDArray[np.int64]]:
    if target.ndim != 1:
        raise ValueError("single-label targets must be one-dimensional")
    classes, inverse, counts = np.unique(target, return_inverse=True, return_counts=True)
    if len(classes) < 2:
        raise ValueError("A PyCOL target must contain at least two classes")
    return (
        classes,
        np.ascontiguousarray(inverse, dtype=np.int64),
        np.ascontiguousarray(counts, dtype=np.int64),
    )


def _prepare_vectors(
    scaled_vectors: NDArray[Any] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(scaled_vectors, torch.Tensor):
        if scaled_vectors.ndim != 2:
            raise ValueError("scaled_vectors must be a two-dimensional array")
        if scaled_vectors.shape[0] < 2 or scaled_vectors.shape[1] < 1:
            raise ValueError("scaled_vectors must contain at least two samples and one feature")
        if scaled_vectors.is_complex():
            raise TypeError("scaled_vectors must contain real numerical values")
        vectors = scaled_vectors.detach().to(device=device, dtype=torch.float32)
    else:
        matrix = np.asarray(scaled_vectors)
        if matrix.ndim != 2:
            raise ValueError("scaled_vectors must be a two-dimensional array")
        if matrix.shape[0] < 2 or matrix.shape[1] < 1:
            raise ValueError("scaled_vectors must contain at least two samples and one feature")
        if not np.issubdtype(matrix.dtype, np.number):
            raise TypeError("scaled_vectors must contain numerical values")
        if np.issubdtype(matrix.dtype, np.complexfloating):
            raise TypeError("scaled_vectors must contain real numerical values")
        if not np.isfinite(matrix).all():
            raise ValueError("scaled_vectors must contain only finite values")
        vectors = torch.as_tensor(
            np.ascontiguousarray(matrix, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
    if not bool(torch.isfinite(vectors).all().item()):
        raise ValueError("scaled_vectors must contain only finite values")
    return vectors


def _prepare_neighbor_indices(
    neighbor_indices: NDArray[Any] | torch.Tensor,
    *,
    device: torch.device,
    n_samples: int,
    neighbors: int | None,
) -> tuple[torch.Tensor, int, int]:
    if isinstance(neighbor_indices, torch.Tensor):
        if neighbor_indices.dtype not in _TORCH_INTEGER_DTYPES:
            raise TypeError("neighbor_indices must contain integer sample indices")
        indices = neighbor_indices.detach().to(device=device, dtype=torch.long)
    else:
        matrix = np.asarray(neighbor_indices)
        if not np.issubdtype(matrix.dtype, np.integer):
            raise TypeError("neighbor_indices must contain integer sample indices")
        indices = torch.as_tensor(
            np.ascontiguousarray(matrix, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )

    if indices.ndim != 2 or indices.shape[0] != n_samples:
        raise ValueError("neighbor_indices must have shape [number of samples, max_k]")
    available = int(indices.shape[1])
    selected = available if neighbors is None else neighbors
    if isinstance(selected, bool) or not isinstance(selected, int):
        raise TypeError("neighbors must be an integer")
    if selected < 1 or selected > available:
        raise ValueError("neighbors must satisfy 1 <= neighbors <= available neighbors")

    chosen = indices[:, :selected]
    if bool(torch.logical_or(chosen < 0, chosen >= n_samples).any().item()):
        raise ValueError("neighbor_indices contains an out-of-range sample index")
    rows = torch.arange(n_samples, dtype=torch.long, device=device).unsqueeze(1)
    if bool((chosen == rows).any().item()):
        raise ValueError("neighbor_indices must exclude each query's own sample index")
    if selected > 1:
        ordered = torch.sort(chosen, dim=1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
            raise ValueError("neighbor_indices cannot repeat a neighbor within a row")
    return chosen, selected, available


def _f1_from_inverse(
    centered: torch.Tensor,
    inverse_numpy: NDArray[np.int64],
    counts_numpy: NDArray[np.int64],
) -> tuple[float, dict[str, int]]:
    """Calculate the equal OVO-pair/feature F1 mean with two-pass variances."""

    device = centered.device
    n_classes = int(len(counts_numpy))
    n_features = int(centered.shape[1])
    inverse = torch.as_tensor(inverse_numpy, dtype=torch.long, device=device)
    counts = torch.as_tensor(counts_numpy, dtype=torch.float32, device=device)

    means = torch.zeros(
        (n_classes, n_features),
        dtype=torch.float32,
        device=device,
    )
    means.index_add_(0, inverse, centered)
    means /= counts.unsqueeze(1)

    residuals = centered - means[inverse]
    variances = torch.zeros_like(means)
    variances.index_add_(0, inverse, residuals.square())
    variances /= counts.unsqueeze(1)

    pair_indices = torch.triu_indices(n_classes, n_classes, offset=1)
    pair_indices = pair_indices.to(device=device)
    pair_count = int(pair_indices.shape[1])
    score_sum = torch.zeros((), dtype=torch.float32, device=device)
    nonfinite_count = torch.zeros((), dtype=torch.long, device=device)
    zero_denominator_count = torch.zeros((), dtype=torch.long, device=device)

    for start in range(0, pair_count, _F1_PAIR_CHUNK_SIZE):
        selected = pair_indices[:, start : start + _F1_PAIR_CHUNK_SIZE]
        left = selected[0]
        right = selected[1]
        numerator = (means[left] - means[right]).square()
        denominator = variances[left] + variances[right]
        ratio = numerator / denominator
        finite = torch.isfinite(ratio)
        nonfinite_count += (~finite).sum()
        zero_denominator_count += (denominator == 0.0).sum()
        # PyCOL 1.0.4 replaces both +/-inf and NaN Fisher ratios with zero
        # before applying 1 / (1 + ratio). Preserve that unusual behavior.
        ratio = torch.where(finite, ratio, torch.zeros_like(ratio))
        score_sum += torch.reciprocal(1.0 + ratio).sum()

    denominator_count = pair_count * n_features
    value = float(score_sum.detach().cpu().item() / denominator_count)
    return value, {
        "class_count": n_classes,
        "class_pair_count": pair_count,
        "feature_count": n_features,
        "pair_feature_count": denominator_count,
        "nonfinite_ratio_count": int(nonfinite_count.detach().cpu().item()),
        "zero_denominator_count": int(zero_denominator_count.detach().cpu().item()),
    }


def _multilabel_aggregates(
    per_label: dict[str, dict[str, float]],
    metric_names: Sequence[str],
    positive_counts: NDArray[np.int64],
) -> tuple[dict[str, float], dict[str, float]]:
    weights = positive_counts.astype(np.float64, copy=False)
    weights /= weights.sum()
    macro = {
        metric: float(np.mean([values[metric] for values in per_label.values()]))
        for metric in metric_names
    }
    weighted = {
        metric: float(
            np.average(
                [values[metric] for values in per_label.values()],
                weights=weights,
            )
        )
        for metric in metric_names
    }
    return macro, weighted


def compute_f1(
    scaled_vectors: NDArray[Any] | torch.Tensor,
    labels: NDArray[Any] | torch.Tensor,
    *,
    device: str | torch.device = "auto",
    multilabel: bool | None = None,
    label_names: Sequence[str] | None = None,
) -> MetricResult:
    """Compute PyCOL 1.0.4 F1 with float32 Torch class statistics.

    Multiclass F1 is the equal arithmetic mean across every one-vs-one class
    pair and every feature. Variances are population variances. For multilabel
    data, each nonconstant indicator column is evaluated one-vs-rest before
    macro and positive-prevalence-weighted aggregation.
    """

    started = time.perf_counter()
    resolved = resolve_device(device)
    target = _as_numpy_labels(labels)
    use_multilabel = _infer_multilabel(target, multilabel)

    transfer_started = time.perf_counter()
    vectors = _prepare_vectors(scaled_vectors, resolved)
    if len(target) != len(vectors):
        raise ValueError("scaled_vectors and labels must contain the same samples")
    synchronize_device(resolved)
    transfer_ms = (time.perf_counter() - transfer_started) * 1_000.0

    metric_started = time.perf_counter()
    # A common translation preserves both pairwise mean differences and
    # within-class variances while improving float32 mean accuracy.
    centered = vectors - vectors.mean(dim=0, keepdim=True)

    per_label: dict[str, dict[str, float]] = {}
    weighted: dict[str, float] = {}
    scientific_details: dict[str, Any]
    valid_names: list[str]
    skipped_names: list[str]

    if not use_multilabel:
        classes, inverse, counts = _prepare_single_label(target)
        value, details = _f1_from_inverse(centered, inverse, counts)
        metrics = {"F1": value}
        scientific_details = details
        valid_names = [_class_name(value, label_names) for value in classes]
        skipped_names = []
        class_values = [_python_scalar(value) for value in classes]
        class_counts = {name: int(count) for name, count in zip(valid_names, counts, strict=True)}
    else:
        plan = _prepare_multilabel(target, label_names)
        details_by_label: dict[str, dict[str, int]] = {}
        for label_index, name in zip(
            plan.valid_indices,
            plan.valid_names,
            strict=True,
        ):
            binary = plan.target[:, label_index]
            inverse = np.ascontiguousarray(binary, dtype=np.int64)
            positive_count = int(binary.sum())
            counts = np.asarray(
                [len(binary) - positive_count, positive_count],
                dtype=np.int64,
            )
            value, details = _f1_from_inverse(centered, inverse, counts)
            per_label[name] = {"F1": value}
            details_by_label[name] = details
        metrics, weighted = _multilabel_aggregates(
            per_label,
            ("F1",),
            plan.positive_counts,
        )
        scientific_details = {"per_label": details_by_label}
        valid_names = list(plan.valid_names)
        skipped_names = list(plan.skipped_names)
        class_values = []
        class_counts = {}

    synchronize_device(resolved)
    metric_ms = (time.perf_counter() - metric_started) * 1_000.0
    total_ms = (time.perf_counter() - started) * 1_000.0
    diagnostics: dict[str, Any] = {
        "algorithm": "torch_two_pass_population_variance_ovo",
        "aggregation": (
            "multilabel_ovr_label_macro" if use_multilabel else "equal_ovo_pair_and_feature_mean"
        ),
        "device": str(resolved),
        "dtype": "float32",
        "n_samples": int(vectors.shape[0]),
        "n_features": int(vectors.shape[1]),
        "multilabel": use_multilabel,
        "valid_label_names": valid_names,
        "skipped_label_names": skipped_names,
        "class_values": class_values,
        "class_counts": class_counts,
        "positive_label_weights": (
            {}
            if not use_multilabel
            else {
                name: int(count)
                for name, count in zip(
                    valid_names,
                    plan.positive_counts,
                    strict=True,
                )
            }
        ),
        "scientific_details": scientific_details,
        "timings_ms": {
            "input_transfer_and_validation": transfer_ms,
            "metric": metric_ms,
            "total": total_ms,
        },
    }
    return MetricResult(
        metrics=metrics,
        per_label=per_label,
        label_weighted_metrics=weighted,
        diagnostics=diagnostics,
    )


def _requested_neighbor_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(metric) for metric in metrics)
    if not selected:
        raise ValueError("metrics cannot be empty")
    if len(set(selected)) != len(selected):
        raise ValueError("metrics cannot contain duplicates")
    unsupported = set(selected) - set(NEIGHBOR_METRICS)
    if unsupported:
        raise ValueError(
            f"Unsupported neighbor metrics {sorted(unsupported)}; choose from {NEIGHBOR_METRICS}"
        )
    return selected


def _single_label_neighbor_scores(
    indices: torch.Tensor,
    target: NDArray[Any],
    *,
    metrics: tuple[str, ...],
    label_names: Sequence[str] | None,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    dict[str, Any],
]:
    classes, inverse_numpy, counts_numpy = _prepare_single_label(target)
    device = indices.device
    inverse = torch.as_tensor(inverse_numpy, dtype=torch.long, device=device)
    same = inverse[indices] == inverse.unsqueeze(1)
    different = ~same

    values: dict[str, float] = {}
    per_class: dict[str, dict[str, float]] = {}
    details: dict[str, Any] = {}
    n_neighbors = int(indices.shape[1])

    if "N3" in metrics:
        errors = different[:, 0]
        error_counts = torch.zeros(
            len(classes),
            dtype=torch.float32,
            device=device,
        )
        error_counts.index_add_(0, inverse, errors.to(dtype=torch.float32))
        counts = torch.as_tensor(
            counts_numpy,
            dtype=torch.float32,
            device=device,
        )
        classwise = error_counts / counts
        values["N3"] = float(classwise.mean().detach().cpu().item())
        classwise_values = classwise.detach().cpu().tolist()
        names = [_class_name(value, label_names) for value in classes]
        per_class = {
            name: {"N3": float(value)} for name, value in zip(names, classwise_values, strict=True)
        }
        error_count = int(errors.sum().detach().cpu().item())
        details["N3_error_count"] = error_count
        details["N3_class_error_counts"] = {
            name: int(value)
            for name, value in zip(
                names,
                error_counts.detach().cpu().tolist(),
                strict=True,
            )
        }
        details["N3_micro"] = float(error_count / len(target))

    if "kDN" in metrics:
        disagreement_count = int(different.sum().detach().cpu().item())
        values["kDN"] = float(disagreement_count / (len(target) * n_neighbors))
        details["kDN_disagreement_count"] = disagreement_count

    if "CM" in metrics:
        disagreement_counts = different.sum(dim=1)
        hard = disagreement_counts * 2 > n_neighbors
        hard_count = int(hard.sum().detach().cpu().item())
        values["CM"] = float(hard_count / len(target))
        details["CM_hard_sample_count"] = hard_count

    if "C1" in metrics:
        ranks = torch.arange(
            1,
            n_neighbors + 1,
            dtype=torch.float32,
            device=device,
        )
        prefix_purity = same.to(dtype=torch.float32).cumsum(dim=1) / ranks
        instance_values = 1.0 - prefix_purity.mean(dim=1)
        values["C1"] = float(instance_values.mean().detach().cpu().item())
        details["C1_instance_sum"] = float(instance_values.sum().detach().cpu().item())

    return values, per_class, details


def _multilabel_neighbor_scores(
    indices: torch.Tensor,
    plan: _MultilabelPlan,
    *,
    metrics: tuple[str, ...],
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    dict[str, float],
    dict[str, Any],
]:
    device = indices.device
    columns = np.asarray(plan.valid_indices, dtype=np.int64)
    selected_target = np.ascontiguousarray(plan.target[:, columns], dtype=np.int64)
    target = torch.as_tensor(selected_target, dtype=torch.long, device=device)
    same = target[indices] == target.unsqueeze(1)
    different = ~same
    n_samples, n_neighbors, n_labels = different.shape

    metric_vectors: dict[str, torch.Tensor] = {}
    auxiliary_vectors: dict[str, torch.Tensor] = {}
    details: dict[str, Any] = {}

    if "N3" in metrics:
        errors = different[:, 0, :].to(dtype=torch.float32)
        positives = target.to(dtype=torch.float32)
        negatives = 1.0 - positives
        positive_counts = positives.sum(dim=0)
        negative_counts = negatives.sum(dim=0)
        positive_errors = (errors * positives).sum(dim=0)
        negative_errors = (errors * negatives).sum(dim=0)
        positive_rates = positive_errors / positive_counts
        negative_rates = negative_errors / negative_counts
        metric_vectors["N3"] = (negative_rates + positive_rates) / 2.0
        auxiliary_vectors["N3_negative"] = negative_rates
        auxiliary_vectors["N3_positive"] = positive_rates
        micro_values = errors.mean(dim=0)
        details["N3_micro_label_macro"] = float(micro_values.mean().detach().cpu().item())
        details["N3_error_counts"] = {
            name: int(value)
            for name, value in zip(
                plan.valid_names,
                errors.sum(dim=0).detach().cpu().tolist(),
                strict=True,
            )
        }

    if "kDN" in metrics:
        disagreement_counts = different.sum(dim=(0, 1))
        metric_vectors["kDN"] = disagreement_counts.to(dtype=torch.float32) / (
            n_samples * n_neighbors
        )
        details["kDN_disagreement_counts"] = {
            name: int(value)
            for name, value in zip(
                plan.valid_names,
                disagreement_counts.detach().cpu().tolist(),
                strict=True,
            )
        }

    if "CM" in metrics:
        hard = different.sum(dim=1) * 2 > n_neighbors
        hard_counts = hard.sum(dim=0)
        metric_vectors["CM"] = hard_counts.to(dtype=torch.float32) / n_samples
        details["CM_hard_sample_counts"] = {
            name: int(value)
            for name, value in zip(
                plan.valid_names,
                hard_counts.detach().cpu().tolist(),
                strict=True,
            )
        }

    if "C1" in metrics:
        ranks = torch.arange(
            1,
            n_neighbors + 1,
            dtype=torch.float32,
            device=device,
        ).reshape(1, n_neighbors, 1)
        prefix_purity = same.to(dtype=torch.float32).cumsum(dim=1) / ranks
        instance_values = 1.0 - prefix_purity.mean(dim=1)
        metric_vectors["C1"] = instance_values.mean(dim=0)
        details["C1_instance_sums"] = {
            name: float(value)
            for name, value in zip(
                plan.valid_names,
                instance_values.sum(dim=0).detach().cpu().tolist(),
                strict=True,
            )
        }

    vector_values = {
        metric: tensor.detach().cpu().tolist() for metric, tensor in metric_vectors.items()
    }
    auxiliary_values = {
        metric: tensor.detach().cpu().tolist() for metric, tensor in auxiliary_vectors.items()
    }
    per_label: dict[str, dict[str, float]] = {}
    for label_position, name in enumerate(plan.valid_names):
        label_values = {metric: float(vector_values[metric][label_position]) for metric in metrics}
        for auxiliary_name, values in auxiliary_values.items():
            label_values[auxiliary_name] = float(values[label_position])
        per_label[name] = label_values

    macro, weighted = _multilabel_aggregates(
        per_label,
        metrics,
        plan.positive_counts,
    )
    return macro, per_label, weighted, details


def compute_neighbor_metrics(
    neighbor_indices: NDArray[Any] | torch.Tensor,
    labels: NDArray[Any] | torch.Tensor,
    metrics: Sequence[str] = NEIGHBOR_METRICS,
    neighbors: int | None = None,
    *,
    device: str | torch.device = "auto",
    multilabel: bool | None = None,
    label_names: Sequence[str] | None = None,
) -> MetricResult:
    """Score N3, kDN, CM, and C1 from one ordered kNN index matrix.

    N3 always uses the first neighbour and is class-balanced. kDN, CM, and C1
    use the requested neighbour prefix and retain PyCOL's sample-micro
    aggregation. The supplied neighbour order must already use PyCOL's
    ``(distance, sample_index)`` tie rule.
    """

    started = time.perf_counter()
    selected_metrics = _requested_neighbor_metrics(metrics)
    resolved = resolve_device(device)
    target = _as_numpy_labels(labels)
    use_multilabel = _infer_multilabel(target, multilabel)
    n_samples = int(target.shape[0])
    if n_samples < 2:
        raise ValueError("neighbor metrics require at least two samples")

    transfer_started = time.perf_counter()
    indices, selected_neighbors, available_neighbors = _prepare_neighbor_indices(
        neighbor_indices,
        device=resolved,
        n_samples=n_samples,
        neighbors=neighbors,
    )
    synchronize_device(resolved)
    transfer_ms = (time.perf_counter() - transfer_started) * 1_000.0

    metric_started = time.perf_counter()
    if not use_multilabel:
        values, per_label, details = _single_label_neighbor_scores(
            indices,
            target,
            metrics=selected_metrics,
            label_names=label_names,
        )
        weighted: dict[str, float] = {}
        valid_names = (
            list(per_label)
            if "N3" in selected_metrics
            else [_class_name(value, label_names) for value in np.unique(target)]
        )
        skipped_names: list[str] = []
        positive_weights: dict[str, int] = {}
    else:
        plan = _prepare_multilabel(target, label_names)
        values, per_label, weighted, details = _multilabel_neighbor_scores(
            indices,
            plan,
            metrics=selected_metrics,
        )
        valid_names = list(plan.valid_names)
        skipped_names = list(plan.skipped_names)
        positive_weights = {
            name: int(count)
            for name, count in zip(
                plan.valid_names,
                plan.positive_counts,
                strict=True,
            )
        }

    synchronize_device(resolved)
    metric_ms = (time.perf_counter() - metric_started) * 1_000.0
    total_ms = (time.perf_counter() - started) * 1_000.0
    diagnostics: dict[str, Any] = {
        **details,
        "algorithm": "shared_ordered_knn_label_scoring",
        "tie_policy_required": "distance_then_smallest_sample_index",
        "aggregation": (
            "multilabel_ovr_label_macro"
            if use_multilabel
            else "N3_class_balanced_other_metrics_sample_micro"
        ),
        "device": str(resolved),
        "dtype": "float32",
        "n_samples": n_samples,
        "neighbors_available": available_neighbors,
        "neighbors_used": selected_neighbors,
        "metrics": list(selected_metrics),
        "multilabel": use_multilabel,
        "valid_label_names": valid_names,
        "skipped_label_names": skipped_names,
        "positive_label_weights": positive_weights,
        "timings_ms": {
            "input_transfer_and_validation": transfer_ms,
            "metric": metric_ms,
            "total": total_ms,
        },
    }
    return MetricResult(
        metrics={metric: values[metric] for metric in selected_metrics},
        per_label=per_label,
        label_weighted_metrics=weighted,
        diagnostics=diagnostics,
    )


__all__ = [
    "NEIGHBOR_METRICS",
    "MetricResult",
    "compute_f1",
    "compute_neighbor_metrics",
]
