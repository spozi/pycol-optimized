"""Feature-based measures, computed without any pairwise distances.

These describe how far individual features separate the classes, so they read
the embedding directly rather than a distance matrix and never touch the tile
engine.  Each is defined one-vs-one and returns one entry per class pair, which
is what the reference produces.

The per-class reductions run on the resolved backend because they are linear in
the sample count.  The small per-pair arithmetic stays on the host: it is cubic
in the feature width but independent of the sample count, and ``pinv`` has
incomplete accelerator coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from .backend import DTYPE, Backend, resolve_backend

Float64Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ClassFeatureStats:
    """Per-class, per-feature summaries shared by every measure here."""

    lower: Float64Array
    upper: Float64Array
    mean: Float64Array
    values: list[Float64Array]
    class_count: NDArray[np.int64]

    @property
    def n_classes(self) -> int:
        """Return the number of distinct classes."""

        return int(self.class_count.shape[0])

    @property
    def n_features(self) -> int:
        """Return the embedding width."""

        return int(self.lower.shape[1])


def class_feature_stats(
    vectors: NDArray[Any] | torch.Tensor,
    labels: NDArray[Any],
    *,
    backend: Backend | str | torch.device = "auto",
) -> ClassFeatureStats:
    """Reduce an embedding to the per-class feature summaries."""

    resolved = backend if isinstance(backend, Backend) else resolve_backend(backend)
    matrix = (
        vectors.detach().cpu().numpy() if isinstance(vectors, torch.Tensor) else np.asarray(vectors)
    )
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional [samples, features] array")
    target = np.asarray(labels)
    if target.ndim != 1 or target.shape[0] != matrix.shape[0]:
        raise ValueError("labels must be one-dimensional with one entry per sample")

    _, codes = np.unique(target, return_inverse=True)
    codes = codes.reshape(-1).astype(np.int64, copy=False)
    n_classes = int(codes.max()) + 1
    if n_classes < 2:
        raise ValueError("feature measures need at least two classes")

    device_matrix = torch.as_tensor(
        np.ascontiguousarray(matrix, dtype=np.float32), dtype=DTYPE, device=resolved.device
    )
    device_codes = torch.as_tensor(codes, device=resolved.device)

    lower = np.empty((n_classes, matrix.shape[1]), dtype=np.float64)
    upper = np.empty_like(lower)
    mean = np.empty_like(lower)
    values: list[Float64Array] = []
    for label in range(n_classes):
        rows = device_matrix[device_codes == label]
        if rows.shape[0] == 0:
            raise ValueError("every class must contain at least one sample")
        lower[label] = rows.amin(dim=0).cpu().numpy()
        upper[label] = rows.amax(dim=0).cpu().numpy()
        mean[label] = rows.mean(dim=0).cpu().numpy()
        values.append(rows.cpu().numpy().astype(np.float64, copy=False))

    return ClassFeatureStats(
        lower=lower,
        upper=upper,
        mean=mean,
        values=values,
        class_count=np.bincount(codes, minlength=n_classes).astype(np.int64),
    )


def _pairs(stats: ClassFeatureStats):
    for first in range(stats.n_classes):
        for second in range(first + 1, stats.n_classes):
            yield first, second


def _overlap_counts(
    sample_values: Float64Array,
    highest_minimum: Float64Array,
    lowest_maximum: Float64Array,
) -> NDArray[np.int64]:
    """Count, per feature, how many samples fall inside the overlap region."""

    within = (sample_values >= highest_minimum) & (sample_values <= lowest_maximum)
    return within.sum(axis=0)


def _safe_ratio(numerator: Float64Array, denominator: Float64Array) -> Float64Array:
    """Divide, mapping the degenerate cases to zero as the reference does."""

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator / denominator
    ratio[~np.isfinite(ratio)] = 0.0
    return ratio


def f1v(stats: ClassFeatureStats) -> list[float]:
    """F1v: class separation along the best single discriminant direction.

    Where F1 scores each feature on its own, F1v first rotates into the
    direction that best separates the pair, so it sees separation that no axis
    reveals alone.  The scatter matrix is pseudo-inverted because collinear or
    constant features make it singular.
    """

    scores: list[float] = []
    for first, second in _pairs(stats):
        left = stats.values[first]
        right = stats.values[second]
        difference = stats.mean[first] - stats.mean[second]

        scatter = (
            len(left) * np.cov(left, rowvar=False, ddof=1)
            + len(right) * np.cov(right, rowvar=False, ddof=1)
        ) / (len(left) + len(right))
        scatter = np.atleast_2d(scatter)

        direction = np.linalg.pinv(scatter) @ difference
        between = np.outer(difference, difference)
        denominator = direction.T @ (scatter @ direction)
        numerator = direction.T @ (between @ direction)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator / denominator
        scores.append(float(1.0 / (1.0 + ratio)))
    return scores


def f2(stats: ClassFeatureStats, *, imb: bool = False) -> list[Any]:
    """F2: how much the two classes' feature ranges overlap, multiplied out.

    One cleanly separated feature drives the product to zero, so F2 rewards any
    single feature that splits the pair.  Under ``imb`` the overlap is expressed
    relative to each class's own spread instead of their combined spread, which
    keeps a wide majority class from hiding a narrow minority one.
    """

    scores: list[Any] = []
    for first, second in _pairs(stats):
        highest_minimum = np.maximum(stats.lower[first], stats.lower[second])
        lowest_maximum = np.minimum(stats.upper[first], stats.upper[second])
        overlap = np.maximum(0.0, lowest_maximum - highest_minimum)

        if imb:
            majority, minority = (
                (first, second)
                if stats.class_count[first] > stats.class_count[second]
                else (second, first)
            )
            per_class = [
                float(np.prod(_safe_ratio(overlap, stats.upper[label] - stats.lower[label])))
                for label in (majority, minority)
            ]
            scores.append(per_class)
        else:
            spread = np.maximum(stats.upper[first], stats.upper[second]) - np.minimum(
                stats.lower[first], stats.lower[second]
            )
            scores.append(float(np.prod(_safe_ratio(overlap, spread))))
    return scores


def f3(
    stats: ClassFeatureStats,
    vectors: NDArray[Any] | torch.Tensor | None = None,
    *,
    imb: bool = False,
) -> list[Any]:
    """F3: the share of samples left ambiguous by the most discriminative feature.

    ``vectors`` is required unless ``imb`` is set, because the reference counts
    the samples falling in the overlap region across the *whole* dataset while
    dividing by only the two classes' sizes.  That is reproduced here rather
    than corrected, so a multiclass result can exceed one.
    """

    if not imb and vectors is None:
        raise ValueError("f3 needs the full embedding unless imb is set")
    if vectors is not None:
        matrix = (
            vectors.detach().cpu().numpy()
            if isinstance(vectors, torch.Tensor)
            else np.asarray(vectors)
        ).astype(np.float64, copy=False)
    else:
        matrix = None

    scores: list[Any] = []
    for first, second in _pairs(stats):
        highest_minimum = np.maximum(stats.lower[first], stats.lower[second])
        lowest_maximum = np.minimum(stats.upper[first], stats.upper[second])

        if imb:
            majority, minority = (
                (first, second)
                if stats.class_count[first] > stats.class_count[second]
                else (second, first)
            )
            scores.append(
                [
                    float(
                        _overlap_counts(
                            stats.values[majority], highest_minimum, lowest_maximum
                        ).min()
                        / stats.class_count[majority]
                    ),
                    float(
                        _overlap_counts(
                            stats.values[minority], highest_minimum, lowest_maximum
                        ).min()
                        / stats.class_count[minority]
                    ),
                ]
            )
        else:
            assert matrix is not None
            pair_size = int(stats.class_count[first] + stats.class_count[second])
            counts = _overlap_counts(matrix, highest_minimum, lowest_maximum)
            scores.append(float(counts.min() / pair_size))
    return scores


def _strictly_inside_count(
    sample_values: Float64Array,
    lower: Float64Array,
    upper: Float64Array,
) -> int:
    """Count feature readings falling strictly inside another class's range.

    Strict on both sides, unlike the closed comparisons F2 and F3 use; a value
    sitting exactly on the other class's boundary is not noise.
    """

    return int(((sample_values > lower) & (sample_values < upper)).sum())


def input_noise(stats: ClassFeatureStats, *, imb: bool = False) -> list[Any]:
    """IN: the share of feature readings that fall inside the other class's range.

    Counts individual readings rather than whole samples, so a sample that
    trespasses on one feature is not treated the same as one that trespasses on
    all of them.  Without ``imb`` the denominator spans the whole dataset rather
    than the class pair, matching the reference, so a multiclass value is
    diluted by classes outside the pair.
    """

    total_samples = int(stats.class_count.sum())
    width = stats.n_features
    scores: list[Any] = []
    for first, second in _pairs(stats):
        majority, minority = (
            (first, second)
            if stats.class_count[first] > stats.class_count[second]
            else (second, first)
        )
        into_minority = _strictly_inside_count(
            stats.values[majority], stats.lower[minority], stats.upper[minority]
        )
        into_majority = _strictly_inside_count(
            stats.values[minority], stats.lower[majority], stats.upper[majority]
        )
        if imb:
            scores.append(
                [
                    into_minority / (stats.class_count[majority] * width),
                    into_majority / (stats.class_count[minority] * width),
                ]
            )
        else:
            scores.append((into_minority + into_majority) / (total_samples * width))
    return scores


__all__ = [
    "input_noise",
    "ClassFeatureStats",
    "class_feature_stats",
    "f1v",
    "f2",
    "f3",
]
