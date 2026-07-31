"""Neighbourhood measures composed from streamed primitives.

Every measure here is a small pure function of a primitive's output rather than
of the distance matrix, so they all inherit one tiling strategy and one
tie-breaking policy instead of restating them.  Adding a measure that reuses an
existing primitive costs a function, not a pass over the data.

Each measure takes ``imb``, matching the reference: when false a single value
is returned for the dataset, and when true a per-class array is returned,
normalized by class size rather than by sample count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

Float64Array: TypeAlias = NDArray[np.float64]
MeasureValue: TypeAlias = "float | Float64Array"


@dataclass(frozen=True, slots=True)
class NeighbourhoodCounts:
    """Per-sample class histogram over a sample's nearest neighbours.

    ``counts`` is ``[samples, classes]``; ``own`` is how many of a sample's
    neighbours carry its own label.  Everything in this module reads from here,
    which is why one k-nearest-neighbour pass serves the whole family.
    """

    counts: NDArray[np.int64]
    own: NDArray[np.int64]
    codes: NDArray[np.int64]
    class_count: NDArray[np.int64]
    max_k: int

    @property
    def n_samples(self) -> int:
        """Return the number of samples described."""

        return int(self.counts.shape[0])

    @property
    def n_classes(self) -> int:
        """Return the number of distinct classes."""

        return int(self.counts.shape[1])


def neighbourhood_counts(
    indices: torch.Tensor | NDArray[Any],
    labels: NDArray[Any],
    *,
    max_k: int | None = None,
) -> NeighbourhoodCounts:
    """Build the per-sample neighbour class histogram from a kNN prefix."""

    prefix = (
        indices.detach().cpu().numpy() if isinstance(indices, torch.Tensor) else np.asarray(indices)
    )
    if prefix.ndim != 2:
        raise ValueError("indices must be a two-dimensional [samples, k] array")
    target = np.asarray(labels)
    if target.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if target.shape[0] != prefix.shape[0]:
        raise ValueError("labels must contain one entry per sample")

    width = int(prefix.shape[1]) if max_k is None else int(max_k)
    if not 1 <= width <= prefix.shape[1]:
        raise ValueError("max_k must satisfy 1 <= max_k <= the available prefix width")
    prefix = prefix[:, :width]

    _, codes = np.unique(target, return_inverse=True)
    codes = codes.reshape(-1).astype(np.int64, copy=False)
    n_classes = int(codes.max()) + 1

    neighbour_codes = codes[prefix]
    counts = np.zeros((prefix.shape[0], n_classes), dtype=np.int64)
    np.add.at(counts, (np.arange(prefix.shape[0])[:, None], neighbour_codes), 1)

    return NeighbourhoodCounts(
        counts=counts,
        own=counts[np.arange(prefix.shape[0]), codes],
        codes=codes,
        class_count=np.bincount(codes, minlength=n_classes).astype(np.int64),
        max_k=width,
    )


def _aggregate(
    per_sample: NDArray[Any],
    counts: NeighbourhoodCounts | NeighbourhoodProfile,
    *,
    imb: bool,
) -> MeasureValue:
    """Sum a per-sample score by class, then normalize by class or by dataset."""

    totals = np.zeros(counts.n_classes, dtype=np.float64)
    np.add.at(totals, counts.codes, per_sample.astype(np.float64, copy=False))
    if imb:
        return totals / counts.class_count
    return float(totals.sum() / counts.n_samples)


def kdn(counts: NeighbourhoodCounts, *, imb: bool = False) -> MeasureValue:
    """k-Disagreeing Neighbours: the mean fraction of neighbours that disagree."""

    return _aggregate((counts.max_k - counts.own) / counts.max_k, counts, imb=imb)


def cm(counts: NeighbourhoodCounts, *, imb: bool = False) -> MeasureValue:
    """Fraction of samples whose neighbours mostly disagree with them."""

    return _aggregate((counts.max_k - counts.own) / counts.max_k > 0.5, counts, imb=imb)


def degree_of_overlap(counts: NeighbourhoodCounts, *, imb: bool = False) -> MeasureValue:
    """Fraction of samples with at least one differently-labelled neighbour."""

    return _aggregate(counts.own != counts.max_k, counts, imb=imb)


def separability_index(counts: NeighbourhoodCounts, *, imb: bool = False) -> MeasureValue:
    """Fraction of samples whose own class is a plurality among its neighbours.

    Ties count as separable, matching the reference, which tests the own-class
    tally against the maximum rather than against every other class.
    """

    return _aggregate(counts.own == counts.counts.max(axis=1), counts, imb=imb)


def d3(counts: NeighbourhoodCounts) -> NDArray[np.int64]:
    """Per-class tally of samples whose neighbours are mostly of other classes.

    Returned as raw per-class counts rather than a rate, which is what the
    reference produces; it takes no ``imb`` argument there either.
    """

    flagged = (counts.own / counts.max_k) < 0.5
    totals = np.zeros(counts.n_classes, dtype=np.int64)
    np.add.at(totals, counts.codes, flagged.astype(np.int64))
    return totals


def local_set_cardinality(
    sizes: torch.Tensor | NDArray[Any],
    labels: NDArray[Any],
    *,
    imb: bool = False,
) -> MeasureValue:
    """LSC: one minus the mean local set size, normalized by the sample count.

    Local sets shrink as classes interleave, so a high value means high
    complexity.  The normalizer is quadratic because the measure compares the
    total local set mass against every ordered pair of samples.
    """

    local = _as_vector(sizes)
    codes, class_count = _encode(labels, expected=local.shape[0])

    if imb:
        totals = np.zeros(class_count.shape[0], dtype=np.float64)
        np.add.at(totals, codes, local)
        return 1.0 - totals / class_count.astype(np.float64) ** 2
    return float(1.0 - local.sum() / float(local.shape[0]) ** 2)


#: The borderline taxonomy is defined on a fixed five-neighbour window; its
#: category boundaries are absolute counts, not fractions of k.
BORDERLINE_NEIGHBOURS = 5


@dataclass(frozen=True, slots=True)
class BorderlineCounts:
    """Share of samples in each of Napierala and Stefanowski's four types."""

    borderline: MeasureValue
    safe: MeasureValue
    rare: MeasureValue
    outlier: MeasureValue


def n2(
    friend_distances: torch.Tensor | NDArray[Any],
    enemy_distances: torch.Tensor | NDArray[Any],
    labels: NDArray[Any],
    *,
    imb: bool = False,
) -> MeasureValue:
    """N2: intra-class spread against inter-class separation, squashed to [0, 1).

    Takes the ratio of summed nearest-same-class distance to summed
    nearest-enemy distance, then maps it through ``r / (1 + r)``.  A class whose
    enemy distances sum to zero scores zero, matching the reference rather than
    dividing by zero.
    """

    intra = _as_vector(friend_distances)
    inter = _as_vector(enemy_distances)
    codes, class_count = _encode(labels, expected=intra.shape[0])
    if inter.shape[0] != intra.shape[0]:
        raise ValueError("friend and enemy distances must describe the same samples")

    n_classes = class_count.shape[0]
    intra_totals = np.zeros(n_classes, dtype=np.float64)
    inter_totals = np.zeros(n_classes, dtype=np.float64)
    np.add.at(intra_totals, codes, intra)
    np.add.at(inter_totals, codes, inter)

    if imb:
        ratio = np.divide(
            intra_totals,
            inter_totals,
            out=np.zeros(n_classes, dtype=np.float64),
            where=inter_totals != 0,
        )
        return ratio / (1.0 + ratio)
    total_inter = float(inter_totals.sum())
    ratio = 0.0 if total_inter == 0 else float(intra_totals.sum()) / total_inter
    return float(ratio / (1.0 + ratio))


def borderline(counts: NeighbourhoodCounts, *, imb: bool = False) -> BorderlineCounts:
    """Split samples into safe, borderline, rare, and outlier neighbourhoods.

    Requires exactly five neighbours because the thresholds are absolute: two
    or three enemies is borderline, four is rare, five is an outlier.
    """

    if counts.max_k != BORDERLINE_NEIGHBOURS:
        raise ValueError(
            f"borderline is defined on exactly {BORDERLINE_NEIGHBOURS} neighbours; "
            f"got {counts.max_k}"
        )
    enemies = counts.max_k - counts.own
    return BorderlineCounts(
        borderline=_aggregate((enemies == 2) | (enemies == 3), counts, imb=imb),
        safe=_aggregate(enemies < 2, counts, imb=imb),
        rare=_aggregate(enemies == 4, counts, imb=imb),
        outlier=_aggregate(enemies == 5, counts, imb=imb),
    )


def r_value(
    counts: NeighbourhoodCounts,
    *,
    theta: int = 2,
    imb: bool = False,
) -> list[Any]:
    """Augmented R-value: pairwise class overlap, weighted by class imbalance.

    A sample counts as invading class ``j`` when more than ``theta`` of its
    neighbours belong to it.  Each class pair is then combined so the smaller
    class's intrusion carries the weight of the imbalance ratio, which is what
    makes this an imbalance-aware overlap measure.  Returns one entry per class
    pair, as the reference does.
    """

    invades = counts.counts > theta
    overlap = np.zeros((counts.n_classes, counts.n_classes), dtype=np.float64)
    np.add.at(overlap, counts.codes, invades.astype(np.float64))
    overlap /= counts.class_count[:, None]

    values: list[Any] = []
    for first in range(counts.n_classes):
        for second in range(first + 1, counts.n_classes):
            if counts.class_count[first] > counts.class_count[second]:
                ratio = counts.class_count[first] / counts.class_count[second]
                major_into_minor = overlap[first, second]
                minor_into_major = overlap[second, first]
            else:
                ratio = counts.class_count[second] / counts.class_count[first]
                major_into_minor = overlap[second, first]
                minor_into_major = overlap[first, second]
            if imb:
                values.append([minor_into_major, major_into_minor])
            else:
                values.append((1.0 / (ratio + 1.0)) * (major_into_minor + ratio * minor_into_major))
    return values


def _as_vector(values: torch.Tensor | NDArray[Any]) -> NDArray[np.float64]:
    array = (
        values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    )
    return array.reshape(-1).astype(np.float64, copy=False)


def _encode(labels: NDArray[Any], *, expected: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    target = np.asarray(labels)
    if target.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if target.shape[0] != expected:
        raise ValueError("labels must contain one entry per sample")
    _, codes = np.unique(target, return_inverse=True)
    codes = codes.reshape(-1).astype(np.int64, copy=False)
    return codes, np.bincount(codes, minlength=int(codes.max()) + 1).astype(np.int64)


@dataclass(frozen=True, slots=True)
class NeighbourhoodProfile:
    """Ordered per-neighbour view of a prefix, for the cumulative measures.

    The C-family scores every prefix length from one to ``max_k`` rather than
    only the widest, so it needs the neighbours in order rather than a
    histogram.  ``same`` marks neighbours sharing the sample's label and
    ``distances`` carries their distances, both ``[samples, max_k]``.
    """

    same: NDArray[np.bool_]
    distances: NDArray[np.float64]
    codes: NDArray[np.int64]
    class_count: NDArray[np.int64]
    max_k: int

    @property
    def n_samples(self) -> int:
        """Return the number of samples described."""

        return int(self.same.shape[0])

    @property
    def n_classes(self) -> int:
        """Return the number of distinct classes."""

        return int(self.class_count.shape[0])


def neighbourhood_profile(
    indices: torch.Tensor | NDArray[Any],
    distances: torch.Tensor | NDArray[Any],
    labels: NDArray[Any],
    *,
    max_k: int | None = None,
) -> NeighbourhoodProfile:
    """Build the ordered prefix view the cumulative measures read."""

    prefix = _as_matrix(indices).astype(np.int64, copy=False)
    prefix_distances = _as_matrix(distances).astype(np.float64, copy=False)
    if prefix.shape != prefix_distances.shape:
        raise ValueError("indices and distances must have the same shape")

    width = int(prefix.shape[1]) if max_k is None else int(max_k)
    if not 1 <= width <= prefix.shape[1]:
        raise ValueError("max_k must satisfy 1 <= max_k <= the available prefix width")
    prefix = prefix[:, :width]
    prefix_distances = prefix_distances[:, :width]

    codes, class_count = _encode(labels, expected=prefix.shape[0])
    return NeighbourhoodProfile(
        same=codes[prefix] == codes[:, None],
        distances=prefix_distances,
        codes=codes,
        class_count=class_count,
        max_k=width,
    )


def _prefix_mean(cumulative: NDArray[np.float64], max_k: int) -> NDArray[np.float64]:
    """Average a cumulative per-neighbour score over every prefix length."""

    lengths = np.arange(1, max_k + 1, dtype=np.float64)
    return 1.0 - (cumulative / lengths).mean(axis=1)


def c1(profile: NeighbourhoodProfile, *, imb: bool = False) -> MeasureValue:
    """C1: same-label purity averaged over every prefix of the neighbour list.

    Scoring all prefixes rather than only the widest makes the measure
    sensitive to *where* in the ranking the first disagreement appears.
    """

    return _aggregate(
        _prefix_mean(np.cumsum(profile.same, axis=1, dtype=np.float64), profile.max_k),
        profile,
        imb=imb,
    )


def c2(profile: NeighbourhoodProfile, *, imb: bool = False) -> MeasureValue:
    """C2: C1 weighted by how close each same-label neighbour actually is.

    A same-label neighbour contributes ``1 - d``, so distant agreement counts
    for less than adjacent agreement.  Distances past one contribute nothing
    rather than turning negative, matching the reference's clamp.
    """

    contribution = np.where(profile.same, 1.0 - np.minimum(profile.distances, 1.0), 0.0)
    return _aggregate(
        _prefix_mean(np.cumsum(contribution, axis=1, dtype=np.float64), profile.max_k),
        profile,
        imb=imb,
    )


def _as_matrix(values: torch.Tensor | NDArray[Any]) -> NDArray[Any]:
    array = (
        values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    )
    if array.ndim != 2:
        raise ValueError("expected a two-dimensional [samples, k] array")
    return array


__all__ = [
    "neighbourhood_profile",
    "c2",
    "c1",
    "NeighbourhoodProfile",
    "r_value",
    "n2",
    "borderline",
    "BorderlineCounts",
    "BORDERLINE_NEIGHBOURS",
    "MeasureValue",
    "NeighbourhoodCounts",
    "cm",
    "d3",
    "degree_of_overlap",
    "kdn",
    "local_set_cardinality",
    "neighbourhood_counts",
    "separability_index",
]
