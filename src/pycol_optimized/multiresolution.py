"""Multiresolution measures.

The feature space is cut into a grid, and the same statistic is taken at every
grid resolution from one cell up to a fine mesh.  Reading the whole sequence
rather than a single resolution shows at what scale the classes separate.

No pairwise distances are involved, so these never touch the tile engine.  The
reference builds cell membership with a per-sample, per-feature, per-boundary
Python loop keyed on concatenated strings; here each resolution is a vectorized
bucketize, which is the difference between minutes and milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

Float64Array: TypeAlias = NDArray[np.float64]
Int64Array: TypeAlias = NDArray[np.int64]

#: Resolutions scored by default, from a single cell up to a fine mesh.
DEFAULT_MAX_RESOLUTION = 32

#: The area under a perfectly pure resolution profile, which the raw area is
#: divided by so a maximally separable dataset scores one.
PURITY_NORMALIZER = 0.702


@dataclass(frozen=True, slots=True)
class ResolutionProfile:
    """A statistic evaluated at each grid resolution, and its summary."""

    per_resolution: Float64Array
    weighted: Float64Array
    value: float


def cell_assignments(vectors: NDArray[Any] | torch.Tensor, resolution: int) -> Int64Array:
    """Label each sample with the grid cell it occupies at ``resolution``.

    Each feature is cut into ``resolution + 1`` equal intervals.  The reference
    tests every boundary in turn and takes the first interval that contains the
    value, and because its intervals are closed at both ends a value sitting
    exactly on a boundary falls to the *lower* cell.

    The interval a value is tested against is ``[bound, bound + step]`` rather
    than ``[bound, next_bound]``.  Those two differ in floating point, because
    ``linspace`` does not accumulate a step the way repeated addition does, and
    at fine resolutions the difference moves values near a boundary between
    cells.  Searching the shifted bounds reproduces the reference exactly
    instead of merely closely.

    A constant feature collapses to a single interval rather than dividing by a
    zero width.
    """

    if resolution < 0:
        raise ValueError("resolution cannot be negative")
    matrix = (
        vectors.detach().cpu().numpy() if isinstance(vectors, torch.Tensor) else np.asarray(vectors)
    ).astype(np.float64, copy=False)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional [samples, features] array")

    coordinates = np.empty(matrix.shape, dtype=np.int64)
    for feature in range(matrix.shape[1]):
        column = matrix[:, feature]
        lowest = float(column.min())
        highest = float(column.max())
        if highest <= lowest:
            coordinates[:, feature] = 0
            continue
        step = (highest - lowest) / (resolution + 1)
        bounds = np.linspace(lowest, highest, num=resolution + 2)
        placed = np.searchsorted(bounds + step, column, side="left")
        coordinates[:, feature] = np.clip(placed, 0, bounds.shape[0] - 1)

    # Distinct coordinate rows are the occupied cells.  Taking them as rows
    # rather than folding them into one integer keeps wide inputs from
    # overflowing at fine resolutions.
    _, cells = np.unique(coordinates, axis=0, return_inverse=True)
    return cells.reshape(-1).astype(np.int64, copy=False)


def _cell_class_purity(cells: Int64Array, codes: Int64Array, n_classes: int) -> float:
    """Score one resolution: how far each cell's mix departs from uniform."""

    n_cells = int(cells.max()) + 1
    counts = np.zeros((n_cells, n_classes), dtype=np.float64)
    np.add.at(counts, (cells, codes), 1.0)

    occupancy = counts.sum(axis=1)
    share = counts / occupancy[:, None]
    deviation = ((share - 1.0 / n_classes) ** 2).sum(axis=1)
    cell_purity = np.sqrt((n_classes / (n_classes - 1)) * deviation)
    return float((cell_purity * occupancy / codes.shape[0]).sum())


def purity(
    vectors: NDArray[Any] | torch.Tensor,
    labels: NDArray[Any],
    *,
    max_resolution: int = DEFAULT_MAX_RESOLUTION,
) -> ResolutionProfile:
    """Purity: how quickly cells become single-class as the grid refines.

    A dataset whose classes occupy distinct regions reaches pure cells at a
    coarse grid; interleaved classes need a fine one.  Each resolution is
    halved in weight relative to the previous, so the coarse end dominates and
    a dataset that only separates under an extremely fine mesh scores low.
    """

    if max_resolution < 2:
        raise ValueError("max_resolution must be at least two to span a range")
    target = np.asarray(labels)
    if target.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    _, codes = np.unique(target, return_inverse=True)
    codes = codes.reshape(-1).astype(np.int64, copy=False)
    n_classes = int(codes.max()) + 1
    if n_classes < 2:
        raise ValueError("purity needs at least two classes")

    scores = np.asarray(
        [
            _cell_class_purity(cell_assignments(vectors, resolution), codes, n_classes)
            for resolution in range(max_resolution)
        ],
        dtype=np.float64,
    )
    return _resolution_profile(scores, max_resolution)


def _resolution_profile(scores: Float64Array, max_resolution: int) -> ResolutionProfile:
    """Weight, normalize, and integrate a per-resolution score sequence."""

    weighted = scores * (0.5 ** np.arange(max_resolution, dtype=np.float64))
    lowest = weighted.min()
    highest = weighted.max()
    span = highest - lowest
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = (weighted - lowest) / span
    positions = np.arange(max_resolution, dtype=np.float64) / (max_resolution - 1)

    area = float(np.trapezoid(normalized, positions))
    return ResolutionProfile(
        per_resolution=scores,
        weighted=weighted,
        value=area / PURITY_NORMALIZER,
    )


__all__ = [
    "DEFAULT_MAX_RESOLUTION",
    "PURITY_NORMALIZER",
    "SEPARABILITY_NEIGHBOUR_CAP",
    "ResolutionProfile",
    "cell_assignments",
    "neighbourhood_separability",
    "purity",
]


#: The reference caps how many same-class neighbours a sample is scored over,
#: so a dense cell costs no more than a sparse one.
SEPARABILITY_NEIGHBOUR_CAP = 11


def _sample_separability(
    local_distances: Float64Array,
    local_codes: Int64Array,
    position: int,
    cell_size: int,
) -> float:
    """Score one sample: how long its own class dominates its cell neighbours."""

    own = local_codes[position]
    same_class_total = int((local_codes == own).sum()) - 1
    depth = min(same_class_total, SEPARABILITY_NEIGHBOUR_CAP)
    if depth <= 0:
        # A lone sample in its cell is trivially separable; one surrounded only
        # by other classes is not separable at all.
        return 1.0 if cell_size == 1 else 0.0

    row = local_distances[position].copy()
    row[position] = np.inf
    # A stable sort breaks equal distances by position, and cell members are
    # listed in ascending sample order, so this is the reference's tie policy.
    order = np.argsort(row, kind="stable")[:depth]
    agreeing = np.cumsum(local_codes[order] == own).astype(np.float64)
    proportions = agreeing / np.arange(1, depth + 1, dtype=np.float64)

    if depth == 1:
        return float(proportions[0])
    positions = np.arange(depth, dtype=np.float64) / same_class_total
    return float(np.trapezoid(proportions, positions))


def neighbourhood_separability(
    vectors: NDArray[Any] | torch.Tensor,
    labels: NDArray[Any],
    distances: NDArray[Any] | torch.Tensor,
    *,
    max_resolution: int = DEFAULT_MAX_RESOLUTION,
) -> ResolutionProfile:
    """How well a sample's own class dominates its neighbours within its cell.

    Purity asks whether a cell is single-class; this asks, for the cells that
    are mixed, whether the classes are still locally ordered inside them.  A
    cell whose classes sit in separate lobes scores high even though the cell
    itself is impure.

    Unlike :func:`purity`, the resolution profile is integrated as-is: the
    reference neither rescales it to the observed range nor divides by a
    constant, so this value is not on the same scale.

    Takes a full ``[samples, samples]`` distance matrix rather than a tile
    kernel.  Scoring a sample needs its distances to arbitrary other members of
    its cell, which is random access rather than a row sweep, so this measure
    does not stream.
    """

    if max_resolution < 2:
        raise ValueError("max_resolution must be at least two to span a range")
    target = np.asarray(labels)
    _, codes = np.unique(target, return_inverse=True)
    codes = codes.reshape(-1).astype(np.int64, copy=False)

    matrix = (
        distances.detach().cpu().numpy()
        if isinstance(distances, torch.Tensor)
        else np.asarray(distances)
    ).astype(np.float64, copy=False)
    n_samples = codes.shape[0]
    if matrix.shape != (n_samples, n_samples):
        raise ValueError("distances must be a square matrix with one row per sample")

    scores = np.empty(max_resolution, dtype=np.float64)
    for resolution in range(max_resolution):
        cells = cell_assignments(vectors, resolution)
        total = 0.0
        for cell in np.unique(cells):
            members = np.flatnonzero(cells == cell)
            local_distances = matrix[np.ix_(members, members)]
            local_codes = codes[members]
            cell_score = sum(
                _sample_separability(local_distances, local_codes, position, members.shape[0])
                for position in range(members.shape[0])
            )
            total += (cell_score / members.shape[0]) * (members.shape[0] / n_samples)
        scores[resolution] = total

    weighted = scores * (0.5 ** np.arange(max_resolution, dtype=np.float64))
    positions = np.arange(max_resolution, dtype=np.float64) / (max_resolution - 1)
    return ResolutionProfile(
        per_resolution=scores,
        weighted=weighted,
        value=float(np.trapezoid(weighted, positions)),
    )
