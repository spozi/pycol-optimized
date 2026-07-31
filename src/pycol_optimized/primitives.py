"""Measure primitives computed from streamed distance tiles.

Almost every PyCOL measure is a small function of one of a handful of
primitives rather than of the distance matrix itself.  Implementing the
primitives once and composing measures on top of them keeps the per-measure
code small and gives every measure the same tiling and tie-breaking guarantees.

This module provides the two primitives the neighbourhood and structural
families are built on:

``TopKPrefix``
    The ordered k-nearest-neighbour prefix, feeding N3, kDN, CM, C1, C2, SI,
    N2, N4, D3, R-value, degOver, and borderline.

``NearestEnemy``
    The nearest sample of a different class, feeding T1, LSC, Clust, DBC, NSG,
    and ICSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from .backend import DTYPE, Backend
from .engine import DistanceTile

Int64Array: TypeAlias = NDArray[np.int64]

#: Neighbours are ranked by distance and ties are resolved by the smaller
#: original sample index.  This matches the reference implementation, whose
#: repeated ``argmin`` returns the first occurrence of an exact minimum.
TIE_POLICY = "distance_then_smallest_sample_index"

#: Only a sample's own diagonal entry is excluded.  Off-diagonal zero-distance
#: duplicates remain eligible neighbours.
SELF_EXCLUSION = "diagonal_only"


def order_by_value_then_index(
    distances: torch.Tensor,
    indices: torch.Tensor,
    *,
    backend: Backend,
    ascending: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort each row lexicographically by distance, then by sample index.

    Set ``ascending`` only when the caller knows the distances already rise per
    row, as ``topk`` with ``sorted=True`` returns them.  The sorts can then be
    skipped whenever no two neighbouring values are equal, which is the common
    case.  The skip is *not* valid for an arbitrary permutation: the straddle
    correction below emits its selection in sample-index order, whose distances
    can descend without any two adjacent ones being equal.
    """

    if ascending and (
        distances.shape[1] < 2 or not bool((distances[:, :-1] == distances[:, 1:]).any().item())
    ):
        return distances, indices

    if backend.supports_stable_sort:
        target_distances, target_indices = distances, indices
        restore = False
    else:
        # The pair is only [block, k]; a host round trip is cheaper than an
        # unstable ordering that would silently break the tie policy.
        target_distances = distances.cpu()
        target_indices = indices.cpu()
        restore = True

    by_index = torch.argsort(target_indices, dim=1, stable=True)
    target_distances = torch.gather(target_distances, 1, by_index)
    target_indices = torch.gather(target_indices, 1, by_index)

    by_value = torch.argsort(target_distances, dim=1, stable=True)
    target_distances = torch.gather(target_distances, 1, by_value)
    target_indices = torch.gather(target_indices, 1, by_value)

    if restore:
        return target_distances.to(distances.device), target_indices.to(indices.device)
    return target_distances, target_indices


def ordered_topk(
    values: torch.Tensor,
    max_k: int,
    *,
    column_index_float: torch.Tensor,
    backend: Backend,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the ``max_k`` smallest entries per row under the tie policy.

    ``torch.topk`` alone is not enough: it gives no guarantee about which of
    several equal values it keeps, so equal distances straddling the cut could
    select the wrong samples, and equal distances inside the cut could be
    returned in the wrong order.  Both are corrected here, and the correction
    for the straddling case runs only when equal distances actually cross the
    boundary.
    """

    block, n_samples = values.shape
    # Selecting one extra neighbour turns straddle detection into an O(block)
    # comparison: equal distances cross the cut exactly when the entries at
    # ranks k and k+1 are equal.  Counting them instead would cost a full
    # sweep of the tile on every pass, and ties are the rare case.
    probe = min(max_k + 1, n_samples)
    selected = torch.topk(values, probe, dim=1, largest=False, sorted=True)
    distances = selected.values[:, :max_k]
    indices = selected.indices[:, :max_k]

    threshold = distances[:, -1:]
    straddling = (
        selected.values[:, max_k] == selected.values[:, max_k - 1]
        if probe > max_k
        else torch.zeros(block, dtype=torch.bool, device=values.device)
    )

    if bool(straddling.any().item()):
        below = (values < threshold).sum(dim=1)
        # Positions are carried as float32 so the selection works identically on
        # every backend; column indices are exact well past any realistic n.
        sentinel = float(n_samples)
        positions = column_index_float.unsqueeze(0).expand(block, n_samples)
        strict_positions = torch.where(values < threshold, positions, sentinel)
        tied_positions = torch.where(values == threshold, positions, sentinel)

        strict_columns = torch.topk(
            strict_positions, max_k, dim=1, largest=False, sorted=True
        ).values.to(torch.long)
        tied_columns = torch.topk(
            tied_positions, max_k, dim=1, largest=False, sorted=True
        ).values.to(torch.long)

        rank = torch.arange(max_k, device=values.device).unsqueeze(0)
        below_column = below.unsqueeze(1)
        from_tied = torch.gather(tied_columns, 1, (rank - below_column).clamp(min=0))
        merged = torch.where(rank < below_column, strict_columns, from_tied)

        indices = torch.where(straddling.unsqueeze(1), merged, indices)
        distances = torch.gather(values, 1, indices)
        # The corrected rows are in sample-index order, not distance order.
        return order_by_value_then_index(distances, indices, backend=backend)

    return order_by_value_then_index(distances, indices, backend=backend, ascending=True)


def nearest_under_mask(
    candidates: torch.Tensor,
    tile: DistanceTile,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce masked rows to their minimum and its smallest achieving index.

    ``torch.min`` gives no guarantee about which of several equal minima it
    reports, so the winning column is taken as the smallest sample index among
    the positions achieving the minimum.  Rows with no candidate at all keep an
    infinite distance and an index of ``-1``.
    """

    best = candidates.min(dim=1).values
    found = torch.isfinite(best)
    sentinel = float(tile.n_samples)
    positions = tile.column_index_float.unsqueeze(0).expand_as(candidates)
    winners = torch.where(candidates == best.unsqueeze(1), positions, sentinel)
    first = winners.min(dim=1).values.to(torch.long)
    return best, torch.where(found, first, torch.full_like(first, -1))


@dataclass(frozen=True, slots=True)
class KNNPrefix:
    """Ordered nearest neighbours for every sample."""

    indices: torch.Tensor
    distances: torch.Tensor
    max_k: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NearestEnemyResult:
    """Nearest differently-labelled sample for every sample.

    Samples with no enemy, which happens when a class fills the dataset, carry
    index ``-1`` and an infinite distance.
    """

    indices: torch.Tensor
    distances: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass(eq=False)
class TopKPrefix:
    """Reducer producing the ordered k-nearest-neighbour prefix."""

    max_k: int
    name: str = "topk"
    _backend: Backend | None = field(default=None, init=False, repr=False)
    _indices: torch.Tensor | None = field(default=None, init=False, repr=False)
    _distances: torch.Tensor | None = field(default=None, init=False, repr=False)

    def begin(self, *, n_samples: int, backend: Backend) -> None:
        """Allocate the ``[n_samples, max_k]`` output buffers."""

        if isinstance(self.max_k, bool) or not isinstance(self.max_k, int):
            raise TypeError("max_k must be an integer")
        if not 1 <= self.max_k < n_samples:
            raise ValueError("max_k must satisfy 1 <= max_k < number of samples")
        self._backend = backend
        self._indices = torch.empty(
            (n_samples, self.max_k), dtype=torch.long, device=backend.device
        )
        self._distances = torch.empty((n_samples, self.max_k), dtype=DTYPE, device=backend.device)

    def update(self, tile: DistanceTile) -> None:
        """Select this tile's neighbour prefix and store it."""

        if self._backend is None or self._indices is None or self._distances is None:
            raise RuntimeError("begin must be called before update")
        distances, indices = ordered_topk(
            tile.self_excluded(),
            self.max_k,
            column_index_float=tile.column_index_float,
            backend=self._backend,
        )
        self._distances[tile.row_start : tile.row_stop] = distances
        self._indices[tile.row_start : tile.row_stop] = indices

    def finish(self) -> KNNPrefix:
        """Return the completed neighbour prefix."""

        if self._indices is None or self._distances is None:
            raise RuntimeError("begin must be called before finish")
        return KNNPrefix(
            indices=self._indices,
            distances=self._distances,
            max_k=self.max_k,
            diagnostics={
                "primitive": "topk_prefix",
                "max_k": self.max_k,
                "tie_policy": TIE_POLICY,
                "self_exclusion": SELF_EXCLUSION,
                "zero_distance_duplicates_retained": True,
            },
        )


@dataclass(eq=False)
class NearestEnemy:
    """Reducer producing each sample's nearest differently-labelled sample.

    ``use_unnormalized`` selects the raw distance matrix, which the hypersphere
    measures are defined on, rather than the range-normalized one.
    """

    labels: NDArray[Any]
    use_unnormalized: bool = False
    name: str = "nearest_enemy"
    _codes: torch.Tensor | None = field(default=None, init=False, repr=False)
    _indices: torch.Tensor | None = field(default=None, init=False, repr=False)
    _distances: torch.Tensor | None = field(default=None, init=False, repr=False)

    def begin(self, *, n_samples: int, backend: Backend) -> None:
        """Encode labels and allocate the per-sample outputs."""

        target = np.asarray(self.labels)
        if target.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if target.shape[0] != n_samples:
            raise ValueError("labels must contain one entry per sample")
        _, codes = np.unique(target, return_inverse=True)
        self._codes = torch.as_tensor(
            codes.reshape(-1).astype(np.int64, copy=False), device=backend.device
        )
        self._indices = torch.full((n_samples,), -1, dtype=torch.long, device=backend.device)
        self._distances = torch.full((n_samples,), torch.inf, dtype=DTYPE, device=backend.device)

    def update(self, tile: DistanceTile) -> None:
        """Reduce this tile to a nearest enemy per row."""

        if self._codes is None or self._indices is None or self._distances is None:
            raise RuntimeError("begin must be called before update")
        source = tile.require_unnormalized() if self.use_unnormalized else tile.normalized

        row_codes = self._codes[tile.row_start : tile.row_stop].unsqueeze(1)
        is_enemy = row_codes != self._codes.unsqueeze(0)
        candidates = torch.where(is_enemy, source, torch.inf)

        best, first = nearest_under_mask(candidates, tile)
        self._distances[tile.row_start : tile.row_stop] = best
        self._indices[tile.row_start : tile.row_stop] = first

    def finish(self) -> NearestEnemyResult:
        """Return the completed nearest-enemy assignment."""

        if self._indices is None or self._distances is None:
            raise RuntimeError("begin must be called before finish")
        return NearestEnemyResult(
            indices=self._indices,
            distances=self._distances,
            diagnostics={
                "primitive": "nearest_enemy",
                "tie_policy": TIE_POLICY,
                "distance_matrix": "unnormalized" if self.use_unnormalized else "normalized",
                "unmatched_sample_count": int((self._indices < 0).sum().item()),
            },
        )


__all__ = [
    "NearestFriend",
    "nearest_under_mask",
    "LocalSetSize",
    "SELF_EXCLUSION",
    "TIE_POLICY",
    "KNNPrefix",
    "NearestEnemy",
    "NearestEnemyResult",
    "TopKPrefix",
    "order_by_value_then_index",
    "ordered_topk",
]


@dataclass(eq=False)
class LocalSetSize:
    """Reducer counting each sample's local set.

    A sample's local set is the set of same-class samples nearer to it than its
    nearest enemy, counting the sample itself.  Both halves live in the same
    row, so row-only tiling lets a single pass produce them: splitting the
    matrix by columns instead would need the enemy distance before the count
    could start.

    Feeds LSC and the cluster cores of Clust.
    """

    labels: NDArray[Any]
    name: str = "local_set_size"
    _codes: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sizes: torch.Tensor | None = field(default=None, init=False, repr=False)

    def begin(self, *, n_samples: int, backend: Backend) -> None:
        """Encode labels and allocate the per-sample counts."""

        target = np.asarray(self.labels)
        if target.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if target.shape[0] != n_samples:
            raise ValueError("labels must contain one entry per sample")
        _, codes = np.unique(target, return_inverse=True)
        self._codes = torch.as_tensor(
            codes.reshape(-1).astype(np.int64, copy=False), device=backend.device
        )
        self._sizes = torch.zeros(n_samples, dtype=torch.long, device=backend.device)

    def update(self, tile: DistanceTile) -> None:
        """Count same-class samples nearer than each row's nearest enemy."""

        if self._codes is None or self._sizes is None:
            raise RuntimeError("begin must be called before update")
        # Deliberately the unmasked tile: a sample belongs to its own local set.
        source = tile.normalized
        row_codes = self._codes[tile.row_start : tile.row_stop].unsqueeze(1)
        is_friend = row_codes == self._codes.unsqueeze(0)

        enemy_distance = torch.where(is_friend, torch.inf, source).min(dim=1).values
        inside = is_friend & (source < enemy_distance.unsqueeze(1))
        self._sizes[tile.row_start : tile.row_stop] = inside.sum(dim=1)

    def finish(self) -> torch.Tensor:
        """Return the per-sample local set sizes."""

        if self._sizes is None:
            raise RuntimeError("begin must be called before finish")
        return self._sizes


@dataclass(eq=False)
class NearestFriend:
    """Reducer producing each sample's nearest same-class neighbour.

    Excludes the sample itself, so it reads the self-excluded tile rather than
    the raw one.  Paired with :class:`NearestEnemy` in a single pass it gives
    the intra- and inter-class distances N2 is built from.

    Samples that are the sole member of their class carry index ``-1`` and an
    infinite distance, which propagates the way the reference's ``inf`` does.
    """

    labels: NDArray[Any]
    name: str = "nearest_friend"
    _codes: torch.Tensor | None = field(default=None, init=False, repr=False)
    _indices: torch.Tensor | None = field(default=None, init=False, repr=False)
    _distances: torch.Tensor | None = field(default=None, init=False, repr=False)

    def begin(self, *, n_samples: int, backend: Backend) -> None:
        """Encode labels and allocate the per-sample outputs."""

        target = np.asarray(self.labels)
        if target.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if target.shape[0] != n_samples:
            raise ValueError("labels must contain one entry per sample")
        _, codes = np.unique(target, return_inverse=True)
        self._codes = torch.as_tensor(
            codes.reshape(-1).astype(np.int64, copy=False), device=backend.device
        )
        self._indices = torch.full((n_samples,), -1, dtype=torch.long, device=backend.device)
        self._distances = torch.full((n_samples,), torch.inf, dtype=DTYPE, device=backend.device)

    def update(self, tile: DistanceTile) -> None:
        """Reduce this tile to a nearest same-class neighbour per row."""

        if self._codes is None or self._indices is None or self._distances is None:
            raise RuntimeError("begin must be called before update")
        row_codes = self._codes[tile.row_start : tile.row_stop].unsqueeze(1)
        is_friend = row_codes == self._codes.unsqueeze(0)
        # The self-excluded tile carries an infinite diagonal, which is exactly
        # the "and i != j" the definition asks for.
        candidates = torch.where(is_friend, tile.self_excluded(), torch.inf)

        best, first = nearest_under_mask(candidates, tile)
        self._distances[tile.row_start : tile.row_stop] = best
        self._indices[tile.row_start : tile.row_stop] = first

    def finish(self) -> NearestEnemyResult:
        """Return the completed nearest-friend assignment."""

        if self._indices is None or self._distances is None:
            raise RuntimeError("begin must be called before finish")
        return NearestEnemyResult(
            indices=self._indices,
            distances=self._distances,
            diagnostics={
                "primitive": "nearest_friend",
                "tie_policy": TIE_POLICY,
                "distance_matrix": "normalized",
                "unmatched_sample_count": int((self._indices < 0).sum().item()),
            },
        )
