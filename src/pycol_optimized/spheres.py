"""Hypersphere measures.

Each sample grows a sphere until it would touch a differently-labelled sample,
then spheres wholly contained in another are absorbed.  What survives describes
how many distinct regions the classes really occupy.

Two parts dominate the cost and both are quadratic in the sample count: finding
each sample's nearest enemy, and testing containment between every pair of
spheres.  Both run through the tile engine.  The radius resolution and the
absorption tally between them are linear and sequential, and stay on the host.

These measures are defined on unscaled distances, so they read a kernel's
unnormalized matrix.  That makes them meaningful only for purely numeric
inputs, matching the reference, which documents them as such: with categorical
columns the reference's containment test and its radii stop agreeing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from .backend import DTYPE, Backend
from .engine import DistanceTile

Float64Array: TypeAlias = NDArray[np.float64]
Int64Array: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SphereCoverage:
    """Sphere radii and how many spheres each surviving sphere absorbed."""

    radius: Float64Array
    absorbed: Int64Array
    diagnostics: dict[str, Any]

    @property
    def surviving(self) -> NDArray[np.bool_]:
        """Return which spheres were not swallowed by a larger one."""

        return self.absorbed > 0


def sphere_radii(
    enemy_indices: torch.Tensor | NDArray[Any],
    enemy_distances: torch.Tensor | NDArray[Any],
) -> Float64Array:
    """Resolve every sphere's radius from the nearest-enemy assignment.

    Following ``i -> enemy[i]`` walks a functional graph whose enemy distances
    never increase, so every chain ends in a mutually-nearest pair.  That pair
    splits its distance evenly and the rest of the chain unwinds from it.

    The reference expresses this as recursion, which costs a Python frame per
    chain link and fails outright on a chain longer than the interpreter's
    recursion limit.  Resolving each chain iteratively removes that ceiling
    while producing the same radii, including the reference's convention that a
    sample revisited mid-resolution contributes a zero radius.
    """

    enemy = _as_int_vector(enemy_indices)
    distance = _as_float_vector(enemy_distances)
    if enemy.shape != distance.shape:
        raise ValueError("enemy indices and distances must describe the same samples")
    if (enemy < 0).any():
        raise ValueError("every sample needs a nearest enemy; a single-class input has none")

    n_samples = enemy.shape[0]
    radius = np.full(n_samples, -1.0, dtype=np.float64)

    for start in range(n_samples):
        if radius[start] >= 0.0:
            continue
        # Walk to the end of this chain, marking visited links with the
        # reference's temporary zero, then assign radii on the way back.
        chain: list[int] = []
        current = start
        while radius[current] < 0.0:
            opponent = int(enemy[current])
            if current == int(enemy[opponent]):
                half = 0.5 * distance[current]
                radius[opponent] = half
                radius[current] = half
                break
            radius[current] = 0.0
            chain.append(current)
            current = opponent
        for link in reversed(chain):
            radius[link] = abs(distance[link] - radius[int(enemy[link])])

    return radius


@dataclass(eq=False)
class ContainingSphere:
    """Reducer finding the sphere that absorbs each smaller sphere.

    The reference scans candidates from the largest radius downwards and stops
    at the first that contains the sphere, so the absorber is the one latest in
    radius order.  That is a masked maximum over each row, which the tile engine
    can take in the same pass that produced the distances.
    """

    radius: NDArray[Any]
    name: str = "containing_sphere"
    _order: torch.Tensor | None = field(default=None, init=False, repr=False)
    _radius: torch.Tensor | None = field(default=None, init=False, repr=False)
    _absorber: torch.Tensor | None = field(default=None, init=False, repr=False)

    def begin(self, *, n_samples: int, backend: Backend) -> None:
        """Rank the spheres by radius and allocate the per-sphere absorber."""

        values = np.asarray(self.radius, dtype=np.float64).reshape(-1)
        if values.shape[0] != n_samples:
            raise ValueError("radius must contain one entry per sample")
        rank = np.empty(n_samples, dtype=np.int64)
        rank[np.argsort(values)] = np.arange(n_samples, dtype=np.int64)

        self._order = torch.as_tensor(rank, device=backend.device)
        self._radius = torch.as_tensor(
            values.astype(np.float32), dtype=DTYPE, device=backend.device
        )
        self._absorber = torch.full((n_samples,), -1, dtype=torch.long, device=backend.device)

    def update(self, tile: DistanceTile) -> None:
        """Find each row's absorbing sphere within this tile."""

        if self._order is None or self._radius is None or self._absorber is None:
            raise RuntimeError("begin must be called before update")
        source = tile.require_unnormalized()
        rows = slice(tile.row_start, tile.row_stop)

        # Containment is "distance + inner radius <= outer radius", but adding a
        # small distance to a large radius loses it: the sum rounds straight
        # back onto the outer radius and swallows spheres that sit just outside.
        # Comparing the distance against the radius *difference* keeps both
        # sides at the scale of the gap, and subtracting two nearby radii is
        # exact in float32.
        inside = source <= (self._radius.unsqueeze(0) - self._radius[rows].unsqueeze(1))
        larger = self._order.unsqueeze(0) > self._order[rows].unsqueeze(1)
        candidates = torch.where(inside & larger, self._order.to(DTYPE).unsqueeze(0), -1.0)

        best = candidates.max(dim=1).values
        self._absorber[rows] = torch.where(
            best >= 0.0, best.to(torch.long), torch.full_like(best, -1, dtype=torch.long)
        )

    def finish(self) -> Int64Array:
        """Return each sphere's absorber as a radius rank, or ``-1`` for none."""

        if self._absorber is None:
            raise RuntimeError("begin must be called before finish")
        return self._absorber.cpu().numpy().astype(np.int64, copy=False)


def sphere_coverage(radius: NDArray[Any], absorber_rank: NDArray[Any]) -> SphereCoverage:
    """Tally absorptions in radius order, smallest sphere first.

    Sequential by nature: a sphere hands on whatever it has already absorbed, so
    a sphere swallowed early carries its own contents into its absorber.
    """

    radii = np.asarray(radius, dtype=np.float64).reshape(-1)
    absorber = np.asarray(absorber_rank, dtype=np.int64).reshape(-1)
    n_samples = radii.shape[0]
    order = np.argsort(radii)

    absorbed = np.ones(n_samples, dtype=np.int64)
    for rank in range(n_samples - 1):
        sphere = int(order[rank])
        target_rank = int(absorber[sphere])
        if target_rank < 0:
            continue
        target = int(order[target_rank])
        absorbed[target] += absorbed[sphere]
        absorbed[sphere] = 0

    return SphereCoverage(
        radius=radii,
        absorbed=absorbed,
        diagnostics={
            "measure_family": "hypersphere",
            "distance_matrix": "unnormalized",
            "sphere_count": int((absorbed > 0).sum()),
            "absorbed_count": int((absorbed == 0).sum()),
        },
    )


def t1(
    coverage: SphereCoverage,
    labels: NDArray[Any],
    *,
    imb: bool = False,
) -> float | Float64Array:
    """T1: how many spheres it takes to cover the data, per sample.

    One sphere per sample means every point sits in its own region and nothing
    generalizes; a few large spheres mean the classes occupy compact areas.
    """

    target = np.asarray(labels)
    if target.shape[0] != coverage.radius.shape[0]:
        raise ValueError("labels must contain one entry per sample")
    if not imb:
        return float(coverage.surviving.sum() / target.shape[0])

    _, codes = np.unique(target, return_inverse=True)
    codes = codes.reshape(-1).astype(np.int64, copy=False)
    class_count = np.bincount(codes).astype(np.float64)
    totals = np.zeros(class_count.shape[0], dtype=np.float64)
    np.add.at(totals, codes, coverage.surviving.astype(np.float64))
    return totals / class_count


def _as_int_vector(values: torch.Tensor | NDArray[Any]) -> Int64Array:
    array = (
        values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    )
    return array.reshape(-1).astype(np.int64, copy=False)


def _as_float_vector(values: torch.Tensor | NDArray[Any]) -> Float64Array:
    array = (
        values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    )
    return array.reshape(-1).astype(np.float64, copy=False)


__all__ = [
    "Ball",
    "ContainingSphere",
    "GreedyCoverStep",
    "SphereCoverage",
    "dbc",
    "icsv",
    "nsg",
    "onb",
    "onb_cover",
    "sphere_coverage",
    "sphere_radii",
    "t1",
]


@dataclass(eq=False)
class GreedyCoverStep:
    """Reducer scoring every candidate ball in one greedy iteration.

    A candidate's ball reaches as far as its nearest enemy, and its score is how
    many still-uncovered samples of its own class that reach covers.  Scoring
    all candidates is a masked row-count, so one tile pass evaluates the whole
    iteration; only the choice between iterations is sequential.

    ``uncovered`` is read fresh on every pass, so the driver shrinks it in place
    between iterations rather than rebuilding the reducer.
    """

    enemy_distance: torch.Tensor
    uncovered: torch.Tensor
    codes: torch.Tensor
    target_class: int
    name: str = "greedy_cover"
    _counts: torch.Tensor | None = field(default=None, init=False, repr=False)

    def begin(self, *, n_samples: int, backend: Backend) -> None:
        """Allocate the per-candidate scores."""

        if self.uncovered.shape[0] != n_samples:
            raise ValueError("uncovered must contain one entry per sample")
        self._counts = torch.full((n_samples,), -1, dtype=torch.long, device=backend.device)

    def update(self, tile: DistanceTile) -> None:
        """Score this tile's candidates against the current uncovered set."""

        if self._counts is None:
            raise RuntimeError("begin must be called before update")
        rows = slice(tile.row_start, tile.row_stop)
        eligible = self.uncovered[rows] & (self.codes[rows] == self.target_class)
        reachable = self.uncovered.unsqueeze(0) & (self.codes == self.target_class).unsqueeze(0)

        covered = (tile.normalized <= self.enemy_distance[rows].unsqueeze(1)) & reachable
        self._counts[rows] = torch.where(eligible, covered.sum(dim=1), -1)

    def finish(self) -> torch.Tensor:
        """Return each candidate's coverage, with ``-1`` for ineligible rows."""

        if self._counts is None:
            raise RuntimeError("begin must be called before finish")
        return self._counts


@dataclass(frozen=True, slots=True)
class Ball:
    """One ball of the greedy cover."""

    size: int
    class_index: int
    center: int
    radius: float


def onb_cover(
    build_engine: Any,
    labels: NDArray[Any],
    enemy_distance: torch.Tensor | NDArray[Any],
    *,
    backend: Backend,
) -> list[Ball]:
    """Cover each class with as few enemy-free balls as greedy selection allows.

    ``build_engine`` is called to obtain a fresh :class:`~.engine.TileEngine`
    per iteration, since the count depends on what is still uncovered.  The
    cover is greedy, so the number of passes is the number of balls rather than
    a function of the sample count.

    Ties go to the smallest sample index, matching the reference's strict
    ``>`` comparison over candidates visited in ascending order.
    """

    target = np.asarray(labels)
    _, codes = np.unique(target, return_inverse=True)
    codes = codes.reshape(-1).astype(np.int64, copy=False)
    n_classes = int(codes.max()) + 1

    device_codes = torch.as_tensor(codes, device=backend.device)
    radius = (
        enemy_distance
        if isinstance(enemy_distance, torch.Tensor)
        else torch.as_tensor(np.asarray(enemy_distance, dtype=np.float32), device=backend.device)
    ).to(device=backend.device, dtype=DTYPE)

    balls: list[Ball] = []
    for class_index in range(n_classes):
        uncovered = torch.as_tensor(codes == class_index, device=backend.device)
        while bool(uncovered.any()):
            engine = build_engine()
            counts = engine.run(
                [
                    GreedyCoverStep(
                        enemy_distance=radius,
                        uncovered=uncovered,
                        codes=device_codes,
                        target_class=class_index,
                    )
                ]
            ).results["greedy_cover"]

            best = int(counts.max().item())
            center = int(torch.nonzero(counts == best)[0].item())

            # One extra single-row tile resolves which samples that winner
            # covers; scoring every candidate up front would have to keep an
            # n x n membership matrix to answer the same question.
            row = engine.kernel.tile(center, center + 1)[0][0]
            covered = (row <= radius[center]) & uncovered & (device_codes == class_index)
            uncovered = uncovered & ~covered

            balls.append(
                Ball(
                    size=best,
                    class_index=class_index,
                    center=center,
                    radius=float(radius[center].item()),
                )
            )
    return balls


def onb(
    balls: list[Ball],
    labels: NDArray[Any],
    *,
    imb: bool = False,
    is_total: bool = False,
) -> float | Float64Array:
    """ONB: balls needed per sample, averaged over classes.

    Few balls per class means the class occupies a compact region an algorithm
    can bound simply; approaching one ball per sample means it does not.
    """

    target = np.asarray(labels)
    _, codes = np.unique(target, return_inverse=True)
    codes = codes.reshape(-1).astype(np.int64, copy=False)
    class_count = np.bincount(codes).astype(np.float64)

    if is_total:
        return float(len(balls) / target.shape[0])

    per_class = np.zeros(class_count.shape[0], dtype=np.float64)
    for ball in balls:
        per_class[ball.class_index] += 1.0
    per_class /= class_count
    if imb:
        return per_class
    return float(per_class.sum() / class_count.shape[0])


def nsg(
    balls: list[Ball],
    labels: NDArray[Any],
    *,
    imb: bool = False,
) -> float | Float64Array:
    """NSG: how many samples an average ball holds.

    Large balls mean the classes group into few broad regions; balls holding
    one sample each mean the data is scattered.
    """

    if not balls:
        raise ValueError("nsg needs at least one ball")
    if not imb:
        return float(np.mean([ball.size for ball in balls]))

    target = np.asarray(labels)
    _, codes = np.unique(target, return_inverse=True)
    class_count = np.bincount(codes.reshape(-1).astype(np.int64, copy=False)).astype(np.float64)
    per_class = np.zeros(class_count.shape[0], dtype=np.float64)
    for ball in balls:
        per_class[ball.class_index] += 1.0
    return class_count / per_class


def icsv(
    balls: list[Ball],
    labels: NDArray[Any],
    n_features: int,
    *,
    normalize: bool = True,
    imb: bool = False,
) -> float | Float64Array:
    """ICSV: how unevenly the balls are packed, as the spread of their densities.

    Density is a ball's sample count over its hypervolume, so a dataset whose
    regions are uniformly dense scores near zero however large those regions
    are.  Radii are scaled by the largest so the volume term stays in range:
    raising an unscaled radius to the feature-count power overflows quickly.

    A zero-radius ball has no volume and yields an infinite density, which
    propagates rather than being silently dropped, as it does in the reference.
    """

    if not balls:
        raise ValueError("icsv needs at least one ball")
    radius = np.asarray([ball.radius for ball in balls], dtype=np.float64)
    if normalize:
        largest = radius.max()
        if largest > 0.0:
            radius = radius / largest

    unit_volume = math.pi ** (n_features / 2) / math.gamma(n_features / 2 + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.asarray([ball.size for ball in balls], dtype=np.float64) / (
            unit_volume * radius**n_features
        )

    if not imb:
        return float(np.std(density))

    target = np.asarray(labels)
    _, codes = np.unique(target, return_inverse=True)
    n_classes = int(codes.reshape(-1).max()) + 1
    return np.asarray(
        [
            float(
                np.std([d for d, ball in zip(density, balls, strict=True) if ball.class_index == c])
            )
            for c in range(n_classes)
        ],
        dtype=np.float64,
    )


def dbc(
    balls: list[Ball],
    vectors: NDArray[Any] | torch.Tensor,
    labels: NDArray[Any],
    *,
    backend: Backend,
    imb: bool = False,
) -> float | Float64Array:
    """DBC: how tangled the boundary between classes is, judged from the cover.

    Reduces each class to its ball centres, spans them with a minimum spanning
    tree, and counts the centres touching an edge that crosses classes.  Working
    from the centres rather than every sample asks whether the *regions* the
    classes occupy interleave, which a handful of stray points cannot fake.

    The centres are re-normalized against their own feature ranges rather than
    the full dataset's, matching the reference: it rebuilds the distance matrix
    from the reduced set, so the ranges come from the centres alone.
    """

    from .distance import heom_kernel
    from .n1 import minimum_spanning_forest

    if not balls:
        raise ValueError("dbc needs at least one ball")
    centres = np.asarray([ball.center for ball in balls], dtype=np.int64)
    if centres.shape[0] < 2:
        raise ValueError("dbc needs at least two balls to span")

    matrix = (
        vectors.detach().cpu().numpy() if isinstance(vectors, torch.Tensor) else np.asarray(vectors)
    )
    target = np.asarray(labels)
    _, codes = np.unique(target, return_inverse=True)
    codes = codes.reshape(-1).astype(np.int64, copy=False)
    centre_codes = codes[centres]

    kernel = heom_kernel(matrix[centres], backend=backend)
    distances = kernel.tile(0, centres.shape[0])[0].cpu().numpy().astype(np.float64)

    edges = minimum_spanning_forest(distances)
    crossing = edges[centre_codes[edges[:, 0]] != centre_codes[edges[:, 1]]]
    touched = np.unique(crossing.reshape(-1)) if crossing.size else np.empty(0, dtype=np.int64)

    if not imb:
        return float(touched.shape[0] / len(balls))

    n_classes = int(codes.max()) + 1
    per_class_touched = np.zeros(n_classes, dtype=np.float64)
    np.add.at(per_class_touched, centre_codes[touched], 1.0)
    per_class_balls = np.zeros(n_classes, dtype=np.float64)
    np.add.at(per_class_balls, centre_codes, 1.0)
    return per_class_touched / per_class_balls
