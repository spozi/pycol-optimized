"""The data-complexity panel, computed with pycol-optimized.

Measures are grouped into a core panel that always runs and an expensive tier
that is opt-in, because a few of them do not scale to the sizes this experiment
reaches:

* ``onb_cover`` runs one engine pass per ball it places.  On scattered
  high-dimensional embeddings the cover can approach one ball per sample, which
  turns ONB/NSG/ICSV/DBC into thousands of full passes.
* ``purity`` grids every feature.  In 768 dimensions each sample lands in its
  own cell almost immediately, so the number is computable but not meaningful.
* ``compute_n1`` builds a full ``n x n`` matrix and a spanning forest over it.

Every measure is guarded individually: one that fails records its error and the
rest of the panel still reports, so a single degenerate statistic cannot cost a
whole run.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pycol_optimized import (
    ContainingSphere,
    LocalSetSize,
    NearestEnemy,
    NearestFriend,
    TileEngine,
    TopKPrefix,
    borderline,
    c1,
    c2,
    class_feature_stats,
    cm,
    compute_metrics,
    d3,
    degree_of_overlap,
    duplicate_group_ids,
    f1v,
    f2,
    f3,
    heom_kernel,
    icsv,
    input_noise,
    kdn,
    local_set_cardinality,
    n2,
    neighbourhood_counts,
    neighbourhood_profile,
    nsg,
    onb,
    onb_cover,
    purity,
    r_value,
    resolve_backend,
    separability_index,
    sphere_coverage,
    sphere_radii,
    t1,
)
from pycol_optimized import dbc as dbc_measure

from .common import as_float

#: borderline() is defined only on exactly five neighbours.
NEIGHBOURS = 5


def _guard(panel: dict[str, float], errors: dict[str, str], name: str, thunk) -> None:
    try:
        panel[name] = as_float(thunk())
    except Exception as error:  # noqa: BLE001 - one bad measure must not stop the panel
        errors[name] = f"{type(error).__name__}: {error}"


def complexity_profile(
    vectors: NDArray[np.float32],
    labels: NDArray[np.int64],
    *,
    device: str = "auto",
    neighbours: int = NEIGHBOURS,
    include_expensive: bool = False,
) -> dict[str, Any]:
    """Score one training set.  Returns the panel, timings, and any failures."""

    started = time.perf_counter()
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    backend = resolve_backend(device)

    def build_engine() -> TileEngine:
        return TileEngine(
            heom_kernel(vectors, backend=backend),
            backend=backend,
            duplicate_group=duplicate_group_ids(vectors),
        )

    panel: dict[str, float] = {}
    errors: dict[str, str] = {}

    # One pass yields every neighbour-derived statistic.
    neighbourhood = build_engine().run(
        [
            TopKPrefix(max_k=neighbours),
            NearestFriend(labels=labels),
            NearestEnemy(labels=labels),
            LocalSetSize(labels=labels),
        ]
    )
    prefix = neighbourhood.results["topk"]
    friend = neighbourhood.results["nearest_friend"]
    enemy = neighbourhood.results["nearest_enemy"]

    counts = neighbourhood_counts(prefix.indices, labels, max_k=neighbours)
    profile = neighbourhood_profile(prefix.indices, prefix.distances, labels, max_k=neighbours)

    _guard(panel, errors, "kDN", lambda: kdn(counts))
    _guard(panel, errors, "CM", lambda: cm(counts))
    _guard(panel, errors, "SI", lambda: separability_index(counts))
    _guard(panel, errors, "degOver", lambda: degree_of_overlap(counts))
    # D3 is a raw per-class count, so it doubles when the arm doubles the data
    # whether or not anything got harder.  Reported as a share of the training
    # set so it is comparable across arms of different sizes; the count is kept
    # alongside it because that is what the reference returns.
    _guard(panel, errors, "D3_rate", lambda: d3(counts).sum() / vectors.shape[0])
    _guard(panel, errors, "D3_count", lambda: d3(counts).sum())
    _guard(panel, errors, "C1", lambda: c1(profile))
    _guard(panel, errors, "C2", lambda: c2(profile))
    _guard(panel, errors, "N2", lambda: n2(friend.distances, enemy.distances, labels))
    _guard(
        panel,
        errors,
        "LSC",
        lambda: local_set_cardinality(neighbourhood.results["local_set_size"], labels),
    )
    _guard(panel, errors, "R_value", lambda: r_value(counts))

    if neighbours == NEIGHBOURS:
        types = borderline(counts)
        for name in ("safe", "borderline", "rare", "outlier"):
            _guard(panel, errors, name, lambda n=name: getattr(types, n))

    # Feature-based measures need no pairwise distances.
    stats = class_feature_stats(vectors, labels, backend=backend)
    _guard(panel, errors, "F1v", lambda: f1v(stats))
    _guard(panel, errors, "F2", lambda: f2(stats))
    _guard(panel, errors, "F3", lambda: f3(stats, vectors))
    _guard(panel, errors, "input_noise", lambda: input_noise(stats))
    _guard(
        panel,
        errors,
        "F1",
        lambda: compute_metrics(vectors, labels, metrics=["F1"], device=device).metrics["F1"],
    )

    # T1 costs two extra passes: one for unnormalized enemy distances, one to
    # find each sphere's absorber.
    def _t1() -> float:
        unnormalized = (
            build_engine()
            .run([NearestEnemy(labels=labels, use_unnormalized=True)])
            .results["nearest_enemy"]
        )
        radius = sphere_radii(unnormalized.indices, unnormalized.distances)
        absorber = (
            build_engine().run([ContainingSphere(radius=radius)]).results["containing_sphere"]
        )
        return t1(sphere_coverage(radius, absorber), labels)

    _guard(panel, errors, "T1", _t1)

    if include_expensive:
        _guard(
            panel,
            errors,
            "N1",
            lambda: compute_metrics(vectors, labels, metrics=["N1"], device=device).metrics["N1"],
        )
        _guard(panel, errors, "purity", lambda: purity(vectors, labels).value)

        balls = onb_cover(build_engine, labels, enemy.distances, backend=backend)
        _guard(panel, errors, "ONB", lambda: onb(balls, labels))
        _guard(panel, errors, "NSG", lambda: nsg(balls, labels))
        _guard(panel, errors, "ICSV", lambda: icsv(balls, labels, vectors.shape[1]))
        _guard(
            panel,
            errors,
            "DBC",
            lambda: dbc_measure(balls, vectors, labels, backend=backend),
        )
        panel["n_balls"] = float(len(balls))

    return {
        "panel": panel,
        "errors": errors,
        "n_samples": int(vectors.shape[0]),
        "n_features": int(vectors.shape[1]),
        "n_duplicate_groups": int(np.unique(duplicate_group_ids(vectors)).size),
        "device": backend.device.type,
        "seconds": time.perf_counter() - started,
    }
