"""Do the duplicate rows already in Bank Marketing move its complexity?

The file as distributed contains exact duplicate rows.  Every one of them gives
its twin a distance-0 nearest neighbour, which is the mechanism that collapsed
N2 to zero in the text experiments.  The question is whether the duplicates
present *before any resampling* are enough to shift the published numbers.

Removing them also removes rows, and sample size moves these measures on its
own -- the finding this whole project rests on.  So the comparison carries its
own null control:

===============  ==========================================================
``full``         the file as distributed
``dedup``        each duplicate group collapsed to one representative
``random_drop``  the *same number* of rows dropped at random
===============  ==========================================================

``dedup`` versus ``random_drop`` isolates the duplicates; ``random_drop``
versus ``full`` shows what the size change alone accounts for.

    python -m tabular.dedup_study
"""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import numpy as np

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
    cm,
    degree_of_overlap,
    duplicate_group_ids,
    heom_kernel,
    kdn,
    local_set_cardinality,
    n2,
    neighbourhood_counts,
    neighbourhood_profile,
    resolve_backend,
    separability_index,
    sphere_coverage,
    sphere_radii,
    t1,
)

from .data import load_bank_marketing

DEVICE = "mps"


def panel(vectors, labels, categorical, missing, *, device: str = DEVICE) -> dict[str, float]:
    """Neighbourhood, boundary, and sphere measures over mixed-type columns."""

    backend = resolve_backend(device)
    groups = duplicate_group_ids(vectors)

    def engine() -> TileEngine:
        return TileEngine(
            heom_kernel(vectors, backend=backend, categorical=categorical, missing=missing),
            backend=backend,
            duplicate_group=groups,
        )

    result = engine().run(
        [
            TopKPrefix(max_k=5),
            NearestFriend(labels=labels),
            NearestEnemy(labels=labels),
            LocalSetSize(labels=labels),
        ]
    )
    prefix = result.results["topk"]
    counts = neighbourhood_counts(prefix.indices, labels, max_k=5)
    profile = neighbourhood_profile(prefix.indices, prefix.distances, labels, max_k=5)
    types = borderline(counts)

    # T1 needs unnormalized distances, which HEOM supplies alongside.
    enemy = engine().run([NearestEnemy(labels=labels, use_unnormalized=True)])
    radius = sphere_radii(
        enemy.results["nearest_enemy"].indices, enemy.results["nearest_enemy"].distances
    )
    absorber = engine().run([ContainingSphere(radius=radius)]).results["containing_sphere"]

    return {
        "n": float(len(labels)),
        "kDN": float(kdn(counts)),
        "CM": float(cm(counts)),
        "SI": float(separability_index(counts)),
        "degOver": float(degree_of_overlap(counts)),
        "C1": float(c1(profile)),
        "C2": float(c2(profile)),
        "N2": float(
            n2(
                result.results["nearest_friend"].distances,
                result.results["nearest_enemy"].distances,
                labels,
            )
        ),
        "LSC": float(local_set_cardinality(result.results["local_set_size"], labels)),
        "T1": float(t1(sphere_coverage(radius, absorber), labels)),
        "safe": float(types.safe),
        "borderline": float(types.borderline),
        "duplicate_rows": float((groups >= 0).sum()),
    }


def main() -> None:
    dataset = load_bank_marketing(Path("cache"))
    vectors, labels = dataset.vectors, dataset.labels
    missing = dataset.missing
    groups = duplicate_group_ids(vectors)

    # --- what the duplicates actually are -------------------------------
    members = collections.defaultdict(list)
    for index, group in enumerate(groups):
        if group >= 0:
            members[int(group)].append(index)

    sizes = [len(m) for m in members.values()]
    conflicted = [m for m in members.values() if len({int(labels[i]) for i in m}) > 1]
    print(f"rows                 {len(vectors):,}")
    print(
        f"rows in a duplicate group {int((groups >= 0).sum()):,} "
        f"({100 * (groups >= 0).sum() / len(vectors):.2f}%)"
    )
    print(
        f"distinct groups      {len(members):,}   largest {max(sizes)}   mean {np.mean(sizes):.2f}"
    )
    print(
        f"groups with CONFLICTING labels  {len(conflicted):,} "
        f"({100 * len(conflicted) / max(1, len(members)):.1f}% of groups)"
    )
    print(f"  -> rows carrying a contradicted label: {sum(len(m) for m in conflicted):,}")

    # --- the three conditions -------------------------------------------
    keep_dedup = np.ones(len(vectors), dtype=bool)
    for group in members.values():
        for index in group[1:]:
            keep_dedup[index] = False
    removed = int((~keep_dedup).sum())

    generator = np.random.default_rng(0)
    keep_random = np.ones(len(vectors), dtype=bool)
    keep_random[generator.choice(len(vectors), size=removed, replace=False)] = False

    conditions = {
        "full": np.ones(len(vectors), dtype=bool),
        "dedup": keep_dedup,
        "random_drop": keep_random,
    }

    print(f"\nremoving {removed:,} rows in both dedup and random_drop\n")
    scores = {}
    for name, mask in conditions.items():
        started = time.perf_counter()
        scores[name] = panel(
            np.ascontiguousarray(vectors[mask]),
            labels[mask],
            dataset.categorical,
            None if missing is None else np.ascontiguousarray(missing[mask]),
        )
        print(f"  {name:12s} done in {time.perf_counter() - started:5.1f}s")

    keys = [k for k in scores["full"] if k != "n"]
    print(f"\n{'measure':<14}{'full':>12}{'random_drop':>14}{'dedup':>12}{'dedup-random':>14}")
    print("-" * 66)
    for key in keys:
        f, r, d = scores["full"][key], scores["random_drop"][key], scores["dedup"][key]
        print(f"{key:<14}{f:>12.4f}{r:>14.4f}{d:>12.4f}{d - r:>+14.4f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/dedup_study.json").write_text(json.dumps(scores, indent=2))
    print("\nwrote results/dedup_study.json")
    print("dedup-random is the duplicates' own effect; random_drop-full is the size change.")


if __name__ == "__main__":
    main()
