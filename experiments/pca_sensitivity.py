"""Is the artifact finding an artifact of 768 dimensions?

Recomputes the complexity panel over a range of PCA ranks and asks, at each
one, whether either treatment beats its own null control.  This is a
sensitivity analysis, not a search for a best rank: tuning the rank until the
conclusion changes would be selecting the dimensionality that produces a
preferred answer.

The projection is fitted **once on the baseline arm** and applied unchanged to
every other arm.  Fitting per arm would put each in its own coordinate system
and destroy exactly the between-arm comparability the design rests on.

    python pca_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from complexity_augmentation.complexity import complexity_profile
from complexity_augmentation.data import load_split
from sklearn.decomposition import PCA

ARMS = ("baseline", "duplicate", "uniform", "minority_duplicate", "minority")
PAIRS = (("uniform", "duplicate"), ("minority", "minority_duplicate"))
#: Measures where lower means simpler, so a treatment "wins" by scoring below
#: its control.
DIRECTED = ("kDN", "CM", "degOver", "C1", "C2", "N2", "T1")
RANKS = (2, 5, 10, 25, 50, 100, 200, 400, 768)

CACHE = Path("cache")
RESULTS = Path("results/phrasebank50_p015/results.json")
TAG = "embed_phrasebank_50_{arm}_p0.15_sequential_n{n}.npy"


def main() -> None:
    stored = json.loads(RESULTS.read_text())
    split = load_split("phrasebank_50", CACHE)
    minority = split.minority_label

    vectors, labels = {}, {}
    for arm in ARMS:
        n = stored["arms"][arm]["n_train"]
        vectors[arm] = np.load(CACHE / TAG.format(arm=arm, n=n)).astype(np.float32)
        # Arm labels are reconstructible: every arm appends to the original
        # training labels, minority arms with the minority class.
        base = split.train_labels
        extra = n - len(base)
        labels[arm] = (
            np.concatenate([base, base])
            if arm in {"duplicate", "uniform"}
            else np.concatenate([base, np.full(extra, minority, dtype=np.int64)])
            if extra
            else base
        )

    # One projection for every arm, fitted on baseline alone.
    full = PCA(n_components=min(RANKS[-1], *vectors["baseline"].shape)).fit(vectors["baseline"])
    variance = np.cumsum(full.explained_variance_ratio_)

    print("rank  var%   " + "  ".join(f"{a:>10.10}" for a in ARMS) + "   uniform  minority")
    print("      " + "-" * 86)
    rows = []
    for rank in RANKS:
        projected = {
            arm: np.ascontiguousarray(
                (vectors[arm] - full.mean_) @ full.components_[:rank].T, dtype=np.float32
            )
            for arm in ARMS
        }
        panels = {
            arm: complexity_profile(projected[arm], labels[arm], device="cpu")["panel"]
            for arm in ARMS
        }
        wins = {aug: [m for m in DIRECTED if panels[aug][m] < panels[ctl][m]] for aug, ctl in PAIRS}
        print(
            f"{rank:<5d} {100 * variance[min(rank, len(variance)) - 1]:>4.1f}  "
            + "  ".join(f"{panels[a]['kDN']:>10.4f}" for a in ARMS)
            + f"   {len(wins['uniform'])}/7      {len(wins['minority'])}/7"
        )
        rows.append(
            {
                "rank": rank,
                "explained_variance": float(variance[min(rank, len(variance)) - 1]),
                "kDN": {a: panels[a]["kDN"] for a in ARMS},
                "uniform_wins": wins["uniform"],
                "minority_wins": wins["minority"],
            }
        )

    print("\n(columns are kDN per arm; the last two are how many of 7 directed")
    print(" measures each treatment beats its own control on)")
    Path("results/pca_sensitivity.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote results/pca_sensitivity.json")


if __name__ == "__main__":
    main()
