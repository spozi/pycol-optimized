"""Does the artifact hold across the standard imbalanced benchmark suite?

Bank Marketing showed that no resampling method beats a matched null control on
complexity.  One dataset is not a result, so this repeats the comparison across
the 27-dataset collection shipped by ``imbalanced-learn`` -- the benchmark suite
this literature actually reports on, spanning ratios from 8.6:1 to 129.5:1.

Only the complexity half is swept.  Complexity is deterministic given the data,
so no cross-validation is needed here: one computation per dataset and arm
answers the question.  The accuracy half stays on Bank Marketing, where the
protocol can be done properly with folds.

For every dataset and method the script reports two differences:

- against ``baseline``, which is what the literature publishes
- against the method's **matched control**, which is what it should publish

The claim under test is that the first is large and the second is not.

    python -m tabular.keel_sweep
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .dedup_study import panel
from .resample import Arm, matched_control

#: All-numeric datasets, so SMOTE proper rather than the mixed-type SMOTENC.
METHODS = ("ros", "rus", "smote", "enn")

#: Measures reported. kDN and N2 are the pre-specified primaries; the rest are
#: descriptive. Every one is oriented so lower means simpler.
MEASURES = ("kDN", "N2", "C1", "C2", "T1", "CM", "degOver")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--max-n",
        type=int,
        default=30_000,
        help="stratified subsample above this size; a resampled arm is roughly "
        "twice the rows and cost grows with the square of them",
    )
    parser.add_argument(
        "--control-draws",
        type=int,
        default=3,
        help="independent control draws to average. A control is one random "
        "resample, so its own sampling noise sits inside every comparison -- "
        "badly so for undersampling on small datasets, where the surviving "
        "subset is tiny and two draws differ a lot",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets", nargs="*", default=None, help="subset by name")
    parser.add_argument("--output", type=Path, default=Path("results/keel_sweep.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/imblearn"))
    return parser.parse_args(argv)


def _subsample(vectors, labels, limit: int, seed: int):
    """Stratified cap, so the class ratio survives the size reduction."""

    if len(labels) <= limit:
        return vectors, labels
    generator = np.random.default_rng(seed)
    share = limit / len(labels)
    keep: list[int] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        take = max(2, int(round(len(members) * share)))
        keep.extend(generator.choice(members, size=min(take, len(members)), replace=False).tolist())
    index = np.asarray(sorted(keep))
    return np.ascontiguousarray(vectors[index]), labels[index]


def resample(name: str, vectors, labels, *, seed: int):
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler

    minority = int(np.bincount(labels).min())
    samplers = {
        "ros": lambda: RandomOverSampler(random_state=seed),
        "rus": lambda: RandomUnderSampler(random_state=seed),
        # SMOTE interpolates towards k neighbours of the same class, so it
        # needs at least that many to exist.
        "smote": lambda: SMOTE(random_state=seed, k_neighbors=min(5, max(1, minority - 1))),
        "enn": EditedNearestNeighbours,
    }
    resampled, new_labels = samplers[name]().fit_resample(vectors, labels)
    return (
        np.ascontiguousarray(resampled, dtype=np.float32),
        np.asarray(new_labels, dtype=np.int64),
    )


def main(argv: list[str] | None = None) -> None:
    from imblearn.datasets import fetch_datasets

    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    collection = fetch_datasets(data_home=str(args.cache_dir))
    names = args.datasets or sorted(collection, key=lambda k: len(collection[k].target))
    results: dict[str, Any] = {}
    started = time.perf_counter()

    for position, name in enumerate(names, 1):
        bundle = collection[name]
        vectors = np.ascontiguousarray(bundle.data, dtype=np.float32)
        # The collection encodes the minority as +1 and the majority as -1.
        labels = (np.asarray(bundle.target) > 0).astype(np.int64)
        vectors, labels = _subsample(vectors, labels, args.max_n, args.seed)
        counts = np.bincount(labels)
        entry: dict[str, Any] = {
            "n": int(len(labels)),
            "n_features": int(vectors.shape[1]),
            "imbalance_ratio": float(counts.max() / counts.min()),
            "arms": {},
        }
        print(
            f"[{position}/{len(names)}] {name:22s} n={len(labels):<7d} "
            f"d={vectors.shape[1]:<4d} ratio={entry['imbalance_ratio']:.1f}:1",
            flush=True,
        )

        try:
            base = panel(
                vectors, labels, np.zeros(vectors.shape[1], dtype=bool), None, device=args.device
            )
        except Exception as error:  # noqa: BLE001 - one bad dataset must not stop the sweep
            print(f"    baseline FAILED {type(error).__name__}: {error}", flush=True)
            results[name] = {**entry, "error": str(error)}
            continue
        entry["arms"]["baseline"] = base

        for method in METHODS:
            try:
                rx, ry = resample(method, vectors, labels, seed=args.seed)
                arm = Arm(method, rx, ry, None)
                nothing = np.zeros(vectors.shape[1], dtype=bool)
                entry["arms"][method] = panel(rx, ry, nothing, None, device=args.device)

                # Several independent control draws, so an arm-versus-control
                # difference can be read against the control's own spread
                # rather than against a single lucky or unlucky sample.
                draws = [
                    panel(control.vectors, control.labels, nothing, None, device=args.device)
                    for control in (
                        matched_control(arm, vectors, labels, missing=None, seed=args.seed + d)
                        for d in range(args.control_draws)
                    )
                ]
                entry["arms"][f"{method}_control"] = {
                    k: float(np.mean([d[k] for d in draws])) for k in draws[0]
                }
                entry["arms"][f"{method}_control_sd"] = {
                    k: float(np.std([d[k] for d in draws], ddof=1)) if len(draws) > 1 else 0.0
                    for k in draws[0]
                }

                line = []
                for measure in ("kDN", "N2"):
                    a = entry["arms"][method][measure]
                    c = entry["arms"][f"{method}_control"][measure]
                    sd = entry["arms"][f"{method}_control_sd"][measure]
                    line.append(
                        f"{measure} base{a - base[measure]:+.4f} ctrl{a - c:+.4f}(sd{sd:.4f})"
                    )
                print(f"    {method:8s} {'  '.join(line)}", flush=True)
            except Exception as error:  # noqa: BLE001
                print(f"    {method:8s} FAILED {type(error).__name__}: {error}", flush=True)
                entry["arms"][method] = {"error": str(error)}

        results[name] = entry
        args.output.write_text(json.dumps(results, indent=2, default=str))

    elapsed = time.perf_counter() - started
    args.output.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.output}  ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
