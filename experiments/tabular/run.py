"""Do resampling methods reduce complexity beyond their own null controls?

Stratified 5-fold cross-validation, which is what the imbalanced-learning
literature uses and which Bank Marketing needs, having no official split.

**Resampling happens inside each fold, on the training portion only.** Doing it
before the split leaks: an oversampled duplicate of a row can land in training
while its twin lands in test, and the score inflates for no good reason. The
test fold is never resampled and never touched by a sampler.

Complexity is measured per fold as well, on the same resampled training rows
the classifier sees. That keeps both halves coherent, and it gives complexity
measures error bars they otherwise lack -- being deterministic, their only
variation comes from which rows the fold happened to contain.

    python -m tabular.run --folds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_bank_marketing
from .dedup_study import panel
from .resample import ARMS, METHODS, build_arm, controls_match, inject_label_noise, matched_control


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument(
        "--noise-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 0.10, 0.20],
        help="label-flip rates for the genuine-difficulty axis",
    )
    parser.add_argument(
        "--skip-noise", action="store_true", help="resampling arms only, no noise sweep"
    )
    parser.add_argument(
        "--skip-complexity", action="store_true", help="classifier only, no complexity panel"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/tabular"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    return parser.parse_args(argv)


def classify(train, test, categorical, *, seed: int) -> dict[str, float]:
    """Fit gradient boosting on a resampled fold, score the untouched test fold."""

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )

    (train_x, train_y), (test_x, test_y) = train, test
    model = HistGradientBoostingClassifier(
        categorical_features=np.flatnonzero(categorical).tolist(),
        random_state=seed,
    )
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    scores = model.predict_proba(test_x)

    # The rarest class in the *original* problem is the one that matters, so it
    # is taken from the test fold, which no arm resamples.
    minority = int(np.bincount(test_y).argmin())
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, predicted, labels=[minority], average=None, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(test_y, predicted, average="macro", zero_division=0)),
        "minority_precision": float(precision[0]),
        "minority_recall": float(recall[0]),
        "minority_f1": float(f1[0]),
        "mcc": float(matthews_corrcoef(test_y, predicted)),
        "minority_ap": float(average_precision_score(test_y == minority, scores[:, minority])),
        "accuracy": float((test_y == predicted).mean()),
    }


def _evaluate(arm, dataset, test, args) -> dict[str, Any]:
    """Complexity of an arm's training rows, plus its score on the test fold."""

    entry: dict[str, Any] = arm.describe()
    if not args.skip_complexity:
        started = time.perf_counter()
        entry["complexity"] = panel(
            arm.vectors, arm.labels, dataset.categorical, arm.missing, device=args.device
        )
        entry["complexity_seconds"] = time.perf_counter() - started
    entry["classification"] = classify(
        (arm.vectors, arm.labels), test, dataset.categorical, seed=args.seed
    )
    return entry


def summarize(folds: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = [k for k, v in folds[0].items() if isinstance(v, (int, float))]
    return {
        k: {
            "mean": float(np.mean([f[k] for f in folds])),
            "std": float(np.std([f[k] for f in folds])),
        }
        for k in keys
    }


def main(argv: list[str] | None = None) -> None:
    from sklearn.model_selection import StratifiedKFold

    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_bank_marketing(args.cache_dir)
    vectors, labels, missing = dataset.vectors, dataset.labels, dataset.missing
    print(f"dataset: {dataset.describe()}", flush=True)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results: dict[str, Any] = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "dataset": dataset.describe(),
        "arms": {},
        "noise": {},
    }
    started = time.perf_counter()

    # ---- resampling arms, each against its matched control ----------------
    collected: dict[str, list[dict[str, Any]]] = {}
    for fold, (train_index, test_index) in enumerate(splitter.split(vectors, labels)):
        train_x, train_y = vectors[train_index], labels[train_index]
        train_missing = None if missing is None else missing[train_index]
        test = (vectors[test_index], labels[test_index])
        print(
            f"\n=== fold {fold + 1}/{args.folds}  train={len(train_y)} test={len(test_index)} ===",
            flush=True,
        )

        for name in args.arms:
            arm = build_arm(
                name,
                train_x,
                train_y,
                categorical=dataset.categorical,
                missing=train_missing,
                seed=args.seed,
            )
            pairs = [arm]
            if name in METHODS:
                control = matched_control(
                    arm, train_x, train_y, missing=train_missing, seed=args.seed
                )
                if not controls_match(arm, control):
                    raise RuntimeError(f"control for {name} does not match its arm")
                pairs.append(control)

            for item in pairs:
                entry = _evaluate(item, dataset, test, args)
                collected.setdefault(item.name, []).append(entry)
                kdn = entry.get("complexity", {}).get("kDN")
                print(
                    f"  {item.name:20s} n={entry['n']:<7d} "
                    f"macro_f1={entry['classification']['macro_f1']:.4f}"
                    + (f"  kDN={kdn:.4f}" if kdn is not None else ""),
                    flush=True,
                )

        (args.output_dir / "raw.json").write_text(json.dumps(collected, indent=2, default=str))

    for name, entries in collected.items():
        results["arms"][name] = {
            "classification": summarize([e["classification"] for e in entries]),
            "complexity": summarize([e["complexity"] for e in entries])
            if "complexity" in entries[0]
            else {},
            "n": float(np.mean([e["n"] for e in entries])),
        }

    # ---- the genuine-difficulty axis --------------------------------------
    if not args.skip_noise:
        n_classes = len(dataset.label_names)
        for rate in args.noise_rates:
            print(f"\n=== label noise {rate:.0%} ===", flush=True)
            folds: list[dict[str, Any]] = []
            for fold, (train_index, test_index) in enumerate(splitter.split(vectors, labels)):
                # Noise goes into training only; the test fold keeps its true
                # labels, or the metric would degrade for a second reason.
                noisy = inject_label_noise(
                    labels[train_index], rate, n_classes=n_classes, seed=args.seed + fold
                )
                arm = build_arm(
                    "baseline",
                    vectors[train_index],
                    noisy,
                    categorical=dataset.categorical,
                    missing=None if missing is None else missing[train_index],
                    seed=args.seed,
                )
                folds.append(
                    _evaluate(arm, dataset, (vectors[test_index], labels[test_index]), args)
                )
                print(
                    f"  fold {fold + 1} macro_f1={folds[-1]['classification']['macro_f1']:.4f}",
                    flush=True,
                )
            results["noise"][f"{rate:.2f}"] = {
                "classification": summarize([f["classification"] for f in folds]),
                "complexity": summarize([f["complexity"] for f in folds])
                if "complexity" in folds[0]
                else {},
            }
            (args.output_dir / "results.json").write_text(
                json.dumps(results, indent=2, default=str)
            )

    results["total_seconds"] = time.perf_counter() - started
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.output_dir / 'results.json'}  ({results['total_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
