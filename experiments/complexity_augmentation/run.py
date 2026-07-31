"""Run the full experiment: complexity before and after augmentation, then
downstream classification on each arm.

Four arms share one fixed test set:

===========  ============================================================
baseline     the training set untouched
duplicate    every text copied verbatim -- doubles n, adds no information,
             so whatever complexity change it produces is a size artifact
uniform      one mask-fill copy of every text (Algorithm 1, 1x)
minority     mask-fill copies of the minority class only, to rebalance
===========  ============================================================

The duplicate arm is the reference point: a complexity drop is only evidence
that the augmentation helped if it exceeds the drop that pure duplication
produces on its own.

    python -m complexity_augmentation.run --output-dir results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .augment import DEFAULT_FILL_MODEL, DEFAULT_MASK_PROBABILITY, MaskFillAugmenter, build_arm
from .common import resolve_device
from .complexity import complexity_profile
from .data import MINORITY_LABEL, load_split
from .embed import DEFAULT_EMBED_MODEL, FrozenEncoder, encode_cached
from .train import TrainConfig, train_and_evaluate

ARMS = ("baseline", "duplicate", "uniform", "minority")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--mask-prob", type=float, default=DEFAULT_MASK_PROBABILITY)
    parser.add_argument("--fill-model", default=DEFAULT_FILL_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--classifier", default="bert-base-uncased")
    parser.add_argument("--fill-strategy", default="sequential", choices=["sequential", "joint"])
    parser.add_argument(
        "--avoid-original",
        action="store_true",
        help="forbid the fill model from predicting the token it just masked "
        "(not in the paper; makes the augmentation a real edit)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--aug-batch-size",
        type=int,
        default=64,
        help="batch size for the mask-fill pass, which is inference-only and so "
        "takes a much larger batch than training; raise it on a big GPU",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="batch size for the frozen encoder (inference-only, same reasoning)",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--neighbours", type=int, default=5)
    parser.add_argument(
        "--include-expensive",
        action="store_true",
        help="also compute N1, purity, and the sphere-cover measures (ONB, NSG, ICSV, DBC); "
        "the cover runs one engine pass per ball and can be very slow at this scale",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="compute the complexity side only, skipping BERT fine-tuning",
    )
    return parser.parse_args(argv)


def summarize(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of each metric across seeds."""

    if not runs:
        return {}
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and k != "seed"]
    return {
        key: {
            "mean": float(np.mean([r[key] for r in runs])),
            "std": float(np.std([r[key] for r in runs])),
        }
        for key in keys
    }


def markdown_report(results: dict[str, Any]) -> str:
    """A readable summary: complexity deltas first, then downstream metrics."""

    lines: list[str] = ["# Complexity-reducing augmentation on SMS Spam", ""]
    dataset = results["dataset"]
    lines += [
        f"Train {dataset['n_train']} / test {dataset['n_test']}, "
        f"{dataset['train_positive']} spam in train "
        f"(ratio {dataset['imbalance_ratio']:.2f}:1).",
        "",
        "## Data complexity",
        "",
    ]

    arms = [a for a in results["arms"] if "complexity" in results["arms"][a]]
    baseline_panel = results["arms"].get("baseline", {}).get("complexity", {}).get("panel", {})
    measures = sorted({m for a in arms for m in results["arms"][a]["complexity"]["panel"]})

    header = "| Measure | " + " | ".join(arms) + " |"
    lines += [header, "| --- | " + " | ".join("---" for _ in arms) + " |"]
    for measure in measures:
        row = [measure]
        for arm in arms:
            value = results["arms"][arm]["complexity"]["panel"].get(measure)
            if value is None:
                row.append("-")
                continue
            cell = f"{value:.4f}"
            base = baseline_panel.get(measure)
            if arm != "baseline" and base is not None and np.isfinite(base):
                cell += f" ({value - base:+.4f})"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "Parenthesised values are the change from baseline.", ""]

    trained = [a for a in results["arms"] if results["arms"][a].get("classification")]
    if trained:
        lines += ["## Classification (held-out test set)", ""]
        metrics = [
            "macro_f1",
            "minority_f1",
            "minority_precision",
            "minority_recall",
            "mcc",
            "average_precision",
            "accuracy",
        ]
        lines += [
            "| Arm | " + " | ".join(metrics) + " |",
            "| --- | " + " | ".join("---" for _ in metrics) + " |",
        ]
        for arm in trained:
            stats = results["arms"][arm]["classification"]["summary"]
            row = [arm] + [
                f"{stats[m]['mean']:.4f} ± {stats[m]['std']:.4f}" if m in stats else "-"
                for m in metrics
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", f"Mean ± sd over seeds {results['config']['seeds']}.", ""]

    aug = {
        a: results["arms"][a]["augmentation"]
        for a in results["arms"]
        if results["arms"][a].get("augmentation", {}).get("changed_rate") is not None
    }
    if aug:
        lines += ["## What the augmentation actually changed", ""]
        lines += [
            "| Arm | masked tokens changed | texts identical to source |",
            "| --- | --- | --- |",
        ]
        for arm, report in aug.items():
            lines.append(
                f"| {arm} | {report['changed_rate']:.1%} | {report['texts_unchanged_rate']:.1%} |"
            )
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print(f"device: {resolve_device(args.device)}", flush=True)
    split = load_split(args.cache_dir, test_fraction=args.test_fraction, seed=args.split_seed)
    print(f"dataset: {split.describe()}", flush=True)

    needs_fill = any(arm in {"uniform", "minority"} for arm in args.arms)
    augmenter = (
        MaskFillAugmenter(
            args.fill_model,
            mask_probability=args.mask_prob,
            device=args.device,
            max_length=args.max_length,
            batch_size=args.aug_batch_size,
            fill_strategy=args.fill_strategy,
            avoid_original=args.avoid_original,
        )
        if needs_fill
        else None
    )
    encoder = FrozenEncoder(
        args.embed_model,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.embed_batch_size,
    )

    results: dict[str, Any] = {
        "config": vars(args)
        | {"output_dir": str(args.output_dir), "cache_dir": str(args.cache_dir)},
        "dataset": split.describe(),
        "arms": {},
    }

    for arm in args.arms:
        print(f"\n=== arm: {arm} ===", flush=True)
        texts, labels, report = build_arm(
            split.train_texts,
            split.train_labels,
            strategy=arm,
            augmenter=augmenter,
            minority_label=MINORITY_LABEL,
            seed=args.split_seed,
        )
        entry: dict[str, Any] = {
            "n_train": len(texts),
            "positive_rate": float(labels.mean()),
            "augmentation": report,
        }
        print(f"  n_train={len(texts)}  positive_rate={labels.mean():.4f}", flush=True)

        tag = f"{arm}_p{args.mask_prob}_{args.fill_strategy}"
        vectors = encode_cached(encoder, texts, args.cache_dir / f"embed_{tag}.npy")
        entry["complexity"] = complexity_profile(
            vectors,
            labels,
            device=args.device,
            neighbours=args.neighbours,
            include_expensive=args.include_expensive,
        )
        print(
            "  complexity: "
            + "  ".join(f"{k}={v:.4f}" for k, v in entry["complexity"]["panel"].items()),
            flush=True,
        )
        if entry["complexity"]["errors"]:
            print(f"  complexity errors: {entry['complexity']['errors']}", flush=True)

        if not args.skip_training:
            config = TrainConfig(
                model_name=args.classifier,
                max_length=args.max_length,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
            )
            runs = []
            for seed in args.seeds:
                print(f"    seed {seed}", flush=True)
                runs.append(
                    train_and_evaluate(
                        texts,
                        labels,
                        split.test_texts,
                        split.test_labels,
                        config=config,
                        seed=seed,
                        device=args.device,
                    )
                )
                print(
                    f"      macro_f1={runs[-1]['macro_f1']:.4f}  "
                    f"minority_f1={runs[-1]['minority_f1']:.4f}",
                    flush=True,
                )
            entry["classification"] = {"runs": runs, "summary": summarize(runs)}

        results["arms"][arm] = entry
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    results["total_seconds"] = time.perf_counter() - started
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (args.output_dir / "report.md").write_text(markdown_report(results))
    print(f"\nwrote {args.output_dir / 'results.json'} and {args.output_dir / 'report.md'}")
    print(f"total {results['total_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
