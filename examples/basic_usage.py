"""Compute all supported PyCOL Optimized metrics on a small binary cohort."""

from __future__ import annotations

import numpy as np

from pycol_optimized import compute_metrics


def main() -> None:
    vectors = np.asarray(
        [
            [0.00, 0.00],
            [0.10, 0.15],
            [0.20, 0.05],
            [0.75, 0.80],
            [0.90, 0.85],
            [1.00, 1.00],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    result = compute_metrics(vectors, labels, neighbors=3, device="auto")

    for name, value in result.metrics.items():
        print(f"{name}: {value:.6f}")
    print(f"project_composite: {result.project_composite:.6f}")
    print(f"device: {result.diagnostics['device']}")


if __name__ == "__main__":
    main()
