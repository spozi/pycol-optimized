from __future__ import annotations

from importlib import metadata
from typing import Any

import numpy as np
import pytest

from pycol_optimized import compute_metrics, compute_n1_reference

scipy_distance = pytest.importorskip(
    "scipy.spatial.distance",
    reason="install the reference extra",
)
pycol_complexity = pytest.importorskip(
    "pycol_complexity.complexity",
    reason="install the reference extra",
)


def _pycol_engine(
    scaled: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
) -> Any:
    classes, counts = np.unique(labels, return_counts=True)
    engine = pycol_complexity.Complexity.__new__(pycol_complexity.Complexity)
    engine.X = scaled
    engine.y = labels
    engine.meta = [0] * scaled.shape[1]
    engine.classes = classes
    engine.class_count = counts.astype(float)
    engine.class_inxs = [np.flatnonzero(labels == label) for label in classes]
    engine.dist_matrix = distances
    engine.unnorm_dist_matrix = distances
    engine.sphere_inst_count_T1 = []
    engine.sphere_tuple_ONB = []
    engine.metrics = {"feature": {}, "struct": {}, "instance": {}, "multi": {}}
    return engine


def test_all_metrics_match_pinned_pycol_scientific_reference() -> None:
    assert metadata.version("pycol-complexity") == "1.0.4"
    vectors = np.asarray(
        [
            [0.000, 0.000],
            [0.125, 0.250],
            [0.250, 0.125],
            [0.375, 0.500],
            [0.625, 0.500],
            [0.750, 0.875],
            [0.875, 0.750],
            [1.000, 1.000],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 0, 1, 2, 2, 2], dtype=np.int64)
    scaled = vectors.astype(np.float64)
    distances = scipy_distance.cdist(scaled, scaled, metric="euclidean")
    engine = _pycol_engine(scaled, labels, distances)

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = {
            "F1": float(np.asarray(engine.F1(), dtype=np.float64).mean()),
            "N3": float(np.asarray(engine.N3(k=1, imb=True)).mean()),
            "kDN": float(engine.kDN(k=3)),
            "CM": float(engine.CM(k=3)),
            "C1": float(engine.C1(max_k=3)),
        }
    expected["N1"] = compute_n1_reference(vectors, labels).value

    actual = compute_metrics(vectors, labels, neighbors=3, device="cpu")

    assert actual.metrics == pytest.approx(expected, abs=2e-6)
