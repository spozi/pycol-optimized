from __future__ import annotations

from importlib import metadata

import numpy as np
import pytest

import pycol_optimized
from pycol_optimized import (
    SUPPORTED_METRICS,
    build_geometry,
    build_knn_graph,
    compute_metrics,
)


def test_distribution_and_package_versions_match() -> None:
    try:
        distribution_version = metadata.version("pycol-optimized")
    except metadata.PackageNotFoundError:
        pytest.skip("source-tree test without installed distribution metadata")
    assert distribution_version == pycol_optimized.__version__


def test_all_public_metrics_are_computed_in_requested_order() -> None:
    vectors = np.asarray(
        [[0.0], [0.1], [0.9], [1.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    result = compute_metrics(vectors, labels, neighbors=1, device="cpu")

    assert tuple(result.metrics) == SUPPORTED_METRICS
    assert result.metrics == pytest.approx(
        {
            "F1": 1.0 / 163.0,
            "N1": 0.5,
            "N3": 0.0,
            "kDN": 0.0,
            "CM": 0.0,
            "C1": 0.0,
        },
        abs=2e-6,
    )
    assert result.project_composite == pytest.approx(np.mean(list(result.metrics.values())))
    assert result.diagnostics["device"] == "cpu"
    assert result.diagnostics["dtype"] == "float32"
    assert result.diagnostics["geometry_build_count"] == 1
    assert result.diagnostics["knn_build_count"] == 1


def test_duplicate_rows_and_equal_distance_ties_are_deterministic() -> None:
    vectors = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    geometry = build_geometry(vectors, device="cpu", return_cpu_distances=True)

    first = build_knn_graph(geometry, max_k=3)
    second = build_knn_graph(geometry, max_k=3)
    expected = np.asarray(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(first.indices.cpu().numpy(), expected)
    np.testing.assert_array_equal(second.indices.cpu().numpy(), expected)
    assert first.distances is not None
    assert first.distances[0, 0].item() == 0.0
    assert geometry.distances_cpu is not None
    assert geometry.distances_cpu[0, 1] == 0.0
    assert first.diagnostics["tie_policy"] == "distance_then_smallest_sample_index"
    assert first.diagnostics["zero_distance_duplicates_retained"] is True


def test_multilabel_uses_valid_columns_and_positive_support_weighting() -> None:
    vectors = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, 0.4],
            [0.6, 0.6],
            [0.8, 0.9],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(
        [
            [1, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.int8,
    )

    result = compute_metrics(
        vectors,
        labels,
        neighbors=2,
        device="cpu",
        multilabel=True,
        label_names=("rare", "constant", "common"),
    )

    assert set(result.per_label) == {"rare", "common"}
    for metric in SUPPORTED_METRICS:
        per_label_values = np.asarray(
            [result.per_label["rare"][metric], result.per_label["common"][metric]]
        )
        assert result.metrics[metric] == pytest.approx(per_label_values.mean())
        assert result.label_weighted_metrics[metric] == pytest.approx(
            np.average(per_label_values, weights=(2.0, 4.0))
        )
    assert "N3_negative" in result.per_label["rare"]
    assert "N3_positive" in result.per_label["common"]


@pytest.mark.parametrize(
    ("vectors", "labels", "kwargs", "message"),
    [
        (
            np.asarray([0.0, 1.0]),
            np.asarray([0, 1]),
            {},
            "two-dimensional",
        ),
        (
            np.asarray([[0.0], [1.0]]),
            np.asarray([0]),
            {},
            "same number of samples",
        ),
        (
            np.asarray([[0.0], [np.nan]]),
            np.asarray([0, 1]),
            {"metrics": ("F1",)},
            "finite",
        ),
        (
            np.asarray([[0.0], [1.0]]),
            np.asarray([0, 1]),
            {"metrics": ()},
            "cannot be empty",
        ),
        (
            np.asarray([[0.0], [1.0]]),
            np.asarray([0, 1]),
            {"metrics": ("F1", "F1")},
            "duplicates",
        ),
        (
            np.asarray([[0.0], [1.0]]),
            np.asarray([0, 1]),
            {"metrics": ("unknown",)},
            "Unsupported",
        ),
        (
            np.asarray([[0.0], [0.5], [1.0]]),
            np.asarray([0, 1, 1]),
            {"metrics": ("N3",), "neighbors": 0},
            "neighbors",
        ),
        (
            np.asarray([[0.0], [1.0]]),
            np.asarray([0, 1]),
            {"multilabel": True},
            "two-dimensional",
        ),
        (
            np.asarray([[0.0], [1.0]]),
            np.asarray([[0, 1], [1, 0]]),
            {"label_names": ("only_one",)},
            "label_names",
        ),
    ],
)
def test_invalid_public_inputs_are_rejected(
    vectors: np.ndarray,
    labels: np.ndarray,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        compute_metrics(vectors, labels, device="cpu", **kwargs)
