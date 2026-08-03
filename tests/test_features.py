from __future__ import annotations

import numpy as np
import pytest

from pycol_optimized.features import class_feature_stats, f1v, f2, f3, input_noise


def _cohort(n_samples: int = 60, n_features: int = 4, classes: int = 3, seed: int = 5):
    generator = np.random.default_rng(seed)
    vectors = generator.normal(size=(n_samples, n_features)).astype(np.float32)
    labels = generator.integers(0, classes, size=n_samples).astype(np.int64)
    return vectors, labels


def _upstream(vectors, labels):
    """Drive the pinned reference over the same embedding."""

    complexity = pytest.importorskip(
        "pycol_complexity.complexity", reason="install the reference extra"
    )
    engine = complexity.Complexity.__new__(complexity.Complexity)
    engine.X = np.asarray(vectors, dtype=np.float64)
    engine.y = np.asarray(labels)
    engine.classes, counts = np.unique(engine.y, return_counts=True)
    engine.class_count = counts.astype(float)
    engine.class_inxs = [np.where(engine.y == label)[0] for label in engine.classes]
    engine.metrics = {"feature": {}, "struct": {}, "instance": {}, "multi": {}}
    return engine


def _stats(vectors, labels):
    return class_feature_stats(vectors, labels, backend="cpu")


def test_f1v_matches_the_pinned_reference() -> None:
    vectors, labels = _cohort()
    assert np.allclose(
        f1v(_stats(vectors, labels)),
        np.asarray(_upstream(vectors, labels).F1v(), dtype=np.float64),
        atol=1e-6,
    )


@pytest.mark.parametrize("imb", [False, True])
def test_f2_matches_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    assert np.allclose(
        np.asarray(f2(_stats(vectors, labels), imb=imb), dtype=np.float64),
        np.asarray(_upstream(vectors, labels).F2(imb=imb), dtype=np.float64),
        atol=1e-9,
    )


@pytest.mark.parametrize("imb", [False, True])
def test_f3_matches_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    assert np.allclose(
        np.asarray(f3(_stats(vectors, labels), vectors, imb=imb), dtype=np.float64),
        np.asarray(_upstream(vectors, labels).F3(imb=imb), dtype=np.float64),
        atol=1e-9,
    )


def test_f3_needs_the_whole_embedding_when_not_imbalanced() -> None:
    # The reference counts the overlap region over every sample, not only the
    # pair's, so the measure cannot be computed from the per-class summaries.
    vectors, labels = _cohort()
    with pytest.raises(ValueError, match="full embedding"):
        f3(_stats(vectors, labels))


def test_f1v_survives_a_singular_scatter_matrix() -> None:
    # A duplicated feature makes the within-class scatter singular; the
    # pseudo-inverse is what keeps this from raising.
    base = np.random.default_rng(2).normal(size=(30, 2)).astype(np.float32)
    vectors = np.hstack([base, base]).astype(np.float32)
    labels = np.asarray([0] * 15 + [1] * 15, dtype=np.int64)

    scores = f1v(_stats(vectors, labels))
    assert len(scores) == 1
    assert np.isfinite(scores[0])


def test_perfectly_separated_classes_score_zero_overlap() -> None:
    vectors = np.asarray(
        [[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    stats = _stats(vectors, labels)

    assert f2(stats) == [0.0]
    assert f3(stats, vectors) == [0.0]


def test_stats_reject_a_dataset_with_one_class() -> None:
    vectors = np.zeros((4, 2), dtype=np.float32)
    labels = np.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError, match="at least two classes"):
        _stats(vectors, labels)


def test_stats_summarize_every_class_and_feature() -> None:
    vectors, labels = _cohort(n_samples=30, n_features=3, classes=2, seed=17)
    stats = _stats(vectors, labels)

    assert stats.n_classes == 2
    assert stats.n_features == 3
    assert stats.class_count.sum() == 30
    assert (stats.lower <= stats.upper).all()
    assert (stats.lower <= stats.mean).all() and (stats.mean <= stats.upper).all()


@pytest.mark.parametrize("imb", [False, True])
def test_input_noise_matches_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    assert np.allclose(
        np.asarray(input_noise(_stats(vectors, labels), imb=imb), dtype=np.float64),
        np.asarray(_upstream(vectors, labels).input_noise(imb=imb), dtype=np.float64),
        atol=1e-9,
    )


def test_input_noise_excludes_readings_exactly_on_the_boundary() -> None:
    # Strict comparisons: a value sitting on the other class's extreme is not
    # inside its domain, which is where IN differs from F2 and F3.
    vectors = np.asarray([[0.0], [1.0], [1.0], [2.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    assert input_noise(_stats(vectors, labels)) == [0.0]


def test_input_noise_counts_readings_not_whole_samples() -> None:
    # One class fully inside the other's box on both features gives every
    # reading, so the pair contributes its full share of the dataset total.
    vectors = np.asarray(
        [[0.0, 0.0], [10.0, 10.0], [5.0, 5.0], [6.0, 6.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    # Four readings from the inner class, none from the outer, over eight total.
    assert input_noise(_stats(vectors, labels)) == [pytest.approx(4 / 8)]
