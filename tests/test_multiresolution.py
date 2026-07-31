from __future__ import annotations

import numpy as np
import pytest

from pycol_optimized.multiresolution import (
    cell_assignments,
    neighbourhood_separability,
    purity,
)


def _cohort(n_samples: int = 60, n_features: int = 3, classes: int = 3, seed: int = 5):
    generator = np.random.default_rng(seed)
    vectors = generator.normal(size=(n_samples, n_features)).astype(np.float32)
    labels = generator.integers(0, classes, size=n_samples).astype(np.int64)
    return vectors, labels


def _upstream(vectors, labels):
    complexity = pytest.importorskip(
        "pycol_complexity.complexity", reason="install the reference extra"
    )
    engine = complexity.Complexity.__new__(complexity.Complexity)
    engine.X = np.asarray(vectors, dtype=np.float64)
    engine.y = np.asarray(labels)
    engine.classes, counts = np.unique(engine.y, return_counts=True)
    engine.class_count = counts.astype(float)
    engine.metrics = {"feature": {}, "struct": {}, "instance": {}, "multi": {}}
    return engine


def _upstream_cells(vectors, labels, resolution):
    """The reference's own cell map, as sample-index groups."""

    engine = _upstream(vectors, labels)
    _, reverse = engine._Complexity__calculate_cells(
        resolution, np.transpose(engine.X), get_labels=1
    )
    return {frozenset(members) for members in reverse.values()}


def _our_cells(vectors, resolution):
    cells = cell_assignments(vectors, resolution)
    return {frozenset(np.flatnonzero(cells == cell).tolist()) for cell in np.unique(cells)}


@pytest.mark.parametrize("resolution", [0, 1, 2, 3, 7])
def test_cell_assignments_partition_exactly_as_the_reference_does(resolution: int) -> None:
    vectors, labels = _cohort(n_samples=40, n_features=2, seed=11)
    assert _our_cells(vectors, resolution) == _upstream_cells(vectors, labels, resolution)


def test_a_value_on_a_boundary_falls_to_the_lower_cell() -> None:
    # The reference's intervals are closed at both ends and it takes the first
    # match, so the midpoint of a two-cell grid belongs to the lower cell.
    vectors = np.asarray([[0.0], [0.5], [1.0]], dtype=np.float32)
    cells = cell_assignments(vectors, 1)

    assert cells[0] == cells[1]
    assert cells[1] != cells[2]


def test_resolution_zero_puts_everything_in_one_cell() -> None:
    vectors, _ = _cohort()
    assert len(np.unique(cell_assignments(vectors, 0))) == 1


def test_a_constant_feature_collapses_instead_of_dividing_by_zero() -> None:
    vectors = np.column_stack(
        [np.linspace(0.0, 1.0, 12), np.full(12, 3.0)],
    ).astype(np.float32)
    cells = cell_assignments(vectors, 3)

    # The constant column contributes nothing, so the split follows the first
    # feature alone.
    assert len(np.unique(cells)) == len(np.unique(cell_assignments(vectors[:, :1], 3)))


def test_purity_matches_the_pinned_reference() -> None:
    vectors, labels = _cohort()
    observed = purity(vectors, labels)
    assert observed.value == pytest.approx(_upstream(vectors, labels).purity(), abs=1e-9)


def test_purity_matches_the_reference_on_separable_classes() -> None:
    generator = np.random.default_rng(2)
    vectors = np.vstack(
        [generator.normal(0.0, 0.2, size=(30, 2)), generator.normal(6.0, 0.2, size=(30, 2))]
    ).astype(np.float32)
    labels = np.asarray([0] * 30 + [1] * 30, dtype=np.int64)

    assert purity(vectors, labels).value == pytest.approx(
        _upstream(vectors, labels).purity(), abs=1e-9
    )


def test_separable_classes_are_purer_at_a_coarse_grid_than_mixed_ones() -> None:
    generator = np.random.default_rng(3)
    separable = np.vstack(
        [generator.normal(0.0, 0.2, size=(40, 2)), generator.normal(9.0, 0.2, size=(40, 2))]
    ).astype(np.float32)
    mixed = generator.normal(0.0, 1.0, size=(80, 2)).astype(np.float32)
    labels = np.asarray([0] * 40 + [1] * 40, dtype=np.int64)

    # At a coarse grid the separable classes already sit in their own cells,
    # while the mixed ones still share every cell.
    assert purity(separable, labels).per_resolution[1] > purity(mixed, labels).per_resolution[1]


def test_purity_reports_the_whole_resolution_profile() -> None:
    vectors, labels = _cohort()
    profile = purity(vectors, labels, max_resolution=8)

    assert profile.per_resolution.shape == (8,)
    assert profile.weighted.shape == (8,)
    # Weights halve at each step, so the profile is damped towards the fine end.
    assert profile.weighted[-1] <= profile.weighted[0]


def test_purity_rejects_inputs_it_cannot_score() -> None:
    vectors, labels = _cohort()
    with pytest.raises(ValueError, match="at least two"):
        purity(vectors, labels, max_resolution=1)
    with pytest.raises(ValueError, match="at least two classes"):
        purity(vectors, np.zeros(len(vectors), dtype=np.int64))
    with pytest.raises(ValueError, match="negative"):
        cell_assignments(vectors, -1)


def _distances(vectors):
    from pycol_optimized.backend import resolve_backend
    from pycol_optimized.distance import heom_kernel

    kernel = heom_kernel(vectors, backend=resolve_backend("cpu"))
    return kernel.tile(0, len(vectors))[0].numpy().astype(np.float64)


def _upstream_with_distances(vectors, labels):
    engine = _upstream(vectors, labels)
    engine.dist_matrix = _distances(vectors)
    return engine


def test_neighbourhood_separability_is_not_reference_parity() -> None:
    # Deliberately not asserted against the reference.  The reference reaches
    # this measure through a helper that looks a neighbour's label up by its
    # position within the cell rather than by its sample index, so it scores
    # against unrelated labels once a cell is a strict subset of the dataset.
    # Reproducing that would make the number meaningless; this implementation
    # uses each neighbour's own label, so it deviates on purpose.
    vectors, labels = _cohort(n_samples=45, n_features=2, seed=7)
    observed = neighbourhood_separability(vectors, labels, _distances(vectors))
    reference = _upstream_with_distances(vectors, labels).neighbourhood_separability()

    assert observed.value != pytest.approx(reference, abs=1e-9)
    assert 0.0 <= observed.value <= 1.0


def test_separable_classes_score_higher_than_interleaved_ones() -> None:
    generator = np.random.default_rng(2)
    separable = np.vstack(
        [generator.normal(0.0, 0.3, size=(25, 2)), generator.normal(6.0, 0.3, size=(25, 2))]
    ).astype(np.float32)
    interleaved = generator.normal(0.0, 1.0, size=(50, 2)).astype(np.float32)
    labels = np.asarray([0] * 25 + [1] * 25, dtype=np.int64)

    apart = neighbourhood_separability(separable, labels, _distances(separable))
    mixed = neighbourhood_separability(interleaved, labels, _distances(interleaved))
    assert apart.value > mixed.value


def test_a_class_that_never_mixes_scores_the_formula_maximum() -> None:
    # Two far-apart groups: no cell ever holds both classes, so every sample's
    # neighbours within its cell always share its label and every proportion is
    # one.  The ceiling is still not one, because the reference integrates those
    # proportions over positions running 0 .. (depth - 1) / same_class_total.
    # With twelve samples per class that is a width of 10/11, so a perfectly
    # separable class scores 10/11 rather than 1.
    generator = np.random.default_rng(5)
    vectors = np.vstack(
        [generator.normal(0.0, 0.05, size=(12, 1)), generator.normal(50.0, 0.05, size=(12, 1))]
    ).astype(np.float32)
    labels = np.asarray([0] * 12 + [1] * 12, dtype=np.int64)

    profile = neighbourhood_separability(vectors, labels, _distances(vectors), max_resolution=4)
    assert profile.per_resolution == pytest.approx(np.full(4, 10 / 11))


def test_a_lone_sample_in_its_cell_counts_as_separable() -> None:
    # A grid fine enough to isolate every sample: the reference treats a lone
    # sample as perfectly separable rather than undefined.
    vectors = np.linspace(0.0, 1.0, 6, dtype=np.float32).reshape(-1, 1)
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)

    profile = neighbourhood_separability(vectors, labels, _distances(vectors), max_resolution=8)
    assert profile.per_resolution[-1] == pytest.approx(1.0)


def test_separability_is_not_rescaled_the_way_purity_is() -> None:
    # purity min-max normalizes its profile and divides by a constant;
    # separability integrates the weighted profile directly, so the two are on
    # different scales and must not be compared to each other.
    vectors, labels = _cohort(n_samples=30, n_features=2, seed=17)
    profile = neighbourhood_separability(vectors, labels, _distances(vectors), max_resolution=8)

    assert profile.weighted[0] == pytest.approx(profile.per_resolution[0])
    assert profile.weighted[-1] == pytest.approx(profile.per_resolution[-1] / 2**7)


def test_separability_rejects_a_mismatched_distance_matrix() -> None:
    vectors, labels = _cohort(n_samples=20, n_features=2, seed=19)
    with pytest.raises(ValueError, match="square matrix"):
        neighbourhood_separability(vectors, labels, np.zeros((5, 5)))
