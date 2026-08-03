from __future__ import annotations

import numpy as np
import pytest
import torch

from pycol_optimized.backend import resolve_backend
from pycol_optimized.distance import heom_kernel
from pycol_optimized.engine import TileEngine, duplicate_group_ids
from pycol_optimized.measures import (
    borderline,
    c1,
    c2,
    cm,
    d3,
    degree_of_overlap,
    kdn,
    local_set_cardinality,
    n2,
    neighbourhood_counts,
    neighbourhood_profile,
    r_value,
    separability_index,
)
from pycol_optimized.primitives import (
    LocalSetSize,
    NearestEnemy,
    NearestFriend,
    TopKPrefix,
)

K = 5


def _cohort(n_samples: int = 60, n_features: int = 4, classes: int = 3, seed: int = 5):
    generator = np.random.default_rng(seed)
    vectors = generator.normal(size=(n_samples, n_features)).astype(np.float32)
    labels = generator.integers(0, classes, size=n_samples).astype(np.int64)
    return vectors, labels


def _primitives(vectors, labels, *, block_size=17, max_k=K):
    backend = resolve_backend("cpu")
    kernel = heom_kernel(vectors, backend=backend)
    engine = TileEngine(
        kernel,
        backend=backend,
        block_size=block_size,
        duplicate_group=duplicate_group_ids(vectors),
    )
    return engine.run([TopKPrefix(max_k=max_k), LocalSetSize(labels=labels)])


def _upstream(vectors, labels):
    """Drive the pinned reference with the same distance matrix we computed."""

    complexity = pytest.importorskip(
        "pycol_complexity.complexity", reason="install the reference extra"
    )
    backend = resolve_backend("cpu")
    kernel = heom_kernel(vectors, backend=backend)
    distances = kernel.tile(0, len(vectors))[0].numpy().astype(np.float64)

    engine = complexity.Complexity.__new__(complexity.Complexity)
    engine.X = np.asarray(vectors, dtype=np.float64)
    engine.y = np.asarray(labels)
    engine.classes, counts = np.unique(engine.y, return_counts=True)
    engine.class_count = counts.astype(float)
    engine.dist_matrix = distances
    engine.unnorm_dist_matrix = distances
    engine.metrics = {"feature": {}, "struct": {}, "instance": {}, "multi": {}}
    return engine


def test_neighbourhood_counts_describe_every_neighbour_exactly_once() -> None:
    vectors, labels = _cohort()
    prefix = _primitives(vectors, labels).results["topk"]
    counts = neighbourhood_counts(prefix.indices, labels)

    assert counts.counts.sum(axis=1).tolist() == [K] * len(vectors)
    assert counts.own.max() <= K
    assert counts.class_count.sum() == len(vectors)


@pytest.mark.parametrize("imb", [False, True])
def test_kdn_cm_and_overlap_match_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    prefix = _primitives(vectors, labels).results["topk"]
    counts = neighbourhood_counts(prefix.indices, labels)
    reference = _upstream(vectors, labels)

    assert kdn(counts, imb=imb) == pytest.approx(reference.kDN(k=K, imb=imb), abs=1e-9)
    assert cm(counts, imb=imb) == pytest.approx(reference.CM(k=K, imb=imb), abs=1e-9)
    assert degree_of_overlap(counts, imb=imb) == pytest.approx(
        reference.deg_overlap(k=K, imb=imb), abs=1e-9
    )


@pytest.mark.parametrize("imb", [False, True])
def test_separability_index_matches_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    prefix = _primitives(vectors, labels).results["topk"]
    counts = neighbourhood_counts(prefix.indices, labels)
    reference = _upstream(vectors, labels)

    assert separability_index(counts, imb=imb) == pytest.approx(
        reference.SI(k=K, imb=imb), abs=1e-9
    )


def test_d3_matches_the_pinned_reference() -> None:
    vectors, labels = _cohort()
    prefix = _primitives(vectors, labels).results["topk"]
    counts = neighbourhood_counts(prefix.indices, labels)
    reference = _upstream(vectors, labels)

    assert d3(counts).tolist() == list(reference.D3_value(k=K))


@pytest.mark.parametrize("imb", [False, True])
def test_local_set_cardinality_matches_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    sizes = _primitives(vectors, labels).results["local_set_size"]
    reference = _upstream(vectors, labels)

    assert local_set_cardinality(sizes, labels, imb=imb) == pytest.approx(
        reference.LSC(imb=imb), abs=1e-9
    )


def test_local_sets_count_the_sample_itself() -> None:
    # A sample is nearer to itself than to any enemy, so its own local set is
    # never empty; reading the self-excluded tile here would be an off-by-one.
    vectors = np.asarray([[0.0], [0.1], [5.0], [5.1]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    sizes = _primitives(vectors, labels, block_size=2, max_k=2).results["local_set_size"]

    assert sizes.tolist() == [2, 2, 2, 2]


def test_measures_are_invariant_to_the_block_size() -> None:
    vectors, labels = _cohort(n_samples=50, seed=9)
    reference = None
    for block_size in (1, 7, 50):
        result = _primitives(vectors, labels, block_size=block_size)
        counts = neighbourhood_counts(result.results["topk"].indices, labels)
        observed = (
            kdn(counts),
            cm(counts),
            separability_index(counts),
            degree_of_overlap(counts),
            local_set_cardinality(result.results["local_set_size"], labels),
        )
        if reference is None:
            reference = observed
        else:
            assert observed == reference


def test_a_shorter_prefix_can_be_reused_without_recomputing() -> None:
    # One pass at the widest k serves every narrower k, which is the point of
    # keeping the measures separate from the primitive.
    vectors, labels = _cohort()
    prefix = _primitives(vectors, labels).results["topk"]
    reference = _upstream(vectors, labels)

    for narrow in (1, 3, K):
        counts = neighbourhood_counts(prefix.indices, labels, max_k=narrow)
        assert kdn(counts) == pytest.approx(reference.kDN(k=narrow), abs=1e-9)

    with pytest.raises(ValueError, match="max_k"):
        neighbourhood_counts(prefix.indices, labels, max_k=K + 1)


def _structural(vectors, labels, *, block_size=17, max_k=K):
    backend = resolve_backend("cpu")
    kernel = heom_kernel(vectors, backend=backend)
    engine = TileEngine(
        kernel,
        backend=backend,
        block_size=block_size,
        duplicate_group=duplicate_group_ids(vectors),
    )
    return engine.run(
        [
            TopKPrefix(max_k=max_k),
            NearestFriend(labels=labels),
            NearestEnemy(labels=labels),
        ]
    )


@pytest.mark.parametrize("imb", [False, True])
def test_n2_matches_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    result = _structural(vectors, labels)
    reference = _upstream(vectors, labels)

    observed = n2(
        result.results["nearest_friend"].distances,
        result.results["nearest_enemy"].distances,
        labels,
        imb=imb,
    )
    assert observed == pytest.approx(reference.N2(imb=imb), abs=1e-6)


@pytest.mark.parametrize("imb", [False, True])
def test_borderline_matches_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    counts = neighbourhood_counts(_primitives(vectors, labels).results["topk"].indices, labels)
    reference = _upstream(vectors, labels)

    expected_borderline, expected_safe, expected_rare, expected_outlier, _ = reference.borderline(
        imb=imb, return_all=True
    )
    observed = borderline(counts, imb=imb)
    assert observed.borderline == pytest.approx(expected_borderline, abs=1e-9)
    assert observed.safe == pytest.approx(expected_safe, abs=1e-9)
    assert observed.rare == pytest.approx(expected_rare, abs=1e-9)
    assert observed.outlier == pytest.approx(expected_outlier, abs=1e-9)


def test_borderline_refuses_a_neighbourhood_it_is_not_defined_on() -> None:
    vectors, labels = _cohort()
    counts = neighbourhood_counts(
        _primitives(vectors, labels).results["topk"].indices, labels, max_k=3
    )
    with pytest.raises(ValueError, match="exactly 5 neighbours"):
        borderline(counts)


@pytest.mark.parametrize("imb", [False, True])
def test_r_value_matches_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    counts = neighbourhood_counts(_primitives(vectors, labels).results["topk"].indices, labels)
    reference = _upstream(vectors, labels)

    assert np.allclose(
        np.asarray(r_value(counts, theta=2, imb=imb), dtype=np.float64),
        np.asarray(reference.R_value(k=K, theta=2, imb=imb), dtype=np.float64),
        atol=1e-9,
    )


def test_nearest_friend_excludes_the_sample_itself() -> None:
    vectors = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    result = _structural(vectors, labels, block_size=2, max_k=2)
    friend = result.results["nearest_friend"]

    assert friend.indices.tolist() == [1, 0, 3, 2]
    assert friend.diagnostics["primitive"] == "nearest_friend"


def test_a_class_of_one_reports_no_friend_rather_than_itself() -> None:
    vectors = np.asarray([[0.0], [1.0], [5.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1], dtype=np.int64)
    result = _structural(vectors, labels, block_size=3, max_k=2)
    friend = result.results["nearest_friend"]

    assert friend.indices.tolist() == [1, 0, -1]
    assert bool(torch.isinf(friend.distances[2]))
    assert friend.diagnostics["unmatched_sample_count"] == 1


@pytest.mark.parametrize("imb", [False, True])
def test_c1_and_c2_match_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    prefix = _primitives(vectors, labels).results["topk"]
    profile = neighbourhood_profile(prefix.indices, prefix.distances, labels)
    reference = _upstream(vectors, labels)

    assert c1(profile, imb=imb) == pytest.approx(reference.C1(max_k=K, imb=imb), abs=1e-6)
    assert c2(profile, imb=imb) == pytest.approx(reference.C2(max_k=K, imb=imb), abs=1e-6)


def test_c2_never_rewards_a_neighbour_beyond_unit_distance() -> None:
    # The reference clamps at one, so a far same-label neighbour contributes
    # nothing rather than a negative amount that would cancel nearer agreement.
    # HEOM normalizes each feature to unit range, so a single feature can never
    # exceed distance one; opposite corners of a two-feature square reach
    # sqrt(2), which is what puts a neighbour past the clamp.
    vectors = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
    prefix = _primitives(vectors, labels, block_size=2, max_k=3).results["topk"]
    profile = neighbourhood_profile(prefix.indices, prefix.distances, labels)

    assert profile.distances.max() > 1.0
    assert 0.0 <= c2(profile) <= 1.0
    assert c2(profile) >= c1(profile)


def test_c1_agrees_with_the_established_compute_metrics_path() -> None:
    # Two independent implementations of C1 now exist; they must not drift.
    from pycol_optimized import compute_metrics

    vectors, labels = _cohort(n_samples=40, seed=61)
    prefix = _primitives(vectors, labels, block_size=13).results["topk"]
    profile = neighbourhood_profile(prefix.indices, prefix.distances, labels)
    established = compute_metrics(vectors, labels, metrics=("C1",), neighbors=K, device="cpu")

    assert c1(profile) == pytest.approx(established.metrics["C1"], abs=1e-5)


def test_profile_rejects_mismatched_indices_and_distances() -> None:
    vectors, labels = _cohort()
    prefix = _primitives(vectors, labels).results["topk"]

    with pytest.raises(ValueError, match="same shape"):
        neighbourhood_profile(prefix.indices, prefix.distances[:, :2], labels)
    with pytest.raises(ValueError, match="max_k"):
        neighbourhood_profile(prefix.indices, prefix.distances, labels, max_k=K + 1)
