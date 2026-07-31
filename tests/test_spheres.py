from __future__ import annotations

import math

import numpy as np
import pytest

from pycol_optimized.backend import resolve_backend
from pycol_optimized.distance import heom_kernel
from pycol_optimized.engine import TileEngine, duplicate_group_ids
from pycol_optimized.primitives import NearestEnemy
from pycol_optimized.spheres import (
    ContainingSphere,
    dbc,
    icsv,
    nsg,
    onb,
    onb_cover,
    sphere_coverage,
    sphere_radii,
    t1,
)


def _cohort(n_samples: int = 60, n_features: int = 4, classes: int = 3, seed: int = 5):
    generator = np.random.default_rng(seed)
    vectors = generator.normal(size=(n_samples, n_features)).astype(np.float32)
    labels = generator.integers(0, classes, size=n_samples).astype(np.int64)
    return vectors, labels


def _upstream(vectors, labels):
    complexity = pytest.importorskip(
        "pycol_complexity.complexity", reason="install the reference extra"
    )
    backend = resolve_backend("cpu")
    kernel = heom_kernel(vectors, backend=backend)
    normalized, unnormalized = kernel.tile(0, len(vectors))

    engine = complexity.Complexity.__new__(complexity.Complexity)
    engine.X = np.asarray(vectors, dtype=np.float64)
    engine.y = np.asarray(labels)
    engine.classes, counts = np.unique(engine.y, return_counts=True)
    engine.class_count = counts.astype(float)
    engine.dist_matrix = normalized.numpy().astype(np.float64)
    engine.unnorm_dist_matrix = unnormalized.numpy().astype(np.float64)
    engine.sphere_inst_count_T1 = []
    engine.radius_T1 = []
    # DBC rebuilds a distance matrix inside the reference, which reads these
    # per-column type flags; zero marks a numeric column.
    engine.meta = np.zeros(vectors.shape[1], dtype=np.int64)
    engine.metrics = {"feature": {}, "struct": {}, "instance": {}, "multi": {}}
    return engine


def _coverage(vectors, labels, *, block_size=17):
    backend = resolve_backend("cpu")
    kernel = heom_kernel(vectors, backend=backend)
    engine = TileEngine(
        kernel,
        backend=backend,
        block_size=block_size,
        duplicate_group=duplicate_group_ids(vectors),
    )
    enemy = engine.run([NearestEnemy(labels=labels, use_unnormalized=True)])
    radius = sphere_radii(
        enemy.results["nearest_enemy"].indices, enemy.results["nearest_enemy"].distances
    )

    second = TileEngine(
        heom_kernel(vectors, backend=backend),
        backend=backend,
        block_size=block_size,
        duplicate_group=duplicate_group_ids(vectors),
    )
    absorber = second.run([ContainingSphere(radius=radius)]).results["containing_sphere"]
    return sphere_coverage(radius, absorber)


def test_sphere_radii_match_the_reference_recursion() -> None:
    vectors, labels = _cohort()
    reference = _upstream(vectors, labels)
    _, expected = reference._Complexity__get_sphere_count()

    backend = resolve_backend("cpu")
    engine = TileEngine(
        heom_kernel(vectors, backend=backend),
        backend=backend,
        block_size=13,
        duplicate_group=duplicate_group_ids(vectors),
    )
    enemy = engine.run([NearestEnemy(labels=labels, use_unnormalized=True)])
    observed = sphere_radii(
        enemy.results["nearest_enemy"].indices, enemy.results["nearest_enemy"].distances
    )
    assert np.allclose(observed, expected, atol=1e-5)


def test_sphere_counts_match_the_reference() -> None:
    vectors, labels = _cohort()
    expected, _ = _upstream(vectors, labels)._Complexity__get_sphere_count()

    assert _coverage(vectors, labels).absorbed.tolist() == list(expected)


@pytest.mark.parametrize("imb", [False, True])
def test_t1_matches_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    observed = t1(_coverage(vectors, labels), labels, imb=imb)
    assert np.allclose(
        np.asarray(observed, dtype=np.float64),
        np.asarray(_upstream(vectors, labels).T1(imb=imb), dtype=np.float64),
        atol=1e-9,
    )


def test_mutually_nearest_enemies_split_the_distance_evenly() -> None:
    # Two samples that are each other's nearest enemy anchor a chain; each takes
    # half the gap so the spheres meet without overlapping.
    vectors = np.asarray([[0.0], [4.0]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int64)
    backend = resolve_backend("cpu")
    engine = TileEngine(heom_kernel(vectors, backend=backend), backend=backend, block_size=2)
    enemy = engine.run([NearestEnemy(labels=labels, use_unnormalized=True)])

    radius = sphere_radii(
        enemy.results["nearest_enemy"].indices, enemy.results["nearest_enemy"].distances
    )
    assert radius.tolist() == [2.0, 2.0]


def test_radius_resolution_survives_a_chain_far_past_the_recursion_limit() -> None:
    # The reference recurses once per chain link, so a long chain would exceed
    # the interpreter's recursion limit; the iterative walk has no such ceiling.
    length = 6_000
    enemy = np.empty(length, dtype=np.int64)
    distance = np.empty(length, dtype=np.float64)
    for index in range(length - 1):
        enemy[index] = index + 1
        distance[index] = float(length - index)
    # The final pair points at each other and terminates the chain.
    enemy[length - 1] = length - 2
    distance[length - 1] = distance[length - 2]

    radius = sphere_radii(enemy, distance)
    assert np.isfinite(radius).all()
    assert (radius >= 0.0).all()


def test_a_single_class_has_no_enemies_to_grow_against() -> None:
    with pytest.raises(ValueError, match="single-class"):
        sphere_radii(np.asarray([-1, -1]), np.asarray([np.inf, np.inf]))


def test_coverage_is_invariant_to_the_block_size() -> None:
    vectors, labels = _cohort(n_samples=45, seed=23)
    baseline = _coverage(vectors, labels, block_size=45)
    for block_size in (1, 7, 16):
        assert _coverage(vectors, labels, block_size=block_size).absorbed.tolist() == (
            baseline.absorbed.tolist()
        )


def test_absorbed_spheres_hand_their_contents_to_the_absorber() -> None:
    vectors, labels = _cohort(n_samples=30, seed=29)
    coverage = _coverage(vectors, labels)

    # Nothing is lost: every sample is accounted for in exactly one surviving
    # sphere, so the tallies must still sum to the sample count.
    assert coverage.absorbed.sum() == len(vectors)
    assert coverage.diagnostics["sphere_count"] == int(coverage.surviving.sum())


def _clusters(spread: float = 0.3, gap: float = 8.0, per_class: int = 40, seed: int = 1):
    """Two well-separated Gaussian blobs, so spheres grow across a wide gap.

    Whether any sphere is actually *absorbed* here is decided within about 1e-7
    of the containment boundary, so it comes out as one absorption on some
    platforms and none on others.  Nothing may assert on that count; the
    absorption path is covered by the explicit-radius tests instead.
    """

    generator = np.random.default_rng(seed)
    first = generator.normal(0.0, spread, size=(per_class, 2))
    second = generator.normal(gap, spread, size=(per_class, 2))
    vectors = np.vstack([first, second]).astype(np.float32)
    labels = np.asarray([0] * per_class + [1] * per_class, dtype=np.int64)
    return vectors, labels


def _explicit_cover(vectors, radius, *, block_size: int = 8):
    """Drive ContainingSphere with chosen radii instead of derived ones.

    Radii that come out of ``sphere_radii`` cannot produce a comfortable
    absorption: a radius is bounded by the distance to the nearest enemy, so the
    triangle inequality forces ``r_j - r_i <= d_ij`` and containment can only
    ever hold at near-equality.  On generated data the winning margin lands
    around 1e-7 -- below float32 epsilon -- so which side it falls on depends on
    the platform's BLAS.  Supplying radii directly decouples the reducer from
    that degeneracy and lets absorption be tested with real headroom.
    """

    backend = resolve_backend("cpu")
    engine = TileEngine(
        heom_kernel(vectors, backend=backend), backend=backend, block_size=block_size
    )
    absorber = engine.run([ContainingSphere(radius=radius)]).results["containing_sphere"]
    return sphere_coverage(np.asarray(radius, dtype=np.float64), absorber)


def test_absorption_actually_fires_and_is_tallied() -> None:
    # The random cohort absorbs nothing, so the parity tests above would pass
    # against a no-op.  Here sphere 0 sits 1.0 away from sphere 1 while the
    # radii differ by 4.9, clearing containment by 3.9 -- seven orders of
    # magnitude above the float32 noise floor, so the outcome is the geometry
    # rather than the rounding.
    vectors = np.asarray([[0.0], [1.0], [10.0]], dtype=np.float32)
    coverage = _explicit_cover(vectors, [0.1, 5.0, 0.2])

    # Sphere 0 is swallowed by sphere 1; sphere 2 is 9.0 away and stays free.
    assert coverage.absorbed.tolist() == [0, 2, 1]
    assert coverage.diagnostics["absorbed_count"] == 1
    assert coverage.diagnostics["sphere_count"] == 2
    assert coverage.absorbed.sum() == len(vectors)


def test_a_sphere_just_outside_another_is_not_swallowed() -> None:
    # The conditioning guard, as an arithmetic fact rather than a parity check.
    # Two equal radii of 1e6 with a 0.02 gap: the correct test asks whether
    # 0.02 <= 0.0 and says no.  Written the reference's way, as
    # "distance + inner <= outer", float32 rounds 1e6 + 0.02 straight back to
    # 1e6 -- one ulp there is 0.0625 -- and the sphere is absorbed.
    assert np.float32(1e6) + np.float32(0.02) == np.float32(1e6)

    vectors = np.asarray([[0.0], [0.02]], dtype=np.float32)
    coverage = _explicit_cover(vectors, [1e6, 1e6], block_size=2)

    assert coverage.diagnostics["absorbed_count"] == 0
    assert coverage.absorbed.tolist() == [1, 1]


def test_several_spheres_accumulate_in_one_absorber() -> None:
    # Two spheres swallowed by the same third.  The reducer keeps the *largest*
    # container, so both go straight to sphere 2 rather than one landing in the
    # other.
    vectors = np.asarray([[0.0], [0.5], [1.0]], dtype=np.float32)
    coverage = _explicit_cover(vectors, [0.1, 2.0, 20.0], block_size=3)

    assert coverage.absorbed.tolist() == [0, 0, 3]
    assert coverage.diagnostics["sphere_count"] == 1


def test_absorption_is_handed_along_a_chain() -> None:
    # sphere_coverage hands a swallowed sphere's contents to its absorber, so a
    # multi-hop chain must accumulate rather than lose the earliest sphere.
    #
    # Geometry alone cannot produce such a chain: if sphere 1 contains sphere 0
    # and sphere 2 contains sphere 1, the triangle inequality puts sphere 0
    # inside sphere 2 as well, and the reducer would pick sphere 2 for both.
    # The tally is written to survive it regardless, so it is exercised here by
    # handing sphere_coverage the absorber ranks directly.
    coverage = sphere_coverage(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1, 2, -1]),  # 0 -> rank 1, 1 -> rank 2, 2 free
    )

    assert coverage.absorbed.tolist() == [0, 0, 3]
    assert coverage.diagnostics["absorbed_count"] == 2


@pytest.mark.parametrize("imb", [False, True])
def test_t1_matches_the_reference_on_separated_clusters(imb: bool) -> None:
    vectors, labels = _clusters()
    observed = t1(_coverage(vectors, labels), labels, imb=imb)
    assert np.allclose(
        np.asarray(observed, dtype=np.float64),
        np.asarray(_upstream(vectors, labels).T1(imb=imb), dtype=np.float64),
        atol=1e-9,
    )


def test_the_absorption_tally_matches_the_reference_on_separated_clusters() -> None:
    # Parity of the whole tally against the reference on clustered data.
    #
    # This is deliberately *not* the guard for the radius-difference
    # conditioning -- see test_a_sphere_just_outside_another_is_not_swallowed
    # for that.  Both sides here read the same distance matrix, so they tend to
    # fall on the same side of a boundary and agree even when the boundary
    # decision itself is arbitrary.
    #
    # If this ever fails on one platform only, that is the reason: on this
    # fixture the closest containment decision sits within about 1e-7 of the
    # boundary, which is below float32 epsilon, and the two implementations use
    # different formulas (ours "d <= r_j - r_i" in float32, the reference's
    # "d + r_i <= r_j" in float64).  Every seed tried behaves the same way, so
    # it is a property of the measure rather than of this fixture.
    vectors, labels = _clusters()
    coverage = _coverage(vectors, labels)
    expected, _ = _upstream(vectors, labels)._Complexity__get_sphere_count()

    assert coverage.absorbed.tolist() == list(expected)


def _balls(vectors, labels, *, block_size=17):
    backend = resolve_backend("cpu")

    def build_engine():
        return TileEngine(
            heom_kernel(vectors, backend=backend),
            backend=backend,
            block_size=block_size,
            duplicate_group=duplicate_group_ids(vectors),
        )

    enemy = build_engine().run([NearestEnemy(labels=labels)])
    return onb_cover(
        build_engine,
        labels,
        enemy.results["nearest_enemy"].distances,
        backend=backend,
    )


@pytest.mark.parametrize("imb", [False, True])
def test_onb_matches_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    reference = _upstream(vectors, labels)
    reference.sphere_tuple_ONB = []
    assert np.allclose(
        np.asarray(onb(_balls(vectors, labels), labels, imb=imb), dtype=np.float64),
        np.asarray(reference.ONB(imb=imb), dtype=np.float64),
        atol=1e-9,
    )


def test_onb_total_matches_the_pinned_reference() -> None:
    vectors, labels = _cohort()
    reference = _upstream(vectors, labels)
    reference.sphere_tuple_ONB = []
    assert onb(_balls(vectors, labels), labels, is_total=True) == pytest.approx(
        reference.ONB(is_tot=True), abs=1e-9
    )


def test_onb_matches_the_reference_on_compact_classes() -> None:
    # Separated clusters need far fewer balls than samples, which is the regime
    # the random cohort never reaches.
    vectors, labels = _clusters()
    reference = _upstream(vectors, labels)
    reference.sphere_tuple_ONB = []

    balls = _balls(vectors, labels)
    assert len(balls) < len(vectors)
    assert onb(balls, labels) == pytest.approx(reference.ONB(), abs=1e-9)


def test_every_sample_ends_up_covered_exactly_once() -> None:
    vectors, labels = _clusters()
    balls = _balls(vectors, labels)

    assert sum(ball.size for ball in balls) == len(vectors)
    assert {ball.class_index for ball in balls} == {0, 1}


def test_the_cover_is_invariant_to_the_block_size() -> None:
    vectors, labels = _clusters(per_class=20, seed=13)
    baseline = [(b.center, b.size) for b in _balls(vectors, labels, block_size=40)]
    for block_size in (1, 7):
        assert [(b.center, b.size) for b in _balls(vectors, labels, block_size=block_size)] == (
            baseline
        )


@pytest.mark.parametrize("imb", [False, True])
def test_nsg_matches_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _clusters()
    reference = _upstream(vectors, labels)
    reference.sphere_tuple_ONB = []
    assert np.allclose(
        np.asarray(nsg(_balls(vectors, labels), labels, imb=imb), dtype=np.float64),
        np.asarray(reference.NSG(imb=imb), dtype=np.float64),
        atol=1e-9,
    )


@pytest.mark.parametrize("imb", [False, True])
@pytest.mark.parametrize("normalize", [True, False])
def test_icsv_matches_the_pinned_reference(imb: bool, normalize: bool) -> None:
    vectors, labels = _clusters()
    reference = _upstream(vectors, labels)
    reference.sphere_tuple_ONB = []
    observed = icsv(_balls(vectors, labels), labels, vectors.shape[1], normalize=normalize, imb=imb)
    assert np.allclose(
        np.asarray(observed, dtype=np.float64),
        np.asarray(reference.ICSV(normalize=normalize, imb=imb), dtype=np.float64),
        atol=1e-6,
    )


def test_nsg_and_icsv_match_the_reference_on_scattered_data() -> None:
    vectors, labels = _cohort()
    reference = _upstream(vectors, labels)
    reference.sphere_tuple_ONB = []
    balls = _balls(vectors, labels)

    assert nsg(balls, labels) == pytest.approx(reference.NSG(), abs=1e-9)
    assert icsv(balls, labels, vectors.shape[1]) == pytest.approx(reference.ICSV(), abs=1e-6)


def _equal_sized_clusters(sizes: tuple[int, ...], seed: int = 4):
    generator = np.random.default_rng(seed)
    blocks = [
        generator.normal(20.0 * index, 0.2, size=(size, 2)) for index, size in enumerate(sizes)
    ]
    vectors = np.vstack(blocks).astype(np.float32)
    labels = np.concatenate(
        [np.full(size, index, dtype=np.int64) for index, size in enumerate(sizes)]
    )
    return vectors, labels


def test_evenly_packed_balls_score_lower_than_unevenly_packed_ones() -> None:
    # ICSV is a spread, so it is only meaningful compared against something.
    # Equal clusters put equal counts in equally sized balls; lopsided ones do
    # not, and must score higher.
    even_vectors, even_labels = _equal_sized_clusters((30, 30, 30))
    uneven_vectors, uneven_labels = _equal_sized_clusters((4, 30, 120))

    even = icsv(_balls(even_vectors, even_labels), even_labels, 2)
    uneven = icsv(_balls(uneven_vectors, uneven_labels), uneven_labels, 2)

    assert even < uneven
    # Equal clusters are not exactly uniform: the middle cluster has enemies on
    # both sides, so its ball is slightly tighter and the densities differ a
    # little.  The spread stays small relative to the densities themselves.
    densities_scale = 30.0 / math.pi
    assert even / densities_scale < 0.05


def test_sphere_measures_reject_an_empty_cover() -> None:
    with pytest.raises(ValueError, match="at least one ball"):
        nsg([], np.asarray([0, 1]))
    with pytest.raises(ValueError, match="at least one ball"):
        icsv([], np.asarray([0, 1]), 2)


@pytest.mark.parametrize("imb", [False, True])
def test_dbc_matches_the_pinned_reference(imb: bool) -> None:
    vectors, labels = _cohort()
    reference = _upstream(vectors, labels)
    reference.sphere_tuple_ONB = []
    observed = dbc(
        _balls(vectors, labels),
        vectors,
        labels,
        backend=resolve_backend("cpu"),
        imb=imb,
    )
    assert np.allclose(
        np.asarray(observed, dtype=np.float64),
        np.asarray(reference.DBC(imb=imb), dtype=np.float64),
        atol=1e-9,
    )


def test_dbc_matches_the_reference_on_compact_classes() -> None:
    vectors, labels = _clusters()
    reference = _upstream(vectors, labels)
    reference.sphere_tuple_ONB = []
    assert dbc(
        _balls(vectors, labels), vectors, labels, backend=resolve_backend("cpu")
    ) == pytest.approx(reference.DBC(), abs=1e-9)


def test_two_balls_of_different_classes_always_cross() -> None:
    # DBC is the share of ball centres touching a class-crossing edge, so it
    # rises as the cover coarsens rather than as the classes overlap: with one
    # ball per class the only spanning edge must cross, giving one exactly.
    vectors, labels = _clusters(spread=0.2, gap=15.0, per_class=30, seed=31)
    balls = _balls(vectors, labels)
    assert len(balls) == 2

    assert dbc(balls, vectors, labels, backend=resolve_backend("cpu")) == pytest.approx(1.0)


def test_dbc_stays_a_share_of_the_cover() -> None:
    backend = resolve_backend("cpu")
    for vectors, labels in (_cohort(), _clusters(), _cohort(n_samples=40, classes=2, seed=43)):
        value = dbc(_balls(vectors, labels), vectors, labels, backend=backend)
        assert 0.0 <= value <= 1.0


def test_dbc_needs_enough_balls_to_span() -> None:
    backend = resolve_backend("cpu")
    vectors, labels = _clusters(per_class=10)
    with pytest.raises(ValueError, match="at least one ball"):
        dbc([], vectors, labels, backend=backend)
    with pytest.raises(ValueError, match="at least two balls"):
        dbc(_balls(vectors, labels)[:1], vectors, labels, backend=backend)
