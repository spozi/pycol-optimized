from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pycol_optimized import build_geometry, build_knn_graph
from pycol_optimized.backend import Backend, resolve_backend, solve_block_size
from pycol_optimized.distance import (
    FAST_COMPUTE_MODE,
    euclidean_kernel,
    heom_kernel,
    resolve_compute_mode,
)
from pycol_optimized.engine import TileEngine, duplicate_group_ids
from pycol_optimized.primitives import NearestEnemy, TopKPrefix

BLOCK_SIZES = (1, 2, 3, 5, 7, 64)


def _backend():
    return resolve_backend("cpu")


def _cohort(n_samples: int = 40, n_features: int = 3, seed: int = 7) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(size=(n_samples, n_features)).astype(np.float32)


def _labels(n_samples: int, classes: int = 3, seed: int = 11) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(0, classes, size=n_samples).astype(np.int64)


def _naive_heom(
    vectors: np.ndarray,
    categorical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transcribe the reference triple loop, including its scaling convention."""

    n_samples, n_features = vectors.shape
    normalized = np.zeros((n_samples, n_samples), dtype=np.float64)
    unnormalized = np.zeros((n_samples, n_samples), dtype=np.float64)
    spans = {}
    for column in range(n_features):
        if not categorical[column]:
            spans[column] = float(vectors[:, column].max() - vectors[:, column].min())

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            scaled_total = 0.0
            raw_total = 0.0
            for column in range(n_features):
                left = float(vectors[i][column])
                right = float(vectors[j][column])
                if categorical[column]:
                    if left != right:
                        scaled_total += 1.0
                        raw_total += 1.0
                    continue
                span = spans[column]
                if span == 0.0:
                    scaled_total += abs(left - right) ** 2
                    raw_total += abs(left - right) ** 2
                else:
                    scaled_total += (abs(left - right) / span) ** 2
                    raw_total += abs(left - right) ** 2
            normalized[i][j] = normalized[j][i] = math.sqrt(scaled_total)
            unnormalized[i][j] = unnormalized[j][i] = math.sqrt(raw_total)
    return normalized, unnormalized


def _run(kernel, reducers, *, block_size, duplicate_group=None):
    engine = TileEngine(
        kernel,
        backend=_backend(),
        block_size=block_size,
        duplicate_group=duplicate_group,
    )
    return engine.run(reducers)


def test_block_size_never_changes_the_result() -> None:
    vectors = _cohort()
    labels = _labels(len(vectors))
    backend = _backend()

    baseline_indices = None
    baseline_distances = None
    baseline_enemy = None

    for block_size in BLOCK_SIZES:
        kernel = euclidean_kernel(vectors, backend=backend)
        result = _run(
            kernel,
            [TopKPrefix(max_k=5), NearestEnemy(labels=labels)],
            block_size=block_size,
        )
        prefix = result.results["topk"]
        enemy = result.results["nearest_enemy"]
        assert result.diagnostics["tiling_axis"] == "rows"

        if baseline_indices is None:
            baseline_indices = prefix.indices.clone()
            baseline_distances = prefix.distances.clone()
            baseline_enemy = (enemy.indices.clone(), enemy.distances.clone())
            continue

        assert torch.equal(prefix.indices, baseline_indices)
        # Bit-identical, not merely close: row-only tiling fixes the reduction
        # order, so a different memory budget must not change the answer.
        assert torch.equal(prefix.distances, baseline_distances)
        assert torch.equal(enemy.indices, baseline_enemy[0])
        assert torch.equal(enemy.distances, baseline_enemy[1])
        assert result.diagnostics["kernel"]["block_size_invariant"] is True


def _backend_of(device_type: str) -> Backend:
    return Backend(
        device=torch.device(device_type),
        supports_float64=device_type != "mps",
        supports_stable_sort=True,
        unified_memory=device_type in {"cpu", "mps"},
        memory_budget_bytes=8 * 1024**3,
    )


def test_auto_compute_mode_picks_the_faster_path_per_device() -> None:
    # CPU measured faster on the direct path, accelerators on the expansion.
    assert resolve_compute_mode("auto", backend=_backend_of("cpu")) == (
        "donot_use_mm_for_euclid_dist",
        "auto:cpu",
    )
    assert resolve_compute_mode("auto", backend=_backend_of("mps")) == (
        FAST_COMPUTE_MODE,
        "auto:mps",
    )
    assert resolve_compute_mode("auto", backend=_backend_of("cuda")) == (
        FAST_COMPUTE_MODE,
        "auto:cuda",
    )


def test_auto_keeps_the_safe_path_for_an_unrecognized_device() -> None:
    # Guessing wrong here would cost a correctness guarantee, not throughput.
    exotic = Backend(
        device=torch.device("meta"),
        supports_float64=False,
        supports_stable_sort=False,
        unified_memory=False,
        memory_budget_bytes=1024**3,
    )
    mode, source = resolve_compute_mode("auto", backend=exotic)
    assert mode == "donot_use_mm_for_euclid_dist"
    assert source == "auto:meta"


def test_an_explicit_compute_mode_overrides_the_device_default() -> None:
    mps = _backend_of("mps")
    assert resolve_compute_mode("donot_use_mm_for_euclid_dist", backend=mps) == (
        "donot_use_mm_for_euclid_dist",
        "explicit",
    )
    with pytest.raises(ValueError, match="compute_mode"):
        resolve_compute_mode("nonsense", backend=mps)


def test_a_run_reports_whether_its_result_is_block_size_invariant() -> None:
    vectors = _cohort(n_samples=30, seed=47)
    backend = _backend()

    auto = _run(euclidean_kernel(vectors, backend=backend), [TopKPrefix(2)], block_size=8)
    assert auto.diagnostics["block_size_invariant"] is True
    assert auto.diagnostics["kernel"]["compute_mode_source"] == "auto:cpu"

    fast = _run(
        euclidean_kernel(vectors, backend=backend, compute_mode=FAST_COMPUTE_MODE),
        [TopKPrefix(2)],
        block_size=8,
    )
    assert fast.diagnostics["block_size_invariant"] is False
    assert fast.diagnostics["kernel"]["compute_mode_source"] == "explicit"


def test_heom_also_honours_the_device_aware_default() -> None:
    vectors = _cohort(n_samples=20, n_features=4, seed=53)
    assert (
        heom_kernel(vectors, backend=_backend()).diagnostics()["compute_mode_source"] == "auto:cpu"
    )
    forced = heom_kernel(vectors, backend=_backend(), compute_mode=FAST_COMPUTE_MODE)
    assert forced.diagnostics()["block_size_invariant"] is False


def test_the_fast_compute_mode_declares_that_it_drops_invariance() -> None:
    # torch.cdist picks the matrix-multiply expansion from the tensor shape, so
    # tiles of different heights round differently under it.  The kernel has to
    # say so rather than let a caller assume the default guarantee still holds.
    vectors = _cohort()
    backend = _backend()
    kernel = euclidean_kernel(vectors, backend=backend, compute_mode=FAST_COMPUTE_MODE)

    assert kernel.diagnostics()["block_size_invariant"] is False

    # Still numerically close, but only close: the expansion leaves ~1e-4 of
    # noise on entries the direct path returns exactly, including the zero
    # self-distance, and how much depends on the block height.
    tall = kernel.tile(0, len(vectors))[0][:1]
    short = kernel.tile(0, 1)[0]
    assert torch.allclose(tall, short, atol=1e-3)

    exact = euclidean_kernel(vectors, backend=backend)
    assert exact.tile(0, 1)[0][0, 0].item() == 0.0
    assert short[0, 0].item() > 0.0

    with pytest.raises(ValueError, match="compute_mode"):
        euclidean_kernel(vectors, backend=backend, compute_mode="nonsense")


def test_euclidean_kernel_tiles_reproduce_cdist() -> None:
    vectors = _cohort(n_samples=23)
    backend = _backend()
    kernel = euclidean_kernel(vectors, backend=backend)

    expected = torch.cdist(
        torch.as_tensor(vectors, dtype=torch.float32),
        torch.as_tensor(vectors, dtype=torch.float32),
        p=2.0,
    )
    observed = torch.cat(
        [kernel.tile(start, min(start + 4, len(vectors)))[0] for start in range(0, len(vectors), 4)]
    )
    assert torch.allclose(observed, expected, atol=1e-5)


def test_heom_tiles_match_the_reference_triple_loop() -> None:
    generator = np.random.default_rng(3)
    numeric = generator.normal(size=(24, 3)).astype(np.float32)
    codes = generator.integers(0, 4, size=(24, 2)).astype(np.float32)
    vectors = np.hstack([numeric, codes]).astype(np.float32)
    categorical = np.asarray([False, False, False, True, True])

    kernel = heom_kernel(vectors, backend=_backend(), categorical=categorical)
    normalized_tiles = []
    unnormalized_tiles = []
    for start in range(0, len(vectors), 5):
        normalized, unnormalized = kernel.tile(start, min(start + 5, len(vectors)))
        normalized_tiles.append(normalized)
        unnormalized_tiles.append(unnormalized)

    expected_normalized, expected_unnormalized = _naive_heom(vectors, categorical)
    assert np.allclose(torch.cat(normalized_tiles).numpy(), expected_normalized, atol=1e-4)
    assert np.allclose(torch.cat(unnormalized_tiles).numpy(), expected_unnormalized, atol=1e-4)


def test_heom_without_categorical_columns_matches_scaled_euclidean() -> None:
    vectors = _cohort(n_samples=12, n_features=4, seed=5)
    kernel = heom_kernel(vectors, backend=_backend())

    spans = vectors.max(axis=0) - vectors.min(axis=0)
    scaled = torch.as_tensor(vectors / spans, dtype=torch.float32)
    expected = torch.cdist(scaled, scaled, p=2.0)

    normalized, unnormalized = kernel.tile(0, len(vectors))
    assert torch.allclose(normalized, expected, atol=1e-5)
    raw = torch.as_tensor(vectors, dtype=torch.float32)
    assert torch.allclose(unnormalized, torch.cdist(raw, raw, p=2.0), atol=1e-5)


def test_topk_prefix_agrees_with_the_validated_dense_graph() -> None:
    vectors = _cohort(n_samples=30, seed=13)
    geometry = build_geometry(vectors, device="cpu")
    dense = build_knn_graph(geometry, max_k=6)

    scaled = geometry.scaled_cpu
    kernel = euclidean_kernel(scaled, backend=_backend())
    result = _run(
        kernel,
        [TopKPrefix(max_k=6)],
        block_size=7,
        duplicate_group=duplicate_group_ids(scaled),
    )
    assert torch.equal(result.results["topk"].indices, dense.indices)


def test_topk_prefix_breaks_equal_distances_by_smallest_index() -> None:
    # Rows 1 and 2 are exact duplicates of row 0 and rows 3 and 4 sit at equal
    # distance on either side, so every selection below is decided by ties.
    vectors = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    kernel = euclidean_kernel(vectors, backend=_backend())
    result = _run(
        kernel,
        [TopKPrefix(max_k=4)],
        block_size=2,
        duplicate_group=duplicate_group_ids(vectors),
    )
    indices = result.results["topk"].indices

    assert indices[0].tolist() == [1, 2, 3, 4]
    assert indices[1].tolist() == [0, 2, 3, 4]
    assert indices[3].tolist() == [0, 1, 2, 4]


def test_topk_prefix_selects_correctly_when_ties_straddle_the_cut() -> None:
    # Four samples sit at distance 1 from the origin sample while max_k is 2,
    # so the boundary correction decides which two are kept.
    vectors = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    kernel = euclidean_kernel(vectors, backend=_backend())
    result = _run(kernel, [TopKPrefix(max_k=2)], block_size=5)

    assert result.results["topk"].indices[0].tolist() == [1, 2]


def test_straddling_correction_still_returns_neighbours_in_distance_order() -> None:
    # Sample 0 sits at the origin.  Samples 1..3 are strictly nearer than the
    # cut but their sample indices run opposite to their distances, and samples
    # 4 and 5 tie exactly on the cut, which triggers the straddle correction.
    # That correction emits sample-index order, so the prefix has to be
    # re-sorted afterwards even though the selected distances are all distinct.
    vectors = np.asarray(
        [[0.0], [3.0], [1.0], [2.0], [4.0], [-4.0]],
        dtype=np.float32,
    )
    kernel = euclidean_kernel(vectors, backend=_backend())
    # One row per tile: a shared tile would hide the defect, because another
    # row's equal distances would trigger the reordering for everyone.
    result = _run(kernel, [TopKPrefix(max_k=4)], block_size=1)
    prefix = result.results["topk"]

    assert prefix.indices[0].tolist() == [2, 3, 1, 4]
    assert prefix.distances[0].tolist() == [1.0, 2.0, 3.0, 4.0]

    rising = prefix.distances[:, :-1] <= prefix.distances[:, 1:]
    assert bool(rising.all()), "every neighbour prefix must be non-decreasing"


def test_duplicate_rows_are_repaired_to_zero_distance() -> None:
    vectors = np.asarray([[0.5, 0.5], [0.5, 0.5], [4.0, 4.0]], dtype=np.float32)
    kernel = euclidean_kernel(vectors, backend=_backend())
    result = _run(
        kernel,
        [TopKPrefix(max_k=1)],
        block_size=1,
        duplicate_group=duplicate_group_ids(vectors),
    )
    prefix = result.results["topk"]

    assert prefix.indices[0].tolist() == [1]
    assert prefix.distances[0].item() == 0.0
    assert result.diagnostics["duplicate_row_count"] == 2


def test_nearest_enemy_matches_a_direct_search() -> None:
    vectors = _cohort(n_samples=25, seed=17)
    labels = _labels(25, classes=3, seed=19)
    kernel = euclidean_kernel(vectors, backend=_backend())
    result = _run(kernel, [NearestEnemy(labels=labels)], block_size=6)
    enemy = result.results["nearest_enemy"]

    reference = torch.cdist(
        torch.as_tensor(vectors, dtype=torch.float32),
        torch.as_tensor(vectors, dtype=torch.float32),
        p=2.0,
    ).numpy()
    for sample in range(len(vectors)):
        candidates = np.flatnonzero(labels != labels[sample])
        best = candidates[np.argmin(reference[sample][candidates])]
        assert enemy.indices[sample].item() == int(best)
        assert enemy.distances[sample].item() == pytest.approx(reference[sample][best], abs=1e-5)


def test_nearest_enemy_reports_missing_enemies_without_failing() -> None:
    vectors = _cohort(n_samples=6, seed=23)
    labels = np.zeros(6, dtype=np.int64)
    kernel = euclidean_kernel(vectors, backend=_backend())
    result = _run(kernel, [NearestEnemy(labels=labels)], block_size=3)
    enemy = result.results["nearest_enemy"]

    assert enemy.indices.tolist() == [-1] * 6
    assert bool(torch.isinf(enemy.distances).all())
    assert enemy.diagnostics["unmatched_sample_count"] == 6


def test_nearest_enemy_can_read_the_unnormalized_matrix() -> None:
    vectors = np.asarray(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 6.0], [3.0, 6.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    kernel = heom_kernel(vectors, backend=_backend())
    result = _run(kernel, [NearestEnemy(labels=labels, use_unnormalized=True)], block_size=2)
    enemy = result.results["nearest_enemy"]

    assert enemy.diagnostics["distance_matrix"] == "unnormalized"
    assert enemy.distances[0].item() == pytest.approx(3.0, abs=1e-5)


def test_unnormalized_distances_are_refused_when_the_kernel_omits_them() -> None:
    vectors = _cohort(n_samples=8, seed=29)
    labels = _labels(8, classes=2, seed=31)
    kernel = euclidean_kernel(vectors, backend=_backend())

    with pytest.raises(RuntimeError, match="unnormalized"):
        _run(kernel, [NearestEnemy(labels=labels, use_unnormalized=True)], block_size=4)


def test_engine_reports_the_memory_it_avoided() -> None:
    vectors = _cohort(n_samples=50, seed=37)
    kernel = euclidean_kernel(vectors, backend=_backend())
    result = _run(kernel, [TopKPrefix(max_k=3)], block_size=10)

    assert result.diagnostics["tile_count"] == 5
    assert result.diagnostics["peak_tile_elements"] == 500
    assert result.diagnostics["dense_matrix_elements"] == 2_500


def test_engine_rejects_duplicate_reducer_names() -> None:
    vectors = _cohort(n_samples=8, seed=41)
    kernel = euclidean_kernel(vectors, backend=_backend())

    with pytest.raises(ValueError, match="unique"):
        _run(kernel, [TopKPrefix(max_k=2), TopKPrefix(max_k=3)], block_size=4)


def test_solved_block_size_stays_within_its_bounds() -> None:
    backend = _backend()

    assert solve_block_size(10, backend=backend) == 10
    assert solve_block_size(100_000, backend=backend) <= 100_000

    with pytest.raises(ValueError, match="budget_ratio"):
        solve_block_size(10, backend=backend, budget_ratio=0.0)
    with pytest.raises(ValueError, match="elements_per_row"):
        solve_block_size(10, backend=backend, elements_per_row=0)


def test_block_sizing_budgets_for_the_kernels_real_working_set() -> None:
    # The reproducible cdist path expands a [block, n, features] intermediate,
    # so budgeting for the output tile alone under-counts by the feature width
    # and asks the allocator for far more than intended.
    backend = _backend()
    wide = euclidean_kernel(_cohort(n_samples=64, n_features=64), backend=backend)
    narrow = euclidean_kernel(_cohort(n_samples=64, n_features=2), backend=backend)

    assert wide.working_elements_per_row() == 66
    assert narrow.working_elements_per_row() == 4
    assert solve_block_size(200_000, backend=backend, elements_per_row=66) < solve_block_size(
        200_000, backend=backend, elements_per_row=4
    )


def test_the_memory_floor_yields_to_the_budget() -> None:
    # A soft floor: when the budget cannot afford the minimum block, taking it
    # anyway would defeat the point of tiling.
    starved = Backend(
        device=torch.device("cpu"),
        supports_float64=True,
        supports_stable_sort=True,
        unified_memory=True,
        memory_budget_bytes=1024 * 1024,
    )
    block = solve_block_size(1_000_000, backend=starved, elements_per_row=32)
    assert block >= 1
    assert block < 64


def test_auto_block_sizing_tiles_a_cohort_too_large_for_one_slab() -> None:
    backend = Backend(
        device=torch.device("cpu"),
        supports_float64=True,
        supports_stable_sort=True,
        unified_memory=True,
        memory_budget_bytes=8 * 1024 * 1024,
    )
    vectors = _cohort(n_samples=600, n_features=16, seed=43)
    kernel = euclidean_kernel(vectors, backend=backend)
    engine = TileEngine(kernel, backend=backend)
    result = engine.run([TopKPrefix(max_k=3)])

    assert result.diagnostics["block_size_source"] == "solved"
    assert result.diagnostics["tile_count"] > 1
    assert torch.equal(
        result.results["topk"].indices,
        _run(
            euclidean_kernel(vectors, backend=backend),
            [TopKPrefix(max_k=3)],
            block_size=600,
        )
        .results["topk"]
        .indices,
    )
