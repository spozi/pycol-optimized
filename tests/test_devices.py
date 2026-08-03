"""Cross-device behaviour.

Every other test file pins the CPU path.  These run the same work on whatever
accelerators the machine actually has and compare against CPU, so a regression
that only shows up on Metal or CUDA is caught rather than discovered by a user.

On a machine without an accelerator the comparisons skip with a visible reason
instead of silently passing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pycol_optimized.backend import resolve_backend
from pycol_optimized.distance import (
    FAST_COMPUTE_MODE,
    REPRODUCIBLE_COMPUTE_MODE,
    euclidean_kernel,
    heom_kernel,
)
from pycol_optimized.engine import TileEngine, duplicate_group_ids
from pycol_optimized.measures import (
    c1,
    kdn,
    local_set_cardinality,
    n2,
    neighbourhood_counts,
    neighbourhood_profile,
    separability_index,
)
from pycol_optimized.primitives import (
    LocalSetSize,
    NearestEnemy,
    NearestFriend,
    TopKPrefix,
)
from pycol_optimized.spheres import ContainingSphere, sphere_coverage, sphere_radii, t1


def _accelerators() -> list[str]:
    found = []
    if torch.backends.mps.is_available():
        found.append("mps")
    if torch.cuda.is_available():
        found.append("cuda")
    return found


ACCELERATORS = _accelerators()
ALL_DEVICES = ["cpu", *ACCELERATORS]

#: Parametrization that stays visible as a skip when no accelerator exists,
#: rather than collecting zero cases and reporting success.
ACCELERATOR_PARAMS = ACCELERATORS or [
    pytest.param(
        "none",
        marks=pytest.mark.skip(reason="no accelerator available on this machine"),
    )
]


def _cohort(n_samples: int = 400, n_features: int = 6, classes: int = 3, seed: int = 5):
    generator = np.random.default_rng(seed)
    vectors = generator.normal(size=(n_samples, n_features)).astype(np.float32)
    labels = generator.integers(0, classes, size=n_samples).astype(np.int64)
    return vectors, labels


def _engine(vectors, device, *, block_size=None, compute_mode="auto"):
    backend = resolve_backend(device)
    return TileEngine(
        heom_kernel(vectors, backend=backend, compute_mode=compute_mode),
        backend=backend,
        block_size=block_size,
        duplicate_group=duplicate_group_ids(vectors),
    )


def _primitives(vectors, labels, device, **kwargs):
    return _engine(vectors, device, **kwargs).run(
        [
            TopKPrefix(max_k=5),
            NearestFriend(labels=labels),
            NearestEnemy(labels=labels),
            LocalSetSize(labels=labels),
        ]
    )


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_backend_probes_describe_the_device_truthfully(device: str) -> None:
    backend = resolve_backend(device)
    assert backend.type == device
    assert backend.memory_budget_bytes > 0

    # Metal has no float64 at all, which is the reason this library commits to
    # float32 everywhere rather than branching per device.
    assert backend.supports_float64 == (device != "mps")
    assert backend.unified_memory == (device in {"cpu", "mps"})

    probe = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=backend.device)
    if backend.supports_stable_sort:
        assert torch.sort(probe, dim=1, stable=True).indices[0].tolist() == [1, 2, 0]


@pytest.mark.parametrize("device", ACCELERATOR_PARAMS)
def test_neighbour_selection_agrees_with_cpu(device: str) -> None:
    vectors, labels = _cohort()
    expected = _primitives(vectors, labels, "cpu")
    observed = _primitives(vectors, labels, device)

    # Indices must agree exactly.  Distances are float32 on both sides and the
    # kernels differ, so they are only required to be close.
    assert torch.equal(observed.results["topk"].indices.cpu(), expected.results["topk"].indices)
    assert torch.allclose(
        observed.results["topk"].distances.cpu(),
        expected.results["topk"].distances,
        atol=1e-4,
    )


@pytest.mark.parametrize("device", ACCELERATOR_PARAMS)
def test_nearest_friend_and_enemy_agree_with_cpu(device: str) -> None:
    vectors, labels = _cohort()
    expected = _primitives(vectors, labels, "cpu")
    observed = _primitives(vectors, labels, device)

    for name in ("nearest_friend", "nearest_enemy"):
        assert torch.equal(observed.results[name].indices.cpu(), expected.results[name].indices), (
            name
        )
    assert torch.equal(observed.results["local_set_size"].cpu(), expected.results["local_set_size"])


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_the_reproducible_mode_is_block_size_invariant_on_every_device(device: str) -> None:
    # The guarantee the reproducible compute mode exists to provide, checked
    # where it actually has to hold rather than only on CPU.
    vectors, labels = _cohort(n_samples=120, seed=17)
    baseline = None
    for block_size in (16, 64, 120):
        result = _primitives(
            vectors,
            labels,
            device,
            block_size=block_size,
            compute_mode=REPRODUCIBLE_COMPUTE_MODE,
        )
        prefix = result.results["topk"]
        assert result.diagnostics["block_size_invariant"] is True
        if baseline is None:
            baseline = (prefix.indices.cpu().clone(), prefix.distances.cpu().clone())
            continue
        assert torch.equal(prefix.indices.cpu(), baseline[0])
        assert torch.equal(prefix.distances.cpu(), baseline[1])


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_auto_resolves_the_compute_mode_from_the_device(device: str) -> None:
    backend = resolve_backend(device)
    kernel = euclidean_kernel(_cohort(n_samples=60)[0], backend=backend)
    diagnostics = kernel.diagnostics()

    assert diagnostics["compute_mode_source"] == f"auto:{device}"
    if device == "cpu":
        assert diagnostics["compute_mode"] == REPRODUCIBLE_COMPUTE_MODE
        assert diagnostics["block_size_invariant"] is True
    else:
        assert diagnostics["compute_mode"] == FAST_COMPUTE_MODE
        assert diagnostics["block_size_invariant"] is False


@pytest.mark.parametrize("device", ACCELERATOR_PARAMS)
def test_measures_agree_with_cpu(device: str) -> None:
    vectors, labels = _cohort()
    results = {name: _primitives(vectors, labels, name) for name in ("cpu", device)}

    scores = {}
    for name, result in results.items():
        counts = neighbourhood_counts(result.results["topk"].indices, labels)
        profile = neighbourhood_profile(
            result.results["topk"].indices, result.results["topk"].distances, labels
        )
        scores[name] = (
            kdn(counts),
            separability_index(counts),
            c1(profile),
            n2(
                result.results["nearest_friend"].distances,
                result.results["nearest_enemy"].distances,
                labels,
            ),
            local_set_cardinality(result.results["local_set_size"], labels),
        )

    assert scores[device] == pytest.approx(scores["cpu"], abs=1e-5)


@pytest.mark.parametrize("device", ACCELERATOR_PARAMS)
def test_sphere_coverage_agrees_with_cpu(device: str) -> None:
    vectors, labels = _cohort(n_samples=150, n_features=2, classes=2, seed=23)

    coverages = {}
    for name in ("cpu", device):
        enemy = _engine(vectors, name).run([NearestEnemy(labels=labels, use_unnormalized=True)])
        radius = sphere_radii(
            enemy.results["nearest_enemy"].indices, enemy.results["nearest_enemy"].distances
        )
        absorber = _engine(vectors, name).run([ContainingSphere(radius=radius)])
        coverages[name] = sphere_coverage(radius, absorber.results["containing_sphere"])

    assert coverages[device].absorbed.tolist() == coverages["cpu"].absorbed.tolist()
    assert t1(coverages[device], labels) == pytest.approx(t1(coverages["cpu"], labels))


@pytest.mark.parametrize("device", ACCELERATOR_PARAMS)
def test_auto_block_sizing_works_against_a_real_device_budget(device: str) -> None:
    # Exercises the device memory query rather than a synthetic budget: the
    # solver must return something usable from what the driver reports.
    vectors, labels = _cohort(n_samples=300, seed=29)
    result = _primitives(vectors, labels, device)

    assert result.diagnostics["block_size_source"] == "solved"
    assert 1 <= result.diagnostics["block_size"] <= len(vectors)
    assert result.diagnostics["backend"]["device"] == device


def test_an_unavailable_device_is_refused_rather_than_silently_downgraded() -> None:
    if "cuda" not in ACCELERATORS:
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_backend("cuda")
    if "mps" not in ACCELERATORS:
        with pytest.raises(RuntimeError, match="MPS"):
            resolve_backend("mps")
    # A device PyTorch understands but this library does not support is refused
    # by our own guard.  A string PyTorch does not recognize at all never
    # reaches it, so torch raises first; both are rejections, not fallbacks.
    with pytest.raises(ValueError, match="cpu, mps, or cuda"):
        resolve_backend("meta")
    with pytest.raises(RuntimeError, match="device type"):
        resolve_backend("tpu")
