"""Device backends and tile sizing for the float32 metric engine.

The engine commits to float32 on every backend.  Apple MPS has no float64
support at all, so a single dtype removes backend-specific numeric paths and
leaves one validated code path for CPU, MPS, and CUDA.

Backend capabilities are probed once at resolution time rather than inferred
from device type, because PyTorch operator coverage on MPS varies by release.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

DTYPE = torch.float32
DTYPE_BYTES = 4

#: Fraction of the reported device memory a single tiled pass may claim.
DEFAULT_BUDGET_RATIO = 0.25

#: Smallest row block the solver will return, independent of memory pressure.
MINIMUM_BLOCK_SIZE = 64

#: Fallback budget when neither the device nor the host reports its memory.
FALLBACK_MEMORY_BYTES = 2 * 1024**3


@dataclass(frozen=True, slots=True)
class Backend:
    """A resolved device together with its probed operator capabilities."""

    device: torch.device
    supports_float64: bool
    supports_stable_sort: bool
    unified_memory: bool
    memory_budget_bytes: int

    @property
    def type(self) -> str:
        """Return the device type string."""

        return self.device.type

    def synchronize(self) -> None:
        """Drain queued accelerator work so timings include it."""

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        """Release cached accelerator blocks between tiled passes."""

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    def diagnostics(self) -> dict[str, Any]:
        """Return a serializable description of this backend."""

        return {
            "device": str(self.device),
            "dtype": "float32",
            "supports_float64": self.supports_float64,
            "supports_stable_sort": self.supports_stable_sort,
            "unified_memory": self.unified_memory,
            "memory_budget_bytes": self.memory_budget_bytes,
        }


def _host_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return FALLBACK_MEMORY_BYTES


def _memory_budget_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    if device.type == "mps":
        recommended = getattr(torch.mps, "recommended_max_memory", None)
        if recommended is not None:
            try:
                reported = int(recommended())
            except (RuntimeError, TypeError):
                reported = 0
            if reported > 0:
                return reported
    return _host_memory_bytes()


def _probe_float64(device: torch.device) -> bool:
    try:
        torch.zeros(2, dtype=torch.float64, device=device).sum().item()
    except (RuntimeError, TypeError):
        return False
    return True


def _probe_stable_sort(device: torch.device) -> bool:
    """Check that ``stable=True`` really preserves the order of equal keys."""

    try:
        probe = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE, device=device)
        order = torch.sort(probe, dim=1, stable=True).indices
        observed = [int(value) for value in order[0].tolist()]
    except (RuntimeError, TypeError, NotImplementedError):
        return False
    return observed == [1, 2, 3, 0]


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve CPU, Apple MPS, or CUDA without silent backend fallback."""

    requested = str(device).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must resolve to cpu, mps, or cuda")
    return resolved


def resolve_backend(device: str | torch.device = "auto") -> Backend:
    """Resolve a device and probe the operator behaviour the engine relies on."""

    resolved = resolve_device(device)
    return Backend(
        device=resolved,
        supports_float64=_probe_float64(resolved),
        supports_stable_sort=_probe_stable_sort(resolved),
        unified_memory=resolved.type in {"cpu", "mps"},
        memory_budget_bytes=_memory_budget_bytes(resolved),
    )


def solve_block_size(
    n_samples: int,
    *,
    backend: Backend,
    elements_per_row: int = 3,
    budget_ratio: float = DEFAULT_BUDGET_RATIO,
    minimum: int = MINIMUM_BLOCK_SIZE,
) -> int:
    """Choose the row-block height that keeps one tiled pass inside budget.

    ``elements_per_row`` counts the ``n_samples``-wide float32 buffers one tile
    row really costs, which is emphatically not one: a distance kernel may
    materialize a per-feature intermediate, and every reducer that masks a tile
    copies it.  Budgeting for the output tile alone under-counts by the feature
    width and asks the allocator for an order of magnitude more than intended.

    Row-block height never changes a result, only peak memory, so an inaccurate
    memory report costs throughput rather than correctness.  The ``minimum``
    floor is therefore soft: when the budget cannot afford it, the budget wins.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if elements_per_row < 1:
        raise ValueError("elements_per_row must be positive")
    if not 0.0 < budget_ratio <= 1.0:
        raise ValueError("budget_ratio must lie in (0, 1]")
    if minimum < 1:
        raise ValueError("minimum must be positive")

    row_bytes = n_samples * DTYPE_BYTES * elements_per_row
    budget = int(backend.memory_budget_bytes * budget_ratio)
    affordable = max(1, budget // row_bytes)
    if affordable >= minimum:
        affordable = max(affordable, minimum)
    return int(min(n_samples, affordable))


__all__ = [
    "DEFAULT_BUDGET_RATIO",
    "DTYPE",
    "DTYPE_BYTES",
    "MINIMUM_BLOCK_SIZE",
    "Backend",
    "resolve_backend",
    "resolve_device",
    "solve_block_size",
]
