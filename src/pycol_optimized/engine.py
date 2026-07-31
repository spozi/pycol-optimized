"""Streaming tile engine that fans one distance pass out to many reducers.

The engine cuts the pairwise distance matrix into row blocks and hands each
block to every registered reducer before releasing it.  Two properties follow
from cutting rows only:

* Peak memory is ``block_size * n_samples`` rather than ``n_samples ** 2``.
* Every row-wise reduction sees its whole row inside one tile, so the block
  height cannot change a result.  Float addition is not associative, and a
  reduction split across blocks would otherwise make the answer depend on the
  memory budget of the machine that produced it.

Computing many measures therefore costs one distance pass rather than one pass
per measure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from .backend import DTYPE, Backend, resolve_backend, solve_block_size
from .distance import TileKernel

Int64Array: TypeAlias = NDArray[np.int64]

#: Row blocks are never split along columns; see the module docstring.
TILING_AXIS = "rows"


@dataclass(slots=True)
class DistanceTile:
    """One ``[block, n_samples]`` slab of the pairwise distance matrix."""

    normalized: torch.Tensor
    unnormalized: torch.Tensor | None
    row_start: int
    row_stop: int
    n_samples: int
    row_index: torch.Tensor
    column_index: torch.Tensor
    column_index_float: torch.Tensor
    _self_excluded: torch.Tensor | None = None

    @property
    def block_size(self) -> int:
        """Return the number of rows in this tile."""

        return self.row_stop - self.row_start

    @property
    def device(self) -> torch.device:
        """Return the device holding the tile."""

        return self.normalized.device

    def self_excluded(self) -> torch.Tensor:
        """Return the tile with each sample's own diagonal entry set to infinity.

        Computed once and shared by every reducer that asks, because most of
        them exclude self and would otherwise each copy the tile.  The write
        touches one entry per row rather than building a ``[block, n_samples]``
        boolean mask, which is the same result for a fraction of the work.

        Reducers that must count a sample within its own neighbourhood, as the
        hypersphere measures do, should read ``normalized`` directly instead.
        """

        if self._self_excluded is None:
            values = self.normalized.clone()
            local_rows = self.row_index - self.row_start
            values[local_rows, self.row_index] = torch.inf
            self._self_excluded = values
        return self._self_excluded

    def require_unnormalized(self) -> torch.Tensor:
        """Return the unnormalized distances, or raise if the kernel omits them."""

        if self.unnormalized is None:
            raise RuntimeError(
                "this reducer needs unnormalized distances; use a kernel that produces them"
            )
        return self.unnormalized


class Reducer(Protocol):
    """Consumes row tiles and produces one measure primitive."""

    name: str

    def begin(self, *, n_samples: int, backend: Backend) -> None:
        """Allocate state for a pass over ``n_samples`` rows."""

    def update(self, tile: DistanceTile) -> None:
        """Fold one row tile into the accumulated state."""

    def finish(self) -> Any:
        """Return the completed primitive."""


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Reducer outputs keyed by reducer name, plus pass diagnostics."""

    results: dict[str, Any]
    diagnostics: dict[str, Any]


def duplicate_group_ids(matrix: NDArray[Any]) -> Int64Array:
    """Label byte-identical float32 rows, using ``-1`` for unique rows.

    Float32 distance expansion can assign a small positive distance to
    identical observations, so the engine repairs those pairs to zero.
    """

    # An explicit copy: the signed-zero fixup below must not reach the caller's
    # array, which an already-float32 contiguous input would otherwise share.
    contiguous = np.array(matrix, dtype=np.float32, order="C", copy=True)
    # Normalize signed zero so -0 and +0 compare byte-wise equal.
    contiguous[contiguous == 0.0] = 0.0
    row_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    keys = contiguous.view(row_dtype).reshape(-1)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1).astype(np.int64, copy=False)
    return np.where(counts[inverse] > 1, inverse, np.int64(-1)).astype(np.int64, copy=False)


class TileEngine:
    """Streams row tiles from a kernel through a set of reducers."""

    def __init__(
        self,
        kernel: TileKernel,
        *,
        backend: Backend | str | torch.device = "auto",
        block_size: int | None = None,
        duplicate_group: NDArray[Any] | None = None,
    ) -> None:
        self.kernel = kernel
        self.backend = backend if isinstance(backend, Backend) else resolve_backend(backend)
        self.n_samples = int(kernel.n_samples)
        self.block_size: int | None

        if block_size is None:
            # Sized in run(), where the reducer count is known: each reducer
            # that masks a tile holds its own copy of it.
            self.block_size = None
            self.block_size_source = "solved"
        else:
            if block_size < 1:
                raise ValueError("block_size must be positive")
            self.block_size = int(min(block_size, self.n_samples))
            self.block_size_source = "explicit"

        self._duplicate_group: torch.Tensor | None = None
        self._duplicate_row_count = 0
        if duplicate_group is not None:
            groups = np.asarray(duplicate_group, dtype=np.int64)
            if groups.shape != (self.n_samples,):
                raise ValueError("duplicate_group must contain one entry per sample")
            if (groups >= 0).any():
                self._duplicate_group = torch.as_tensor(groups, device=self.backend.device)
                self._duplicate_row_count = int((groups >= 0).sum())

        self._column_index = torch.arange(
            self.n_samples,
            dtype=torch.long,
            device=self.backend.device,
        )
        # Reducers resolve index ties by comparing column positions as float32;
        # building that vector once keeps it off the per-tile path.
        self._column_index_float = self._column_index.to(DTYPE)

    def _repair_duplicates(self, tile: torch.Tensor, row_start: int, row_stop: int) -> None:
        if self._duplicate_group is None:
            return
        rows = self._duplicate_group[row_start:row_stop].unsqueeze(1)
        shared = (rows >= 0) & (rows == self._duplicate_group.unsqueeze(0))
        tile[shared] = 0.0

    def run(self, reducers: list[Reducer]) -> EngineResult:
        """Stream every row tile through ``reducers`` in one distance pass."""

        if not reducers:
            raise ValueError("at least one reducer is required")
        names = [reducer.name for reducer in reducers]
        if len(set(names)) != len(names):
            raise ValueError("reducer names must be unique")

        if self.block_size is None:
            self.block_size = solve_block_size(
                self.n_samples,
                backend=self.backend,
                elements_per_row=self.kernel.working_elements_per_row() + len(reducers),
            )

        for reducer in reducers:
            reducer.begin(n_samples=self.n_samples, backend=self.backend)

        self.backend.synchronize()
        started = time.perf_counter()
        distance_seconds = 0.0
        reduce_seconds = 0.0
        tile_count = 0

        for row_start in range(0, self.n_samples, self.block_size):
            row_stop = min(row_start + self.block_size, self.n_samples)

            distance_started = time.perf_counter()
            normalized, unnormalized = self.kernel.tile(row_start, row_stop)
            self._repair_duplicates(normalized, row_start, row_stop)
            if unnormalized is not None:
                self._repair_duplicates(unnormalized, row_start, row_stop)
            self.backend.synchronize()
            distance_seconds += time.perf_counter() - distance_started

            tile = DistanceTile(
                normalized=normalized,
                unnormalized=unnormalized,
                row_start=row_start,
                row_stop=row_stop,
                n_samples=self.n_samples,
                row_index=self._column_index[row_start:row_stop],
                column_index=self._column_index,
                column_index_float=self._column_index_float,
            )

            reduce_started = time.perf_counter()
            for reducer in reducers:
                reducer.update(tile)
            self.backend.synchronize()
            reduce_seconds += time.perf_counter() - reduce_started
            tile_count += 1

        results = {reducer.name: reducer.finish() for reducer in reducers}
        self.backend.synchronize()
        total_ms = (time.perf_counter() - started) * 1_000.0

        kernel_diagnostics = self.kernel.diagnostics()
        diagnostics: dict[str, Any] = {
            "backend": self.backend.diagnostics(),
            "kernel": kernel_diagnostics,
            "tiling_axis": TILING_AXIS,
            # Promoted from the kernel because it is a property of the run, not
            # a detail of the distance metric: it says whether this result
            # would reproduce bit-for-bit under a different memory budget.
            "block_size_invariant": bool(kernel_diagnostics.get("block_size_invariant", False)),
            "block_size": self.block_size,
            "block_size_source": self.block_size_source,
            "tile_count": tile_count,
            "reducers": names,
            "duplicate_row_count": self._duplicate_row_count,
            "peak_tile_elements": self.block_size * self.n_samples,
            "dense_matrix_elements": self.n_samples * self.n_samples,
            "timings_ms": {
                "distance": distance_seconds * 1_000.0,
                "reduce": reduce_seconds * 1_000.0,
                "total": total_ms,
            },
        }
        return EngineResult(results=results, diagnostics=diagnostics)


__all__ = [
    "TILING_AXIS",
    "DistanceTile",
    "EngineResult",
    "Reducer",
    "TileEngine",
    "duplicate_group_ids",
]
