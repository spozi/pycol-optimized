"""Row-tile distance kernels in float32.

A kernel materializes one ``[block, n_samples]`` slab at a time instead of the
full ``n x n`` matrix.  Tiles are cut along rows only, so every row-wise
reduction a consumer performs completes inside a single tile and the block
height cannot change a result.

Two metrics are provided:

``euclidean``
    Plain L2 over pre-scaled embeddings, matching the existing geometry layer.

``heom``
    The Heterogeneous Euclidean Overlap Metric used by PyCOL, which mixes
    range-normalized numeric differences with categorical mismatch counts and
    therefore accepts mixed-type inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

from .backend import DTYPE, Backend

Float32Array: TypeAlias = NDArray[np.float32]

#: Sample count above which ``torch.cdist`` switches to its matrix-multiply
#: expansion.  Mean-centering the inputs is translation invariant and limits
#: float32 cancellation in that implementation.
MM_CENTER_THRESHOLD = 25

#: Direct per-pair reduction.  Its arithmetic depends only on the feature width,
#: so a row block rounds identically however tall it is, and it avoids the
#: cancellation the ``x^2 + y^2 - 2xy`` expansion suffers in float32.
REPRODUCIBLE_COMPUTE_MODE = "donot_use_mm_for_euclid_dist"

#: BLAS expansion, which is faster for wide inputs.  ``torch.cdist`` selects it
#: from the tensor shape, so tiles of different heights can take different code
#: paths and round differently; opting in trades block-size invariance for
#: throughput.
FAST_COMPUTE_MODE = "use_mm_for_euclid_dist_if_necessary"

#: Pick the mode from the device.  This is the default, and it means the
#: block-size invariance guarantee depends on where the code runs; kernels
#: report which one they actually got.
AUTO_COMPUTE_MODE = "auto"

COMPUTE_MODES = (REPRODUCIBLE_COMPUTE_MODE, FAST_COMPUTE_MODE)
SELECTABLE_COMPUTE_MODES = (AUTO_COMPUTE_MODE, *COMPUTE_MODES)

#: Measured on one Apple Silicon machine at n = 32,000, d = 20, k = 5: the
#: direct path took 2.9 s on CPU against 12.9 s on MPS, and the matrix-multiply
#: path took 5.3 s on CPU against 2.1 s on MPS.  CPU therefore keeps the
#: reproducible path at no cost, while accelerators need the expansion to be
#: worth using at all.
DEVICE_COMPUTE_MODES = {
    "cpu": REPRODUCIBLE_COMPUTE_MODE,
    "mps": FAST_COMPUTE_MODE,
    # Inferred rather than measured: CUDA has the same matrix-multiply-first
    # cost profile as MPS, but no NVIDIA benchmark has been run.
    "cuda": FAST_COMPUTE_MODE,
}


def resolve_compute_mode(compute_mode: str, *, backend: Backend) -> tuple[str, str]:
    """Return the effective ``(compute_mode, source)`` for a backend.

    An unrecognized device keeps the reproducible path, because a wrong guess
    there costs correctness guarantees rather than only throughput.
    """

    if compute_mode not in SELECTABLE_COMPUTE_MODES:
        raise ValueError(f"compute_mode must be one of {SELECTABLE_COMPUTE_MODES}")
    if compute_mode != AUTO_COMPUTE_MODE:
        return compute_mode, "explicit"
    resolved = DEVICE_COMPUTE_MODES.get(backend.device.type, REPRODUCIBLE_COMPUTE_MODE)
    return resolved, f"auto:{backend.device.type}"


#: How a column pair with a missing value on either side is scored.  Upstream
#: PyCOL adds one *and* falls through to the numeric or categorical term rather
#: than skipping it, so the stored fill value still contributes.
MISSING_VALUE_POLICY = "penalty_plus_stored_value"


class TileKernel(Protocol):
    """Produces one row-block of a pairwise distance matrix on demand."""

    n_samples: int
    produces_unnormalized: bool

    def tile(self, row_start: int, row_stop: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(normalized, unnormalized)`` distances for the row block."""

    def working_elements_per_row(self) -> int:
        """Return the ``n_samples``-wide float32 buffers one tile row costs."""

    def diagnostics(self) -> dict[str, Any]:
        """Return a serializable description of this kernel."""


def _as_float32_matrix(vectors: NDArray[Any] | torch.Tensor, *, name: str) -> Float32Array:
    if isinstance(vectors, torch.Tensor):
        matrix = vectors.detach().cpu().numpy()
    else:
        matrix = np.asarray(vectors)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional [samples, features] array")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{name} must contain numerical values")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _validate_rows(row_start: int, row_stop: int, n_samples: int) -> None:
    if not 0 <= row_start < row_stop <= n_samples:
        raise ValueError("row block must satisfy 0 <= row_start < row_stop <= n_samples")


@dataclass(frozen=True, slots=True)
class EuclideanKernel:
    """L2 distances over pre-scaled float32 embeddings."""

    vectors: torch.Tensor
    centering_input: torch.Tensor
    n_samples: int
    n_features: int
    centered: bool
    compute_mode: str
    compute_mode_source: str
    produces_unnormalized: bool = False

    def tile(self, row_start: int, row_stop: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the L2 distances from rows ``[row_start, row_stop)`` to all samples."""

        _validate_rows(row_start, row_stop, self.n_samples)
        rows = self.centering_input[row_start:row_stop]
        distances = torch.cdist(
            rows,
            self.centering_input,
            p=2.0,
            compute_mode=self.compute_mode,
        )
        return distances, None

    def working_elements_per_row(self) -> int:
        """Return the ``n_samples``-wide float32 buffers one tile row costs."""

        if self.compute_mode == REPRODUCIBLE_COMPUTE_MODE:
            # The direct path expands a [block, n_samples, n_features]
            # difference before reducing it.
            return self.n_features + 2
        return 3

    def diagnostics(self) -> dict[str, Any]:
        """Return a serializable description of this kernel."""

        return {
            "metric": "euclidean",
            "working_elements_per_row": self.working_elements_per_row(),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "compute_mode": self.compute_mode,
            "compute_mode_source": self.compute_mode_source,
            "block_size_invariant": self.compute_mode == REPRODUCIBLE_COMPUTE_MODE,
            "mean_centered_for_mm_cdist": self.centered,
            "produces_unnormalized": False,
        }


def euclidean_kernel(
    scaled_vectors: NDArray[Any] | torch.Tensor,
    *,
    backend: Backend,
    compute_mode: str = AUTO_COMPUTE_MODE,
) -> EuclideanKernel:
    """Build an L2 tile kernel over already-scaled embeddings.

    ``compute_mode`` defaults to picking from the device, so an accelerator gets
    the matrix-multiply expansion and gives up block-size invariance to do it.
    Pass :data:`REPRODUCIBLE_COMPUTE_MODE` to keep that guarantee everywhere;
    the kernel's ``block_size_invariant`` diagnostic reports which applies.
    """

    resolved, source = resolve_compute_mode(compute_mode, backend=backend)
    scaled = _as_float32_matrix(scaled_vectors, name="scaled_vectors")
    # Centering only guards the matrix-multiply expansion; the direct path has
    # no cancellation to protect against.
    centered = resolved == FAST_COMPUTE_MODE and scaled.shape[0] > MM_CENTER_THRESHOLD
    if centered:
        shifted = scaled - scaled.mean(axis=0, dtype=np.float32, keepdims=True)
        centering_input = np.ascontiguousarray(shifted, dtype=np.float32)
    else:
        centering_input = scaled

    device_vectors = torch.as_tensor(scaled, dtype=DTYPE, device=backend.device)
    device_centering = (
        device_vectors
        if centering_input is scaled
        else torch.as_tensor(centering_input, dtype=DTYPE, device=backend.device)
    )
    return EuclideanKernel(
        vectors=device_vectors,
        centering_input=device_centering,
        n_samples=int(scaled.shape[0]),
        n_features=int(scaled.shape[1]),
        centered=centered,
        compute_mode=resolved,
        compute_mode_source=source,
    )


@dataclass(frozen=True, slots=True)
class HEOMKernel:
    """PyCOL's Heterogeneous Euclidean Overlap Metric over mixed-type columns.

    Numeric columns contribute a range-normalized squared difference, and a
    parallel unnormalized matrix keeps the raw squared difference because the
    hypersphere measures are defined on unscaled distances.  Constant numeric
    columns divide by one rather than by zero, which is exact: a constant column
    contributes a zero difference under either convention.
    """

    numeric_scaled: torch.Tensor
    numeric_raw: torch.Tensor
    categorical: torch.Tensor
    missing: torch.Tensor | None
    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    compute_mode: str = REPRODUCIBLE_COMPUTE_MODE
    compute_mode_source: str = "explicit"
    produces_unnormalized: bool = True

    def _overlap(self, row_start: int, row_stop: int) -> torch.Tensor | None:
        """Return the categorical mismatch plus missing-value penalty, or None."""

        total: torch.Tensor | None = None
        for column in range(self.n_categorical):
            rows = self.categorical[row_start:row_stop, column].unsqueeze(1)
            mismatch = (rows != self.categorical[:, column].unsqueeze(0)).to(DTYPE)
            total = mismatch if total is None else total + mismatch
        if self.missing is not None:
            for column in range(self.n_features):
                rows = self.missing[row_start:row_stop, column].unsqueeze(1)
                penalty = (rows | self.missing[:, column].unsqueeze(0)).to(DTYPE)
                total = penalty if total is None else total + penalty
        return total

    def tile(self, row_start: int, row_stop: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(normalized, unnormalized)`` HEOM distances for the row block."""

        _validate_rows(row_start, row_stop, self.n_samples)
        scaled = torch.cdist(
            self.numeric_scaled[row_start:row_stop],
            self.numeric_scaled,
            p=2.0,
            compute_mode=self.compute_mode,
        )
        raw = torch.cdist(
            self.numeric_raw[row_start:row_stop],
            self.numeric_raw,
            p=2.0,
            compute_mode=self.compute_mode,
        )

        overlap = self._overlap(row_start, row_stop)
        if overlap is None:
            # Purely numeric input: the square-then-root round trip would only
            # add avoidable float32 error.
            return scaled, raw
        return torch.sqrt(scaled.square() + overlap), torch.sqrt(raw.square() + overlap)

    def working_elements_per_row(self) -> int:
        """Return the ``n_samples``-wide float32 buffers one tile row costs."""

        reproducible = self.compute_mode == REPRODUCIBLE_COMPUTE_MODE
        per_distance = self.n_numeric + 2 if reproducible else 3
        # Both the normalized and unnormalized matrices are built, and the
        # overlap accumulator plus the square-and-root temporaries ride along.
        return 2 * per_distance + 3

    def diagnostics(self) -> dict[str, Any]:
        """Return a serializable description of this kernel."""

        return {
            "metric": "heom",
            "working_elements_per_row": self.working_elements_per_row(),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_numeric_features": self.n_numeric,
            "n_categorical_features": self.n_categorical,
            "compute_mode": self.compute_mode,
            "compute_mode_source": self.compute_mode_source,
            "block_size_invariant": self.compute_mode == REPRODUCIBLE_COMPUTE_MODE,
            "has_missing_mask": self.missing is not None,
            "missing_value_policy": MISSING_VALUE_POLICY,
            "produces_unnormalized": True,
        }


def heom_kernel(
    vectors: NDArray[Any] | torch.Tensor,
    *,
    backend: Backend,
    categorical: NDArray[Any] | None = None,
    missing: NDArray[Any] | None = None,
    compute_mode: str = AUTO_COMPUTE_MODE,
) -> HEOMKernel:
    """Build a HEOM tile kernel over raw, unscaled mixed-type columns.

    ``categorical`` is a boolean mask over features; omitting it treats every
    column as numeric.  ``missing`` is an optional ``[samples, features]``
    boolean mask; each column pair with a missing value on either side adds one
    to the squared distance, on top of the stored value's own contribution.
    """

    resolved, source = resolve_compute_mode(compute_mode, backend=backend)
    matrix = _as_float32_matrix(vectors, name="vectors")
    n_samples, n_features = matrix.shape

    if categorical is None:
        categorical_mask = np.zeros(n_features, dtype=bool)
    else:
        categorical_mask = np.asarray(categorical, dtype=bool)
        if categorical_mask.shape != (n_features,):
            raise ValueError("categorical must be a boolean mask over features")

    numeric_mask = ~categorical_mask
    numeric = matrix[:, numeric_mask]
    if numeric.shape[1] == 0:
        # cdist needs a non-degenerate width; a zero-width block contributes
        # nothing, so stand in a single constant column.
        numeric = np.zeros((n_samples, 1), dtype=np.float32)
        spans = np.ones(1, dtype=np.float32)
    else:
        lower = numeric.min(axis=0)
        upper = numeric.max(axis=0)
        spans = upper - lower
        if not np.isfinite(spans).all():
            raise ValueError("numeric feature ranges must remain finite in float32")
    safe_spans = np.where(spans <= 0.0, np.float32(1.0), spans).astype(np.float32)
    scaled = np.ascontiguousarray(numeric / safe_spans, dtype=np.float32)

    missing_device: torch.Tensor | None = None
    if missing is not None:
        missing_mask = np.asarray(missing, dtype=bool)
        if missing_mask.shape != (n_samples, n_features):
            raise ValueError("missing must be a boolean [samples, features] mask")
        if missing_mask.any():
            missing_device = torch.as_tensor(missing_mask, device=backend.device)

    return HEOMKernel(
        numeric_scaled=torch.as_tensor(scaled, dtype=DTYPE, device=backend.device),
        numeric_raw=torch.as_tensor(
            np.ascontiguousarray(numeric, dtype=np.float32),
            dtype=DTYPE,
            device=backend.device,
        ),
        categorical=torch.as_tensor(
            np.ascontiguousarray(matrix[:, categorical_mask], dtype=np.float32),
            dtype=DTYPE,
            device=backend.device,
        ),
        missing=missing_device,
        n_samples=int(n_samples),
        n_features=int(n_features),
        n_numeric=int(numeric_mask.sum()),
        n_categorical=int(categorical_mask.sum()),
        compute_mode=resolved,
        compute_mode_source=source,
    )


__all__ = [
    "AUTO_COMPUTE_MODE",
    "COMPUTE_MODES",
    "DEVICE_COMPUTE_MODES",
    "FAST_COMPUTE_MODE",
    "MISSING_VALUE_POLICY",
    "MM_CENTER_THRESHOLD",
    "REPRODUCIBLE_COMPUTE_MODE",
    "SELECTABLE_COMPUTE_MODES",
    "EuclideanKernel",
    "HEOMKernel",
    "TileKernel",
    "euclidean_kernel",
    "heom_kernel",
    "resolve_compute_mode",
]
