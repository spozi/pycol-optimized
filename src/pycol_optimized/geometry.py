"""Shared float32 geometry construction for optimized PyCOL metrics.

The geometry layer deliberately separates the immutable CPU representation
used for auditing from the device tensors used by accelerated metrics.  N1
continues to use its existing validated implementation; this module is the
common foundation for the remaining metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

Float32Array: TypeAlias = NDArray[np.float32]
Int64Array: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DistanceResult:
    """Device-resident float32 pairwise distances and optional CPU snapshot."""

    distances_device: torch.Tensor
    distances_cpu: Float32Array | None
    device: torch.device
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GeometryResult:
    """Scaled embeddings and their reusable dense pairwise geometry."""

    scaled_cpu: Float32Array
    scaled_device: torch.Tensor
    distance: DistanceResult
    reference_min: Float32Array
    reference_max: Float32Array
    duplicate_groups: tuple[Int64Array, ...]
    diagnostics: dict[str, Any]

    @property
    def distances_device(self) -> torch.Tensor:
        """Return the device-resident dense distance matrix."""

        return self.distance.distances_device

    @property
    def distances_cpu(self) -> Float32Array | None:
        """Return the optional immutable-by-convention CPU distance snapshot."""

        return self.distance.distances_cpu

    @property
    def device(self) -> torch.device:
        """Return the selected PyTorch device."""

        return self.distance.device


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


def synchronize_device(device: torch.device) -> None:
    """Synchronize an accelerator so reported timings include queued work."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _validate_and_scale(
    vectors: NDArray[Any],
    *,
    reference_min: NDArray[Any] | None,
    reference_max: NDArray[Any] | None,
) -> tuple[Float32Array, Float32Array, Float32Array, NDArray[np.bool_]]:
    matrix = np.asarray(vectors)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional [samples, features] array")
    if matrix.shape[0] < 2:
        raise ValueError("geometry construction requires at least two samples")
    if matrix.shape[1] < 1:
        raise ValueError("vectors must contain at least one feature")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("vectors must contain numerical values")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must contain only finite values")
    if (reference_min is None) != (reference_max is None):
        raise ValueError("reference_min and reference_max must be supplied together")

    matrix32 = np.ascontiguousarray(matrix, dtype=np.float32)
    expected_shape = (matrix32.shape[1],)
    if reference_min is None:
        lower = matrix32.min(axis=0)
        upper = matrix32.max(axis=0)
    else:
        lower = np.asarray(reference_min, dtype=np.float32)
        upper = np.asarray(reference_max, dtype=np.float32)
        if lower.shape != expected_shape or upper.shape != expected_shape:
            raise ValueError("reference bounds must match the embedding width")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("reference bounds must contain only finite values")
        if np.any(upper < lower):
            raise ValueError("reference_max cannot be smaller than reference_min")

    lower = np.ascontiguousarray(lower, dtype=np.float32)
    upper = np.ascontiguousarray(upper, dtype=np.float32)
    widths = upper - lower
    if not np.isfinite(widths).all():
        raise ValueError("reference feature ranges must remain finite in float32")
    constant_features = widths <= 0.0
    safe_widths = np.where(constant_features, np.float32(1.0), widths)
    scaled = (matrix32 - lower) / safe_widths
    scaled[:, constant_features] = 0.0
    if not np.isfinite(scaled).all():
        raise ValueError("fixed-range scaling produced non-finite float32 values")

    # Normalize signed zero so byte-wise duplicate detection treats -0 and +0
    # as the same observation.
    scaled[scaled == 0.0] = 0.0
    return (
        np.ascontiguousarray(scaled, dtype=np.float32),
        lower,
        upper,
        constant_features,
    )


def fixed_range_scale(
    vectors: NDArray[Any],
    *,
    reference_min: NDArray[Any] | None = None,
    reference_max: NDArray[Any] | None = None,
) -> Float32Array:
    """Apply the project's PyCOL fixed-range transform in float32."""

    scaled, _, _, _ = _validate_and_scale(
        vectors,
        reference_min=reference_min,
        reference_max=reference_max,
    )
    return scaled


def _duplicate_groups(matrix: Float32Array) -> tuple[tuple[Int64Array, ...], int]:
    """Find exact float32-identical rows without an O(n²d) broadcast."""

    row_dtype = np.dtype((np.void, matrix.dtype.itemsize * matrix.shape[1]))
    row_keys = np.ascontiguousarray(matrix).view(row_dtype).reshape(-1)
    _, inverse, counts = np.unique(row_keys, return_inverse=True, return_counts=True)
    groups = tuple(
        np.flatnonzero(inverse == group_id).astype(np.int64, copy=False)
        for group_id in np.flatnonzero(counts > 1)
    )
    pair_count = int(sum(len(group) * (len(group) - 1) // 2 for group in groups))
    return groups, pair_count


def _repair_duplicate_distances(
    distances: torch.Tensor,
    duplicate_groups: tuple[Int64Array, ...],
) -> None:
    """Set exact-duplicate pairs to zero on the active device."""

    for group in duplicate_groups:
        indices = torch.as_tensor(group, dtype=torch.long, device=distances.device)
        rows = indices.repeat_interleave(len(group))
        columns = indices.repeat(len(group))
        distances[rows, columns] = 0.0
    distances.fill_diagonal_(0.0)


def build_geometry(
    vectors: NDArray[Any],
    *,
    reference_min: NDArray[Any] | None = None,
    reference_max: NDArray[Any] | None = None,
    device: str | torch.device = "auto",
    return_cpu_distances: bool = False,
) -> GeometryResult:
    """Scale vectors and build one reusable float32 dense distance matrix.

    Exact duplicate rows are repaired to zero after ``torch.cdist``.  This is
    important in float32 because matrix-multiplication distance expansion can
    otherwise assign a small positive distance to identical observations.
    Only callers, such as kNN, decide whether diagonal zeroes are valid edges.
    """

    total_start = time.perf_counter()
    resolved = resolve_device(device)

    scale_start = time.perf_counter()
    scaled, lower, upper, constant_features = _validate_and_scale(
        vectors,
        reference_min=reference_min,
        reference_max=reference_max,
    )
    duplicate_groups, duplicate_pair_count = _duplicate_groups(scaled)
    scaling_ms = (time.perf_counter() - scale_start) * 1_000.0

    # torch.cdist switches to its matrix-multiplication implementation for
    # larger inputs. Subtracting a common centroid is translation invariant
    # and limits float32 cancellation in that implementation.
    centered_for_mm = scaled.shape[0] > 25
    if centered_for_mm:
        distance_input = scaled - scaled.mean(axis=0, dtype=np.float32, keepdims=True)
        distance_input = np.ascontiguousarray(distance_input, dtype=np.float32)
    else:
        distance_input = scaled

    transfer_start = time.perf_counter()
    scaled_device = torch.as_tensor(scaled, dtype=torch.float32, device=resolved)
    if distance_input is scaled:
        distance_input_device = scaled_device
    else:
        distance_input_device = torch.as_tensor(
            distance_input,
            dtype=torch.float32,
            device=resolved,
        )
    synchronize_device(resolved)
    input_transfer_ms = (time.perf_counter() - transfer_start) * 1_000.0

    distance_start = time.perf_counter()
    distances_device = torch.cdist(
        distance_input_device,
        distance_input_device,
        p=2.0,
        compute_mode="use_mm_for_euclid_dist_if_necessary",
    )
    synchronize_device(resolved)
    distance_ms = (time.perf_counter() - distance_start) * 1_000.0

    repair_start = time.perf_counter()
    _repair_duplicate_distances(distances_device, duplicate_groups)
    synchronize_device(resolved)
    duplicate_repair_ms = (time.perf_counter() - repair_start) * 1_000.0

    output_transfer_start = time.perf_counter()
    distances_cpu: Float32Array | None = None
    if return_cpu_distances:
        distances_cpu = np.array(
            distances_device.detach().cpu().numpy(),
            dtype=np.float32,
            order="C",
            copy=True,
        )
    output_transfer_ms = (time.perf_counter() - output_transfer_start) * 1_000.0

    total_ms = (time.perf_counter() - total_start) * 1_000.0
    distance_diagnostics: dict[str, Any] = {
        "device": str(resolved),
        "dtype": "float32",
        "n_samples": int(scaled.shape[0]),
        "n_features": int(scaled.shape[1]),
        "mean_centered_for_mm_cdist": centered_for_mm,
        "exact_duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_pair_count": duplicate_pair_count,
        "cpu_distance_snapshot": return_cpu_distances,
        "timings_ms": {
            "input_transfer": input_transfer_ms,
            "distance": distance_ms,
            "duplicate_repair": duplicate_repair_ms,
            "output_transfer": output_transfer_ms,
        },
    }
    distance = DistanceResult(
        distances_device=distances_device,
        distances_cpu=distances_cpu,
        device=resolved,
        diagnostics=distance_diagnostics,
    )
    diagnostics: dict[str, Any] = {
        **distance_diagnostics,
        "range_source": "input" if reference_min is None else "provided",
        "constant_feature_count": int(np.count_nonzero(constant_features)),
        "timings_ms": {
            "scaling_and_duplicate_detection": scaling_ms,
            **distance_diagnostics["timings_ms"],
            "total": total_ms,
        },
    }
    return GeometryResult(
        scaled_cpu=scaled,
        scaled_device=scaled_device,
        distance=distance,
        reference_min=lower,
        reference_max=upper,
        duplicate_groups=duplicate_groups,
        diagnostics=diagnostics,
    )
