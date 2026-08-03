"""Shared plumbing: device selection, seeding, and small coercions."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

#: Several ops still have no Metal kernel; without this they raise instead of
#: falling back to CPU.  Set before the first MPS allocation to take effect.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick a torch device, preferring CUDA, then Metal, then CPU."""

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed every generator the pipeline draws from."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_float(value: Any) -> float:
    """Collapse a measure result to one number.

    Measures return a scalar under ``imb=False`` but a per-class array under
    ``imb=True``, and a few always return a list.  The mean is the summary used
    throughout, and NaN is preserved rather than silently zeroed so a degenerate
    measure stays visible in the report.
    """

    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.nanmean(array))
