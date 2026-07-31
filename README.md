# PyCOL Optimized

`pycol-optimized` is a standalone float32 PyTorch implementation of the six
PyCOL data-complexity metrics used by its originating research project:

- F1;
- N1;
- class-balanced N3;
- kDN;
- CM;
- C1.

The distribution name is `pycol-optimized`; the Python import is
`pycol_optimized`.

> This is an independent compatibility implementation. It is not affiliated
> with or endorsed by the maintainers of
> [PyCOL](https://github.com/DiogoApostolo/pycol).

## Installation

Install the optimized implementation:

```bash
python -m pip install pycol-optimized
```

Install the optional pinned scientific-reference adapter:

```bash
python -m pip install "pycol-optimized[reference]"
```

For local development:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np

from pycol_optimized import compute_metrics

vectors = np.asarray(
    [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.8, 0.7],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)
labels = np.asarray([0, 0, 1, 1])

result = compute_metrics(
    vectors,
    labels,
    metrics=("F1", "N1", "N3", "kDN", "CM", "C1"),
    neighbors=2,
    device="auto",
)

print(result.metrics)
print(result.project_composite)
print(result.diagnostics["device"])
```

`device="auto"` selects CUDA, then Apple MPS, then CPU. Explicit values are
`"cpu"`, `"mps"`, and `"cuda"`.

For a before/after study, freeze the scaling range to the original cohort:

```python
reference_min = original_vectors.min(axis=0)
reference_max = original_vectors.max(axis=0)

before = compute_metrics(
    original_vectors,
    original_labels,
    device="cpu",
    reference_min=reference_min,
    reference_max=reference_max,
)
after = compute_metrics(
    augmented_vectors,
    augmented_labels,
    device="cpu",
    reference_min=reference_min,
    reference_max=reference_max,
)
```

## Multilabel targets

Pass a binary `[samples, labels]` indicator matrix:

```python
result = compute_metrics(
    vectors,
    multihot_labels,
    multilabel=True,
    label_names=["payment", "termination", "liability"],
    neighbors=5,
    device="mps",
)
```

Constant label columns are skipped. `result.metrics` contains the unweighted
macro average across valid one-vs-rest problems. Per-label values and
positive-prevalence-weighted values are available through `result.per_label`
and `result.label_weighted_metrics`.

## Scientific compatibility

- F1 uses equal one-vs-one class-pair and feature weighting, population
  variance, and PyCOL 1.0.4's nonfinite-ratio behavior.
- N3 uses one neighbor and is averaged equally across true classes.
- kDN is the sample-micro fraction of disagreeing labels over the requested
  neighbor prefix.
- CM uses the strict `different_neighbors * 2 > k` condition.
- C1 averages same-label purity over every ordered neighbor prefix.
- N1 uses a deterministic minimum-spanning forest and excludes distances
  `<= 1e-8`.

The kNN implementation excludes only the diagonal, retains eligible
zero-distance duplicates, and resolves computed equal distances using the
smallest original sample index.

The optional reference can be called with:

```python
from pycol_optimized import compute_n1_reference

reference = compute_n1_reference(vectors, labels)
```

## Devices and numerical behavior

The CPU and Apple MPS paths are validated. The CUDA code path uses the same
PyTorch operations but has not yet been benchmarked on NVIDIA hardware.

The implementation intentionally uses float32. Results normally agree with
the official float64 reference within the documented tolerances, but
float32-collapsed points and equal or nearly equal distances can alter
neighbor or minimum-spanning-tree order. In the validation suite, the only
out-of-tolerance non-N1 result was C1 on a deliberately adversarial tied,
cross-class-duplicate fixture; no reported metric or dataset ranking changed.

## Scaling limitation

The current exact implementation materializes a dense `n x n` float32 distance
matrix and a same-sized working copy for kNN construction. Memory and distance
computation are therefore quadratic in the number of samples. Sampling is
recommended for large datasets; exact blockwise kNN is not implemented yet.

## Development

```bash
ruff check .
ruff format --check .
pytest
python -m build
twine check dist/*
check-wheel-contents dist/*.whl
```

See `RELEASING.md` for the publication checklist and `NOTICE` for the
relationship to the separately maintained PyCOL project.
