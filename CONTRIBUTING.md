# Contributing

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Required checks

```bash
ruff check .
ruff format --check .
pytest
python -m build
twine check dist/*
```

Changes to metric definitions, tie handling, float precision, aggregation, or
device selection must include comparison tests against the pinned scientific
reference. CUDA performance claims require results from actual NVIDIA
hardware; Apple MPS results cannot be generalized to CUDA.

Do not silently replace exact N1 with a kNN-graph approximation. Approximate
metrics must have distinct names and documented scientific interpretations.
