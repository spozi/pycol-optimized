# PyCOL Optimized

`pycol-optimized` is a standalone float32 PyTorch implementation of the PyCOL
data-complexity measures, covering twenty-eight of the reference's twenty-nine
across the feature-based, neighbourhood, structural, and multiresolution
families. See [Measure coverage and known issues](#measure-coverage-and-known-issues)
for the full list, the deviations, and what is left out.

Distance-based measures run through a tile engine that streams the distance
matrix in row blocks, so memory is bounded and one pass serves many measures at
once. It runs on CPU, Apple MPS, and CUDA.

`compute_metrics` remains the compatibility entry point for the six measures
the originating research project uses — F1, N1, class-balanced N3, kDN, CM, and
C1 — with its previous behaviour unchanged.

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

## Scaling

`compute_metrics` still materializes a dense `n x n` float32 distance matrix,
so its memory is quadratic in the number of samples.

The `TileEngine` path removes that limit. It streams the distance matrix in row
blocks and fans each block out to every registered reducer, so peak memory is
`block_size x n` rather than `n x n`, and several measures share one distance
pass instead of one pass each:

```python
from pycol_optimized import (
    NearestEnemy,
    TileEngine,
    TopKPrefix,
    duplicate_group_ids,
    euclidean_kernel,
    resolve_backend,
)

backend = resolve_backend("auto")
kernel = euclidean_kernel(scaled_vectors, backend=backend)
engine = TileEngine(kernel, backend=backend, duplicate_group=duplicate_group_ids(scaled_vectors))
result = engine.run([TopKPrefix(max_k=5), NearestEnemy(labels=labels)])
```

Blocks are cut along rows only, so every row-wise reduction completes inside a
single tile. Float addition is not associative, and a reduction split across
blocks would otherwise make results depend on the memory budget of the machine
that produced them. Leaving `block_size=None` sizes blocks from the device's
reported memory and the kernel's real working set.

For mixed numeric and categorical columns, `heom_kernel` implements PyCOL's
Heterogeneous Euclidean Overlap Metric and returns both the range-normalized
and unnormalized matrices, the latter being what the hypersphere measures are
defined on.

### Measures on top of the primitives

Measures in `pycol_optimized.measures` are pure functions of a primitive's
output rather than of the distance matrix, so one pass serves the whole family
and a narrower `k` needs no recomputation:

```python
from pycol_optimized import (
    LocalSetSize,
    TopKPrefix,
    cm,
    degree_of_overlap,
    kdn,
    local_set_cardinality,
    neighbourhood_counts,
    separability_index,
)

result = engine.run([TopKPrefix(max_k=5), LocalSetSize(labels=labels)])
counts = neighbourhood_counts(result.results["topk"].indices, labels)

kdn(counts)  # k-disagreeing neighbours
cm(counts)  # neighbours mostly disagree
separability_index(counts)  # own class is a plurality
degree_of_overlap(counts)  # at least one enemy among the neighbours
local_set_cardinality(result.results["local_set_size"], labels)

# A narrower prefix reuses the same pass, with no distances recomputed.
kdn(neighbourhood_counts(result.results["topk"].indices, labels, max_k=3))
```

Structural measures come from the same pass by registering more reducers:

```python
from pycol_optimized import NearestEnemy, NearestFriend, borderline, n2, r_value

result = engine.run(
    [TopKPrefix(max_k=5), NearestFriend(labels=labels), NearestEnemy(labels=labels)]
)
n2(
    result.results["nearest_friend"].distances,
    result.results["nearest_enemy"].distances,
    labels,
)
borderline(counts)  # safe / borderline / rare / outlier shares
r_value(counts, theta=2)  # imbalance-weighted pairwise class overlap
```

The C-family scores every prefix length rather than only the widest, so it
reads the neighbours in order instead of as a histogram:

```python
from pycol_optimized import c1, c2, neighbourhood_profile

profile = neighbourhood_profile(
    result.results["topk"].indices, result.results["topk"].distances, labels
)
c1(profile)  # same-label purity across every prefix
c2(profile)  # the same, weighted by how close each agreeing neighbour is
```

Every measure takes `imb=True` to return per-class values normalized by class
size rather than one dataset value. `kDN`, `CM`, `SI`, `degOver`, `D3`, `LSC`,
`N2`, `borderline`, `R-value`, `C1`, and `C2` are all verified against the
pinned `pycol-complexity==1.0.4` reference, in both `imb` settings.

`borderline` requires exactly five neighbours, because its categories are
absolute enemy counts rather than fractions of `k`; it raises instead of
silently rescaling. `D3` and `R-value` return per-class and per-class-pair
values respectively, as the reference does.

### Feature-based measures

`pycol_optimized.features` describes how far individual features separate the
classes, so it reads the embedding directly and never builds a distance matrix
or touches the tile engine:

```python
from pycol_optimized import class_feature_stats, f1v, f2, f3, input_noise

stats = class_feature_stats(vectors, labels, backend="auto")
f1v(stats)  # separation along the best discriminant direction
f2(stats)  # feature-range overlap, multiplied across features
f3(stats, vectors)  # share of samples the best feature leaves ambiguous
input_noise(stats)  # share of readings inside the other class's range
```

These are one-vs-one, returning one entry per class pair as the reference does.
`f2`, `f3`, and `input_noise` take `imb=True`.

Three reference behaviours are reproduced rather than corrected. `f3` needs the
embedding as well as the summaries, because it counts the overlap region across
every sample while dividing by only the pair's size, so a multiclass value can
exceed one. `input_noise` likewise divides by the whole dataset, so a multiclass
value is diluted by classes outside the pair. And `input_noise` compares
strictly where `f2` and `f3` use closed comparisons, so a reading sitting
exactly on the other class's boundary does not count as noise.

`F4` is not implemented. The reference's own `F4_optimized` evaluates
`overlapped_region and c1_inds` on two arrays, which raises, and `F4` has a
branch that appends an undefined `sample`. Reproducing either faithfully is not
meaningful, and deviating silently would break the parity guarantee the rest of
this module holds to.

### Hypersphere measures

Each sample grows a sphere until it would touch a differently-labelled sample,
then spheres wholly inside another are absorbed. Both quadratic steps go
through the tile engine; the linear, sequential steps between them stay on the
host:

```python
from pycol_optimized import (
    ContainingSphere,
    NearestEnemy,
    sphere_coverage,
    sphere_radii,
    t1,
)

enemy = engine.run([NearestEnemy(labels=labels, use_unnormalized=True)]).results
radius = sphere_radii(enemy["nearest_enemy"].indices, enemy["nearest_enemy"].distances)

absorber = engine.run([ContainingSphere(radius=radius)]).results["containing_sphere"]
t1(sphere_coverage(radius, absorber), labels)
```

These read a kernel's **unnormalized** matrix, so they need `heom_kernel`, and
they are meaningful only for purely numeric inputs — the same restriction the
reference documents.

`sphere_radii` resolves each chain iteratively rather than recursively. The
reference recurses once per chain link and raises `RecursionError` past the
interpreter's limit; the iterative walk produces the same radii with no such
ceiling, which a 6,000-link regression test covers.

ONB covers each class greedily with enemy-free balls. Scoring every candidate
in an iteration is a masked row-count, so one tile pass evaluates the whole
iteration and the number of passes is the number of balls:

```python
from pycol_optimized import NearestEnemy, onb, onb_cover, resolve_backend

backend = resolve_backend("auto")
enemy = build_engine().run([NearestEnemy(labels=labels)]).results
balls = onb_cover(build_engine, labels, enemy["nearest_enemy"].distances, backend=backend)
onb(balls, labels)
```

`onb_cover` takes a factory rather than an engine because the candidate scores
depend on what is still uncovered, so each iteration needs its own pass.

`nsg` and `icsv` read the same cover: NSG is the mean ball occupancy, ICSV the
spread of ball densities. Both are pure functions of the ball list, so they cost
nothing beyond the cover itself:

```python
from pycol_optimized import icsv, nsg

nsg(balls, labels)  # samples per ball
icsv(balls, labels, vectors.shape[1])  # how unevenly the balls are packed
```

`icsv` needs the feature count because a ball's volume is its radius raised to
that power; radii are scaled by the largest first, since an unscaled radius
overflows quickly at realistic feature counts. A zero-radius ball yields an
infinite density, which propagates rather than being silently dropped, as in
the reference.

`dbc` reduces each class to its ball centres, spans them with a minimum
spanning tree, and reports the share of centres touching a class-crossing edge:

```python
from pycol_optimized import dbc

dbc(balls, vectors, labels, backend=backend)
```

It rises as the cover coarsens rather than as the classes overlap — with one
ball per class the single spanning edge must cross, giving exactly one. The
centres are re-normalized against their own feature ranges rather than the full
dataset's, matching the reference, which rebuilds its distance matrix from the
reduced set.

Containment and coverage tests compare a distance against a **radius
difference** rather than adding a distance to a radius. The two are equivalent
in exact arithmetic, but in float32 the sum rounds back onto the larger radius
and swallows spheres that sit just outside it. Regression tests cover the
boundary, on clustered data — on random data no sphere ever absorbs another, so
the test would pass against a no-op.

### Multiresolution measures

`pycol_optimized.multiresolution` cuts the feature space into a grid and scores
the same statistic at every resolution, from a single cell up to a fine mesh.
No pairwise distances are involved:

```python
from pycol_optimized import purity

profile = purity(vectors, labels)
profile.value  # the summary the reference reports
profile.per_resolution  # the score at each grid resolution
```

Cell membership is a vectorized search per resolution rather than the
reference's per-sample, per-feature, per-boundary loop keyed on concatenated
strings: 14x faster at 200 samples and 22x at 400, growing with the sample
count.

Two boundary conventions are reproduced exactly. Intervals are closed at both
ends and the first match wins, so a value sitting on a boundary falls to the
*lower* cell. And a value is tested against `[bound, bound + step]` rather than
`[bound, next_bound]` — those differ in floating point, because `linspace` does
not accumulate a step the way repeated addition does, and at fine resolutions
the difference moves values between cells.

`neighbourhood_separability` asks a different question of the same grid: for
cells that are *not* single-class, are the classes still locally ordered
inside them? It takes a full distance matrix rather than a tile kernel, because
scoring a sample needs its distances to arbitrary members of its cell:

```python
from pycol_optimized import neighbourhood_separability

neighbourhood_separability(vectors, labels, distances).value
```

> **This one is not reference-parity, unlike every other measure here.** The
> reference reaches it through a neighbour-counting helper that looks a
> neighbour's label up by its position *within the cell* rather than by its
> sample index. Once a cell is a strict subset of the dataset those are
> different samples, so the reference scores against unrelated labels.
> This implementation uses each neighbour's own label. Reproducing the
> reference's behaviour would make the value meaningless, so the deviation is
> deliberate and is asserted by a test rather than hidden.

Its profile is also integrated as-is — the reference neither rescales it to the
observed range nor divides by a constant, as it does for purity — so the two
values are not on the same scale. A perfectly separable class scores
`(depth - 1) / same_class_total` rather than one, because the profile is
integrated over positions that stop short of the final neighbour.

### Choosing a compute mode

`euclidean_kernel` and `heom_kernel` take `compute_mode="auto"`, which picks
from the device because the faster path is not the same on all of them:

| Device | Resolved mode | Block-size invariant |
| --- | --- | --- |
| `cpu` | `donot_use_mm_for_euclid_dist` | yes |
| `mps` | `use_mm_for_euclid_dist_if_necessary` | no |
| `cuda` | `use_mm_for_euclid_dist_if_necessary` | no |
| anything else | `donot_use_mm_for_euclid_dist` | yes |

The direct mode's arithmetic depends only on the feature width, so results are
bit-identical at any block size. The matrix-multiply mode is faster on
accelerators, but `torch.cdist` selects it from the tensor shape, so tiles of
different heights round differently.

**The default therefore trades a reproducibility guarantee for speed on
accelerators.** Pass `compute_mode="donot_use_mm_for_euclid_dist"` to keep the
guarantee everywhere. Every run reports what it actually got:

```python
result.diagnostics["block_size_invariant"]  # True or False
result.diagnostics["kernel"]["compute_mode_source"]  # "auto:mps" or "explicit"
```

The CUDA row is inferred from the MPS result, not measured; no NVIDIA benchmark
has been run.

Measurements on one Apple Silicon machine, `n = 32,000`, 20 features, `k = 5`,
computing a k-nearest-neighbour prefix and nearest-enemy assignment in one
pass: MPS under `auto` took 6.5 s against 16.2 s when forced onto the
reproducible path, a 2.5x gain. CPU under `auto` took 5.4 s. On this hardware
MPS does not beat CPU for this workload, because neighbour selection rather
than distance computation dominates; measure before assuming an accelerator
helps.

## Measure coverage and known issues

Twenty-eight of the reference's twenty-nine measures are implemented. Every one
listed below is verified against the pinned `pycol-complexity==1.0.4` reference,
in both `imb` settings where the reference offers them, except where noted.

| Family | Measures |
| --- | --- |
| Feature-based | F1, F1v, F2, F3, IN |
| Neighbourhood | N3, kDN, CM, SI, degOver, D3, R-value, borderline |
| Structural | N1, N2, LSC, T1, ONB, NSG, ICSV, DBC |
| Multiresolution | C1, C2, purity, neighbourhood_separability\* |

### Deviations from the reference

**`neighbourhood_separability` does not match the reference, deliberately.**
The reference reaches it through a neighbour-counting helper that looks a
neighbour's label up by its position *within the cell* rather than by its sample
index. Once a cell is a strict subset of the dataset those are different
samples, so the reference scores neighbours against unrelated labels. This
implementation uses each neighbour's own label. Reproducing the reference would
make the number meaningless, so the deviation is asserted by a test rather than
hidden. Note that correcting the label lookup does not on its own account for
the whole difference, so at least one further divergence remains unidentified.

**`sphere_radii` resolves chains iteratively, not recursively.** Same radii, but
without the reference's `RecursionError` on long chains.

**`ONB` indexes classes by position, not by label value.** The reference tallies
into an array indexed by the class label itself, which silently requires labels
to be `0..C-1`. This implementation is correct for arbitrary labels, and agrees
with the reference wherever the reference is well defined.

### Not implemented

**`F4`** — the reference cannot run it. `F4_optimized` evaluates
`overlapped_region and c1_inds` on two arrays, which raises, and `F4` has a
branch that appends an undefined `sample`. With no working reference there is
nothing to establish parity against, and shipping an unverified value would
break the guarantee every other measure here holds to.

**`N4`** — needs a rectangular query-against-reference distance matrix, which
the tile engine does not model: it assumes a dataset compared against itself,
throughout self-exclusion, duplicate repair, and block sizing. The measure also
interpolates synthetic samples with unseeded `numpy.random`, so the reference
returns a different value on every call and parity is not well defined until a
seeding convention is chosen.

**`MRCA`** — needs k-means clustering, which would add scikit-learn as a
dependency this project otherwise avoids.

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
