# Experiments

Research code that *uses* `pycol-optimized`. Nothing here ships in the wheel —
`src/` is the library, this directory is work built on top of it.

## complexity_augmentation

**Question.** Mask-and-fill augmentation is reported to improve a downstream
classifier. Does it do so by making the training data *less complex*, and if so,
does the complexity drop predict the accuracy gain?

The augmentation is Algorithm 1 of Pozi & Sato (2025), *A data-augmented model
routing framework for efficient LLM deployment in edge–cloud environments*, The
Journal of Supercomputing 81:1573,
[doi:10.1007/s11227-025-08034-8](https://doi.org/10.1007/s11227-025-08034-8)
(open access): tokenize, replace each token with `[MASK]` at probability
`p_mask`, fill each mask with a masked language model's argmax, and append the
result to the original set.

The paper applies this to code prompts, filling with GraphCodeBERT and routing
with CodeBERT-python. Here the domain is ordinary English SMS, so the direct
analogue is `bert-base-uncased` for both the fill and the classifier.

### Design

Four arms share one fixed, never-augmented test set:

| Arm | Training set | Purpose |
| --- | --- | --- |
| `baseline` | original | reference point |
| `duplicate` | original + verbatim copy | **size control** |
| `uniform` | original + one mask-fill copy of each | Algorithm 1 at 1× |
| `minority` | original + mask-fill copies of the minority class | rebalancing |

Three decisions carry most of the weight, and each exists to stop a specific
way the result could be an artifact rather than a finding.

**The complexity encoder is frozen and shared.** Complexity is measured on
mean-pooled `bert-base-uncased` embeddings from a model that is never updated,
and the same weights embed every arm. Embedding with the fine-tuned classifier
instead would make the complexity drop tautological: that encoder was optimized
to separate exactly these classes, so it would report training success, not data
difficulty.

**The `duplicate` arm is the control that makes the comparison mean anything.**
Augmenting at 1× doubles the sample count, and complexity measures move with
sample count on their own — T1's sphere counts, N1's spanning forest, and kDN's
neighbourhoods all shift when density changes. Duplicating adds zero
information while doubling `n`, so whatever it does to a measure is pure size
artifact. A drop in the `uniform` arm is only evidence of real simplification
insofar as it *exceeds* the drop in `duplicate`.

Exact duplicates also land on the library's `duplicate_group_ids` path as
zero-distance pairs, which is worth watching: with `k = 5`, every duplicated
sample has its twin as a distance-0 nearest neighbour, so agreement-based
measures like kDN improve mechanically.

**The fill is an argmax, so the augmentation can be close to a no-op.** The
model is free to predict the token that was just masked out. The pipeline
measures how often that happens (`changed_rate`, `texts_unchanged_rate`) rather
than assuming augmentation occurred. `--avoid-original` forbids it — this is
*not* in the paper, and is provided to bound how much the conservatism matters.

Two consequences of masking at the *token* level are inherent to Algorithm 1 and
are left as-is: `bert-base-uncased` lowercases its input, and masking a subword
piece can corrupt a word (`Go until jurong` → `gorong gorong`). Since the fill
model, the complexity encoder, and the classifier are all uncased, no arm sees a
casing difference the others do not.

### Running

```bash
cd experiments
pip install -r requirements.txt
pip install -e ..

# complexity only, no fine-tuning (minutes)
python -m complexity_augmentation.run --skip-training

# the full experiment: 4 arms x 3 seeds of BERT fine-tuning
python -m complexity_augmentation.run --output-dir results
```

On a GPU server, install torch from the index matching the driver's CUDA
version *before* the rest, then run as above — `--device` defaults to `auto`
and picks CUDA on its own:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt && pip install -e ..
python -m complexity_augmentation.run --output-dir results --batch-size 32
```

Mixed precision and pinned-memory loading switch on automatically when the
device is CUDA and stay off everywhere else (`TrainConfig.amp`,
`TrainConfig.num_workers`). A larger `--batch-size` is usually worth it on a
discrete GPU; 16 is the default because it is the safer figure on a laptop.

Runtime has only been measured on Metal, and only for a 96-sample smoke run, so
there is no honest estimate for the full run yet. Get one cheaply before
committing the machine to it:

```bash
python -m complexity_augmentation.run --arms baseline --seeds 0 --epochs 1
```

Writes `results.json` (everything) and `report.md` (complexity deltas, then
downstream metrics). Embeddings are cached in `cache/`, so re-running the
complexity side after a code change costs only the measures.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--mask-prob` | 0.10 / 0.15 / 0.20, the values swept in the paper |
| `--fill-strategy joint` | fill every mask in one pass instead of one at a time |
| `--avoid-original` | forbid the fill model from restoring the masked token |
| `--include-expensive` | add N1, purity, ONB, NSG, ICSV, DBC |
| `--arms baseline uniform` | run a subset |
| `--seeds 0` | one seed instead of three |

`--include-expensive` is off by default because `onb_cover` runs one engine pass
per ball it places, and on scattered high-dimensional embeddings the cover can
approach one ball per sample. `purity` grids every feature, which in 768
dimensions puts almost every sample in its own cell — computable, but not
informative.

### Reading the output

Accuracy is not the headline. At a 1:6.5 class ratio, predicting "ham"
everywhere scores about 87%. The metrics that move are macro F1, the minority
class's own precision/recall/F1, and average precision.

The result this is built to detect is a *dissociation*: augmentation that lowers
complexity without improving the classifier, or improves the classifier without
lowering complexity. Either outcome is more interesting than the two moving
together, and the `duplicate` arm is what makes it possible to tell them apart.

### Devices

Runs on CPU, MPS, and CUDA; `--device` defaults to `auto` (CUDA → MPS → CPU).
`PYTORCH_ENABLE_MPS_FALLBACK=1` is set on import for the few ops that still have
no Metal kernel.

Mixed precision is enabled on CUDA and deliberately **not** on MPS: fp16
autocast on Metal still returns wrong values in places, and BERT-base at this
scale does not need it. The gradient scaler is constructed either way but stays
disabled off CUDA, where it is a pass-through. Pinned memory and DataLoader
workers are likewise CUDA-only — on unified memory there is no host-to-device
transfer to overlap, so they would only add process overhead.

Because AMP changes the arithmetic, CUDA numbers are not bit-comparable with
Metal ones. Run all arms of a comparison on the same device; `--amp` can be
turned off through `TrainConfig` if you need CUDA and CPU to agree more closely.
