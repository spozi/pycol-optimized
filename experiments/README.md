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

### Datasets

`--dataset` selects the corpus. Every one downloads from its origin on first
use with nothing but the standard library — no dataset hub, no loading scripts,
no authentication — and is cached thereafter. Figures below were measured
through the loader, not quoted from papers:

| `--dataset` | Train | Test | Classes | Skew | Minority | Character |
| --- | --- | --- | --- | --- | --- | --- |
| `phrasebank` | 1,811 | 453 | 3 | 4.6:1 | `negative` (242) | tiny; finance jargon; **non-commercial** |
| `sms_spam` | 4,460 | 1,114 | 2 | 6.5:1 | `spam` (598) | saturates BERT — ceiling control |
| `trec` | 4,761 | 1,191 | 6 | 14.1:1 | `ABBR` (76) | small, clean, fast |
| `tweeteval` | 9,576 | 2,394 | 2 | 1.4:1 | `hate` (4,028) | hard but barely skewed |
| `banking77` | 10,469 | 2,614 | 77 | 3.0:1 | `contactless…` (60) | many fine-grained classes |
| `davidson` | 19,826 | 4,957 | 3 | 13.4:1 | `hate` (1,144) | noisy tweets; **contains slurs** |
| `goemotions` | 36,358 | 9,088 | 28 | **337:1** | `grief` (38) | most extreme skew here |
| `agnews` | 102,080 | 25,520 | 4 | 1.0:1 | — | balanced; large |

**Which to use.** `sms_spam` is a poor primary choice: BERT reaches macro-F1
above 0.9 from 400 samples and one epoch, so the baseline saturates and
augmentation has no room to show an effect. On `trec` at 600 samples, kDN is
0.47 against SMS Spam's 0.03 — the same panel, an order of magnitude more
difficulty. Prefer `davidson` and `goemotions` for skew, `trec` for fast
iteration, `agnews` for scale. Keep `sms_spam` as a ceiling control: a dataset
where augmentation demonstrably *cannot* help is a useful negative reference.

Binary corpora barely exercise the panel. R-value, D3, degOver, F1v, F2, and F3
are all per-class-pair, and two classes means exactly one pair; `goemotions`
gives 378 and `banking77` gives 2,926.

**Reshaping.** `agnews` is balanced by construction, so it becomes an imbalance
study only under `--imbalance-ratio`, which downsamples every class but the
largest until majority:minority hits the given figure. `--max-train` then caps
the total, keeping class shares. Both touch the **training half only** — the
test half keeps its natural distribution, because a test set squeezed to the
same skew would leave too few minority samples to measure recall on, and recall
is the metric the whole experiment turns on.

```bash
# 20:1 imbalance, capped at 8k training samples
python -m complexity_augmentation.run --dataset agnews \
  --imbalance-ratio 20 --max-train 8000
```

**One quirk to expect.** F3 can exceed 1 on multi-class data. That is not a
bug: the reference counts samples in the overlap region across the whole
dataset while dividing by only the pair's size, and the library reproduces that
rather than silently correcting it.

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

On a GPU server, install torch from the index matching the *card*, not just the
driver, then run as above — `--device` defaults to `auto` and picks CUDA on its
own:

```bash
python -m pip install "torch==2.13.0" \
  --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt && pip install -e ..
python -m complexity_augmentation.run --output-dir results
```

**Check the wheel actually has kernels for the card before anything else.** A
torch build can import, report `cuda.is_available() == True`, and still fail at
the first matmul if it carries no cubin for that architecture — Blackwell
(RTX 50-series, `sm_120`) is the current instance of this. CUDA 13.2 (`cu132`)
is the pinned environment for this experiment. One command settles it:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, \
torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); \
print(torch.cuda.get_arch_list()); \
print((torch.randn(4096,4096,device='cuda')@torch.randn(4096,4096,device='cuda')).sum().item())"
```

The capability must appear in the arch list and the matmul must return a number.
`no kernel image is available for execution on the device` means the wheel is
too old for the card, regardless of what the driver supports.

Mixed precision, pinned memory, and loader workers switch on automatically under
CUDA and stay off elsewhere. The autocast dtype is bf16 wherever the card
supports it and fp16 otherwise; bf16 keeps fp32's exponent range, so no loss
scaling is involved. The precision used is recorded in every result row.

Runtime has only been measured on Metal, and only for a 96-sample smoke run, so
there is no honest estimate for the full run yet. Get one cheaply before
committing the machine to it:

```bash
python -m complexity_augmentation.run --arms baseline --seeds 0 --epochs 1
```

### Choosing parameters

The dataset is small — 4,460 training texts — so **steps, not memory, are the
binding constraint**, and a large-VRAM card does not change that. Batch size
sets how many optimizer steps each arm gets across three epochs:

| `--batch-size` | baseline | duplicate / uniform | minority |
| --- | --- | --- | --- |
| 16 | 837 | 1,674 | 1,449 |
| 32 | 420 | 837 | 726 |
| 64 | 210 | 419 | 363 |
| 128 | 105 | 210 | 182 |

At 128 the baseline arm gets about a hundred updates, which is not enough to
fine-tune BERT reliably; the arms would differ by optimization budget as much as
by data. Stay at 16 or 32 and raise `--learning-rate` only if you move up:
2e-5 at batch 16–32 is the well-characterised setting.

Spare GPU capacity is better spent on `--seeds 0 1 2 3 4` and on sweeping
`--mask-prob 0.10 0.15 0.20` (the paper's three values, one run each) than on
larger batches. Seeds buy confidence in a difference; batch size past 32 only
buys utilization.

One confound to keep in view: at fixed epochs the augmented arms take roughly
twice as many optimizer steps as the baseline, simply because they hold twice
the data. The `duplicate` arm controls for this as well as for sample size — it
gets the same doubled step count while adding no information, so a `uniform`
gain over `duplicate` is not just extra training.

Writes `results.json` (everything) and `report.md` (complexity deltas, then
downstream metrics). Embeddings are cached in `cache/`, so re-running the
complexity side after a code change costs only the measures.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--dataset` | corpus to run on; see the table above |
| `--imbalance-ratio` | downsample the training half to this majority:minority skew |
| `--max-train` | cap the training half, keeping class shares |
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

Accuracy is not the headline. At SMS Spam's 6.5:1 ratio, predicting "ham"
everywhere already scores about 87%, and on `goemotions` at 337:1 the majority
baseline is far higher still. The metrics that move are macro F1, the minority
class's own precision/recall/F1, and its one-vs-rest average precision — the
last being the one that shifts when a model trades minority recall for
precision. `minority_average_precision` is defined identically whether there
are two classes or seventy-seven, which is what makes arms comparable across
corpora.

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
