# Does mask-and-fill augmentation work by reducing data complexity?

Results from `complexity_augmentation`, run on the Financial PhraseBank,
2026-08-01/02. Every figure here was produced by the code in this directory and
is reproducible from `results/*/results.json`.

---

## 1. The question

Data augmentation is reported to improve downstream classifiers. A common
explanation is that it makes the training data *easier* — better separated
classes, cleaner neighbourhoods, less overlap. Data-complexity measures claim to
quantify exactly that. So the explanation is testable:

> **If augmentation helps by simplifying the data, then a training set that
> scores lower on data-complexity measures should produce a better classifier.**

The augmentation under test is Algorithm 1 of Pozi & Sato (2025), *A
data-augmented model routing framework for efficient LLM deployment in
edge–cloud environments*, The Journal of Supercomputing 81:1573
([doi:10.1007/s11227-025-08034-8](https://doi.org/10.1007/s11227-025-08034-8)):
tokenize, replace each token with `[MASK]` at probability `p_mask`, fill each
mask with a masked language model's argmax, append the result to the original
set.

### Hypotheses

| | Statement |
| --- | --- |
| **H1** | Mask-and-fill augmentation reduces measured data complexity. |
| **H2** | Augmented training sets produce better classifiers. |
| **H3** | The complexity reduction *explains* the accuracy gain — lower complexity predicts higher accuracy. |

H3 is the claim of interest. H1 and H2 are its preconditions: if the
augmentation does not reduce complexity, or does not improve accuracy, H3 has
nothing to explain.

---

## 2. Methodology

### 2.1 The core problem: complexity measures move for the wrong reasons

Complexity measures are computed on a *sample*. Change the sample and they move,
whether or not the underlying problem got easier. Augmenting at 1× doubles the
training set, and:

- **kDN, N2, N1** count or measure distances to nearest neighbours. Add copies
  and every point acquires a near-twin, so neighbourhoods look cleaner.
- **T1** counts hyperspheres. More points in the same volume means fewer
  surviving spheres.
- **F1v, F3** weight class statistics by class size. Change the class balance
  and they move even if no class's distribution changed at all.

A naive before/after comparison therefore cannot distinguish "the data got
simpler" from "the sample got denser".

### 2.2 The design: every treatment paired with a null control

Five arms share one fixed test set that is never augmented and never resampled.

| Arm | Training set | Role |
| --- | --- | --- |
| `baseline` | original | reference |
| `duplicate` | original + verbatim copy | **null control for `uniform`** |
| `uniform` | original + one mask-fill copy of each | Algorithm 1 at 1× |
| `minority_duplicate` | original + verbatim minority copies to 1:1 | **null control for `minority`** |
| `minority` | original + mask-fill minority copies to 1:1 | rebalancing |

Each control reproduces its treatment's *sample-level side effect* while adding
**no information**:

- `duplicate` matches `uniform`'s sample count exactly (7,752 both).
- `minority_duplicate` matches `minority`'s count and class balance exactly
  (6,786 both, minority rate 0.5000 both), drawing **the same source samples
  from the same seed**. The only difference between the pair is whether those
  copies were mask-filled.

**The decision rule, fixed before results were seen:** a treatment counts as
reducing complexity only where it beats *its own control*, not where it beats
`baseline`.

### 2.3 Controls on the measurement itself

**Complexity is measured in a frozen space.** All arms are embedded with the
same never-updated `bert-base-uncased`, mean-pooled over the last hidden state.
Embedding with the fine-tuned classifier would make any complexity drop
tautological — that encoder was optimised to separate exactly these classes, so
it would report training success, not data difficulty.

**The augmentation is verified to have occurred.** The fill is an argmax, so the
model may predict back the token it just masked. Measured, not assumed:
`changed_rate = 0.529` on `uniform`, with 18.4% of texts token-identical to
their source. The augmentation is real.

**Inference is deterministic.** All inference models run under `eval()` and
`torch.inference_mode()` with frozen parameters, verified by tests that assert
embeddings repeat *exactly* — plus a test that puts the encoder back into
training mode to confirm the check would catch live dropout rather than passing
vacuously.

### 2.4 Configuration

Corpus `phrasebank_50` (n = 4,846; 3,876 train / 970 test; 3 classes; 4.77:1;
minority `negative`, 483 training samples). `p_mask = 0.15`.
Classifier `bert-base-uncased`, 3 epochs, batch 16, lr 2e-5, **5 seeds per
arm** — 25 fine-tuning runs, 298.6 minutes. Headline metrics are macro-F1 and
the minority class's own precision/recall/F1, because accuracy is uninformative
under skew.

---

## 3. Results

### 3.1 Complexity

Bold marks each treatment's control — the number it must beat.

| Measure | baseline | **duplicate** | uniform | **minority_dup** | minority |
| --- | --- | --- | --- | --- | --- |
| kDN | 0.3694 | **0.2715** | 0.2812 | **0.2180** | 0.2295 |
| C1 | 0.3512 | **0.1817** | 0.1883 | **0.1868** | 0.2036 |
| C2 | 0.9947 | **0.5390** | 0.8630 | **0.5064** | 0.8595 |
| N2 | 0.4810 | **0.0000** | 0.2464 | **0.3358** | 0.3901 |
| T1 | 0.9979 | **0.4990** | 0.9043 | **0.5700** | 0.9207 |
| borderline | 0.3945 | **0.3455** | 0.3327 | **0.1640** | 0.1823 |
| F1v | 0.0719 | 0.0719 | *0.0838* | 0.0633 | *0.0789* |

> **`uniform` beats `duplicate` on 0 of 7. `minority` beats `minority_duplicate`
> on 0 of 7.**

Against `baseline` alone, `uniform` improves kDN by 24% and looks effective.
Verbatim copying improves it by 26.5% — more, on every measure.

The single exception is **F1v**, which moves only for the mask-fill arms:
+16.5% for `uniform` over `duplicate`, +24.7% for `minority` over its control.
The augmentation *does* change the class-conditional feature distribution. It
simply does not make the classes more separable.

### 3.2 Classification

| Arm | n | macro-F1 | minority-F1 | min-recall | min-precision |
| --- | --- | --- | --- | --- | --- |
| baseline | 3,876 | 0.8544 ± 0.0054 | 0.8648 | 0.8777 | 0.853 |
| duplicate | 7,752 | 0.8550 ± 0.0048 | 0.8623 | 0.8595 | 0.866 |
| uniform | 7,752 | 0.8536 ± 0.0044 | 0.8576 | 0.8512 | 0.865 |
| minority_duplicate | 6,786 | 0.8534 ± 0.0024 | 0.8597 | 0.8711 | 0.849 |
| **minority** | 6,786 | **0.8278 ± 0.0099** | **0.8078** | **0.9190** | **0.721** |

Spreads above are population standard deviations over five seeds; §3.3 uses
the sample standard deviation, which is marginally larger.

`uniform`: **−0.0007** against baseline, **−0.0014** against its control. Four
of five arms lie within 0.0016 of each other while per-arm standard deviations
are 0.0024–0.0054. With ~14 points of headroom below the ceiling and five seeds,
this is a genuine null rather than an underpowered result (§3.3, p = 0.676).

`minority`: **−0.0255** against its control (t = −5.03, p = 0.0055). The
control isolates the cause — `minority_duplicate` reaches the *identical* 1:1 balance from the *identical*
source samples and lands on baseline, so the damage is **not** rebalancing. It
is the mask-fill corrupting minority-class labels: +4 points of recall bought
for **−13 points of precision**.

### 3.3 Null hypotheses and tests

The word "null" is used here in two senses that are worth keeping apart. A
**null control** is a design element — an arm reproducing a treatment's
sample-level side effect while adding no information. A **null hypothesis** is
the formal statement a test tries to reject. The first determines the second,
and that is the whole point of the design:

| Framing | H₀ | Why |
| --- | --- | --- |
| Naive | μ(uniform) = μ(baseline) | Rejecting this proves nothing: duplication alone changes sample size and optimizer-step count |
| **Used here** | μ(uniform) = μ(**duplicate**) | Isolates what the mask-fill contributed over replication |

Welch two-sided t-tests, five seeds per arm:

| H₀ | difference | t | p | Cohen's d | Outcome |
| --- | --- | --- | --- | --- | --- |
| μ(uniform) = μ(duplicate) | −0.0014 | −0.43 | **0.676** | −0.27 | fail to reject |
| μ(minority) = μ(minority_duplicate) | −0.0255 | −5.03 | **0.0055** | −3.18 | **reject**, harmful direction |
| μ(duplicate) = μ(baseline) | +0.0007 | +0.19 | **0.854** | +0.12 | fail to reject |

Two primary comparisons, so a Bonferroni threshold is α = 0.025; `minority`
clears it and `uniform` is nowhere near.

**H1 has no null hypothesis, because complexity measures are deterministic.**
One dataset yields one number, with no sampling distribution and nothing to
reject. The "0 of 7" counts are direct numerical comparisons rather than
inference. This cuts both ways: there is no seed noise to contend with, but
also no interval, and 0.2812 against 0.2715 is reported as a fact rather than
an estimate. The uncertainty that genuinely exists there concerns the choice of
embedding space and corpus, which no p-value addresses — §7 covers the former
and §8 the latter.

**What the tests license.** The seeds are reruns of one procedure on one
corpus, so the inference is *whether another seed would change the answer*, not
*whether another corpus would*. p = 0.676 means the `uniform` null is robust to
seed variation on `phrasebank_50`. It says nothing about `davidson` or
`goemotions`.

### 3.4 The decisive contrast

| | duplicate | minority (vs its control) |
| --- | --- | --- |
| Complexity change | N2 → 0.0000, T1 halved, kDN −26.5% | worse on all 7 |
| Information added | **none** | mask-filled text |
| Accuracy effect | **+0.0006** | **−0.0255** |

The largest measured simplification in the entire study came from copying rows,
and changed downstream accuracy by six ten-thousandths.

---

## 4. Verdict on each hypothesis

### H1 — augmentation reduces complexity: **rejected**

0 of 7 measures beaten by either treatment against its own control. Replicated
on `phrasebank` (unanimous labels) and `phrasebank_50` (bare-majority labels);
the `duplicate` signature reproduces on SMS Spam and TREC as well, across 2, 3
and 6 classes.

### H2 — augmentation improves the classifier: **rejected**

`uniform` −0.0014 against its control, indistinguishable from zero. `minority`
−0.0255, significantly *worse*.

### H3 — complexity explains accuracy: **not testable as designed, and the design was the finding**

H3 could not be evaluated, because **no arm delivered a genuine complexity
reduction to test with.** Every arm whose numbers fell got there by artifact.
The experiment answered a different and more useful question: *can these
measures tell a real simplification from a sample-level artifact?* They cannot.

### Where the hypothesis *is* supported

The same data contains a clean natural experiment in the opposite direction.
The PhraseBank ships at four annotator-agreement thresholds — the same corpus,
domain and skew, differing only in label noise. Two were run:

| | `phrasebank` (unanimous) | `phrasebank_50` (bare majority) |
| --- | --- | --- |
| kDN | 0.2711 | **0.3694** (+36%) |
| C1 | 0.2525 | 0.3512 |
| borderline | 0.2938 | 0.3945 |
| safe | 0.6052 | 0.4515 |
| **baseline macro-F1** | **0.9616 ± 0.0036** | **0.8544 ± 0.0054** |

Complexity up 36%; accuracy down **10.7 points**, about 20σ. Every measure moved
in the predicted direction and the classifier followed.

**So complexity does predict accuracy — when the complexity difference is
genuine.** The measures are not broken. They are not *robust*.

---

## 5. What "genuine" means

This is the distinction the whole study turns on, so it is worth stating
precisely.

A complexity measure is meant to estimate how hard the classification problem
is — a property of the joint distribution `P(x, y)`. But it is computed from a
*sample*. That gap is where the trouble lives.

> **A complexity change is *genuine* if it reflects a change in how separable
> the classes actually are.**
> **It is an *artifact* if it reflects a change in the sample's density or
> composition while separability is untouched.**

| | Genuine | Artifact |
| --- | --- | --- |
| What changed | `P(y\|x)` — how distinguishable the classes are | the sample's size, density, or class prior |
| Bayes error | moves | unchanged |
| Best achievable accuracy | moves | unchanged |
| Example here | label noise: unanimous → bare-majority labels | duplication; rebalancing; adding near-copies |
| Accuracy effect | **−10.7 points** | **+0.0006** |

**Why duplication is an artifact.** Copying every row leaves the empirical
distribution identical — same means, same variances, same class-conditional
densities. Nothing about the problem changed. But kDN asks "how many of my five
nearest neighbours disagree with me?", and after duplication every point has a
twin at distance zero that always agrees. The statistic collapses while the
problem is untouched. N2 → 0.0000 is this in its purest form: nearest same-class
distance is now zero for every sample.

**Why rebalancing is an artifact.** Oversampling the minority class changes
`P(y)`, not `P(x|y)`. The classes are exactly as distinguishable as before. But
kDN improves because minority points now have more same-class neighbours to
find, and F1v shifts because it weights its scatter matrix by class size. The
test set keeps the original prior, so relative to the actual task nothing was
simplified — only the training set's composition changed.

**Why label noise is genuine.** Moving from unanimous to bare-majority labels
means samples whose true class is genuinely ambiguous now carry confident
labels. `P(y|x)` really is noisier, the Bayes error really is higher, and no
model can recover the lost information. The measures register it and accuracy
falls accordingly.

### The operational test

The definition above is not directly observable, but it has a practical proxy —
the one this study is built on:

> **Construct a null control that reproduces the treatment's sample-level side
> effect while adding no information. If the measure moves as much for the null
> as for the treatment, the movement is an artifact.**

`duplicate` is that null for sample size. `minority_duplicate` is that null for
class composition. Neither is a competing method; both exist to be subtracted.

---

## 6. Consequences

**For anyone using these measures to evaluate resampling or augmentation.** A
before/after complexity comparison in which `n` or the class balance changes is
uninterpretable without a matched null control. This applies to SMOTE, random
over/under-sampling, and every text-augmentation method — not only to the one
tested here.

**A correction worth recording.** Mid-study it appeared that the feature-based
measures (F1, F1v, F2, F3) were the artifact-proof subset, since `duplicate`
left them bit-identical. `minority_duplicate` disproved it: F1v moved
0.0719 → 0.0633 on rebalancing alone. They are robust to **sample size**, not to
**class composition**.

**What the study does not claim.** It does not refute the source paper. That
paper tested whether augmentation improves a *router* on *code prompts* using
GraphCodeBERT, and reported +2–3 points. This tests a proposed *explanation* for
such gains, on financial sentiment with `bert-base-uncased`. The explanation
finds no support and the benefit does not transfer to this domain. Both can hold
at once: augmentation may help through regularisation or decision-boundary
effects that these measures cannot see.

---

## 7. Threats to validity

Two properties of the measurement space could weaken the conclusions, and
neither is fully controlled.

### 7.1 Complexity is measured in a space the classifier does not use

The encoder is frozen precisely to avoid tautology — embedding with the
fine-tuned classifier would make any complexity drop circular. The cost is that
fine-tuning *learns its own representation*. Complexity is read in pretrained
BERT space; classification happens in a space the model reshapes during
training.

This is the most credible competing explanation for the §3.2 null that is not
"the measures are artifact-dominated": fine-tuning may simply undo or bypass
whatever structure the augmentation created in frozen space.

Two things argue it is not the whole story. First, the label-noise contrast
(§4) shows frozen-space complexity tracking fine-tuned accuracy across a 10.7
point gap — so the frozen space does carry real signal about achievable
accuracy. Second, the artifact result does not depend on the choice of space at
all: duplication collapses N2 to zero in *any* metric space, because the copies
are at distance zero by construction.

There is no clean fix. Measuring in the fine-tuned space is circular; measuring
in the frozen space is a proxy. The honest position is that these measures
describe the data as a *fixed representation* sees it, which is a weaker claim
than describing the learning problem.

### 7.2 768 dimensions is high for distance-based measures

Distances concentrate as dimensionality grows: beyond roughly 10–15 dimensions
the nearest and farthest neighbours of a query begin to converge, which erodes
the very contrasts kDN, N2 and T1 depend on
[[Beyer et al. 1999]](https://consensus.app/papers/details/e31ec13b3aa95105823b0b752325c6d6/?utm_source=claude_desktop),
[[Aggarwal et al. 2001]](https://consensus.app/papers/details/22400863605a593198b19b5720b75902/?utm_source=claude_desktop).
High dimensionality also induces *hubness*, where a few points appear in
disproportionately many k-nearest-neighbour lists, skewing exactly the
k-occurrence statistics kDN aggregates
[[Radovanović et al. 2010]](https://consensus.app/papers/details/cbd6ac0d08ba50dfb454cc505a7314f4/?utm_source=claude_desktop).

Three considerations bound the concern:

1. **Intrinsic, not ambient, dimensionality governs the effect.** Real data
   typically occupies a manifold of far lower dimension than its coordinates
   suggest, and that intrinsic figure is what predicts behaviour
   [[Korn et al. 2001]](https://consensus.app/papers/details/7e703aacc14f52a79877b2ffc191897b/?utm_source=claude_desktop).
   Distances provably fail to concentrate whenever the number of *relevant*
   dimensions grows with the ambient count
   [[Durrant & Kabán 2009]](https://consensus.app/papers/details/23145179359251c88cddf285a94af12b/?utm_source=claude_desktop).
2. **Text embeddings are empirically resilient.** Nearest-neighbour search over
   high-dimensional text embeddings degrades markedly less than over random
   vectors of the same dimension, and remains meaningful in practice
   [[Chen et al. 2024]](https://consensus.app/papers/details/7c0114488eba5f459707852fc50d22c7/?utm_source=claude_desktop).
3. **The artifact finding is invariant to any linear projection.** A duplicated
   row maps to a duplicated row under every linear map, so N2 → 0.0000 and the
   duplicate signature survive PCA at any rank, by construction.

So dimensionality could in principle affect the *magnitudes* in §3.1 — T1 and
the sphere measures most, since they depend on absolute radii — but it cannot
overturn §3.3.

### 7.3 If this is to be checked: PCA, not t-SNE

**PCA fitted once on `baseline` and applied unchanged to every arm** is the
defensible option. It is linear, deterministic, and has an out-of-sample
extension, so all arms land in one coordinate system. Fitting separately per
arm would be an error: the arms would no longer be comparable, which is the
whole point of the design.

**t-SNE and UMAP are the wrong tool here**, for three independent and
individually disqualifying reasons:

- **They do not preserve distances.** t-SNE discards large-scale structure by
  construction
  [[Zhou et al. 2018]](https://consensus.app/papers/details/1476bbfab20e5eda8691b899c7dbe5ae/?utm_source=claude_desktop)
  and distorts inter-cluster distances
  [[Wu et al. 2018]](https://consensus.app/papers/details/64fc663421cd5c258b0bc7be5f4e3d81/?utm_source=claude_desktop);
  the local/global trade-off is intrinsic to the method family
  [[Wang et al. 2020]](https://consensus.app/papers/details/191c7af069bb5ab4a81437bc861e5967/?utm_source=claude_desktop).
  Even work defending these embeddings concedes they "do not preserve
  high-dimensional distances"
  [[Lause et al. 2024]](https://consensus.app/papers/details/67ccf180fb17563299602e387fad1166/?utm_source=claude_desktop).
  N1, N2 and T1 are defined *on distances*; computing them on a t-SNE layout
  measures the embedding, not the data.
- **No out-of-sample extension, and non-deterministic.** t-SNE is
  non-parametric, so two initialisations give two different embeddings
  [[Candel et al. 2021]](https://consensus.app/papers/details/ad8ee592f0325b4bace82ec6be79048b/?utm_source=claude_desktop).
  Each arm would receive its own incomparable layout — fatal for a design whose
  every conclusion is a between-arm comparison.
- **Results hinge on initialisation.** Whether global structure survives is
  governed by the initialisation rather than the algorithm
  [[Kobak & Linderman 2021]](https://consensus.app/papers/details/81bfb91099435f69af85637fc5f4413c/?utm_source=claude_desktop),
  making any complexity figure a function of a visualisation hyperparameter.

t-SNE and UMAP are visualisation tools. They are appropriate for *looking* at
what the augmentation did to the embedding space, and inappropriate for
computing any number reported here.

### 7.4 The PCA check, run

`pca_sensitivity.py` recomputes the whole panel across PCA ranks, fitting the
projection **once on `baseline`** and applying it unchanged to every arm. It
uses the cached embeddings, so no retraining is involved.

This is a sensitivity sweep, not a search for a best rank. There is no
objective to optimise: tuning the rank until the conclusion changes would be
choosing the dimensionality that yields a preferred answer.

| rank | variance | `uniform` beats `duplicate` | `minority` beats `minority_duplicate` |
| --- | --- | --- | --- |
| 2 | 15.4% | **0 / 7** | **0 / 7** |
| 5 | 27.1% | **0 / 7** | **0 / 7** |
| 10 | 38.8% | **0 / 7** | **0 / 7** |
| 25 | 55.6% | **0 / 7** | **0 / 7** |
| 50 | 67.8% | **0 / 7** | **0 / 7** |
| 100 | 79.8% | **0 / 7** | **0 / 7** |
| 200 | 90.1% | **0 / 7** | **0 / 7** |
| 400 | 97.0% | 1 / 7 | 2 / 7 |
| 768 | 100% | 2 / 7 | 3 / 7 |

**From 2 to 200 dimensions — 15% to 90% of the variance — the result is
identical.** No rank in that range rescues either treatment. §7.2 is therefore
answered rather than merely bounded: dimensionality did not produce the null.

### 7.5 A trap: PCA and HEOM interact catastrophically at high rank

The rank-400 and rank-768 rows above are **not** evidence that the finding
weakens in high dimensions. They are a measurement pathology, and the
discrepancy that exposes it is worth recording.

Full-rank PCA is a rotation plus a translation, which preserves Euclidean
distances exactly, so rank 768 should reproduce the raw 768-dimensional numbers.
It does not — baseline kDN is 0.7272 under full-rank PCA against 0.3694 raw.

The cause is that **HEOM divides each feature by its range**
(`distance.py`), so it is not rotation-invariant. After that division every
column carries equal weight whatever its variance:

```text
raw BERT features,  range ratio max/min:          5.7x
PCA components,     range ratio max/min:  3,019,430x
```

At full rank HEOM therefore amplifies the lowest-variance principal component —
numerical noise — to roughly three million times its natural weight, level with
PC1. The metric becomes noise-dominated. The distortion is negligible where the
sweep is informative (1× at rank 2, 3× at rank 50, 8× at rank 200) and explodes
only past rank ~200.

**Practical consequence for anyone using `pycol-optimized`: do not run PCA at or
near full rank with the HEOM kernel.** If a high-rank projection is wanted,
either use the Euclidean kernel, which has no per-feature normalisation, or
whiten so that the components carry comparable ranges by construction.

## 8. Scope and limits

One domain (financial sentiment), one mask probability (0.15 of the paper's
0.10/0.15/0.20), one classifier, 3 classes, 4.77:1 skew.

- **Best supported:** the artifact finding (§3.4, §5). Invariant to linear
  projection by construction, and confirmed unchanged across PCA ranks 2-200
  in §7.4. Deterministic, no seed
  variance, reproduced on four corpora spanning 2/3/6 classes.
- **Single-corpus, needs replication:** the H1 and H2 rejections. `davidson`
  (13.4:1), `goemotions` (337:1), and the 0.10/0.20 mask probabilities are
  wired and unrun.
- **Not yet run:** the direct test of H3. The four agreement thresholds form a
  genuine complexity axis with domain and skew fixed. Running `baseline` alone
  at each — four datasets, five seeds, no mask-fill pass — and correlating kDN
  against macro-F1 across the four points would establish whether these measures
  predict accuracy when the underlying difference is real. Two of the four
  points already exist and lie on a steep line.

## Reproducing

```bash
cd experiments
pip install -r requirements.txt && pip install -e ..

python -m complexity_augmentation.run \
  --dataset phrasebank_50 --mask-prob 0.15 --seeds 0 1 2 3 4
```

Raw output: `results/phrasebank50_p015/results.json` (full run) and
`results/phrasebank_p015/results.json` (the unanimous-label corpus).

The PhraseBank is licensed **CC BY-NC-SA 3.0, non-commercial**; the authors ask
to be contacted for commercial use.
