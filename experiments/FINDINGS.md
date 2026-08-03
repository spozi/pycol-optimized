# Data-complexity measures respond to resampling, not just to difficulty

Two studies run in this directory, on tabular and text data. Every figure comes
from the code here and is reproducible from `results/`.

---

## The short version

Data-complexity measures are supposed to tell you how hard a classification
problem is. They are widely used to check whether a preprocessing method —
SMOTE, oversampling, undersampling — has made a training set easier: measure
complexity before, measure it after, and if the number went down, the method
worked.

**That check does not work, and this is why.** Every resampling method changes
two things about a dataset: how many rows it has, and how the classes are
balanced. These measures respond to both of those *by themselves*, with no
change in how separable the classes actually are. So a complexity drop after
resampling might mean the method helped — or it might mean nothing at all.

To tell those apart you need a **null control**: a second dataset with exactly
the same number of rows and exactly the same class balance, built by copying
existing rows at random rather than by any clever method. If the clever method
does no better than random copying, its complexity drop was not about the
method.

We ran that comparison. **No method beat its own control.** Random oversampling
appears to reduce N2 by 0.151 — a large, publishable-looking effect — and beats
its control by **0.0000**. SMOTE's mixed-type variant does *worse* than random
copying.

But the measures are not broken. When we made the data genuinely harder, by
flipping a known fraction of labels, they tracked it perfectly (Kendall
τ = +1.00). So the conclusion is precise:

> **These measures detect genuine difficulty reliably, and cannot tell it apart
> from a resampling artifact.** Valid, but not robust.

One measure, T1, did something worse than that: it moved in the *wrong*
direction as the data got genuinely harder. §6 explains exactly why.

---

## 1. Why this needs checking at all

### 1.1 What complexity measures do

Given a labelled dataset, these measures score how tangled the classes are.
Some common ones, all oriented so **higher means harder**:

| Measure | What it asks |
| --- | --- |
| **kDN** | Of my 5 nearest neighbours, how many have a different label? |
| **N2** | How far is my nearest same-class point, relative to my nearest different-class point? |
| **C1** | An entropy-style summary of neighbourhood label mixing |
| **T1** | How many hyperspheres does it take to cover the data? |

They are well validated for comparing **different datasets** — telling you that
problem A is harder than problem B. That use is supported by studies across
OpenML, 1,060 binary datasets, and microarray collections.

### 1.2 The use that has not been validated

They are also used to compare **one dataset against itself**, before and after a
preprocessing step. In the imbalanced-learning literature this is a standard
way to argue that a resampling method works. One IEEE TKDE paper with 119
citations tracks complexity changes across 24 datasets after applying SMOTE
variants; another monitors 22 measures across 20 datasets for the same purpose.

Neither uses a control, and that is the problem.

### 1.3 Why the two uses are not the same

When you compare two different datasets, their sizes and class balances are
just facts about them.

When you resample one dataset, **you change its size and class balance on
purpose** — that is the entire operation. And these measures are computed from
a finite sample, not from the underlying distribution, so they move when the
sample changes even if the problem does not:

- **Neighbourhood measures** (kDN, N2, N1, N3) look at distances to nearest
  neighbours. Add copies of existing rows and everyone's neighbours get closer.
- **Geometric measures** (T1 and the sphere family) depend on how densely
  packed the points are in space.
- **Feature-based measures** (F1v, F3) weight class statistics by class size, so
  they shift the moment you rebalance.

An analogy: measure a crowd's average height, add a hundred copies of the
tallest person, and conclude the crowd got taller. The number moved. The crowd
did not change.

---

## 2. The design: every method gets a matched control

We used **UCI Bank Marketing** — 41,188 rows, 19 columns (10 categorical,
9 numeric), predicting term-deposit subscription. 11.3% positive, a 7.88:1
imbalance arising from the domain rather than from subsampling. Mixed column
types are exactly what HEOM, the distance function these measures use, was
designed for.

### 2.1 The arms

Four resampling methods, each paired with a control:

| Arm | What it does |
| --- | --- |
| `baseline` | nothing; the untouched data |
| `ros` | random oversampling — duplicate minority rows until balanced |
| `rus` | random undersampling — drop majority rows until balanced |
| `smotenc` | SMOTE for mixed types — *synthesise* new minority rows by interpolation |
| `enn` | Edited Nearest Neighbours — remove rows their own neighbours disagree with |

### 2.2 What a control is, exactly

For each method, its control draws rows **at random from the original training
data** until it reaches *the same count in every class* as the method produced.
It synthesises nothing and selects nothing cleverly.

So the control has:

- the same number of rows,
- the same class balance,
- no new information whatsoever.

**Anything the control does to a complexity measure is what resampling to those
counts does on its own.** A method has earned a complexity reduction only where
it beats its control — never where it merely beats `baseline`.

One detail matters. The control is built *from* the method's output rather than
fixed in advance. Our first attempt used random undersampling as the control
for ENN, and it did not match: ENN removes however many rows its rule rejects
(7,501 majority rows, leaving 6.26:1), while random undersampling targets
balance and lands somewhere else entirely. **A control that does not match is
not a control.** All four pairs are now verified to match exactly.

### 2.3 Protocol

Stratified 5-fold cross-validation, which is what this literature uses and
which Bank Marketing needs, having no official split.

**Resampling happens inside each fold, on the training portion only.** This
matters more than it sounds. Resample *before* splitting and an oversampled
duplicate of a row can land in training while its twin lands in test — the
model has effectively seen the answer, and every score inflates. The test fold
is never resampled and never touched by a sampler.

Complexity is measured per fold too, on the same rows the classifier trains on.
The classifier is histogram gradient boosting, which takes categorical columns
natively rather than through an encoding that would invent an ordering.

---

## 3. Result 1: no method beats its control

### 3.1 The two columns that matter

Primary measures were **pre-specified as kDN and N2** before results were seen —
the two most reported in this literature. Lower is simpler. Averaged over
5 folds:

| Method | Measure | vs `baseline` | vs **its own control** |
| --- | --- | --- | --- |
| `ros` | kDN | **−0.0433** | −0.0022 |
| `ros` | N2 | **−0.1511** | **+0.0000** |
| `rus` | kDN | **+0.2072** | +0.0019 |
| `rus` | N2 | +0.0329 | +0.0004 |
| `smotenc` | kDN | **−0.0218** | **+0.0193** |
| `smotenc` | N2 | **−0.1031** | **+0.0480** |
| `enn` | kDN | −0.0196 | −0.0456 |
| `enn` | N2 | −0.0053 | −0.0117 |

**Read the two right-hand columns against each other.** That contrast is the
entire finding.

- **`ros` against baseline** reduces N2 by 0.151 — a big number, exactly the
  sort reported as evidence a method works. Against a control that just copies
  rows at random, its advantage is **+0.0000**. Every bit of it was the
  resampling.
- **`rus` against baseline** raises kDN by 0.207, appearing to make the data
  more than twice as hard. Against its control: **+0.0019**. Again, essentially
  all of it is the class balance changing, not the data.
- **`smotenc`** is the interesting one. Against baseline it reduces kDN by
  0.022 and N2 by 0.103, and looks effective. Against its control both flip
  sign: **+0.019 and +0.048**. Interpolating new points makes the data
  measurably *harder* than simply copying existing ones — consistently, with a
  fold-to-fold standard deviation of 0.0011 on a difference of 0.0193.

### 3.2 Full picture

| arm | n | kDN | N2 | C1 | T1 | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 32,950 | 0.1546 | 0.4497 | 0.1536 | 0.9501 | 0.6564 ± 0.0074 |
| ros | 58,476 | 0.1113 | 0.2986 | 0.0835 | 0.5354 | 0.6906 ± 0.0041 |
| ros_control | 58,476 | 0.1135 | 0.2985 | 0.0811 | 0.5355 | 0.6910 ± 0.0058 |
| rus | 7,424 | 0.3617 | 0.4825 | 0.3601 | 0.9853 | 0.6694 ± 0.0053 |
| rus_control | 7,424 | 0.3598 | 0.4821 | 0.3569 | 0.9848 | 0.6691 ± 0.0049 |
| smotenc | 58,476 | 0.1328 | 0.3466 | 0.1185 | 0.7101 | 0.7013 ± 0.0040 |
| smotenc_control | 58,476 | 0.1135 | 0.2985 | 0.0811 | 0.5355 | 0.6910 ± 0.0058 |
| enn | 26,902 | 0.1349 | 0.4444 | 0.1282 | 0.9544 | 0.7145 ± 0.0064 |
| enn_control | 26,902 | 0.1805 | 0.4561 | 0.1791 | 0.9576 | 0.6769 ± 0.0098 |

Notice `ros` and `ros_control` agree to three decimal places on everything.
They should: random oversampling *is* its own control, so this is the machinery
verifying itself.

### 3.3 Downstream accuracy

| Method | vs `baseline` | vs **its control** | effect size |
| --- | --- | --- | --- |
| `ros` | +0.0342 | −0.0004 | −0.1 |
| `rus` | +0.0130 | +0.0004 | +0.1 |
| `smotenc` | +0.0449 | **+0.0103** | 2.3 |
| `enn` | +0.0580 | **+0.0376** | 6.0 |

The same pattern. `ros` gains 0.034 macro-F1 over baseline and **nothing** over
random copying — its benefit is having more minority rows, however you get them.

SMOTENC and ENN *do* beat their controls on accuracy. SMOTENC's case is the
sharpest dissociation in the study: it makes the data **measurably harder** by
every complexity measure and **classifies better anyway**. Whatever it does for
the classifier, the complexity panel cannot see it.

### 3.4 What happens underneath

Minority-class precision and recall show the actual mechanism:

| arm | minority F1 | recall | precision |
| --- | --- | --- | --- |
| baseline | 0.3668 | 0.2554 | 0.6512 |
| ros | 0.4729 | 0.6218 | 0.3816 |
| smotenc | 0.4688 | 0.4610 | 0.4772 |
| enn | 0.4986 | 0.5399 | 0.4632 |

Every method trades precision for recall. Baseline catches only 25.5% of
positives but is right 65% of the time it fires; `ros` catches 62% at 38%
precision. This trade is what resampling is *for*, and **no complexity measure
in the panel can see it**, because it is a property of where the decision
threshold ends up, not of how the points sit in space.

---

## 4. Result 2: the measures work when the difficulty is real

If methods never beat controls, one explanation is that these measures simply
do not work. So we tested that directly.

We flipped a known fraction of training labels at random. This raises the Bayes
error — the error no classifier can avoid — by a controlled amount, while
leaving the features, the sample size, and very nearly the class balance
untouched. **That is what makes it genuine where resampling is not.**

| noise | kDN | N2 | C1 | T1 | macro-F1 |
| --- | --- | --- | --- | --- | --- |
| 0% | 0.1546 | 0.4497 | 0.1536 | 0.9501 | 0.6564 |
| 2% | 0.1818 | 0.4564 | 0.1810 | 0.9488 | 0.6575 |
| 5% | 0.2201 | 0.4643 | 0.2192 | 0.9475 | 0.6535 |
| 10% | 0.2792 | 0.4742 | 0.2788 | 0.9440 | 0.6451 |
| 20% | 0.3764 | 0.4867 | 0.3759 | 0.9375 | 0.6424 |

| measure | Kendall τ vs noise |
| --- | --- |
| kDN | **+1.00** |
| N2 | **+1.00** |
| C1 | **+1.00** |
| T1 | **−1.00** |
| macro-F1 | −0.80 (manipulation check) |

kDN, N2, and C1 rise **perfectly monotonically** with injected noise. Accuracy
falls, confirming the axis really did make the problem harder.

**So the measures are not broken.** They detect genuine difficulty flawlessly.
They just cannot distinguish it from having resampled the data.

---

## 5. The combined claim

| | Needed | Observed |
| --- | --- | --- |
| Methods do **not** beat their controls | ✔ | 0 of 4 on both primary measures |
| Measures **do** track genuine difficulty | ✔ | τ = +1.00 for kDN, N2, C1 |

> **Valid, but not robust.**

Had only the first held, the honest reading would be "these measures don't
work" — weaker, and as §4 shows, wrong.

---

## 6. T1 moves the wrong way, and here is why

T1 counts how many hyperspheres it takes to cover the data, divided by the
sample count. More spheres means less structure, so **higher should mean
harder**.

It fell steadily as we injected label noise: 0.9501 → 0.9375, τ = **−1.00**.
Perfectly monotone in the wrong direction. That is not noise, so we traced it.

Each point's sphere grows until it reaches the nearest point of another class.
Flip a label and some point suddenly has an opposite-class point sitting right
next to it, so its sphere shrinks to almost nothing. Measured on a 12,000-row
subsample:

| noise | median radius | spheres under 1% of max | absorbed |
| --- | --- | --- | --- |
| 0% | 1.4142 | 0.4% | 243 |
| 20% | 1.0249 | **1.5%** | **289** |

A sphere is absorbed by another when `distance ≤ r_outer − r_inner`. When
`r_inner` is nearly zero that condition is trivially satisfied — **a tiny sphere
gets swallowed by anything near it**. So label noise creates many tiny spheres,
tiny spheres get absorbed, fewer survive, and T1 falls.

The mechanism is real and the implementation is correct. But the consequence is
that **T1 reads label noise as simplification**, the opposite of what a
complexity measure should do. On this evidence T1 should not be used where label
noise is plausible — which is most real datasets.

---

## 7. Supporting evidence: the same effect on text

Before the tabular work we ran the same design on text, with mask-and-fill
augmentation (Pozi & Sato 2025) on the Financial PhraseBank, 5 arms × 5 seeds.

- Augmentation beat its controls on **0 of 7** measures.
- Downstream accuracy: **−0.0014** against control, p = 0.676.
- Minority-class augmentation *hurt* — −0.0255, p = 0.0055 — and the control
  isolated why: rebalancing was fine, but the mask-fill corrupted minority
  labels, buying +4 points of recall for −13 of precision.
- **Verbatim duplication produced the single largest complexity reduction in
  the whole project** — N2 to exactly 0.0000, T1 exactly halved, kDN −26.5% —
  while adding provably no information, and changed accuracy by **+0.0006**
  (p = 0.854).

Two corpora, two domains, two representations, same conclusion.

### 7.1 A dose-response

Bank Marketing carries **3,731 exact duplicate rows (9.06%)** as distributed.
Removing them — against a control that removes the same number at random —
moves N2 by +0.015 and T1 by +0.056, while kDN barely moves at all.

| duplication | N2 | T1 |
| --- | --- | --- |
| 9% (natural) | +3% | +6% |
| 100% (constructed) | → exactly 0.0000 | halved |

The artifact scales with the perturbation. That is stronger evidence than
either point alone.

### 7.2 Contradicted labels in a standard benchmark

237 of those duplicate groups carry **conflicting labels** — 542 rows share a
feature vector with a row labelled differently. No classifier can be right
about both. That is directly measurable irreducible error sitting in a widely
used benchmark, computable in seconds, and as far as we have found nobody
reports it. Deduplication silently picks a winner.

---

## 8. What "genuine" means

The distinction the whole project turns on:

> A complexity change is **genuine** if it reflects a change in how separable
> the classes actually are.
> It is an **artifact** if it reflects a change in the sample's density or class
> prior while separability is untouched.

| | Genuine | Artifact |
| --- | --- | --- |
| What changed | how distinguishable the classes are | the sample's size or class prior |
| Bayes error | moves | unchanged |
| Best achievable accuracy | moves | unchanged |
| Example | label noise | duplication; rebalancing; interpolation |
| Effect on accuracy | **−0.014 macro-F1 at 20% noise** | **+0.0000 for `ros` vs control** |

**Why duplication is an artifact.** Copying rows leaves the distribution
identical — same means, same variances, same class-conditional densities. But
kDN asks "do my 5 nearest neighbours agree?", and a duplicate is a neighbour at
distance zero that always agrees. The statistic moves; the problem does not.

**Why rebalancing is an artifact.** Oversampling changes `P(y)`, not `P(x|y)`.
The classes are exactly as distinguishable as before. kDN improves because
minority points now have more same-class neighbours available, and the test set
keeps the original prior, so relative to the actual task nothing was simplified.

**Why label noise is genuine.** Samples whose true class is ambiguous now carry
confident labels. `P(y|x)` really is noisier, the Bayes error really is higher,
and no model can recover the lost information.

### The operational test

> **Build a null control that reproduces the intervention's sample-level side
> effect while adding no information. If the measure moves as much for the null
> as for the intervention, the movement is an artifact.**

---

## 9. Consequences

**If you compare complexity before and after resampling, you need a control.**
This applies to SMOTE and its variants, random over- and undersampling,
instance selection, and text augmentation. Without one, the comparison cannot
distinguish a better dataset from a bigger or more balanced one.

**The control is cheap.** Reproduce the method's per-class counts by drawing
rows at random from the original data. It costs one extra complexity
computation and no modelling.

**Report the duplicate count.** `pycol-optimized` already emits
`duplicate_row_count` in its diagnostics. A nonzero value means the
distance-based measures are partly reading density. Bank Marketing sits at
9.06% before anyone touches it.

**Two corrections we had to make ourselves, recorded so others need not.**

1. We first believed the feature-based measures (F1, F1v, F2, F3) were the
   artifact-proof subset, since verbatim duplication left them bit-identical.
   The composition control disproved it: F1v moved 0.0719 → 0.0633 under
   rebalancing alone. They are robust to **sample size**, not to **class
   composition**.
2. We first used random undersampling as ENN's control. It did not match ENN's
   class counts, and an unmatched control is not a control.

---

## 10. Limitations

**One tabular dataset.** Bank Marketing only. The text results show the effect
is not domain-specific, but a KEEL sweep is the obvious next step, and KEEL is
the benchmark suite this literature actually uses.

**Cross-validation folds are not independent.** Any two training folds share
3/5 of their rows, so paired tests across them understate variance. We therefore
report **paired differences and effect sizes**, and treat p-values as
indicative. With 5 folds a one-sided Wilcoxon bottoms out at p = 0.031 anyway.

**Multiplicity.** Four methods against roughly eleven measures is 44 tests.
kDN and N2 were pre-specified as primary; everything else is descriptive.

**ENN's complexity win is close to circular.** ENN removes points that their own
*k* nearest neighbours misclassify, which is very nearly the definition of kDN.
"ENN reduces kDN" is largely ENN doing what it says it does. Its **accuracy**
gain of +0.0376 over control is the part that is not circular, and the part
worth reporting.

**One classifier.** Histogram gradient boosting. A distance-based classifier
such as k-NN might track these measures far more closely, since they are built
from the same neighbourhood structure — a genuinely open question, and a
plausible partial explanation for the dissociation.

---

## Reproducing

```bash
cd experiments
pip install -r requirements.txt && pip install -e ..

# tabular: 5 folds x 9 conditions, plus the label-noise sweep (~72 min)
python -m tabular.run --folds 5

# the duplicates already in the file, against a random-removal control
python -m tabular.dedup_study

# text (~5 hours; needs transformers)
python -m complexity_augmentation.run --dataset phrasebank_50 --seeds 0 1 2 3 4
```

Raw output in `results/tabular/` and `results/phrasebank50_p015/`.

Bank Marketing is CC BY 4.0 (UCI). The Financial PhraseBank is
**CC BY-NC-SA 3.0, non-commercial**.
