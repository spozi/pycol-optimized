# Problem statement, hypotheses, and objectives

Framing for the work in `tabular/` and `complexity_augmentation/`. Results are
in [FINDINGS.md](FINDINGS.md) (text) and `results/tabular/` (in progress); this
document states what is being asked and why.

---

## 1. Problem statement

Data-complexity measures estimate how hard a classification problem is from
geometric and neighbourhood properties of a labelled sample — class overlap,
boundary shape, neighbourhood purity. They are well established as
**meta-features for algorithm recommendation**: given a new dataset, predict
which classifier will do well, or how well any classifier can be expected to
do. That use is validated across OpenML repositories, 1,060 binary datasets,
and microarray collections, where N1 and N3 in particular correlate strongly
with achievable accuracy.

**Every one of those validations compares different datasets.** The question
answered is *which problem is harder than which*.

The measures are also used for a second purpose, which has not been validated
in the same way. In the imbalanced-learning literature they are used to judge
whether a **resampling method** has made a training set easier — complexity
before against complexity after, on the *same* dataset, offered as evidence
that a method works. Examples are prominent: one IEEE TKDE study with 119
citations tracks complexity changes across 24 datasets after applying SMOTE
variants; another monitors 22 complexity measures across 20 datasets to assess
how resampling affects them. Neither uses a null control.

**The two uses are not equivalent, and the difference is consequential.**
Between datasets, sample size and class balance are properties of the problems
being compared. Within a dataset, a resampler *deliberately changes both* —
that is what a resampler is. And because these measures are computed on a
finite sample rather than on the underlying distribution, they respond to
sample size and class prior whether or not class separability moved at all:

- neighbourhood measures (kDN, N1, N2, N3) depend on distances to nearest
  neighbours, which shrink when copies or interpolated points are added;
- geometric measures (T1, the sphere-cover family) depend on point density in
  a fixed volume;
- feature-based measures (F1v, F3) weight class statistics by class size, so
  they move whenever the class prior changes.

If this is right, complexity reductions reported after resampling may reflect
density and composition rather than difficulty, and conclusions drawn from them
are unsupported. The general risk is acknowledged in the literature — Al Hosni
et al. note that "little attention has been paid to validating the meta-feature
decisions in reflecting the actual data properties" — but the specific confound
has not, as far as we have found, been isolated and quantified with a null
control.

**The gap.** There is no established procedure for distinguishing a genuine
reduction in class separability from a sample-level artifact, and therefore no
basis on which a within-dataset complexity comparison can be interpreted.

---

## 2. Formal hypotheses

### 2.1 Notation

| Symbol | Meaning |
| --- | --- |
| $D$ | the dataset; stratified $k$-fold gives training index $T_i$, test index $E_i$ |
| $M$ | a resampling method, $M \in \{\text{ROS}, \text{RUS}, \text{SMOTENC}, \text{ENN}\}$ |
| $M(T_i)$ | the resampled training set for fold $i$ |
| $C_M(T_i)$ | the **matched null control** |
| $\phi$ | a complexity measure, oriented so **lower = simpler** |
| $A$ | macro-F1 of a classifier trained on the arm, scored on $E_i$ |
| $N_\rho(y)$ | labels with fraction $\rho$ flipped uniformly at random |

$C_M(T_i)$ draws rows at random **from $T_i$ only**, reproducing $M(T_i)$'s
per-class counts exactly. By construction it matches on sample size and class
balance and synthesises nothing. $E_i$ is never resampled.

### 2.2 The artifact hypothesis (complexity)

For each method $M$ and measure $\phi$:

$$H_0^{\phi,M}:\quad \mathbb{E}_i\big[\phi(M(T_i))\big] = \mathbb{E}_i\big[\phi(C_M(T_i))\big]$$

$$H_1^{\phi,M}:\quad \mathbb{E}_i\big[\phi(M(T_i))\big] < \mathbb{E}_i\big[\phi(C_M(T_i))\big]$$

One-sided, because the method's own claim is that it reduces complexity.

**This deliberately replaces the hypothesis the literature tests:**

$$H_0^{\text{naive}}:\quad \mathbb{E}_i\big[\phi(M(T_i))\big] = \mathbb{E}_i\big[\phi(T_i)\big]$$

Rejecting $H_0^{\text{naive}}$ is uninformative, because $C_M$ — which adds no
information whatsoever — rejects it too.

### 2.3 The downstream hypothesis (accuracy)

$$H_0^{A,M}:\ \mathbb{E}_i\big[A(f_{M(T_i)}, E_i)\big] = \mathbb{E}_i\big[A(f_{C_M(T_i)}, E_i)\big]
\qquad H_1^{A,M}:\ >$$

### 2.4 The validity hypothesis (genuine difficulty)

With $\rho \in \{0, 0.02, 0.05, 0.10, 0.20\}$ and $\tau$ Kendall's rank
correlation across those points:

$$H_0^{\text{valid}}:\ \tau\big(\rho,\ \mathbb{E}_i[\phi(T_i, N_\rho)]\big) = 0
\qquad H_1^{\text{valid}}:\ \tau > 0$$

with a manipulation check that the axis is genuine: $\tau(\rho, A) < 0$.

Label flipping raises the Bayes error by a known amount while leaving the
feature distribution, sample size, and very nearly the class balance untouched.
That is what makes it *genuine* where resampling is not.

### 2.5 The composite claim

The contribution requires **both** outcomes:

| Hypothesis | Required | Meaning |
| --- | --- | --- |
| $H_0^{\phi,M}$ | **fail to reject** | methods do not beat their own controls |
| $H_0^{\text{valid}}$ | **reject** | measures do track genuine difficulty |

Together: **valid but not robust.** The first alone would read as "these
measures are broken," which is both weaker and probably false.

### 2.6 How these are tested, and why cautiously

Three limitations are stated rather than papered over.

**Folds are not independent.** Any two training folds share $(k-2)/k$ of their
rows, so paired tests across CV folds understate variance and are
anti-conservative. Results are therefore reported as **paired differences with
effect sizes**, and any p-value is indicative rather than decisive.

**$k = 5$ has very little power.** A one-sided Wilcoxon signed-rank test at
$n = 5$ bottoms out at $p = 1/32 = 0.031$; it can just clear 0.05 and no more.

**Multiplicity.** Four methods against roughly eleven measures is 44 tests.
**kDN and N2 are pre-specified as primary** — the two most reported in this
literature — and every other measure is descriptive.

---

## 3. Research objectives

| | Objective | Serves |
| --- | --- | --- |
| **O1** | Formalise genuine complexity reduction (separability changes; Bayes error moves) against artifactual (density or class prior changes; separability untouched). | §2.2 |
| **O2** | Design matched null controls reproducing a method's per-class counts by random draw, adding no information. | §2.2 |
| **O3** | Quantify how far each measure moves under those controls, and identify which measures resist which perturbation. | §2.2 |
| **O4** | Establish a genuine-difficulty axis with features, size, and balance fixed, and test whether complexity tracks it. | §2.4 |
| **O5** | Evaluate whether resampling methods beat their controls on complexity and on held-out accuracy. | §2.2, §2.3 |
| **O6** | Test robustness to representation and dimensionality. | — |
| **O7** | Formulate a reporting protocol for within-dataset complexity comparisons. | §2.5 |

---

## 4. Status

| Objective | Status | Evidence |
| --- | --- | --- |
| O1 | Done | [FINDINGS.md §5](FINDINGS.md) |
| O2 | Done | `matched_control` in `tabular/resample.py`; verified matched for all four methods |
| O3 | Done, two perturbations | Duplication: N2 → 0.0000, T1 exactly halved, kDN −26.5%. Feature-based measures robust to size but **not** to composition. |
| O4 | **Running** | Label-noise sweep, 0–20%, in `tabular/run.py` |
| O5 | **Running** | 5-fold CV on Bank Marketing, 9 conditions per fold |
| O6 | Partial | PCA ranks 2–200 leave conclusions unchanged; see the HEOM interaction in FINDINGS §7.5 |
| O7 | Not started | — |

### Supporting evidence already in hand

**Text (complete, 5 arms × 5 seeds).** Mask-and-fill augmentation beat its
controls on **0 of 7** measures and produced no accuracy effect (p = 0.676).
Verbatim duplication — provably zero information — produced the largest
complexity reduction in the study (N2 → 0.0000, T1 halved, kDN −26.5%) and
changed macro-F1 by **+0.0006** (p = 0.854).

**Dose-response.** Bank Marketing carries 9.06% duplicate rows as distributed;
at that rate N2 moves 3% and T1 6%, against total collapse at 100%. The
artifact scales with the perturbation.

**Naturally contradicted labels.** 237 duplicate groups in Bank Marketing carry
conflicting labels — 542 rows share a feature vector with a differently
labelled row. Directly measurable irreducible error in a standard benchmark.

---

## 5. Scope

**In scope.** Single-label classification on mixed-type tabular data;
complexity via HEOM as implemented in `pycol-optimized`; interventions that
change sample size or class prior; the neighbourhood, geometric, and
feature-based measures.

**Out of scope, and why.**

- *Designing new complexity measures.* The claim concerns how existing measures
  are used. A replacement would need the same null-control protocol to be
  validated.
- *Text-specific complexity measures.* Complexity of text is undefined without
  a representation, so the question reduces to encoder choice. The text
  experiments remain as supporting evidence that the effect is not
  domain-specific.
- *Information-theoretic difficulty measures (PVI).* Considered and parked; the
  design is preserved in history at commit `9d3db4c`. It is a second paper, and
  pursuing it now would leave both unfinished.
- *Multi-label and regression settings.*

---

## 6. Contribution claimed

A methodological correction, not a new measure:

> Data-complexity measures are validated for comparison **between** datasets and
> are routinely applied **within** a dataset, before and after resampling. The
> second use is unsound without a null control, because these measures respond
> to sample density and class composition independently of class separability.
> We quantify the confound, give the control that removes it, and show which
> published claims it changes.

The strongest single piece of evidence remains that **verbatim duplication —
adding provably no information — produced the largest complexity reduction
observed and changed downstream accuracy by +0.0006.**
