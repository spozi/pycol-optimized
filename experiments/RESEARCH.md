# Problem statement, research questions, and objectives

Framing for the work in `complexity_augmentation`. Results to date are in
[FINDINGS.md](FINDINGS.md); this document states what is being asked and why.

---

## 1. Problem statement

Data-complexity measures estimate how hard a classification problem is from
geometric and neighbourhood properties of a labelled sample — class overlap,
boundary shape, neighbourhood purity. The canonical catalogue is Lorena et al.'s
survey, and the measures are well established as **meta-features for algorithm
recommendation**: given a new dataset, predict which classifier will do well, or
how well any classifier can be expected to do.

That use is validated. Studies across OpenML repositories, 1,060 binary
datasets, and microarray collections consistently find complexity measures
correlate with achievable accuracy, and N1 and N3 in particular show strong
negative correlation with classifier performance.

**Every one of those validations compares different datasets.** The question
answered is *which problem is harder than which*.

The measures are also used for a second purpose, which has not been validated
in the same way: judging whether a **data-level intervention** — resampling,
instance selection, or augmentation — has made a training set easier. This
compares complexity before and after a transformation of the *same* dataset,
and is common in the imbalanced-learning and augmentation literatures, where a
reported complexity reduction is offered as evidence that a method works.

**The two uses are not equivalent, and the difference is consequential.**
Between datasets, sample size and class balance are properties of the problem
being compared. Within a dataset, an intervention *deliberately changes them*.
Because these measures are computed on a finite sample rather than on the
underlying distribution, they respond to sample size and class prior whether or
not class separability moved at all:

- neighbourhood measures (kDN, N1, N2, N3) depend on distances to nearest
  neighbours, which shrink when copies or near-copies are added;
- geometric measures (T1, and the sphere-cover family) depend on point density
  in a fixed volume;
- feature-based measures (F1v, F3) weight class statistics by class size, so
  they move whenever the class prior changes.

If this is right, then complexity reductions reported after resampling or
augmentation may reflect density and composition rather than difficulty, and
conclusions drawn from them are unsupported. The literature acknowledges the
general risk — Al Hosni et al. note that "little attention has been paid to
validating the meta-feature decisions in reflecting the actual data
properties" — but the specific confound has not, as far as we have found, been
isolated and quantified with a null control.

**The gap.** There is no established procedure for distinguishing a genuine
reduction in class separability from a sample-level artifact, and therefore no
basis on which a within-dataset complexity comparison can be interpreted.

---

## 2. Research questions

**RQ1 — Discrimination.** Can data-complexity measures distinguish a genuine
change in class separability from a change in sample density or class
composition that leaves separability untouched?

> *Testable form:* do the measures move as much under an intervention that adds
> no information as under one that does?

**RQ2 — Validity floor.** Do the measures track achievable accuracy when
difficulty genuinely differs and representation, domain, and class skew are
held fixed?

> *Testable form:* along a controlled label-noise axis, does complexity
> correlate with downstream classifier performance?
>
> RQ2 is the necessary complement to RQ1. Without it, a negative RQ1 result
> cannot be told apart from the measures being uninformative in general.

**RQ3 — Remedy.** What experimental control renders a within-dataset complexity
comparison interpretable, and does applying it alter the conclusions such
comparisons support?

> *Testable form:* when each intervention is paired with a null control
> reproducing its sample-level side effect, do previously apparent complexity
> reductions survive?

**RQ4 — Robustness.** Are the answers to RQ1–RQ3 artifacts of the
representation in which complexity is measured, or of its dimensionality?

> *Testable form:* do the conclusions hold across embedding models and across
> projection ranks?

**RQ5 — Downstream consequence.** Does a complexity change, genuine or
artifactual, predict a change in classifier performance?

> *Testable form:* across intervention arms, does the ranking by complexity
> match the ranking by held-out accuracy?

---

## 3. Research objectives

| | Objective | Serves |
| --- | --- | --- |
| **O1** | Formalise the distinction between a genuine complexity reduction (class separability changes; Bayes error moves) and an artifactual one (sample density or class prior changes; separability untouched). | RQ1 |
| **O2** | Design paired **null controls** that reproduce an intervention's sample-level side effects — sample count, class balance, source samples, random seed — while adding no information. | RQ1, RQ3 |
| **O3** | Quantify how far each complexity measure moves under those null controls, and identify which measures are robust to which perturbation. | RQ1 |
| **O4** | Establish a genuine-difficulty axis with domain, skew, and representation held fixed, and measure whether complexity tracks achievable accuracy along it. | RQ2 |
| **O5** | Evaluate whether data-level interventions beat their own null controls, on complexity and on held-out classifier performance, with adequate statistical power. | RQ3, RQ5 |
| **O6** | Test robustness of the conclusions to embedding model and projection rank. | RQ4 |
| **O7** | Formulate a reporting protocol for within-dataset complexity comparisons, and identify which published claims it would change. | RQ3 |

---

## 4. Status

| Objective | Status | Evidence |
| --- | --- | --- |
| O1 | Done | [FINDINGS.md §5](FINDINGS.md) |
| O2 | Done | `duplicate` (sample size), `minority_duplicate` (class composition) |
| O3 | Done for two perturbations | §3.1: N2 → 0.0000, T1 exactly halved, kDN −26.5% under verbatim duplication. Feature-based measures robust to size but **not** to composition. |
| O4 | **Two points of four** | Label-noise axis: kDN 0.2711 → 0.3694 with macro-F1 0.9616 → 0.8544. The remaining two agreement thresholds are unrun. |
| O5 | Done for one corpus | 0/7 on complexity for both interventions; accuracy p = 0.676 (`uniform`), p = 0.0055 harmful (`minority`). |
| O6 | Partial | PCA ranks 2–200 unchanged (§7.4). Encoder sweep not run. |
| O7 | Not started | — |

**The critical gap is O4.** RQ1 and RQ3 are answered; RQ2 is not. Without the
validity floor, the artifact result is open to the reading that these measures
are simply uninformative, which the two existing points argue against but do
not establish. It is also the cheapest outstanding experiment: `baseline` only,
four corpora of 1,811–3,876 samples, no mask-fill pass.

---

## 5. Scope

**In scope.** Single-label classification; complexity measured on a fixed,
frozen representation; interventions that change sample size or class prior;
neighbourhood, geometric, and feature-based measures as implemented in
`pycol-optimized`.

**Out of scope, and why.**

- *Designing new complexity measures.* The claim is about how existing measures
  are used, not that they need replacing. A repaired measure would still need
  the null-control protocol to be validated.
- *Text-specific complexity measures.* Complexity of text is undefined without
  a representation, and representation choice is already known to dominate text
  classification outcomes. That question reduces to encoder selection and is
  better handled as robustness (O6) than as a contribution.
- *Multi-label and regression settings.*
- *Whether the augmentation under test is useful in its original domain.* This
  work tests a proposed *explanation* for augmentation gains, not the gains
  themselves. The source paper reports its result on code prompts with a
  different fill model and a different task; nothing here bears on that.

---

## 6. Contribution claimed

A methodological correction, not a new measure:

> Data-complexity measures are validated for comparison **between** datasets and
> are routinely applied **within** a dataset, before and after an intervention.
> The second use is unsound without a null control, because these measures
> respond to sample density and class composition independently of class
> separability. We quantify the confound, give the control that removes it, and
> show that a widely-cited augmentation method survives neither.

The strongest single piece of evidence is that **verbatim duplication — adding
provably no information — produced the largest complexity reduction observed in
this study (N2 → 0.0000, T1 exactly halved, kDN −26.5%) and changed downstream
macro-F1 by +0.0006 (p = 0.854).**
