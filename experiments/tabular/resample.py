"""Resampling arms, each paired with a matched null control.

The imbalanced-learning literature routinely reports complexity before and
after resampling as evidence that a method works.  But every resampler changes
sample size and class balance by construction, and these measures move with
both.  Each method here is therefore paired with a control that reaches **the
same per-class counts by drawing at random from the original rows**, adding
nothing the data did not already hold.

The control is derived from its arm rather than fixed in advance, because what
a method does is not always known beforehand.  Random oversampling always
targets balance, so it would serve as a control for SMOTENC either way — but
Edited Nearest Neighbours removes however many rows its neighbourhood rule
happens to reject (7,501 majority rows here, leaving 6.26:1, not balance).  A
fixed random undersampler would not match that, and a control that does not
match is not a control.  :func:`matched_control` reproduces any arm's class
counts exactly:

- where a class was grown, it duplicates real rows (random oversampling)
- where a class was cut, it drops rows at random (random undersampling)

So a method beats its control only where it exceeds it, never where it merely
exceeds ``baseline``.

SMOTE proper interpolates between points, which is meaningless for a
categorical column: halfway between "married" and "single" is not a marital
status.  SMOTENC is the variant defined for mixed-type data, and is the
correct choice here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Methods under test. Each is scored against its own matched control, built by
#: :func:`matched_control`; ``baseline`` is the untouched reference.
METHODS = ("ros", "rus", "smotenc", "enn")
ARMS = ("baseline", *METHODS)


@dataclass(frozen=True, slots=True)
class Arm:
    """One resampled training set."""

    name: str
    vectors: NDArray[np.float32]
    labels: NDArray[np.int64]
    missing: NDArray[np.bool_] | None

    def describe(self) -> dict[str, object]:
        counts = np.bincount(self.labels)
        present = counts[counts > 0]
        return {
            "arm": self.name,
            "n": int(len(self.labels)),
            "class_counts": counts.tolist(),
            "imbalance_ratio": float(present.max() / present.min()),
        }


def _regather_missing(
    original: NDArray[np.float32],
    missing: NDArray[np.bool_] | None,
    resampled: NDArray[np.float32],
) -> NDArray[np.bool_] | None:
    """Carry the missing mask onto resampled rows.

    Rows surviving unchanged keep their own mask.  Synthesised rows have no
    counterpart in the original and are treated as fully observed, which is
    what a synthetic row is: an artefact with no provenance and no missing
    values to inherit.
    """

    if missing is None:
        return None
    lookup = {row.tobytes(): index for index, row in enumerate(original)}
    out = np.zeros(resampled.shape, dtype=bool)
    for position, row in enumerate(resampled):
        source = lookup.get(row.tobytes())
        if source is not None:
            out[position] = missing[source]
    return out if out.any() else None


def build_arm(
    name: str,
    vectors: NDArray[np.float32],
    labels: NDArray[np.int64],
    *,
    categorical: NDArray[np.bool_],
    missing: NDArray[np.bool_] | None = None,
    seed: int = 0,
) -> Arm:
    """Apply one resampling strategy, or none for ``baseline``."""

    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; choose from {ARMS}")
    if name == "baseline":
        return Arm(name, vectors.copy(), labels.copy(), missing)

    from imblearn.over_sampling import SMOTENC, RandomOverSampler
    from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler

    samplers = {
        "ros": lambda: RandomOverSampler(random_state=seed),
        "rus": lambda: RandomUnderSampler(random_state=seed),
        "smotenc": lambda: SMOTENC(
            categorical_features=np.flatnonzero(categorical).tolist(), random_state=seed
        ),
        "enn": EditedNearestNeighbours,
    }
    resampled, new_labels = samplers[name]().fit_resample(vectors, labels)
    resampled = np.ascontiguousarray(resampled, dtype=np.float32)
    return Arm(
        name,
        resampled,
        np.asarray(new_labels, dtype=np.int64),
        _regather_missing(vectors, missing, resampled),
    )


def matched_control(
    arm: Arm,
    vectors: NDArray[np.float32],
    labels: NDArray[np.int64],
    *,
    missing: NDArray[np.bool_] | None = None,
    seed: int = 0,
) -> Arm:
    """Reproduce ``arm``'s per-class counts by drawing from the original rows.

    This is the null: identical sample size, identical class balance, zero
    synthesis.  Whatever it does to a complexity measure is what resampling to
    those counts does on its own, independent of the method's cleverness.
    """

    generator = np.random.default_rng(seed + 991)
    target = np.bincount(arm.labels, minlength=int(labels.max()) + 1)

    chosen: list[int] = []
    for label, wanted in enumerate(target):
        available = np.flatnonzero(labels == label)
        if wanted == 0 or available.size == 0:
            continue
        if wanted <= available.size:
            chosen.extend(generator.choice(available, size=wanted, replace=False).tolist())
        else:
            # Keep every real row, then duplicate to make up the shortfall --
            # exactly what random oversampling does.
            chosen.extend(available.tolist())
            chosen.extend(
                generator.choice(available, size=wanted - available.size, replace=True).tolist()
            )

    index = np.asarray(chosen, dtype=np.int64)
    generator.shuffle(index)
    return Arm(
        f"{arm.name}_control",
        np.ascontiguousarray(vectors[index]),
        labels[index],
        None if missing is None else np.ascontiguousarray(missing[index]),
    )


def controls_match(arm: Arm, control: Arm) -> bool:
    """A control is only a control if its class counts match its arm's."""

    return np.array_equal(np.bincount(arm.labels), np.bincount(control.labels))


def inject_label_noise(
    labels: NDArray[np.int64], rate: float, *, n_classes: int, seed: int = 0
) -> NDArray[np.int64]:
    """Flip ``rate`` of labels uniformly at random to a different class.

    This is the genuine-difficulty axis.  It raises the Bayes error by a known
    amount while leaving the feature distribution, the sample size, and very
    nearly the class balance untouched.  Anything that moves here moves because
    the problem really did get harder — which is exactly what distinguishes it
    from every resampling perturbation above.
    """

    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must lie in [0, 1]")
    if rate == 0.0:
        return labels.copy()

    generator = np.random.default_rng(seed)
    noisy = labels.copy()
    chosen = generator.choice(len(labels), size=int(round(rate * len(labels))), replace=False)
    for index in chosen:
        alternatives = [c for c in range(n_classes) if c != labels[index]]
        noisy[index] = generator.choice(alternatives)
    return noisy
