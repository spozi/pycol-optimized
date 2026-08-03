"""The models used for inference must stay in inference mode.

Two things go wrong if they do not, and neither raises:

* **Dropout stays active.** The frozen encoder produces the coordinates every
  complexity measure is computed on. Dropout there makes those coordinates
  noisy *and* different on every call, so two arms would be compared in two
  different spaces and the whole experiment would quietly stop meaning
  anything.
* **Autograd records.** The fill and encode passes are forward-only. A graph
  built over them is wasted memory during the phase that already holds the most
  weights.

These tests are not collected by the library's CI: the root ``pyproject.toml``
sets ``testpaths = ["tests"]``, so a bare ``pytest`` at the repository root
never reaches this directory. Run them from ``experiments/``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers", reason="install experiments/requirements.txt")

from complexity_augmentation.augment import MaskFillAugmenter  # noqa: E402
from complexity_augmentation.embed import FrozenEncoder  # noqa: E402

TEXTS = [
    "free entry to win a prize now",
    "how did serfdom develop in russia",
    "call me back later please",
]


@pytest.fixture(scope="module")
def encoder() -> FrozenEncoder:
    return FrozenEncoder(device="cpu")


@pytest.fixture(scope="module")
def augmenter() -> MaskFillAugmenter:
    return MaskFillAugmenter(device="cpu", mask_probability=0.3)


def test_the_encoder_is_frozen_and_in_eval_mode(encoder: FrozenEncoder) -> None:
    assert encoder.model.training is False
    assert not any(parameter.requires_grad for parameter in encoder.model.parameters())


def test_the_augmenter_is_frozen_and_in_eval_mode(augmenter: MaskFillAugmenter) -> None:
    # The augmenter is only ever run forward, so it is frozen for the same
    # reason the encoder is, even though the inference_mode decorator already
    # prevents a graph from forming today.
    assert augmenter.model.training is False
    assert not any(parameter.requires_grad for parameter in augmenter.model.parameters())


def test_embeddings_repeat_exactly(encoder: FrozenEncoder) -> None:
    first = encoder.encode(TEXTS)
    second = encoder.encode(TEXTS)

    # Exact equality, not approximate: any active dropout would separate these.
    assert np.array_equal(first, second)


def test_that_check_would_actually_catch_live_dropout(encoder: FrozenEncoder) -> None:
    """Proves the repeatability test above has power rather than passing vacuously.

    Put the same encoder into training mode and the two passes must diverge.
    If they did not, ``test_embeddings_repeat_exactly`` would be asserting
    nothing at all.
    """

    encoder.model.train()
    try:
        drifted = not np.array_equal(encoder.encode(TEXTS), encoder.encode(TEXTS))
    finally:
        encoder.model.eval()

    assert drifted, "dropout appears inactive even in train mode; the guard proves nothing"
    # And the fixture is left as it was found.
    assert encoder.model.training is False
    assert np.array_equal(encoder.encode(TEXTS), encoder.encode(TEXTS))


def test_embeddings_carry_no_autograd_history(encoder: FrozenEncoder) -> None:
    # encode returns numpy, so the check is that building it never needed a
    # graph: run the model directly under the same guard the encoder uses.
    with torch.inference_mode():
        encoded = encoder.tokenizer(TEXTS, return_tensors="pt", padding=True)
        hidden = encoder.model(**encoded).last_hidden_state

    assert hidden.grad_fn is None
    assert hidden.requires_grad is False


@pytest.mark.parametrize("strategy", ["sequential", "joint"])
def test_the_fill_pass_repeats_exactly(strategy: str) -> None:
    filler = MaskFillAugmenter(device="cpu", mask_probability=0.3, fill_strategy=strategy)
    first, _ = filler.augment(TEXTS, seed=11)
    second, _ = filler.augment(TEXTS, seed=11)

    assert first == second


def test_the_classifier_scores_the_same_run_twice(tmp_path) -> None:
    """The evaluation half of training must also be dropout-free.

    Fine-tuning twice from the same seed and getting the same metrics covers
    the whole path at once: the training loop is reproducible, and evaluation
    runs with dropout off.  Were ``model.eval()`` missing before the test loop,
    the two runs would disagree.

    Deliberately on CPU -- accelerator kernels carry their own nondeterminism,
    which would make this flaky for reasons that have nothing to do with
    inference mode.
    """

    from pathlib import Path

    from complexity_augmentation.data import load_split
    from complexity_augmentation.train import TrainConfig, train_and_evaluate

    split = load_split("sms_spam", Path(__file__).resolve().parent.parent / "cache")
    arguments = {
        "config": TrainConfig(epochs=1, batch_size=8),
        "seed": 5,
        "device": "cpu",
        "verbose": False,
    }
    scored = [
        train_and_evaluate(
            split.train_texts[:64],
            split.train_labels[:64],
            split.test_texts[:48],
            split.test_labels[:48],
            **arguments,
        )
        for _ in range(2)
    ]

    for metric in ("macro_f1", "minority_f1", "minority_recall", "mcc", "accuracy"):
        assert scored[0][metric] == scored[1][metric], metric


def test_a_borrowed_encoder_matches_a_freshly_loaded_one(
    augmenter: MaskFillAugmenter, encoder: FrozenEncoder
) -> None:
    # The saving is only legitimate if it changes nothing.  A masked-LM
    # checkpoint contains the encoder AutoModel would load, so these must agree
    # exactly rather than merely closely.
    borrowed = FrozenEncoder.borrowing_from(augmenter, device="cpu")

    assert borrowed.shared is True
    assert borrowed.model.training is False
    assert np.array_equal(borrowed.encode(TEXTS), encoder.encode(TEXTS))
