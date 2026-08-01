"""Mask-and-fill augmentation, following Algorithm 1 of the routing paper.

Each text is tokenized, every non-special token is replaced by ``[MASK]`` with
probability ``p_mask``, and each mask is then filled with the masked language
model's most likely token.  The augmented copies are appended to the original
set, so an augmentation of size ``1x`` doubles the data.

Algorithm 1 fills masks one at a time, each conditioned on the fills already
made.  That is reproduced here as ``sequential`` (the default): one forward pass
per mask *round*, batched across texts, so a batch costs as many passes as its
deepest text has masks rather than one pass per mask per text.  ``joint`` fills
every mask in a single pass, which is faster but scores each mask against the
other masks rather than against their replacements.

Because the fill is an argmax, the model frequently predicts the original token
straight back, so a nontrivial share of "augmented" texts are token-identical to
their source.  That rate is measured rather than assumed -- see
:class:`AugmentationReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .common import resolve_device

DEFAULT_FILL_MODEL = "bert-base-uncased"
DEFAULT_MASK_PROBABILITY = 0.10


@dataclass
class AugmentationReport:
    """What the augmentation actually did, as opposed to what it attempted."""

    n_texts: int = 0
    n_tokens: int = 0
    n_masked: int = 0
    n_restored: int = 0
    n_texts_unchanged: int = 0
    examples: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        masked = max(1, self.n_masked)
        texts = max(1, self.n_texts)
        return {
            "n_texts": self.n_texts,
            "mask_rate_observed": self.n_masked / max(1, self.n_tokens),
            # Share of masked positions the model filled with the original token
            # again.  High values mean the augmentation is close to a no-op.
            "restored_rate": self.n_restored / masked,
            "changed_rate": 1.0 - (self.n_restored / masked),
            "texts_unchanged_rate": self.n_texts_unchanged / texts,
        }


class MaskFillAugmenter:
    """Applies Algorithm 1 with a HuggingFace masked language model."""

    def __init__(
        self,
        model_name: str = DEFAULT_FILL_MODEL,
        *,
        mask_probability: float = DEFAULT_MASK_PROBABILITY,
        device: str = "auto",
        max_length: int = 128,
        batch_size: int = 64,
        fill_strategy: str = "sequential",
        avoid_original: bool = False,
    ) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if not 0.0 < mask_probability < 1.0:
            raise ValueError("mask_probability must lie in (0, 1)")
        if fill_strategy not in {"sequential", "joint"}:
            raise ValueError("fill_strategy must be 'sequential' or 'joint'")

        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # eval() puts dropout in inference behaviour; freezing the parameters
        # means a future edit that drops the inference_mode decorator cannot
        # quietly start building autograd graphs over a model that is only ever
        # used forward.
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mask_probability = mask_probability
        self.max_length = max_length
        self.batch_size = batch_size
        self.fill_strategy = fill_strategy
        # Not in the paper: forbids the model from predicting the token it just
        # saw removed, which turns a near no-op into a real edit.  Off by
        # default so the faithful behaviour is what runs.
        self.avoid_original = avoid_original
        self.mask_id = int(self.tokenizer.mask_token_id)

    def _maskable(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Positions eligible for masking: real tokens, not [CLS]/[SEP]/[PAD]."""

        special = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.tokenizer.all_special_ids:
            special |= input_ids == token_id
        return ~special

    @torch.inference_mode()
    def _fill(self, input_ids: torch.Tensor, attention: torch.Tensor, original: torch.Tensor):
        """Replace every [MASK] with a predicted token."""

        if self.fill_strategy == "joint":
            logits = self.model(input_ids=input_ids, attention_mask=attention).logits
            if self.avoid_original:
                logits.scatter_(2, original.unsqueeze(2), float("-inf"))
            predicted = logits.argmax(dim=2)
            holes = input_ids == self.mask_id
            return torch.where(holes, predicted, input_ids)

        # Sequential: fill the earliest remaining mask in every row, then loop.
        working = input_ids.clone()
        for _ in range(int((working == self.mask_id).sum(dim=1).max().item())):
            holes = working == self.mask_id
            if not bool(holes.any()):
                break
            logits = self.model(input_ids=working, attention_mask=attention).logits
            if self.avoid_original:
                logits.scatter_(2, original.unsqueeze(2), float("-inf"))
            predicted = logits.argmax(dim=2)

            # One position per row: the first still-masked column.
            first = torch.argmax(holes.to(torch.int64), dim=1)
            rows = torch.arange(working.shape[0], device=working.device)
            active = holes.any(dim=1)
            working[rows[active], first[active]] = predicted[rows[active], first[active]]
        return working

    def augment(self, texts: list[str], *, seed: int = 0) -> tuple[list[str], AugmentationReport]:
        """Return one augmented copy of every text, plus what changed."""

        generator = torch.Generator(device="cpu").manual_seed(seed)
        report = AugmentationReport()
        produced: list[str] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            original = encoded["input_ids"]
            attention = encoded["attention_mask"]

            eligible = self._maskable(original) & attention.bool()
            draws = torch.rand(original.shape, generator=generator)
            holes = eligible & (draws < self.mask_probability)

            masked = original.masked_fill(holes, self.mask_id)
            filled = self._fill(
                masked.to(self.device), attention.to(self.device), original.to(self.device)
            ).cpu()

            restored = (filled == original) & holes
            report.n_texts += len(batch)
            report.n_tokens += int(eligible.sum())
            report.n_masked += int(holes.sum())
            report.n_restored += int(restored.sum())
            report.n_texts_unchanged += int(((filled == original) | ~attention.bool()).all(1).sum())

            for row in range(len(batch)):
                keep = attention[row].bool()
                text = self.tokenizer.decode(filled[row][keep], skip_special_tokens=True)
                produced.append(text)
                if len(report.examples) < 10 and bool(holes[row].any()):
                    report.examples.append((batch[row], text))

        return produced, report


def build_arm(
    texts: list[str],
    labels: NDArray[np.int64],
    *,
    strategy: str,
    augmenter: MaskFillAugmenter | None = None,
    minority_label: int = 1,
    target_ratio: float = 1.0,
    seed: int = 0,
) -> tuple[list[str], NDArray[np.int64], dict[str, Any]]:
    """Build one experimental arm's training set.

    ===================  ====================================================
    ``baseline``         the data untouched
    ``duplicate``        every text copied verbatim.  Doubles the sample count
                         while adding no information, which is the control
                         that separates genuine simplification from a
                         sample-size artifact
    ``uniform``          one mask-fill copy of every text, as the paper says
    ``minority_duplicate``  verbatim copies of the minority class up to
                         ``target_ratio`` -- classic random oversampling
    ``minority``         mask-fill copies of the minority class, same target
    ===================  ====================================================

    The two minority arms draw *the same* source samples from the same seed, so
    the only difference between them is whether those copies were mask-filled.
    That is what makes ``minority_duplicate`` a control rather than merely
    another condition: ``duplicate`` holds sample size fixed for ``uniform``,
    and this holds class composition fixed for ``minority``.  Rebalancing moves
    neighbourhood measures on its own -- kDN counts neighbours that disagree, so
    handing the minority class more same-class neighbours lowers it whether or
    not the classes became easier to tell apart.
    """

    if strategy == "baseline":
        return list(texts), labels.copy(), {"strategy": strategy}

    if strategy == "duplicate":
        return list(texts) * 2, np.concatenate([labels, labels]), {"strategy": strategy}

    if strategy in {"minority", "minority_duplicate"}:
        minority = np.flatnonzero(labels == minority_label)
        majority = int((labels != minority_label).sum())
        wanted = max(0, int(round(majority * target_ratio)) - len(minority))
        if wanted == 0:
            return list(texts), labels.copy(), {"strategy": strategy, "n_generated": 0}

        # Drawn before the branch, from a seed both arms share, so the two
        # differ only in the mask-fill.  With replacement, so a repeated draw
        # still diverges once masking is applied.
        picks = np.random.default_rng(seed).choice(minority, size=wanted, replace=True)
        source = [texts[i] for i in picks]

        if strategy == "minority_duplicate":
            extra: list[str] = list(source)
            details: dict[str, Any] = {"strategy": strategy, "n_generated": wanted}
        else:
            if augmenter is None:
                raise ValueError(f"strategy {strategy!r} needs an augmenter")
            extra, report = augmenter.augment(source, seed=seed)
            details = {
                "strategy": strategy,
                "n_generated": wanted,
                **report.summary(),
                "examples": report.examples,
            }

        return (
            list(texts) + extra,
            np.concatenate([labels, np.full(wanted, minority_label, dtype=np.int64)]),
            details,
        )

    if augmenter is None:
        raise ValueError(f"strategy {strategy!r} needs an augmenter")

    if strategy == "uniform":
        extra, report = augmenter.augment(list(texts), seed=seed)
        return (
            list(texts) + extra,
            np.concatenate([labels, labels]),
            {"strategy": strategy, **report.summary(), "examples": report.examples},
        )

    raise ValueError(f"unknown strategy {strategy!r}")
