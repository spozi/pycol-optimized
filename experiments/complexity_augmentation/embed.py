"""Frozen sentence embeddings, so complexity is measured on fixed coordinates.

The complexity of a training set has to be read in a space that does not itself
depend on the arm being measured.  Embedding with the *fine-tuned* classifier
would report training success rather than data difficulty: that encoder was
optimized to pull the classes apart, so complexity would fall by construction.

So the encoder here is pretrained and never updated, and the same weights embed
every arm.  Mean pooling over the last hidden state is used rather than the
``[CLS]`` vector, which carries little without fine-tuning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from .common import resolve_device

DEFAULT_EMBED_MODEL = "bert-base-uncased"


class FrozenEncoder:
    """Mean-pooled last hidden state from a pretrained, frozen transformer."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        *,
        device: str = "auto",
        max_length: int = 128,
        batch_size: int = 64,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed ``texts`` into a float32 ``[samples, hidden]`` matrix."""

        chunks: list[NDArray[np.float32]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            # Mean over real tokens only; padding must not drag the vector.
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            chunks.append(pooled.to(torch.float32).cpu().numpy())

        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def encode_cached(
    encoder: FrozenEncoder, texts: list[str], cache: Path | None
) -> NDArray[np.float32]:
    """Embed, reusing a ``.npy`` cache when the sample count still matches."""

    if cache is not None and cache.exists():
        stored = np.load(cache)
        if stored.shape[0] == len(texts):
            return stored.astype(np.float32, copy=False)

    vectors = encoder.encode(texts)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
    return vectors
