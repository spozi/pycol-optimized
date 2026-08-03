"""Does complexity-reducing data augmentation help imbalanced text classification?

Measures data complexity before and after the mask-and-fill augmentation of
Pozi and Sato (2025), then fine-tunes BERT on each arm to see whether a
complexity drop actually buys downstream accuracy.
"""

from __future__ import annotations

__all__ = ["augment", "common", "complexity", "data", "embed", "run", "train"]
