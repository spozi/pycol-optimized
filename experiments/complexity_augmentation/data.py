"""The SMS Spam Collection, split once and held fixed across every arm.

Downloaded straight from the UCI archive rather than through a dataset hub, so
the experiment has no hub dependency and the bytes are pinned by checksum.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SOURCE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ARCHIVE_MEMBER = "SMSSpamCollection"

#: sha256 of the archive as downloaded on 2026-07-31.  A mismatch means the
#: upstream file moved, which must fail loudly rather than silently changing
#: the dataset under a published result.
SOURCE_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"

LABEL_NAMES = ("ham", "spam")
MINORITY_LABEL = 1  # spam


@dataclass(frozen=True, slots=True)
class Split:
    """One train/test partition.  ``test`` never changes between arms."""

    train_texts: list[str]
    train_labels: NDArray[np.int64]
    test_texts: list[str]
    test_labels: NDArray[np.int64]

    def describe(self) -> dict[str, object]:
        return {
            "n_train": len(self.train_texts),
            "n_test": len(self.test_texts),
            "train_positive": int(self.train_labels.sum()),
            "test_positive": int(self.test_labels.sum()),
            "train_positive_rate": float(self.train_labels.mean()),
            "imbalance_ratio": float(
                (self.train_labels == 0).sum() / max(1, (self.train_labels == 1).sum())
            ),
        }


def download(cache_dir: Path, *, verify: bool = False) -> Path:
    """Fetch the collection into ``cache_dir``, reusing it once present."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / ARCHIVE_MEMBER
    if target.exists():
        return target

    with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:  # noqa: S310
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if verify and digest != SOURCE_SHA256:
        raise RuntimeError(
            f"SMS Spam archive checksum changed: expected {SOURCE_SHA256}, got {digest}"
        )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        target.write_bytes(archive.read(ARCHIVE_MEMBER))
    return target


def load_raw(path: Path) -> tuple[list[str], NDArray[np.int64]]:
    """Read the tab-separated ``label<TAB>text`` file."""

    texts: list[str] = []
    labels: list[int] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        label, _, text = line.partition("\t")
        if not text:
            continue
        texts.append(text.strip())
        labels.append(1 if label.strip() == "spam" else 0)
    return texts, np.asarray(labels, dtype=np.int64)


def stratified_split(
    texts: list[str],
    labels: NDArray[np.int64],
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Split:
    """Split preserving the class ratio, so the test set stays imbalanced too."""

    generator = np.random.default_rng(seed)
    train_index: list[int] = []
    test_index: list[int] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        generator.shuffle(members)
        cut = int(round(len(members) * test_fraction))
        test_index.extend(members[:cut].tolist())
        train_index.extend(members[cut:].tolist())

    generator.shuffle(train_index)
    generator.shuffle(test_index)
    return Split(
        train_texts=[texts[i] for i in train_index],
        train_labels=labels[train_index],
        test_texts=[texts[i] for i in test_index],
        test_labels=labels[test_index],
    )


def load_split(cache_dir: Path, *, test_fraction: float = 0.2, seed: int = 0) -> Split:
    texts, labels = load_raw(download(cache_dir))
    return stratified_split(texts, labels, test_fraction=test_fraction, seed=seed)
