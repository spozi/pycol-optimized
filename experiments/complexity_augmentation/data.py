"""Dataset registry: text classification corpora, fetched on demand.

Every corpus here downloads from its origin with nothing but the standard
library -- no dataset hub, no loading scripts, no authentication.  Files land in
the cache directory and are reused.

Each loader returns ``(texts, labels, label_names)`` with labels already encoded
as contiguous integers.  ``load_split`` turns that into a :class:`Split` whose
test half is fixed once and never augmented.

Sizes and skews below were measured, not quoted:

===============  ========  =======  ==========  =====================
Dataset          Samples   Classes  Skew        Note
===============  ========  =======  ==========  =====================
``sms_spam``        5,574        2      6.5:1    saturates BERT easily
``phrasebank``      2,264        3      4.6:1    NON-COMMERCIAL licence
``trec``            5,952        6     14.5:1    clean, fast
``tweeteval``      11,970        2      1.4:1    hard but barely skewed
``banking77``      13,083       77      5.3:1    many fine-grained classes
``davidson``       24,783        3     13.4:1    slurs; noisy tweets
``goemotions``     36,308       28    328.8:1    most extreme skew here
``agnews``        127,600        4      1.0:1    balanced; large
===============  ========  =======  ==========  =====================

``agnews`` is balanced by construction, so it is only an imbalance study once
``--imbalance-ratio`` is applied.  It is the size at which the tiled engine
matters: augmented, it is a quarter of a million rows.
"""

from __future__ import annotations

import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# AG News packs a whole article into one field.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

Loaded = tuple[list[str], NDArray[np.int64], tuple[str, ...]]

#: sha256 of the SMS archive as downloaded on 2026-07-31.  The other corpora are
#: served from mutable branches, so pinning them here would be theatre.
SMS_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"


@dataclass(frozen=True, slots=True)
class Split:
    """One train/test partition.  ``test`` never changes between arms."""

    name: str
    train_texts: list[str]
    train_labels: NDArray[np.int64]
    test_texts: list[str]
    test_labels: NDArray[np.int64]
    label_names: tuple[str, ...]

    @property
    def minority_label(self) -> int:
        """The rarest class in the training half."""

        counts = np.bincount(self.train_labels, minlength=len(self.label_names))
        # Ignore classes the training split does not contain at all.
        counts = np.where(counts == 0, counts.max() + 1, counts)
        return int(counts.argmin())

    def describe(self) -> dict[str, object]:
        counts = np.bincount(self.train_labels, minlength=len(self.label_names))
        present = counts[counts > 0]
        return {
            "dataset": self.name,
            "n_train": len(self.train_texts),
            "n_test": len(self.test_texts),
            "n_classes": len(self.label_names),
            "minority_label": self.minority_label,
            "minority_name": self.label_names[self.minority_label],
            "minority_count": int(counts[self.minority_label]),
            "imbalance_ratio": float(present.max() / present.min()),
            "train_class_counts": counts.tolist(),
        }


def _fetch(url: str, cache_dir: Path, filename: str, *, sha256: str | None = None) -> bytes:
    """Download once, then serve from ``cache_dir``."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if target.exists():
        return target.read_bytes()

    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
        payload = response.read()
    if sha256 is not None:
        digest = hashlib.sha256(payload).hexdigest()
        if digest != sha256:
            raise RuntimeError(f"{filename} checksum changed: expected {sha256}, got {digest}")
    target.write_bytes(payload)
    return payload


def _encode(raw: list[str]) -> tuple[NDArray[np.int64], tuple[str, ...]]:
    """Map string labels to contiguous integers, ordered by name."""

    names = tuple(sorted(set(raw)))
    lookup = {name: index for index, name in enumerate(names)}
    return np.asarray([lookup[value] for value in raw], dtype=np.int64), names


def _load_sms_spam(cache_dir: Path) -> Loaded:
    payload = _fetch(
        "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        cache_dir,
        "sms_spam.zip",
        sha256=SMS_SHA256,
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        body = archive.read("SMSSpamCollection").decode("latin-1")

    texts, labels = [], []
    for line in body.splitlines():
        label, _, text = line.partition("\t")
        if text.strip():
            texts.append(text.strip())
            labels.append(label.strip())
    encoded, names = _encode(labels)
    return texts, encoded, names


def _load_davidson(cache_dir: Path) -> Loaded:
    payload = _fetch(
        "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language"
        "/master/data/labeled_data.csv",
        cache_dir,
        "davidson.csv",
    )
    names = {"0": "hate", "1": "offensive", "2": "neither"}
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    texts = [row["tweet"].strip() for row in rows]
    encoded, label_names = _encode([names[row["class"]] for row in rows])
    return texts, encoded, label_names


def _load_trec(cache_dir: Path) -> Loaded:
    base = "https://cogcomp.seas.upenn.edu/Data/QA/QC/"
    texts, labels = [], []
    for filename, cached in (
        ("train_5500.label", "trec_train.label"),
        ("TREC_10.label", "trec_test.label"),
    ):
        body = _fetch(base + filename, cache_dir, cached).decode("latin-1")
        for line in body.splitlines():
            if not line.strip():
                continue
            head, _, question = line.partition(" ")
            texts.append(question.strip())
            # Coarse label only: the fine-grained one has 50 classes.
            labels.append(head.split(":")[0])
    encoded, names = _encode(labels)
    return texts, encoded, names


def _load_phrasebank(cache_dir: Path) -> Loaded:
    # Licensed CC BY-NC-SA 3.0: research only.  The authors ask to be contacted
    # for commercial use -- see License.txt inside the archive.
    payload = _fetch(
        "https://huggingface.co/datasets/takala/financial_phrasebank"
        "/resolve/main/data/FinancialPhraseBank-v1.0.zip",
        cache_dir,
        "phrasebank.zip",
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(n for n in archive.namelist() if n.endswith("Sentences_AllAgree.txt"))
        body = archive.read(member).decode("latin-1")

    texts, labels = [], []
    for line in body.splitlines():
        sentence, separator, label = line.rpartition("@")
        if separator and sentence.strip():
            texts.append(sentence.strip())
            labels.append(label.strip())
    encoded, names = _encode(labels)
    return texts, encoded, names


def _load_tweeteval(cache_dir: Path) -> Loaded:
    base = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/hate/"
    texts, labels = [], []
    for split in ("train", "test"):
        body = _fetch(base + f"{split}_text.txt", cache_dir, f"tweeteval_{split}_text.txt")
        tags = _fetch(base + f"{split}_labels.txt", cache_dir, f"tweeteval_{split}_labels.txt")
        texts.extend(body.decode("utf-8").splitlines())
        labels.extend("hate" if t == "1" else "not_hate" for t in tags.decode().split())
    encoded, names = _encode(labels)
    return texts, encoded, names


def _load_banking77(cache_dir: Path) -> Loaded:
    base = (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
    )
    texts, labels = [], []
    for split in ("train", "test"):
        body = _fetch(base + f"{split}.csv", cache_dir, f"banking77_{split}.csv")
        for row in csv.DictReader(io.StringIO(body.decode("utf-8"))):
            texts.append(row["text"].strip())
            labels.append(row["category"].strip())
    encoded, names = _encode(labels)
    return texts, encoded, names


def _load_goemotions(cache_dir: Path) -> Loaded:
    base = (
        "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
    )
    emotions = _fetch(base + "emotions.txt", cache_dir, "goemotions_labels.txt").decode().split()

    texts, labels = [], []
    for split in ("train", "dev", "test"):
        body = _fetch(base + f"{split}.tsv", cache_dir, f"goemotions_{split}.tsv").decode("utf-8")
        for line in body.splitlines():
            parts = line.split("\t")
            # About 16% of GoEmotions carries several labels at once.  The
            # complexity measures assume one label per sample, so multi-label
            # rows are dropped rather than collapsed to an arbitrary choice.
            if len(parts) >= 2 and parts[0].strip() and "," not in parts[1]:
                texts.append(parts[0].strip())
                labels.append(emotions[int(parts[1])])
    encoded, names = _encode(labels)
    return texts, encoded, names


def _load_agnews(cache_dir: Path) -> Loaded:
    base = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
    names = {"1": "world", "2": "sports", "3": "business", "4": "sci_tech"}
    texts, labels = [], []
    for split in ("train", "test"):
        body = _fetch(base + f"{split}.csv", cache_dir, f"agnews_{split}.csv").decode("utf-8")
        for row in csv.reader(io.StringIO(body)):
            if len(row) >= 3:
                texts.append(f"{row[1].strip()} {row[2].strip()}")
                labels.append(names[row[0]])
    encoded, label_names = _encode(labels)
    return texts, encoded, label_names


#: Loader per dataset name, as accepted by ``--dataset``.
DATASETS: dict[str, Callable[[Path], Loaded]] = {
    "sms_spam": _load_sms_spam,
    "phrasebank": _load_phrasebank,
    "trec": _load_trec,
    "tweeteval": _load_tweeteval,
    "banking77": _load_banking77,
    "davidson": _load_davidson,
    "goemotions": _load_goemotions,
    "agnews": _load_agnews,
}

#: Corpora that may not be used commercially without a separate licence.
NON_COMMERCIAL = frozenset({"phrasebank"})


def stratified_split(
    texts: list[str],
    labels: NDArray[np.int64],
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Index split preserving each class's share on both sides."""

    generator = np.random.default_rng(seed)
    train_index: list[int] = []
    test_index: list[int] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        generator.shuffle(members)
        cut = int(round(len(members) * test_fraction))
        # Never let a class vanish from either side.
        cut = min(max(cut, 1), len(members) - 1) if len(members) > 1 else 0
        test_index.extend(members[:cut].tolist())
        train_index.extend(members[cut:].tolist())

    generator.shuffle(train_index)
    generator.shuffle(test_index)
    return train_index, test_index


def _impose_imbalance(
    labels: NDArray[np.int64], index: list[int], ratio: float, seed: int
) -> list[int]:
    """Downsample every class but the largest until majority:minority is ``ratio``."""

    generator = np.random.default_rng(seed + 1)
    selected = np.asarray(index)
    counts = np.bincount(labels[selected])
    keep = max(1, int(round(counts.max() / ratio)))

    chosen: list[int] = []
    for label in np.unique(labels[selected]):
        members = selected[labels[selected] == label]
        if len(members) > keep and label != counts.argmax():
            members = generator.choice(members, size=keep, replace=False)
        chosen.extend(members.tolist())
    generator.shuffle(chosen)
    return chosen


def _cap(labels: NDArray[np.int64], index: list[int], limit: int, seed: int) -> list[int]:
    """Shrink to ``limit`` samples, keeping each class's share."""

    if len(index) <= limit:
        return index
    generator = np.random.default_rng(seed + 2)
    selected = np.asarray(index)
    share = limit / len(selected)

    chosen: list[int] = []
    for label in np.unique(labels[selected]):
        members = selected[labels[selected] == label]
        take = max(1, int(round(len(members) * share)))
        chosen.extend(
            generator.choice(members, size=min(take, len(members)), replace=False).tolist()
        )
    generator.shuffle(chosen)
    return chosen


def load_split(
    dataset: str,
    cache_dir: Path,
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
    max_train: int | None = None,
    imbalance_ratio: float | None = None,
) -> Split:
    """Fetch ``dataset`` and cut it into a fixed train/test split.

    ``imbalance_ratio`` and ``max_train`` reshape the *training* half only.  The
    test half keeps the corpus's natural distribution, because a test set
    squeezed to the same skew would leave too few minority samples to measure
    recall on -- the metric the whole experiment turns on.
    """

    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")

    texts, labels, label_names = DATASETS[dataset](cache_dir)
    train_index, test_index = stratified_split(
        texts, labels, test_fraction=test_fraction, seed=seed
    )

    if imbalance_ratio is not None:
        if imbalance_ratio < 1.0:
            raise ValueError("imbalance_ratio must be at least 1.0")
        train_index = _impose_imbalance(labels, train_index, imbalance_ratio, seed)
    if max_train is not None:
        train_index = _cap(labels, train_index, max_train, seed)

    return Split(
        name=dataset,
        train_texts=[texts[i] for i in train_index],
        train_labels=labels[train_index],
        test_texts=[texts[i] for i in test_index],
        test_labels=labels[test_index],
        label_names=label_names,
    )
