"""Mixed-type tabular corpora, in the form HEOM was designed for.

Unlike the text side of these experiments, nothing here needs an embedding: the
columns *are* the feature space, and they carry genuine categorical values and
genuine missing values.  All three of HEOM's mechanisms are live — the
range-normalised numeric term, the categorical overlap term, and the
missing-value penalty — where on sentence embeddings only the first was.

Downloaded from the UCI archive with the standard library alone.
"""

from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

#: Columns holding categories rather than magnitudes.
BANK_CATEGORICAL = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
)

#: ``duration`` is the last call's length, which is unknown until the call has
#: happened and essentially determines the outcome.  The UCI documentation says
#: to discard it for any realistic model; it is dropped unless asked for.
LEAKY_COLUMNS = ("duration",)

#: ``pdays`` uses 999 for "never previously contacted", which is a flag wearing
#: a number's clothes.  It covers 96.3% of rows, and because HEOM normalises by
#: a column's range it would otherwise squeeze the real values (0..27) into
#: under 3% of the scale.
PDAYS_SENTINEL = 999

#: Several categorical columns spell missing data as a level.
UNKNOWN_TOKEN = "unknown"


@dataclass(frozen=True, slots=True)
class TabularDataset:
    """Feature matrix plus the masks HEOM needs to read it correctly."""

    name: str
    vectors: NDArray[np.float32]
    labels: NDArray[np.int64]
    categorical: NDArray[np.bool_]
    missing: NDArray[np.bool_] | None
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]

    def describe(self) -> dict[str, object]:
        counts = np.bincount(self.labels, minlength=len(self.label_names))
        present = counts[counts > 0]
        minority = int(np.where(counts == 0, counts.max() + 1, counts).argmin())
        return {
            "dataset": self.name,
            "n_samples": int(self.vectors.shape[0]),
            "n_features": int(self.vectors.shape[1]),
            "n_categorical": int(self.categorical.sum()),
            "n_numeric": int((~self.categorical).sum()),
            "n_classes": len(self.label_names),
            "minority_label": minority,
            "minority_name": self.label_names[minority],
            "minority_count": int(counts[minority]),
            "imbalance_ratio": float(present.max() / present.min()),
            "missing_cells": 0 if self.missing is None else int(self.missing.sum()),
        }


def _fetch(url: str, cache_dir: Path, filename: str) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if target.exists():
        return target.read_bytes()
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
        payload = response.read()
    target.write_bytes(payload)
    return payload


def load_bank_marketing(
    cache_dir: Path,
    *,
    drop_leaky: bool = True,
    pdays_sentinel_as_missing: bool = True,
    unknown_as_missing: bool = True,
) -> TabularDataset:
    """UCI Bank Marketing: 41,188 rows, 20 columns, 10 of them categorical.

    Subscription to a term deposit, positive in 11.3% of rows — a 7.9:1 skew
    that arises from the domain rather than from subsampling.

    The three defaults each exist to stop a column being read as something it
    is not; see :data:`LEAKY_COLUMNS`, :data:`PDAYS_SENTINEL`, and
    :data:`UNKNOWN_TOKEN`. Turning them off reproduces the raw file, which is
    what most published results on this dataset actually use.
    """

    outer = zipfile.ZipFile(io.BytesIO(_fetch(BANK_URL, cache_dir, "bank_marketing.zip")))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
    member = next(n for n in inner.namelist() if n.endswith("bank-additional-full.csv"))
    rows = list(csv.reader(io.StringIO(inner.read(member).decode()), delimiter=";"))

    header, body = rows[0], [r for r in rows[1:] if r]
    keep = [
        i
        for i, name in enumerate(header)
        if name != "y" and not (drop_leaky and name in LEAKY_COLUMNS)
    ]
    feature_names = tuple(header[i] for i in keep)
    target = header.index("y")

    n_samples, n_features = len(body), len(keep)
    vectors = np.zeros((n_samples, n_features), dtype=np.float32)
    categorical = np.zeros(n_features, dtype=bool)
    missing = np.zeros((n_samples, n_features), dtype=bool)

    for column, source in enumerate(keep):
        name = header[source]
        values = [row[source] for row in body]

        if name in BANK_CATEGORICAL:
            categorical[column] = True
            # Ordinal codes: HEOM only ever tests these for equality, so the
            # ordering carries no meaning and none is implied.
            levels = {value: code for code, value in enumerate(sorted(set(values)))}
            vectors[:, column] = [levels[value] for value in values]
            if unknown_as_missing:
                missing[:, column] = [value == UNKNOWN_TOKEN for value in values]
        else:
            numeric = np.asarray([float(value) for value in values], dtype=np.float32)
            if name == "pdays" and pdays_sentinel_as_missing:
                flagged = numeric == PDAYS_SENTINEL
                missing[:, column] = flagged
                # Leave the sentinel out of the range so the real values keep
                # their spread; the missing mask carries the "never contacted"
                # signal on its own.
                numeric = np.where(flagged, np.float32(0.0), numeric)
            vectors[:, column] = numeric

    raw_labels = [row[target].strip() for row in body]
    label_names = tuple(sorted(set(raw_labels)))
    lookup = {name: index for index, name in enumerate(label_names)}
    labels = np.asarray([lookup[value] for value in raw_labels], dtype=np.int64)

    return TabularDataset(
        name="bank_marketing",
        vectors=vectors,
        labels=labels,
        categorical=categorical,
        missing=missing if missing.any() else None,
        feature_names=feature_names,
        label_names=label_names,
    )


DATASETS = {"bank_marketing": load_bank_marketing}
