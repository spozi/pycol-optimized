"""Fine-tune BERT on one arm and score it on the held-out test set.

Accuracy is close to meaningless at a 1:6.5 class ratio -- predicting "ham"
everywhere already scores about 87% -- so the reported headline is macro F1 and
the minority class's own precision, recall, and F1.  Average precision is
included because it is the metric that actually moves when a model trades
minority recall for precision.

A plain training loop is used rather than ``Trainer`` so that behaviour does not
shift underneath the experiment when transformers changes its defaults.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from .common import resolve_device, seed_everything

DEFAULT_MODEL = "bert-base-uncased"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    model_name: str = DEFAULT_MODEL
    max_length: int = 128
    batch_size: int = 16
    eval_batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    #: Mixed precision. Honoured on CUDA only -- see the note at the training
    #: loop for why Metal is excluded.
    amp: bool = True
    #: DataLoader workers. Applied on CUDA only; on unified memory the extra
    #: processes cost more than the overlap they buy.
    num_workers: int = 2


class TextDataset(Dataset):
    """Unpadded encodings; padding happens per batch in the collator.

    SMS messages run about fifteen tokens against a 128-token limit, so padding
    everything to the limit would spend most of the compute on [PAD].
    """

    def __init__(self, encodings: list[dict[str, list[int]]], labels: NDArray[np.int64]) -> None:
        self.encodings = encodings
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {**self.encodings[index], "labels": int(self.labels[index])}


class PadCollator:
    """Pad a batch to its own longest sequence rather than the global maximum."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([item.pop("labels") for item in batch], dtype=torch.long)
        padded = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
        padded["labels"] = labels
        return dict(padded)


def _metrics(labels: NDArray[np.int64], predicted: NDArray[np.int64], scores: NDArray[np.float64]):
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=[1], average=None, zero_division=0
    )
    return {
        "accuracy": float((labels == predicted).mean()),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "minority_precision": float(precision[0]),
        "minority_recall": float(recall[0]),
        "minority_f1": float(f1[0]),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def train_and_evaluate(
    train_texts: list[str],
    train_labels: NDArray[np.int64],
    test_texts: list[str],
    test_labels: NDArray[np.int64],
    *,
    config: TrainConfig | None = None,
    seed: int = 0,
    device: str = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Fine-tune once and return test-set metrics."""

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    config = config or TrainConfig()
    seed_everything(seed)
    torch_device = resolve_device(device)
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def encode(texts: list[str]) -> list[dict[str, list[int]]]:
        encoded = tokenizer(texts, truncation=True, max_length=config.max_length)
        return [
            {key: encoded[key][i] for key in encoded}  # noqa: SIM118 - BatchEncoding
            for i in range(len(texts))
        ]

    on_cuda = torch_device.type == "cuda"
    # Host-to-device copies overlap with compute only from pinned memory, and
    # only a discrete GPU has a transfer to overlap.  On unified memory (CPU,
    # MPS) pinning buys nothing and workers just add process overhead.
    loader_kwargs: dict[str, Any] = (
        {
            "pin_memory": True,
            "num_workers": config.num_workers,
            "persistent_workers": config.num_workers > 0,
        }
        if on_cuda
        else {}
    )

    collator = PadCollator(tokenizer)
    train_loader = DataLoader(
        TextDataset(encode(train_texts), train_labels),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        TextDataset(encode(test_texts), test_labels),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        **loader_kwargs,
    )

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2).to(
        torch_device
    )

    decay = [
        p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm"))
    ]
    no_decay = [
        p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm"))
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )

    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy="linear",
    )

    # Mixed precision is a large win on CUDA and reliable there.  It is NOT
    # enabled on MPS: fp16 autocast on Metal still produces wrong results in
    # places, and BERT-base at this scale does not need it.
    use_amp = on_cuda and config.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    for epoch in range(config.epochs):
        running = 0.0
        for batch in train_loader:
            batch = {
                key: value.to(torch_device, non_blocking=on_cuda) for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**batch)
            scaler.scale(outputs.loss).backward()
            # Unscale before clipping, or the threshold applies to scaled grads.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(outputs.loss.detach())
        if verbose:
            print(
                f"      epoch {epoch + 1}/{config.epochs}  loss {running / len(train_loader):.4f}",
                flush=True,
            )

    model.eval()
    logits: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for batch in test_loader:
            batch.pop("labels")
            batch = {
                key: value.to(torch_device, non_blocking=on_cuda) for key, value in batch.items()
            }
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(**batch).logits
            # Back to fp32 before softmax so the probabilities are comparable
            # across devices regardless of what precision produced them.
            logits.append(output.float().cpu().numpy())

    stacked = np.concatenate(logits, axis=0)
    predicted = stacked.argmax(axis=1).astype(np.int64)
    scores = torch.softmax(torch.as_tensor(stacked), dim=1).numpy()[:, 1]

    del model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    elif torch_device.type == "mps":
        torch.mps.empty_cache()

    return {
        "seed": seed,
        "device": torch_device.type,
        "n_train": len(train_texts),
        "seconds": time.perf_counter() - started,
        **_metrics(np.asarray(test_labels, dtype=np.int64), predicted, scores.astype(np.float64)),
    }
