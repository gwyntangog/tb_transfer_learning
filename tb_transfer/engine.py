"""Epoch loop. Selection AUPRC picks the checkpoint. Val is not passed in."""

from __future__ import annotations

import gc
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from tb_transfer.config import WEIGHT_DECAY
from tb_transfer.dataset import make_loader
from tb_transfer.io_utils import atomic_torch_save, load_checkpoint
from tb_transfer.metrics import average_precision


def train_run(
    model: nn.Module,
    fit_paths,
    fit_labels,
    sel_paths,
    sel_labels,
    *,
    lr: float,
    patience: int,
    max_epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
    run_name: str,
    num_workers: int = 0,
) -> tuple[dict, nn.Module]:
    pin = device.type == "cuda"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    fit_loader = make_loader(
        fit_paths,
        fit_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin,
    )
    sel_loader = make_loader(
        sel_paths,
        sel_labels,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin,
    )
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    epochs_ran = 0
    for index in range(max_epochs):
        model.train()
        for images, labels in fit_loader:
            images = images.to(device, non_blocking=pin)
            labels = labels.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            logits = _positive_logits(model(images))
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        labels_np, scores = collect_scores(model, sel_loader, device)
        score = average_precision(labels_np, scores)
        epochs_ran = index + 1
        if not math.isfinite(score):
            raise RuntimeError(f"{run_name} selection AUPRC is {score}.")
        if score > best_score:
            best_score = score
            best_epoch = epochs_ran
            stale = 0
            atomic_torch_save(
                {
                    "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "epoch": best_epoch,
                    "selection_auprc": best_score,
                    "batch_size": batch_size,
                    "lr": lr,
                    "seed": seed,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        print(
            f"{run_name} epoch {epochs_ran}/{max_epochs} "
            f"selection_auprc={score:.6f} best={best_score:.6f}",
            flush=True,
        )
        if stale >= patience:
            print(
                f"{run_name} stop: {stale} epoch(s) without a better selection AUPRC.",
                flush=True,
            )
            break
    saved = load_checkpoint(checkpoint_path)
    model.load_state_dict(saved["model"])
    info = {
        "epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "selection_auprc": best_score,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
    }
    return info, model


def train_with_retry(model_factory, *, batch_size: int, **kwargs):
    """On a CUDA out-of-memory error at batch 32, start that run over at batch 16."""
    attempt = batch_size
    while True:
        model = None
        try:
            model = model_factory()
            return train_run(model, batch_size=attempt, **kwargs)
        except Exception as exc:
            if model is not None:
                del model
            gc.collect()
            if attempt == 32 and _is_oom(exc):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("CUDA out of memory at batch 32. Retrying this run at batch 16.", flush=True)
                attempt = 16
                continue
            raise


def collect_scores(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels = []
    scores = []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = _positive_logits(model(images.to(device)))
            probability = torch.sigmoid(logits).detach().cpu()
            scores.append(probability)
            labels.append(batch_labels.detach().cpu())
    return torch.cat(labels).numpy(), torch.cat(scores).numpy()


def score_paths(model, paths, labels, *, batch_size: int, device: torch.device, num_workers: int = 0):
    loader = make_loader(
        paths,
        labels,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return collect_scores(model, loader, device)


def _positive_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    return logits


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
