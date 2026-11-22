"""Score a finished checkpoint and record the operating point.

Sensitivity and specificity use the Youden cutoff. For Stage B that cutoff is
chosen on the frozen holdout, then applied once to validation. For Stage A it
is chosen on the selection films, which also chose the epoch, so those rates
describe the selection set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tb_transfer.engine import score_paths
from tb_transfer.io_utils import load_checkpoint
from tb_transfer.metrics import auroc, holdout_report, rates_at_threshold, val_report, youden_threshold
from tb_transfer.models import build_model


def load_finished_model(architecture: str, checkpoint: Path) -> torch.nn.Module:
    model = build_model(architecture, pretrained=False)
    saved = load_checkpoint(checkpoint)
    donor = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
    model.load_state_dict(donor)
    return model


def write_scores(path: Path, rel_paths, groups, labels, scores) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rel_path=np.asarray(list(rel_paths), dtype=str),
        group=np.asarray(list(groups), dtype=str),
        label=np.asarray(labels, dtype=np.int8),
        score=np.asarray(scores, dtype=np.float32),
    )


def selection_operating_point(
    model,
    paths,
    labels,
    rel_paths,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    score_path: Path,
) -> dict:
    """Youden rates on the films that picked the epoch. Not a test set."""
    observed, scores = score_paths(
        model,
        paths,
        labels,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    groups = ["pneumonia" if int(label) == 1 else "no_pneumonia" for label in observed]
    write_scores(score_path, rel_paths, groups, observed, scores)
    threshold = youden_threshold(observed, scores)
    rates = rates_at_threshold(observed, scores, threshold)
    return {
        "selection_auroc": auroc(observed, scores),
        "selection_threshold": threshold,
        "selection_sensitivity": rates["sensitivity"],
        "selection_specificity": rates["specificity"],
        "selection_youden_j": rates["youden_j"],
        "selection_tp": rates["tp"],
        "selection_fp": rates["fp"],
        "selection_tn": rates["tn"],
        "selection_fn": rates["fn"],
        "selection_scores": str(score_path),
    }


def evaluate_tb_run(
    model,
    hold,
    load_val,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    hold_score_path: Path,
    val_score_path: Path,
) -> dict:
    """Choose Youden on the holdout, then score validation once."""
    hold_y, hold_scores = score_paths(
        model,
        [film.path for film in hold],
        [film.label for film in hold],
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    threshold = youden_threshold(hold_y, hold_scores)
    write_scores(
        hold_score_path,
        [film.rel_path for film in hold],
        [film.group for film in hold],
        hold_y,
        hold_scores,
    )
    report = holdout_report(hold_y, hold_scores, [film.group for film in hold], threshold)
    val_films = load_val()
    val_y, val_scores = score_paths(
        model,
        [film.path for film in val_films],
        [film.label for film in val_films],
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    write_scores(
        val_score_path,
        [film.rel_path for film in val_films],
        [film.group for film in val_films],
        val_y,
        val_scores,
    )
    report.update(
        val_report(val_y, val_scores, [film.group for film in val_films], threshold)
    )
    report["threshold"] = threshold
    report["holdout_scores"] = str(hold_score_path)
    report["val_scores"] = str(val_score_path)
    return report
