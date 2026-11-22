"""Stage A: pneumonia on frontal CheXpert films. The patient split stays on seed 0."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch

from tb_transfer.config import (
    ARCHS,
    BATCH_SIZE,
    DEFAULT_CHEXPERT_ROOT,
    DEFAULT_OUTPUT,
    MAX_EPOCHS,
    PROTOCOL,
    STAGE_A_LR,
    STAGE_A_PATIENCE,
    STAGE_A_SEEDS,
)
from tb_transfer.data_chexpert import (
    MissingFilmsError,
    attach_pair,
    freeze_patient_split,
    load_filtered,
)
from tb_transfer.engine import resolve_device, train_with_retry
from tb_transfer.evaluate import selection_operating_point
from tb_transfer.io_utils import write_json
from tb_transfer.models import make_model
from tb_transfer.reporting import publish_best, refresh_stage_a_csv, reusable


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not args.arch:
        args.arch = list(ARCHS)
    root = Path(args.chexpert_root)
    output = Path(args.output)
    device = resolve_device(args.device)
    frame = load_filtered(root)
    split_path = output / "stage_a" / "patient_split_seed0.json"
    fit, sel = freeze_patient_split(frame, split_path, seed=0)
    _require_both_classes(fit, "fit")
    _require_both_classes(sel, "selection")
    _describe(root, fit, sel, device)
    try:
        fit, sel = attach_pair(fit, sel, root)
    except MissingFilmsError as exc:
        print(exc)
        raise SystemExit(1) from exc
    if args.check:
        print(f"Check passed. {len(fit) + len(sel)} filtered films are on disk. No training was started.")
        return

    fit_paths = fit["abs_path"].tolist()
    fit_labels = fit["label"].astype(float).tolist()
    sel_paths = sel["abs_path"].tolist()
    sel_labels = sel["label"].astype(float).tolist()
    counts = _counts(fit, sel)
    for architecture in args.arch:
        arch_dir = output / "stage_a" / architecture
        finished = []
        for seed in STAGE_A_SEEDS:
            json_path = arch_dir / f"seed_{seed}.json"
            existing = reusable(json_path, args.force, "selection_auprc")
            if existing is not None:
                print(f"Reusing {json_path}")
                finished.append(existing)
                continue
            print(
                f"Stage A {architecture} seed {seed}: AdamW lr {STAGE_A_LR} "
                f"patience {STAGE_A_PATIENCE} max_epochs {MAX_EPOCHS}",
                flush=True,
            )
            info, model = train_with_retry(
                lambda seed=seed, architecture=architecture: make_model(architecture, seed),
                batch_size=args.batch_size,
                fit_paths=fit_paths,
                fit_labels=fit_labels,
                sel_paths=sel_paths,
                sel_labels=sel_labels,
                lr=STAGE_A_LR,
                patience=STAGE_A_PATIENCE,
                max_epochs=MAX_EPOCHS,
                seed=seed,
                device=device,
                checkpoint_path=arch_dir / f"seed_{seed}.pt",
                run_name=f"{architecture} seed {seed}",
                num_workers=args.num_workers,
            )
            operating = selection_operating_point(
                model,
                sel_paths,
                sel_labels,
                sel["Path"].tolist(),
                device=device,
                batch_size=info["batch_size"],
                num_workers=args.num_workers,
                score_path=arch_dir / f"seed_{seed}_selection_scores.npz",
            )
            row = {
                "protocol": PROTOCOL,
                "architecture": architecture,
                "init": "imagenet",
                "device": str(device),
                **counts,
                **info,
                **operating,
            }
            write_json(json_path, row)
            finished.append(row)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        publish_best(arch_dir, finished)
        table = refresh_stage_a_csv(output)
        print(f"Wrote {table}")


def _counts(fit, sel) -> dict:
    return {
        "n_fit": int(len(fit)),
        "n_sel": int(len(sel)),
        "n_pos_fit": int((fit["label"] == 1).sum()),
        "n_pos_sel": int((sel["label"] == 1).sum()),
        "n_patients_fit": int(fit["patient_id"].nunique()),
        "n_patients_sel": int(sel["patient_id"].nunique()),
    }


def _describe(root: Path, fit, sel, device: torch.device) -> None:
    total = len(fit) + len(sel)
    positives = int((fit["label"] == 1).sum() + (sel["label"] == 1).sum())
    print(f"CheXpert root: {root}", flush=True)
    print(
        f"Filtered frontal pneumonia films: {total} "
        f"({positives} positive, {total - positives} negative)",
        flush=True,
    )
    print(
        f"Fit: {len(fit)} films, {fit['patient_id'].nunique()} patients. "
        f"Selection: {len(sel)} films, {sel['patient_id'].nunique()} patients.",
        flush=True,
    )
    print(f"Device: {device}", flush=True)


def _require_both_classes(frame, name: str) -> None:
    if frame["label"].nunique() < 2:
        raise SystemExit(f"CheXpert {name} split has only one class.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage A: CheXpert frontal pneumonia pretraining.")
    parser.add_argument(
        "--chexpert-root",
        default=os.environ.get("CHEXPERT_ROOT", str(DEFAULT_CHEXPERT_ROOT)),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--arch", action="append", choices=ARCHS)
    parser.add_argument("--force", action="store_true", help="Retrain even if a finished seed exists.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Resolve the filtered films and exit before training.",
    )
    return parser


if __name__ == "__main__":
    main()
