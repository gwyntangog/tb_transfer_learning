"""Stage B: fine-tune on TBX11K. Val is scored once, after the holdout choices."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch

from tb_transfer.config import (
    ARCHS,
    BATCH_SIZE,
    DEFAULT_OUTPUT,
    MAX_EPOCHS,
    PROTOCOL,
    STAGE_B_LR,
    STAGE_B_PATIENCE,
    STAGE_B_SEED,
)
from tb_transfer.data_tbx11k import freeze_holdout, group_counts, load_tbx_split
from tb_transfer.engine import resolve_device, score_paths, train_with_retry
from tb_transfer.io_utils import write_json
from tb_transfer.metrics import val_report, youden_threshold
from tb_transfer.models import make_model
from tb_transfer.reporting import refresh_stage_b_csv, reusable


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not args.arch:
        args.arch = list(ARCHS)
    if not args.init:
        args.init = ["imagenet", "chexpert"]
    root = Path(args.tbx_root) if args.tbx_root else None
    if root is None:
        env = os.environ.get("TBX11K_ROOT")
        if not env:
            raise SystemExit("Set TBX11K_ROOT to the folder that contains lists and imgs.")
        root = Path(env)
    output = Path(args.output)
    if "chexpert" in args.init:
        missing = [
            architecture
            for architecture in args.arch
            if not (output / "stage_a" / architecture / "best.pt").is_file()
        ]
        if missing:
            joined = ", ".join(missing)
            raise SystemExit(f"Stage A checkpoints missing for: {joined}")

    device = resolve_device(args.device)
    train_films = load_tbx_split(root, "train")
    hold_path = output / "stage_b" / "tbx11k_holdout_seed0.json"
    fit, hold = freeze_holdout(train_films, hold_path, seed=0)
    print(f"TBX11K root: {root}", flush=True)
    print(f"Fit {group_counts(fit)}. Holdout {group_counts(hold)}.", flush=True)
    print(f"Device: {device}", flush=True)
    if args.check:
        print("Check passed. Train list and frozen holdout are ready. Val was not opened.")
        return

    fit_paths = [film.path for film in fit]
    fit_labels = [film.label for film in fit]
    hold_paths = [film.path for film in hold]
    hold_labels = [film.label for film in hold]
    # Val stays unread until a run has chosen its epoch and threshold.
    val_films = None
    for architecture in args.arch:
        for init in args.init:
            json_path = output / "stage_b" / f"{architecture}__{init}.json"
            existing = reusable(json_path, args.force, "val_auprc")
            if existing is not None:
                print(f"Reusing {json_path}")
                continue
            checkpoint = None
            if init == "chexpert":
                checkpoint = output / "stage_a" / architecture / "best.pt"
            print(
                f"Stage B {architecture} from {init}: AdamW lr {STAGE_B_LR} "
                f"patience {STAGE_B_PATIENCE} max_epochs {MAX_EPOCHS}",
                flush=True,
            )
            info, model = train_with_retry(
                lambda architecture=architecture, checkpoint=checkpoint: make_model(
                    architecture, STAGE_B_SEED, checkpoint
                ),
                batch_size=args.batch_size,
                fit_paths=fit_paths,
                fit_labels=fit_labels,
                sel_paths=hold_paths,
                sel_labels=hold_labels,
                lr=STAGE_B_LR,
                patience=STAGE_B_PATIENCE,
                max_epochs=MAX_EPOCHS,
                seed=STAGE_B_SEED,
                device=device,
                checkpoint_path=output / "stage_b" / f"{architecture}__{init}.pt",
                run_name=f"{architecture} {init}",
                num_workers=args.num_workers,
            )
            hold_y, hold_scores = score_paths(
                model,
                hold_paths,
                hold_labels,
                batch_size=info["batch_size"],
                device=device,
                num_workers=args.num_workers,
            )
            threshold = youden_threshold(hold_y, hold_scores)
            if val_films is None:
                val_films = load_tbx_split(root, "val")
            val_y, val_scores = score_paths(
                model,
                [film.path for film in val_films],
                [film.label for film in val_films],
                batch_size=info["batch_size"],
                device=device,
                num_workers=args.num_workers,
            )
            report = val_report(
                val_y, val_scores, [film.group for film in val_films], threshold
            )
            row = {
                "protocol": PROTOCOL,
                "architecture": architecture,
                "init": init,
                "device": str(device),
                "n_fit": len(fit),
                "n_holdout": len(hold),
                "threshold": threshold,
                **info,
                **report,
            }
            write_json(json_path, row)
            table = refresh_stage_b_csv(output)
            print(
                f"{architecture} {init}: val AUPRC {report['val_auprc']:.4f}, "
                f"without sick {report['val_auprc_no_sick']:.4f}, "
                f"FPR healthy {report['fpr_healthy']:.4f}, "
                f"FPR sick {report['fpr_sick']:.4f}",
                flush=True,
            )
            print(f"Wrote {table}")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage B: TBX11K fine-tuning from ImageNet and Stage A.")
    parser.add_argument("--tbx-root", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--arch", action="append", choices=ARCHS)
    parser.add_argument("--init", action="append", choices=("imagenet", "chexpert"))
    parser.add_argument("--force", action="store_true", help="Retrain even if a finished run exists.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the train list and write the frozen holdout. Do not open val.",
    )
    return parser


if __name__ == "__main__":
    main()
