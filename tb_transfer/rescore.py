"""Fill sensitivity and specificity from checkpoints that already finished.

This does not train. Stage B recomputes the Youden cutoff on the frozen
holdout, then scores validation once. Stage A reports the Youden cutoff on
the selection films only.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from tb_transfer.config import ARCHS, DEFAULT_CHEXPERT_ROOT, DEFAULT_OUTPUT, STAGE_A_FIELDS, STAGE_B_FIELDS
from tb_transfer.data_chexpert import freeze_patient_split, load_filtered
from tb_transfer.data_tbx11k import freeze_holdout, load_tbx_split
from tb_transfer.engine import resolve_device
from tb_transfer.evaluate import evaluate_tb_run, load_finished_model, selection_operating_point
from tb_transfer.io_utils import read_json, write_csv, write_json
from tb_transfer.reporting import refresh_stage_a_csv, refresh_stage_b_csv
from tb_transfer.train_a_subset import films_on_disk


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Recompute threshold stats from saved checkpoints.")
    parser.add_argument("--which", choices=("subset", "full", "both"), default="both")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tbx-root", default=os.environ.get("TBX11K_ROOT"))
    parser.add_argument(
        "--chexpert-root",
        default=os.environ.get("CHEXPERT_ROOT", str(DEFAULT_CHEXPERT_ROOT)),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    output = Path(args.output)
    which = ("subset", "full") if args.which == "both" else (args.which,)
    for kind in which:
        _rescore_stage_a(kind, output, Path(args.chexpert_root), device, args)
        if args.tbx_root:
            _rescore_stage_b(kind, output, Path(args.tbx_root), device, args)
        else:
            print("Skipping Stage B. Set --tbx-root to rescore those checkpoints.")


def _rescore_stage_a(kind: str, output: Path, root: Path, device, args) -> None:
    folder = output / ("stage_a_subset" if kind == "subset" else "stage_a")
    if not folder.is_dir():
        print(f"No {folder}. Skipping Stage A {kind}.")
        return
    if not (root / "train.csv").is_file():
        print(f"No train.csv under {root}. Skipping Stage A {kind}.")
        return
    _, sel = _stage_a_films(kind, output, root)
    for architecture in ARCHS:
        checkpoints = [folder / architecture / "seed_0.pt"] if kind == "subset" else [
            folder / architecture / f"seed_{seed}.pt" for seed in (0, 1, 2)
        ]
        for checkpoint in checkpoints:
            if not checkpoint.is_file():
                print(f"Missing {checkpoint}")
                continue
            json_path = checkpoint.with_suffix(".json")
            row = read_json(json_path) if json_path.is_file() else {"architecture": architecture}
            print(f"Scoring {checkpoint}", flush=True)
            model = load_finished_model(architecture, checkpoint).to(device)
            metrics = selection_operating_point(
                model,
                sel["abs_path"].tolist(),
                sel["label"].astype(float).tolist(),
                sel["Path"].tolist(),
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                score_path=checkpoint.with_name(checkpoint.stem + "_selection_scores.npz"),
            )
            row.update(metrics)
            write_json(json_path, row)
            print(
                f"{architecture}: selection sensitivity {metrics['selection_sensitivity']:.4f}, "
                f"specificity {metrics['selection_specificity']:.4f}, "
                f"TP {metrics['selection_tp']} FP {metrics['selection_fp']} "
                f"TN {metrics['selection_tn']} FN {metrics['selection_fn']}",
                flush=True,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if kind == "full":
        table = refresh_stage_a_csv(output)
    else:
        rows = [read_json(path) for path in sorted(folder.glob("*/seed_*.json"))]
        table = output / "stage_a_subset_metrics.csv"
        write_csv(table, rows, STAGE_A_FIELDS)
    print(f"Wrote {table}")


def _rescore_stage_b(kind: str, output: Path, root: Path, device, args) -> None:
    folder = output / ("stage_b_subset" if kind == "subset" else "stage_b")
    if not folder.is_dir():
        print(f"No {folder}. Skipping Stage B {kind}.")
        return
    train_films = load_tbx_split(root, "train")
    hold_path = folder / "tbx11k_holdout_seed0.json"
    _, hold = freeze_holdout(train_films, hold_path, seed=0)
    val_films = None

    def load_val():
        nonlocal val_films
        if val_films is None:
            val_films = load_tbx_split(root, "val")
        return val_films

    found = False
    for architecture in ARCHS:
        for init in ("imagenet", "chexpert"):
            checkpoint = folder / f"{architecture}__{init}.pt"
            if not checkpoint.is_file():
                print(f"Missing {checkpoint}")
                continue
            found = True
            json_path = checkpoint.with_suffix(".json")
            row = read_json(json_path) if json_path.is_file() else {
                "architecture": architecture,
                "init": init,
            }
            print(f"Scoring {checkpoint}", flush=True)
            model = load_finished_model(architecture, checkpoint).to(device)
            metrics = evaluate_tb_run(
                model,
                hold,
                load_val,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                hold_score_path=folder / f"{architecture}__{init}_holdout_scores.npz",
                val_score_path=folder / f"{architecture}__{init}_val_scores.npz",
            )
            row.update(metrics)
            write_json(json_path, row)
            print(
                f"{architecture} {init}: sensitivity {metrics['val_sensitivity']:.4f}, "
                f"specificity {metrics['val_specificity']:.4f}, "
                f"specificity healthy {metrics['val_specificity_healthy']:.4f}, "
                f"specificity sick {metrics['val_specificity_sick']:.4f}, "
                f"TP {metrics['val_tp']} FP {metrics['val_fp']} "
                f"TN {metrics['val_tn']} FN {metrics['val_fn']}",
                flush=True,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not found:
        return
    if kind == "full":
        table = refresh_stage_b_csv(output)
    else:
        rows = [read_json(path) for path in sorted(folder.glob("*__*.json"))]
        table = output / "stage_b_subset_metrics.csv"
        write_csv(table, rows, STAGE_B_FIELDS)
    print(f"Wrote {table}")


def _stage_a_films(kind: str, output: Path, root: Path):
    split_path = output / ("stage_a_subset" if kind == "subset" else "stage_a") / "patient_split_seed0.json"
    if not split_path.is_file():
        raise SystemExit(f"No saved patient split at {split_path}.")
    filtered = load_filtered(root)
    if kind == "subset":
        present = films_on_disk(filtered, root)
        saved = read_json(split_path)
        ids = {str(patient) for patient in saved["fit_patients"] + saved["sel_patients"]}
        present = present[present["patient_id"].isin(ids)].reset_index(drop=True)
        frame = present
    else:
        from tb_transfer.data_chexpert import attach_pair

        fit, sel = freeze_patient_split(filtered, split_path, seed=0)
        return attach_pair(fit, sel, root)
    return freeze_patient_split(frame, split_path, seed=0)


if __name__ == "__main__":
    main()
