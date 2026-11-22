"""Short Stage B from the partial Stage A checkpoints.

Uses stage_a_subset/<arch>/seed_0.pt, not stage_a/<arch>/best.pt. At most 3
epochs, patience 1. Val is opened only after the holdout epoch and threshold
are fixed. Results go under stage_b_subset.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import torch

from tb_transfer.config import ARCHS, PROTOCOL, STAGE_B_FIELDS, STAGE_B_LR, STAGE_B_SEED
from tb_transfer.data_tbx11k import freeze_holdout, group_counts, load_tbx_split
from tb_transfer.engine import resolve_device, train_with_retry
from tb_transfer.evaluate import evaluate_tb_run
from tb_transfer.io_utils import write_csv, write_json
from tb_transfer.models import make_model
from tb_transfer.reporting import reusable
from tb_transfer.train_b import _parser

SUBSET_EPOCHS = 3
SUBSET_PATIENCE = 1


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    architectures = list(args.arch) if args.arch else list(ARCHS)
    inits = list(args.init) if args.init else ["imagenet", "chexpert"]
    root = Path(args.tbx_root) if args.tbx_root else None
    if root is None:
        env = os.environ.get("TBX11K_ROOT")
        if not env:
            raise SystemExit("Set TBX11K_ROOT to the folder that contains lists and imgs.")
        root = Path(env)
    output = Path(args.output)
    if "chexpert" in inits:
        missing = [
            architecture
            for architecture in architectures
            if not (output / "stage_a_subset" / architecture / "seed_0.pt").is_file()
        ]
        if missing:
            raise SystemExit("Subset Stage A checkpoints missing for: " + ", ".join(missing))

    device = resolve_device(args.device)
    train_films = load_tbx_split(root, "train")
    hold_path = output / "stage_b_subset" / "tbx11k_holdout_seed0.json"
    fit, hold = freeze_holdout(train_films, hold_path, seed=0)
    print(f"TBX11K root: {root}", flush=True)
    print(f"Partial Stage B from subset checkpoints. Fit {group_counts(fit)}. Holdout {group_counts(hold)}.", flush=True)
    print(f"Device: {device}", flush=True)
    if args.check:
        print("Check passed. Train list and frozen holdout are ready. Val was not opened.")
        return

    fit_paths = [film.path for film in fit]
    fit_labels = [film.label for film in fit]
    hold_paths = [film.path for film in hold]
    hold_labels = [film.label for film in hold]
    val_films = None
    rows = []
    table = output / "stage_b_subset_metrics.csv"
    for architecture in architectures:
        for init in inits:
            json_path = output / "stage_b_subset" / f"{architecture}__{init}.json"
            existing = reusable(json_path, args.force, "val_auprc")
            if existing is None:
                checkpoint = None
                if init == "chexpert":
                    checkpoint = output / "stage_a_subset" / architecture / "seed_0.pt"
                print(
                    f"Subset Stage B {architecture} from {init}: AdamW lr {STAGE_B_LR} "
                    f"patience {SUBSET_PATIENCE} max_epochs {SUBSET_EPOCHS}",
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
                    patience=SUBSET_PATIENCE,
                    max_epochs=SUBSET_EPOCHS,
                    seed=STAGE_B_SEED,
                    device=device,
                    checkpoint_path=output / "stage_b_subset" / f"{architecture}__{init}.pt",
                    run_name=f"subset {architecture} {init}",
                    num_workers=args.num_workers,
                )
                def load_val(root=root):
                    nonlocal val_films
                    if val_films is None:
                        val_films = load_tbx_split(root, "val")
                    return val_films

                report = evaluate_tb_run(
                    model,
                    hold,
                    load_val,
                    device=device,
                    batch_size=info["batch_size"],
                    num_workers=args.num_workers,
                    hold_score_path=output / "stage_b_subset" / f"{architecture}__{init}_holdout_scores.npz",
                    val_score_path=output / "stage_b_subset" / f"{architecture}__{init}_val_scores.npz",
                )
                row = {
                    "protocol": PROTOCOL,
                    "variant": "subset",
                    "architecture": architecture,
                    "init": init,
                    "device": str(device),
                    "n_fit": len(fit),
                    "n_holdout": len(hold),
                    **info,
                    **report,
                }
                write_json(json_path, row)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"Reusing {json_path}")
                row = existing
            rows.append(row)
            write_csv(table, rows, STAGE_B_FIELDS)
            if "val_sensitivity" in row:
                print(
                    f"Subset {architecture} {init}: val AUPRC {float(row['val_auprc']):.4f}, "
                    f"sensitivity {float(row['val_sensitivity']):.4f}, "
                    f"specificity {float(row['val_specificity']):.4f}, "
                    f"FPR healthy {float(row['fpr_healthy']):.4f}, "
                    f"FPR sick {float(row['fpr_sick']):.4f}",
                    flush=True,
                )
            else:
                print(
                    f"Subset {architecture} {init}: val AUPRC {float(row['val_auprc']):.4f}, "
                    f"without sick {float(row['val_auprc_no_sick']):.4f}, "
                    f"FPR healthy {float(row['fpr_healthy']):.4f}, "
                    f"FPR sick {float(row['fpr_sick']):.4f}",
                    flush=True,
                )
    print(f"Wrote {table}")


if __name__ == "__main__":
    main()
