"""Short Stage A for a deadline. The full run stays in train_a.py.

All three architectures, seed 0 only, at most 3 epochs, stop after one epoch
that does not improve. Same films, same patient split, and the same learning
rate as the full experiment. Checkpoints go under stage_a_quick so they do
not replace stage_a.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from tb_transfer.config import ARCHS, PROTOCOL, STAGE_A_FIELDS, STAGE_A_LR
from tb_transfer.data_chexpert import MissingFilmsError, attach_pair, freeze_patient_split, load_filtered
from tb_transfer.engine import resolve_device, train_with_retry
from tb_transfer.io_utils import write_csv, write_json
from tb_transfer.models import make_model
from tb_transfer.reporting import reusable
from tb_transfer.train_a import _counts, _describe, _parser, _require_both_classes

QUICK_EPOCHS = 3
QUICK_PATIENCE = 1
QUICK_SEED = 0


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    architectures = list(args.arch) if args.arch else list(ARCHS)
    root = Path(args.chexpert_root)
    output = Path(args.output)
    device = resolve_device(args.device)
    frame = load_filtered(root)
    split_path = output / "stage_a_quick" / "patient_split_seed0.json"
    fit, sel = freeze_patient_split(frame, split_path, seed=0)
    _require_both_classes(fit, "fit")
    _require_both_classes(sel, "selection")
    print(
        f"Quick Stage A: {', '.join(architectures)}, seed {QUICK_SEED}, "
        f"at most {QUICK_EPOCHS} epochs, patience {QUICK_PATIENCE}.",
        flush=True,
    )
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
    rows = []
    table = output / "stage_a_quick_metrics.csv"
    for architecture in architectures:
        arch_dir = output / "stage_a_quick" / architecture
        json_path = arch_dir / f"seed_{QUICK_SEED}.json"
        existing = reusable(json_path, args.force, "selection_auprc")
        if existing is None:
            print(
                f"Quick {architecture} seed {QUICK_SEED}: AdamW lr {STAGE_A_LR} "
                f"patience {QUICK_PATIENCE} max_epochs {QUICK_EPOCHS}",
                flush=True,
            )
            info, model = train_with_retry(
                lambda architecture=architecture: make_model(architecture, QUICK_SEED),
                batch_size=args.batch_size,
                fit_paths=fit_paths,
                fit_labels=fit_labels,
                sel_paths=sel_paths,
                sel_labels=sel_labels,
                lr=STAGE_A_LR,
                patience=QUICK_PATIENCE,
                max_epochs=QUICK_EPOCHS,
                seed=QUICK_SEED,
                device=device,
                checkpoint_path=arch_dir / f"seed_{QUICK_SEED}.pt",
                run_name=f"quick {architecture} seed {QUICK_SEED}",
                num_workers=args.num_workers,
            )
            row = {
                "protocol": PROTOCOL,
                "variant": "quick",
                "architecture": architecture,
                "init": "imagenet",
                "device": str(device),
                "selected": 1,
                **counts,
                **info,
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
        write_csv(table, rows, STAGE_A_FIELDS)
        print(
            f"Quick {architecture}: selection AUPRC {float(row['selection_auprc']):.4f} "
            f"at epoch {int(row['epoch'])}.",
            flush=True,
        )
    print(f"Wrote {table}")


if __name__ == "__main__":
    main()
