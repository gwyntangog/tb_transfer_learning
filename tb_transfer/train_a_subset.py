"""Stage A on whatever frontal pneumonia films are already on disk.

The full run stays in train_a.py. This split is only the patients whose
images exist, written under stage_a_subset. It is a partial cohort.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from tb_transfer.config import ARCHS, PROTOCOL, STAGE_A_FIELDS, STAGE_A_LR
from tb_transfer.data_chexpert import freeze_patient_split, load_filtered, resolve_image
from tb_transfer.engine import resolve_device, train_with_retry
from tb_transfer.io_utils import write_csv, write_json
from tb_transfer.models import make_model
from tb_transfer.reporting import reusable
from tb_transfer.train_a import _counts, _parser, _require_both_classes

SUBSET_EPOCHS = 3
SUBSET_PATIENCE = 1
SUBSET_SEED = 0


def films_on_disk(frame, root: Path):
    paths = []
    keep = []
    for csv_path in frame["Path"].tolist():
        found = resolve_image(root, str(csv_path))
        if found is None:
            keep.append(False)
            continue
        keep.append(True)
        paths.append(found)
    present = frame.loc[keep].reset_index(drop=True)
    present["abs_path"] = paths
    return present


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    architectures = list(args.arch) if args.arch else list(ARCHS)
    root = Path(args.chexpert_root)
    output = Path(args.output)
    device = resolve_device(args.device)
    filtered = load_filtered(root)
    present = films_on_disk(filtered, root)
    print(
        f"Partial cohort: {len(present)} of {len(filtered)} frontal pneumonia films are on disk.",
        flush=True,
    )
    if present.empty:
        raise SystemExit("No films have been extracted yet.")
    if present["label"].nunique() < 2:
        raise SystemExit("Films on disk contain only one class. Wait for more of the zip.")
    split_path = output / "stage_a_subset" / "patient_split_seed0.json"
    fit, sel = freeze_patient_split(present, split_path, seed=0)
    _require_both_classes(fit, "fit")
    _require_both_classes(sel, "selection")
    print(
        f"Subset Stage A: {', '.join(architectures)}, seed {SUBSET_SEED}, "
        f"at most {SUBSET_EPOCHS} epochs, patience {SUBSET_PATIENCE}.",
        flush=True,
    )
    print(
        f"Fit: {len(fit)} films, {fit['patient_id'].nunique()} patients. "
        f"Selection: {len(sel)} films, {sel['patient_id'].nunique()} patients.",
        flush=True,
    )
    print(f"Device: {device}", flush=True)
    if args.check:
        print("Check passed. No training was started.")
        return

    counts = _counts(fit, sel)
    rows = []
    table = output / "stage_a_subset_metrics.csv"
    for architecture in architectures:
        arch_dir = output / "stage_a_subset" / architecture
        json_path = arch_dir / f"seed_{SUBSET_SEED}.json"
        existing = reusable(json_path, args.force, "selection_auprc")
        if existing is None:
            info, model = train_with_retry(
                lambda architecture=architecture: make_model(architecture, SUBSET_SEED),
                batch_size=args.batch_size,
                fit_paths=fit["abs_path"].tolist(),
                fit_labels=fit["label"].astype(float).tolist(),
                sel_paths=sel["abs_path"].tolist(),
                sel_labels=sel["label"].astype(float).tolist(),
                lr=STAGE_A_LR,
                patience=SUBSET_PATIENCE,
                max_epochs=SUBSET_EPOCHS,
                seed=SUBSET_SEED,
                device=device,
                checkpoint_path=arch_dir / f"seed_{SUBSET_SEED}.pt",
                run_name=f"subset {architecture} seed {SUBSET_SEED}",
                num_workers=args.num_workers,
            )
            row = {
                "protocol": PROTOCOL,
                "variant": "subset",
                "architecture": architecture,
                "init": "imagenet",
                "device": str(device),
                "selected": 1,
                "n_on_disk": int(len(present)),
                "n_filtered": int(len(filtered)),
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
            f"Subset {architecture}: selection AUPRC {float(row['selection_auprc']):.4f} "
            f"at epoch {int(row['epoch'])}.",
            flush=True,
        )
    print(f"Wrote {table}")


if __name__ == "__main__":
    main()
