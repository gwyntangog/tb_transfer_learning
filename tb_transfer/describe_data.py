"""Write dataset counts for the paper. No model is trained."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

from tb_transfer.config import DEFAULT_CHEXPERT_ROOT, DEFAULT_OUTPUT, TBX_TRAIN_COUNTS, TBX_VAL_COUNTS
from tb_transfer.data_chexpert import load_filtered, resolve_image
from tb_transfer.data_tbx11k import (
    official_list_path,
    parse_list_text,
    require_counts,
    resolve_tbx_image,
)
from tb_transfer.io_utils import write_json

_META = {"Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Describe the CheXpert cohort and the TBX11K lists.")
    parser.add_argument(
        "--chexpert-root",
        default=os.environ.get("CHEXPERT_ROOT", str(DEFAULT_CHEXPERT_ROOT)),
    )
    parser.add_argument("--tbx-root", default=os.environ.get("TBX11K_ROOT"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-image-sizes",
        action="store_true",
        help="Skip reading image width and height.",
    )
    args = parser.parse_args(argv)
    payload = {}
    text = []
    chexpert_root = Path(args.chexpert_root)
    if (chexpert_root / "train.csv").is_file():
        payload["chexpert"] = describe_chexpert(chexpert_root, sizes=not args.skip_image_sizes)
        text.append(_chexpert_text(payload["chexpert"]))
    else:
        text.append(f"CheXpert train.csv was not found at {chexpert_root}.")
    if args.tbx_root:
        payload["tbx11k"] = describe_tbx(Path(args.tbx_root), sizes=not args.skip_image_sizes)
        text.append(_tbx_text(payload["tbx11k"]))
    else:
        text.append("TBX11K root was not set. Pass --tbx-root to describe those lists.")
    text.append(_definitions())
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "data_description.json"
    text_path = args.output / "data_description.txt"
    write_json(json_path, payload)
    text_path.write_text("\n\n".join(text) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {text_path}")


def describe_chexpert(root: Path, sizes: bool = True) -> dict:
    raw = pd.read_csv(root / "train.csv")
    view = raw["Frontal/Lateral"].astype(str)
    pneumonia = pd.to_numeric(raw["Pneumonia"], errors="coerce") if "Pneumonia" in raw.columns else None
    frontal = view == "Frontal"
    funnel = {
        "n_rows": int(len(raw)),
        "n_frontal": int(frontal.sum()),
        "n_lateral": int((view == "Lateral").sum()),
    }
    if pneumonia is not None:
        frontal_labels = pneumonia.loc[frontal]
        funnel["frontal_pneumonia_positive"] = int((frontal_labels == 1.0).sum())
        funnel["frontal_pneumonia_negative"] = int((frontal_labels == 0.0).sum())
        funnel["frontal_pneumonia_uncertain"] = int((frontal_labels == -1.0).sum())
        funnel["frontal_pneumonia_blank"] = int(frontal_labels.isna().sum())
    kept = load_filtered(root)
    resolved = [resolve_image(root, str(path)) for path in kept["Path"].tolist()]
    n_on_disk = sum(path is not None for path in resolved)
    cohort = {
        "n_films": int(len(kept)),
        "n_patients": int(kept["patient_id"].nunique()),
        "n_positive": int((kept["label"] == 1).sum()),
        "n_negative": int((kept["label"] == 0).sum()),
        "prevalence": float((kept["label"] == 1).mean()),
        "films_per_patient": _films_per_patient(kept["patient_id"]),
        "n_on_disk": n_on_disk,
        "n_missing": int(len(kept) - n_on_disk),
    }
    if "Sex" in raw.columns:
        cohort["sex"] = _value_counts(raw.loc[raw["Path"].isin(set(kept["Path"])), "Sex"])
    if "Age" in raw.columns:
        cohort["age"] = _numeric_summary(raw.loc[raw["Path"].isin(set(kept["Path"])), "Age"])
    if "AP/PA" in raw.columns:
        cohort["ap_pa"] = _value_counts(raw.loc[raw["Path"].isin(set(kept["Path"])), "AP/PA"])
    labels = _observation_counts(raw, kept["Path"])
    image_sizes = None
    if sizes:
        image_sizes = _image_sizes([path for path in resolved if path is not None])
    return {"funnel": funnel, "kept_cohort": cohort, "observations_on_kept_films": labels, "image_sizes": image_sizes}


def describe_tbx(root: Path, sizes: bool = True) -> dict:
    splits = {}
    for split, expected_name in (("train", "train"), ("val", "val")):
        list_path = official_list_path(root, split)
        rows = parse_list_text(list_path.read_text(encoding="utf-8-sig"))
        expected = TBX_TRAIN_COUNTS if split == "train" else TBX_VAL_COUNTS
        require_counts([group for _, group, _ in rows], expected, list_path.name)
        found = []
        missing = []
        for rel, _, _ in rows:
            path = resolve_tbx_image(root, rel)
            if path is None:
                missing.append(rel)
            else:
                found.append(path)
        summary = list_summary(rows)
        summary["n_on_disk"] = len(found)
        summary["n_missing"] = len(missing)
        summary["list"] = str(list_path)
        if sizes:
            summary["image_sizes"] = _image_sizes(found)
        splits[expected_name] = summary
    return {
        "splits": splits,
        "positive": "Directory tb, including active and latent tuberculosis.",
        "negative": "Healthy (health, healthy, or normal) or sick but not tuberculosis.",
        "ignored_lists": "all_*.txt files are not read. They mix other collections into the lists.",
    }


def list_summary(rows: list[tuple[str, str, int]]) -> dict:
    counts = Counter(group for _, group, _ in rows)
    n = len(rows)
    n_pos = int(counts["tb"])
    return {
        "n": n,
        "tb": int(counts["tb"]),
        "healthy": int(counts["healthy"]),
        "sick": int(counts["sick"]),
        "n_positive": n_pos,
        "n_negative": n - n_pos,
        "prevalence": n_pos / n if n else None,
    }


def _observation_counts(raw: pd.DataFrame, kept_paths: pd.Series) -> dict:
    kept = raw.loc[raw["Path"].isin(set(kept_paths))]
    columns = [column for column in raw.columns if column not in _META]
    out = {}
    for column in columns:
        values = pd.to_numeric(kept[column], errors="coerce")
        out[column] = {
            "positive": int((values == 1.0).sum()),
            "negative": int((values == 0.0).sum()),
            "uncertain": int((values == -1.0).sum()),
            "blank": int(values.isna().sum()),
        }
    return out


def _films_per_patient(patient_ids: pd.Series) -> dict:
    counts = patient_ids.value_counts()
    return {
        "min": int(counts.min()),
        "median": float(counts.median()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "patients_with_more_than_one_film": int((counts > 1).sum()),
    }


def _value_counts(series: pd.Series) -> dict:
    filled = series.fillna("blank").astype(str)
    return {str(key): int(value) for key, value in filled.value_counts().items()}


def _numeric_summary(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def _image_sizes(paths: list[Path]) -> dict | None:
    if not paths:
        return None
    widths = []
    heights = []
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        widths.append(width)
        heights.append(height)
    return {
        "n": len(paths),
        "width_min": int(min(widths)),
        "width_median": float(pd.Series(widths).median()),
        "width_max": int(max(widths)),
        "height_min": int(min(heights)),
        "height_median": float(pd.Series(heights).median()),
        "height_max": int(max(heights)),
    }


def _chexpert_text(description: dict) -> str:
    funnel = description["funnel"]
    cohort = description["kept_cohort"]
    lines = [
        "CheXpert train.csv",
        f"Rows in train.csv: {funnel['n_rows']}",
        f"Frontal: {funnel['n_frontal']}. Lateral: {funnel['n_lateral']}.",
    ]
    if "frontal_pneumonia_positive" in funnel:
        lines.append(
            "Frontal pneumonia cells: "
            f"{funnel['frontal_pneumonia_positive']} positive, "
            f"{funnel['frontal_pneumonia_negative']} negative, "
            f"{funnel['frontal_pneumonia_uncertain']} uncertain (-1), "
            f"{funnel['frontal_pneumonia_blank']} blank."
        )
    lines.append(
        "Kept cohort (frontal and pneumonia exactly 0 or 1): "
        f"{cohort['n_films']} films, {cohort['n_patients']} patients, "
        f"{cohort['n_positive']} positive, {cohort['n_negative']} negative, "
        f"prevalence {cohort['prevalence']:.4f}."
    )
    lines.append(
        f"Films on disk: {cohort['n_on_disk']}. Missing: {cohort['n_missing']}."
    )
    return "\n".join(lines)


def _tbx_text(description: dict) -> str:
    lines = ["TBX11K official lists"]
    for name, summary in description["splits"].items():
        lines.append(
            f"{name}: {summary['n']} films, {summary['tb']} TB, "
            f"{summary['healthy']} healthy, {summary['sick']} sick, "
            f"prevalence {summary['prevalence']:.4f}, "
            f"on disk {summary['n_on_disk']}, missing {summary['n_missing']}."
        )
    lines.append(description["positive"])
    lines.append(description["negative"])
    lines.append(description["ignored_lists"])
    return "\n".join(lines)


def _definitions() -> str:
    return "\n".join(
        [
            "Definitions used for the tables",
            "CheXpert positive: frontal film whose Pneumonia cell is 1.",
            "CheXpert negative: frontal film whose Pneumonia cell is 0.",
            "Dropped before training: lateral films, Pneumonia -1, and blank Pneumonia.",
            "TBX11K positive: a film in a tb directory. Active and latent tuberculosis both count.",
            "TBX11K negative: healthy or sick but not tuberculosis.",
            "Sensitivity: true positive rate, true positives divided by all positive films.",
            "Specificity: true negative rate, true negatives divided by all negative films.",
            "Stage B cutoff: Youden threshold fit on the frozen 10% holdout, then applied to validation.",
            "Stage A cutoff: Youden threshold on the selection films. Those films also chose the epoch.",
        ]
    )


if __name__ == "__main__":
    main()
