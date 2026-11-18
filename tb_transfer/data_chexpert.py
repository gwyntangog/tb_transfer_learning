"""Frontal CheXpert films with an explicit pneumonia label, split by patient."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from tb_transfer.config import SPLIT_SEED
from tb_transfer.io_utils import read_json, write_json

_PATIENT = re.compile(r"(patient\d+)", re.IGNORECASE)
_REQUIRED = ("Path", "Frontal/Lateral", "Pneumonia")


class MissingFilmsError(FileNotFoundError):
    def __init__(self, missing: list[str], total: int, root: Path):
        self.missing = missing
        self.total = total
        preview = missing[0] if missing else ""
        super().__init__(
            f"{len(missing)} of {total} filtered films are not under {root} yet. "
            f"First missing CSV path: {preview}. "
            "Wait until the image fetch finishes, then run this again."
        )


def load_filtered(root: Path) -> pd.DataFrame:
    """Keep frontal rows whose Pneumonia value is 0.0 or 1.0.

    Uncertain labels (-1) and blank labels are dropped. valid.csv is not read.
    """
    csv_path = root / "train.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected {csv_path}. CHEXPERT_ROOT must be the folder that contains train.csv."
        )
    frame = pd.read_csv(csv_path)
    missing = [column for column in _REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}.")
    frontal = frame["Frontal/Lateral"].astype(str) == "Frontal"
    pneumonia = pd.to_numeric(frame["Pneumonia"], errors="coerce")
    keep = frontal & pneumonia.isin([0.0, 1.0])
    filtered = frame.loc[keep, ["Path"]].copy()
    filtered["label"] = (pneumonia.loc[keep] == 1.0).astype(int)
    filtered["patient_id"] = filtered["Path"].map(patient_id_from_path)
    filtered = filtered.reset_index(drop=True)
    if filtered.empty:
        raise RuntimeError(f"No frontal rows with Pneumonia 0.0 or 1.0 in {csv_path}.")
    return filtered


def patient_id_from_path(path: str) -> str:
    match = _PATIENT.search(str(path).replace("\\", "/"))
    if match is None:
        raise ValueError(f"No patient id in {path}")
    return match.group(1).lower()


def candidate_paths(root: Path, csv_path: str) -> list[Path]:
    """Places a CSV Path might point once the archive is unpacked.

    train.csv stores CheXpert-v1.0/train/patient.../view.jpg. The local copy
    under C:\\data\\chexpert\\CheXpert-v1.0 keeps the files at train/patient...
    """
    parts = tuple(part for part in str(csv_path).replace("\\", "/").split("/") if part)
    candidates: list[Path] = []

    def add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    add(root / Path(*parts))
    if parts and parts[0].lower().startswith("chexpert"):
        add(root / Path(*parts[1:]))
    for index, part in enumerate(parts):
        if part.lower().startswith("patient"):
            suffix = Path(*parts[index:])
            add(root / suffix)
            add(root / "train" / suffix)
            break
    return candidates


def resolve_image(root: Path, csv_path: str) -> Path | None:
    for candidate in candidate_paths(root, csv_path):
        if candidate.is_file():
            return candidate
    return None


def split_patients(frame: pd.DataFrame, seed: int = SPLIT_SEED, test_size: float = 0.1):
    """90/10 patient split. Every image from a patient stays on one side."""
    patients = sorted({str(patient) for patient in frame["patient_id"].unique()})
    fit_patients, sel_patients = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    fit_ids = {str(patient) for patient in fit_patients}
    sel_ids = {str(patient) for patient in sel_patients}
    _require_partition(set(patients), fit_ids, sel_ids)
    fit = frame[frame["patient_id"].isin(fit_ids)].reset_index(drop=True)
    sel = frame[frame["patient_id"].isin(sel_ids)].reset_index(drop=True)
    if len(fit) + len(sel) != len(frame):
        raise RuntimeError("Patient split dropped or duplicated rows.")
    return fit, sel, sorted(fit_ids), sorted(sel_ids)


def freeze_patient_split(frame: pd.DataFrame, path: Path, seed: int = SPLIT_SEED):
    """Reuse a written split when one exists. Otherwise create the seed-0 split."""
    current = {str(patient) for patient in frame["patient_id"].unique()}
    if path.is_file():
        saved = read_json(path)
        if int(saved["seed"]) != seed:
            raise RuntimeError(f"{path} was built with seed {saved['seed']}, not {seed}.")
        fit_ids = {str(patient) for patient in saved["fit_patients"]}
        sel_ids = {str(patient) for patient in saved["sel_patients"]}
        _require_partition(current, fit_ids, sel_ids)
    else:
        _, _, fit_list, sel_list = split_patients(frame, seed=seed)
        fit_ids, sel_ids = set(fit_list), set(sel_list)
        write_json(
            path,
            {
                "seed": seed,
                "test_size": 0.1,
                "fit_patients": sorted(fit_ids),
                "sel_patients": sorted(sel_ids),
            },
        )
    fit = frame[frame["patient_id"].isin(fit_ids)].reset_index(drop=True)
    sel = frame[frame["patient_id"].isin(sel_ids)].reset_index(drop=True)
    return fit, sel


def attach_pair(fit: pd.DataFrame, sel: pd.DataFrame, root: Path):
    both = pd.concat([fit, sel], ignore_index=True)
    both = attach_files(both, root)
    fit_ids = set(fit["patient_id"])
    sel_ids = set(sel["patient_id"])
    return (
        both[both["patient_id"].isin(fit_ids)].reset_index(drop=True),
        both[both["patient_id"].isin(sel_ids)].reset_index(drop=True),
    )


def attach_files(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    resolved: list[Path] = []
    missing: list[str] = []
    for csv_path in frame["Path"].tolist():
        found = resolve_image(root, str(csv_path))
        if found is None:
            missing.append(str(csv_path))
        else:
            resolved.append(found)
    if missing:
        raise MissingFilmsError(missing, len(frame), root)
    out = frame.copy()
    out["abs_path"] = resolved
    return out


def _require_partition(current: set[str], fit_ids: set[str], sel_ids: set[str]) -> None:
    if fit_ids & sel_ids:
        raise RuntimeError("Patient split puts the same patient on both sides.")
    if fit_ids | sel_ids != current:
        raise RuntimeError("Frozen patient split does not match the filtered cohort.")
