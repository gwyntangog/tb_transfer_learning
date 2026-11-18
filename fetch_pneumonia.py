"""Download frontal CheXpert images whose Pneumonia label is 0.0 or 1.0.

The training images are split across three large zips. This script downloads
one zip, extracts only the needed files into CheXpert-v1.0/train, keeps
the zip, then continues. Safe to run again if it stops.
"""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("CHEXPERT_FETCH_ROOT", r"C:\data\chexpert"))
DATASET = ROOT / "CheXpert-v1.0"
BATCH1 = ROOT / "CheXpert-v1.0 batch 1 (validate & csv).zip"
AZCOPY = Path(
    os.environ.get(
        "AZCOPY",
        str(Path.home() / r"azcopy\azcopy_windows_amd64_10.34.2\azcopy.exe"),
    )
)
def _secret(name: str) -> str:
    path = Path(__file__).resolve().parent / "secrets.txt"
    if not path.is_file():
        sys.exit(f"Missing {path}. Put ACCOUNT and SAS there. That file is not committed.")
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        found[key.strip()] = value.strip()
    if name not in found or not found[name]:
        sys.exit(f"{path} has no {name}")
    return found[name]


ACCOUNT = _secret("ACCOUNT")
SAS = _secret("SAS")
# Smallest zip first, so a path mismatch is caught before the 185 GB file.
BATCHES = [
    ("CheXpert-v1.0 batch 4 (train 3).zip", 97_806_638_774),
    ("CheXpert-v1.0 batch 2 (train 1).zip", 174_369_842_258),
    ("CheXpert-v1.0 batch 3 (train 2).zip", 198_450_558_994),
]
MARGIN = 20 * 1024**3
TRAIN_CSV_IN_ZIP = "CheXpert-v1.0 batch 1 (validate & csv)/train.csv"


def log(message: str) -> None:
    print(message, flush=True)


def image_key(path: str) -> str | None:
    path = path.replace("\\", "/")
    marker = "train/patient"
    index = path.find(marker)
    if index >= 0:
        return path[index:]
    # The zips store patient folders directly, with no train/ directory.
    match = re.search(r"(?:^|/)(patient\d+/.+)", path)
    if match is None:
        return None
    return "train/" + match.group(1)


def load_needed_keys() -> set[str]:
    if not BATCH1.is_file():
        sys.exit(f"Missing {BATCH1}")
    with zipfile.ZipFile(BATCH1) as archive:
        text = archive.read(TRAIN_CSV_IN_ZIP).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    DATASET.mkdir(parents=True, exist_ok=True)
    (DATASET / "train.csv").write_text(text, encoding="utf-8", newline="")
    keys = set()
    for row in rows:
        if row["Frontal/Lateral"] != "Frontal":
            continue
        if row["Pneumonia"] not in ("0.0", "1.0"):
            continue
        key = image_key(row["Path"])
        if key is None:
            sys.exit(f"Unexpected path: {row['Path']}")
        keys.add(key)
    log(f"Need {len(keys)} frontal pneumonia images.")
    return keys


def still_missing(keys: set[str]) -> set[str]:
    return {key for key in keys if not (DATASET / key).is_file()}


def run_azcopy(args: list[str]) -> None:
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        sys.exit(
            "AzCopy stopped, usually because the connection dropped. "
            "Reconnect, then run this same command again. "
            "It keeps images already saved and continues the unfinished zip."
        )


def download_zip(blob_name: str, expected_size: int, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    existing = [path for path in folder.rglob("*.zip") if path.is_file()]
    # A finished download is renamed to the real zip name. AzCopy's
    # in-progress file starts with .azDownload- and is already full size.
    complete = [path for path in existing if not path.name.startswith(".azDownload-")]
    partial = [path for path in existing if path.name.startswith(".azDownload-")]
    if len(complete) == 1 and complete[0].stat().st_size == expected_size:
        log(f"Already downloaded {blob_name}")
        return complete[0]
    if len(complete) > 1:
        sys.exit(f"More than one finished zip in {folder}: {complete}")
    if not AZCOPY.is_file():
        sys.exit(f"AzCopy not found at {AZCOPY}")

    if partial:
        # .azDownload-<job-id>-<file name>
        job_id = partial[-1].name[len(".azDownload-"):][:36]
        log(f"Resuming unfinished download {blob_name} (job {job_id}).")
        run_azcopy([str(AZCOPY), "jobs", "resume", job_id])
    else:
        free = shutil.disk_usage(ROOT).free
        if free < expected_size + MARGIN:
            sys.exit(
                f"Not enough free space for {blob_name}. "
                f"Need about {(expected_size + MARGIN) / 1024**3:.0f} GB, "
                f"have {free / 1024**3:.0f} GB."
            )
        url = f"{ACCOUNT}/{urllib.parse.quote(blob_name)}?{SAS}"
        log(f"Downloading {blob_name} ({expected_size / 1024**3:.0f} GB). Leave this window open.")
        run_azcopy([str(AZCOPY), "copy", url, str(folder), "--overwrite=ifSourceNewer"])

    found = [path for path in folder.rglob("*.zip") if path.is_file() and not path.name.startswith(".azDownload-")]
    if len(found) != 1:
        sys.exit(f"Expected one downloaded zip in {folder}, found {found}")
    if found[0].stat().st_size != expected_size:
        sys.exit(f"Download size mismatch for {blob_name}: {found[0].stat().st_size} != {expected_size}")
    return found[0]


def extract_matches(zip_path: Path, missing: set[str]) -> int:
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        matches = []
        samples = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            if len(samples) < 5:
                samples.append(info.filename)
            key = image_key(info.filename)
            if key in missing:
                matches.append((info, key))
        log("Example archive paths: " + " | ".join(samples))
        log(f"{zip_path.name} contains {len(matches)} needed images.")
        for number, (info, key) in enumerate(matches, start=1):
            dest = DATASET / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, dest.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            extracted += 1
            if number % 200 == 0 or number == len(matches):
                log(f"  extracted {number} / {len(matches)}")
    return extracted


def main() -> None:
    keys = load_needed_keys()
    missing = still_missing(keys)
    log(f"{len(keys) - len(missing)} already on disk, {len(missing)} still to fetch.")
    if not missing:
        log(f"Done. Images are in {DATASET}")
        return

    for blob_name, expected_size in BATCHES:
        missing = still_missing(keys)
        if not missing:
            break
        folder = ROOT / "_zip" / blob_name.replace(".zip", "")
        zip_path = download_zip(blob_name, expected_size, folder)
        found = extract_matches(zip_path, missing)
        log(f"Kept {zip_path}.")
        if found == 0:
            sys.exit(
                "This zip had none of the remaining images. "
                "Stopped before downloading the next zip."
            )

    missing = still_missing(keys)
    if missing:
        sample = "\n".join(sorted(missing)[:10])
        sys.exit(f"Still missing {len(missing)} images. First ones:\n{sample}")
    log(f"Done. {len(keys)} images and train.csv are in {DATASET}")


if __name__ == "__main__":
    main()
