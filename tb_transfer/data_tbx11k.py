"""TBX11K train and val lists. Extended all_*.txt lists are never opened."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from tb_transfer.config import (
    SPLIT_SEED,
    TBX_HOLDOUT_COUNTS,
    TBX_TRAIN_COUNTS,
    TBX_VAL_COUNTS,
)
from tb_transfer.io_utils import read_json, write_json

_HEALTHY_DIRS = {"health", "healthy", "normal"}
_EXPECTED = {"train": TBX_TRAIN_COUNTS, "val": TBX_VAL_COUNTS}


@dataclass(frozen=True)
class TBFilm:
    rel_path: str
    path: Path
    group: str
    label: int


class MissingFilmsError(FileNotFoundError):
    def __init__(self, missing: list[str], root: Path):
        self.missing = missing
        preview = missing[0] if missing else ""
        super().__init__(
            f"{len(missing)} TBX11K films are missing under {root}. First path: {preview}."
        )


def official_list_path(root: Path, split: str) -> Path:
    if split not in {"train", "val"}:
        raise ValueError("Only TBX11K_train.txt and TBX11K_val.txt are used.")
    return root / "lists" / f"TBX11K_{split}.txt"


def classify_rel_path(rel: str) -> str:
    parts = [part.lower() for part in str(rel).replace("\\", "/").split("/") if part]
    if not parts:
        raise ValueError("Empty list path.")
    parent, name = parts[:-1], parts[-1]
    from_dirs = []
    if "tb" in parent:
        from_dirs.append("tb")
    if any(part in _HEALTHY_DIRS for part in parent):
        from_dirs.append("healthy")
    if "sick" in parent:
        from_dirs.append("sick")
    if len(from_dirs) > 1:
        raise ValueError(f"Conflicting labels in path: {rel}")
    if len(from_dirs) == 1:
        return from_dirs[0]
    from_name = []
    if name.startswith("tb"):
        from_name.append("tb")
    if name.startswith("normal") or name.startswith("health"):
        from_name.append("healthy")
    if name.startswith("sick"):
        from_name.append("sick")
    if len(from_name) != 1:
        raise ValueError(f"Cannot label {rel}.")
    return from_name[0]


def parse_list_text(text: str) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        token = stripped.split()[0].strip('"').replace("\\", "/")
        group = classify_rel_path(token)
        rows.append((token, group, 1 if group == "tb" else 0))
    if not rows:
        raise ValueError("The list is empty.")
    rels = [rel for rel, _, _ in rows]
    if len(rels) != len(set(rels)):
        raise ValueError("The list contains a duplicate path.")
    return rows


def require_counts(groups: list[str], expected: dict[str, int], what: str) -> None:
    counts = Counter(groups)
    got = {name: int(counts[name]) for name in expected}
    if got != expected or sum(counts.values()) != sum(expected.values()):
        raise RuntimeError(f"{what} counts are {dict(counts)}, expected {expected}.")


def group_counts(films: list[TBFilm]) -> dict[str, int]:
    counts = Counter(film.group for film in films)
    return {name: int(counts[name]) for name in ("tb", "healthy", "sick")}


def resolve_tbx_image(root: Path, rel: str) -> Path | None:
    relative = Path(rel)
    candidates = [root / relative]
    if not rel.startswith("imgs/"):
        candidates.append(root / "imgs" / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_tbx_split(root: Path, split: str) -> list[TBFilm]:
    list_path = official_list_path(root, split)
    if not list_path.is_file():
        raise FileNotFoundError(
            f"Expected {list_path}. This experiment reads TBX11K_train.txt and TBX11K_val.txt only."
        )
    rows = parse_list_text(list_path.read_text(encoding="utf-8-sig"))
    require_counts([group for _, group, _ in rows], _EXPECTED[split], list_path.name)
    missing: list[str] = []
    films: list[TBFilm] = []
    for rel, group, label in rows:
        found = resolve_tbx_image(root, rel)
        if found is None:
            missing.append(rel)
            continue
        films.append(TBFilm(rel, found, group, label))
    if missing:
        raise MissingFilmsError(missing, root)
    return films


def split_holdout(films: list[TBFilm], seed: int = SPLIT_SEED, frac: float = 0.1):
    groups = [film.group for film in films]
    indices = np.arange(len(films))
    fit_index, hold_index = train_test_split(
        indices,
        test_size=frac,
        random_state=seed,
        shuffle=True,
        stratify=groups,
    )
    fit = [films[int(index)] for index in fit_index]
    hold = [films[int(index)] for index in hold_index]
    return fit, hold


def freeze_holdout(films: list[TBFilm], path: Path, seed: int = SPLIT_SEED):
    """Load the written 10% holdout, or create it and write it."""
    by_rel: dict[str, TBFilm] = {}
    for film in films:
        if film.rel_path in by_rel:
            raise ValueError(f"Duplicate list path: {film.rel_path}")
        by_rel[film.rel_path] = film
    if path.is_file():
        saved = read_json(path)
        if int(saved["seed"]) != seed:
            raise RuntimeError(f"{path} was built with seed {saved['seed']}, not {seed}.")
        hold = [_take(by_rel, rel) for rel in saved["holdout"]]
        fit = [_take(by_rel, rel) for rel in saved["fit"]]
        used = {film.rel_path for film in fit + hold}
        if used != set(by_rel) or len(fit) + len(hold) != len(films):
            raise RuntimeError(f"Frozen holdout {path} does not match the current train list.")
    else:
        fit, hold = split_holdout(films, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            path,
            {
                "seed": seed,
                "fit": [film.rel_path for film in fit],
                "holdout": [film.rel_path for film in hold],
            },
        )
    _assert_official_holdout(fit, hold)
    return fit, hold


def _take(by_rel: dict[str, TBFilm], rel: str) -> TBFilm:
    try:
        return by_rel[rel]
    except KeyError as exc:
        raise RuntimeError(f"Frozen holdout path is not in the train list: {rel}") from exc


def _assert_official_holdout(fit: list[TBFilm], hold: list[TBFilm]) -> None:
    total = group_counts(fit + hold)
    if total != TBX_TRAIN_COUNTS:
        return
    expected_fit = {
        name: TBX_TRAIN_COUNTS[name] - TBX_HOLDOUT_COUNTS[name] for name in TBX_TRAIN_COUNTS
    }
    if group_counts(hold) != TBX_HOLDOUT_COUNTS or group_counts(fit) != expected_fit:
        raise RuntimeError(
            "The frozen 10% split is not 60 TB, 300 healthy, and 300 sick. "
            f"Holdout counts: {group_counts(hold)}."
        )
