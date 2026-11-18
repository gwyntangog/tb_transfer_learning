"""Read finished run JSON files and write the tables for the writeup."""

from __future__ import annotations

from pathlib import Path

from tb_transfer.config import ARCHS, PROTOCOL, STAGE_A_FIELDS, STAGE_A_SEEDS, STAGE_B_FIELDS
from tb_transfer.io_utils import read_json, write_csv, write_json


def reusable(path: Path, force: bool, required_key: str) -> dict | None:
    if force or not path.is_file():
        return None
    row = read_json(path)
    if row.get("protocol") != PROTOCOL:
        raise RuntimeError(
            f"{path} was written by protocol {row.get('protocol')!r}. "
            f"This code expects {PROTOCOL!r}. Delete the file to retrain."
        )
    if required_key not in row:
        return None
    return row


def choose_best_seed(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: (float(row["selection_auprc"]), -int(row["seed"])))


def mark_selected(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["architecture"], []).append(row)
    winners = set()
    for architecture, group in grouped.items():
        seeds = {int(row["seed"]) for row in group}
        if set(STAGE_A_SEEDS) <= seeds:
            winner = choose_best_seed(group)
            winners.add((architecture, int(winner["seed"])))
    for row in rows:
        row["selected"] = int((row["architecture"], int(row["seed"])) in winners)
    return rows


def publish_best(arch_dir: Path, rows: list[dict]) -> None:
    seeds = {int(row["seed"]) for row in rows}
    if set(STAGE_A_SEEDS) != seeds and not set(STAGE_A_SEEDS) <= seeds:
        print(f"Not writing {arch_dir / 'best.pt'} until seeds 0, 1, and 2 are finished.")
        return
    complete = [row for row in rows if int(row["seed"]) in STAGE_A_SEEDS]
    winner = choose_best_seed(complete)
    source = arch_dir / f"seed_{int(winner['seed'])}.pt"
    target = arch_dir / "best.pt"
    target.write_bytes(source.read_bytes())
    write_json(target.with_suffix(".json"), {**winner, "checkpoint": target.name})
    print(
        f"{winner['architecture']}: keeping seed {int(winner['seed'])} "
        f"(selection AUPRC {float(winner['selection_auprc']):.6f}) -> {target}",
        flush=True,
    )


def refresh_stage_a_csv(output: Path) -> Path:
    rows = [read_json(path) for path in sorted((output / "stage_a").glob("*/seed_*.json"))]
    rows = _sort_rows(rows)
    mark_selected(rows)
    destination = output / "stage_a_metrics.csv"
    write_csv(destination, rows, STAGE_A_FIELDS)
    return destination


def refresh_stage_b_csv(output: Path) -> Path:
    rows = [read_json(path) for path in sorted((output / "stage_b").glob("*__*.json"))]
    rows = _sort_rows(rows)
    destination = output / "stage_b_metrics.csv"
    write_csv(destination, rows, STAGE_B_FIELDS)
    return destination


def _sort_rows(rows: list[dict]) -> list[dict]:
    order = {name: index for index, name in enumerate(ARCHS)}
    return sorted(
        rows,
        key=lambda row: (
            order.get(row["architecture"], 99),
            str(row.get("init", "")),
            int(row["seed"]),
        ),
    )
