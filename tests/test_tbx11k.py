"""TBX11K labels, official counts, and the frozen 10% holdout."""

import json
import tempfile
import unittest
from pathlib import Path

from tb_transfer.data_tbx11k import (
    TBFilm,
    classify_rel_path,
    freeze_holdout,
    group_counts,
    official_list_path,
    parse_list_text,
    require_counts,
    split_holdout,
)
from tb_transfer.config import TBX_HOLDOUT_COUNTS, TBX_TRAIN_COUNTS, TBX_VAL_COUNTS


def _films(n_tb: int, n_healthy: int, n_sick: int) -> list[TBFilm]:
    films = []
    for group, count, label in (("tb", n_tb, 1), ("healthy", n_healthy, 0), ("sick", n_sick, 0)):
        for index in range(count):
            rel = f"imgs/{group}/{index:05d}.png"
            films.append(TBFilm(rel, Path(rel), group, label))
    return films


class TBXTests(unittest.TestCase):
    def test_labels_from_directories_and_filenames(self):
        cases = {
            "imgs/tb/tb0001.png": "tb",
            "imgs/tb/active/a.png": "tb",
            "imgs/tb/latent/b.png": "tb",
            "imgs/health/h.png": "healthy",
            "imgs/healthy/h.png": "healthy",
            "imgs/normal/n.png": "healthy",
            "imgs/sick/s.png": "sick",
            "imgs/tb0421.png": "tb",
            "imgs/normal_0001.png": "healthy",
            "imgs/sick_0001.png": "sick",
        }
        for rel, group in cases.items():
            self.assertEqual(classify_rel_path(rel), group, rel)
        with self.assertRaises(ValueError):
            classify_rel_path("imgs/extra/unknown.png")

    def test_extra_column_is_ignored_and_extended_lists_are_rejected(self):
        rows = parse_list_text("imgs/tb/a.png 1\nimgs/sick/b.png 0\n")
        self.assertEqual([group for _, group, _ in rows], ["tb", "sick"])
        with self.assertRaises(ValueError):
            official_list_path(Path("."), "all_train")

    def test_official_counts(self):
        train = (["tb"] * 600) + (["healthy"] * 3000) + (["sick"] * 3000)
        val = (["tb"] * 200) + (["healthy"] * 800) + (["sick"] * 800)
        require_counts(train, TBX_TRAIN_COUNTS, "train")
        require_counts(val, TBX_VAL_COUNTS, "val")
        with self.assertRaises(RuntimeError):
            require_counts((["tb"] * 599) + (["healthy"] * 3000) + (["sick"] * 3000), TBX_TRAIN_COUNTS, "train")

    def test_stratified_holdout_on_the_official_train_size(self):
        fit, hold = split_holdout(_films(600, 3000, 3000), seed=0)
        self.assertEqual(group_counts(hold), TBX_HOLDOUT_COUNTS)
        self.assertEqual(group_counts(fit)["tb"], 540)
        self.assertEqual(len(fit) + len(hold), 6600)

    def test_frozen_file_is_not_resampled(self):
        films = _films(10, 10, 10)
        fit, hold = split_holdout(films, seed=0)
        hold_rels = {film.rel_path for film in hold}
        fit_healthy = next(film for film in fit if film.group == "healthy")
        hold_healthy = next(film for film in hold if film.group == "healthy")
        swapped_hold = (hold_rels - {hold_healthy.rel_path}) | {fit_healthy.rel_path}
        swapped_fit = ({film.rel_path for film in films} - swapped_hold)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "holdout.json"
            path.write_text(
                json.dumps({"seed": 0, "fit": sorted(swapped_fit), "holdout": sorted(swapped_hold)}),
                encoding="utf-8",
            )
            _, loaded = freeze_holdout(films, path, seed=0)
        self.assertEqual({film.rel_path for film in loaded}, swapped_hold)
        self.assertNotEqual({film.rel_path for film in loaded}, hold_rels)

    def test_reader_ignores_all_txt_even_if_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lists = root / "lists"
            lists.mkdir()
            (lists / "all_train.txt").write_text("this would not parse\n", encoding="utf-8")
            (lists / "TBX11K_train.txt").write_text("imgs/tb/a.png\nimgs/sick/b.png\n", encoding="utf-8")
            text = official_list_path(root, "train").read_text(encoding="utf-8")
        self.assertNotIn("would not parse", text)
        self.assertEqual(len(parse_list_text(text)), 2)


if __name__ == "__main__":
    unittest.main()
