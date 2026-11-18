"""CheXpert filter, patient split, and path resolution."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from tb_transfer.config import DEFAULT_CHEXPERT_ROOT
from tb_transfer.data_chexpert import (
    MissingFilmsError,
    attach_files,
    freeze_patient_split,
    load_filtered,
    resolve_image,
    split_patients,
)


def _csv(rows: list[tuple[str, str, str]]) -> str:
    lines = ["Path,Frontal/Lateral,Pneumonia"]
    lines.extend(f"{path},{view},{label}" for path, view, label in rows)
    return "\n".join(lines) + "\n"


class CheXpertTests(unittest.TestCase):
    def test_filter_keeps_explicit_frontal_labels(self):
        rows = [
            ("CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg", "Frontal", "1.0"),
            ("CheXpert-v1.0/train/patient00001/study1/view1_lateral.jpg", "Lateral", "1.0"),
            ("CheXpert-v1.0/train/patient00002/study1/view1_frontal.jpg", "Frontal", "0.0"),
            ("CheXpert-v1.0/train/patient00003/study1/view1_frontal.jpg", "Frontal", "-1.0"),
            ("CheXpert-v1.0/train/patient00004/study1/view1_frontal.jpg", "Frontal", ""),
            ("CheXpert-v1.0/train/patient00005/study1/view2_frontal.jpg", "Frontal", "1.0"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.csv").write_text(_csv(rows), encoding="utf-8")
            filtered = load_filtered(root)
        self.assertEqual(sorted(filtered["patient_id"]), ["patient00001", "patient00002", "patient00005"])
        labels = dict(zip(filtered["patient_id"], filtered["label"]))
        self.assertEqual(labels["patient00001"], 1)
        self.assertEqual(labels["patient00002"], 0)
        self.assertEqual(labels["patient00005"], 1)

    def test_missing_pneumonia_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.csv").write_text("Path,Frontal/Lateral\na,Frontal\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_filtered(root)

    def test_patient_images_stay_together_and_split_is_stable(self):
        records = []
        for index in range(20):
            patient = f"patient{index:05d}"
            label = "1.0" if index % 2 == 0 else "0.0"
            records.append((f"CheXpert-v1.0/train/{patient}/study1/view1_frontal.jpg", "Frontal", label))
            if index == 0:
                records.append((f"CheXpert-v1.0/train/{patient}/study2/view1_frontal.jpg", "Frontal", label))
        frame = load_filtered_from_rows(records)
        fit_a, sel_a, _, _ = split_patients(frame, seed=0)
        fit_b, sel_b, _, _ = split_patients(frame, seed=0)
        self.assertEqual(set(fit_a["patient_id"]), set(fit_b["patient_id"]))
        self.assertTrue(set(fit_a["patient_id"]).isdisjoint(sel_a["patient_id"]))
        self.assertEqual(
            set(fit_a["patient_id"]) | set(sel_a["patient_id"]),
            set(frame["patient_id"]),
        )
        patient0_sides = []
        if "patient00000" in set(fit_a["patient_id"]):
            patient0_sides.append("fit")
            self.assertEqual(int((fit_a["patient_id"] == "patient00000").sum()), 2)
        if "patient00000" in set(sel_a["patient_id"]):
            patient0_sides.append("sel")
            self.assertEqual(int((sel_a["patient_id"] == "patient00000").sum()), 2)
        self.assertEqual(patient0_sides, ["fit"] if patient0_sides == ["fit"] else ["sel"])
        self.assertEqual(len(patient0_sides), 1)
        self.assertNotEqual(set(fit_a["patient_id"]), set(split_patients(frame, seed=1)[0]["patient_id"]))

    def test_frozen_split_is_reloaded(self):
        frame = load_filtered_from_rows(
            [
                (f"CheXpert-v1.0/train/patient{index:05d}/study1/view1_frontal.jpg", "Frontal", "1.0" if index < 2 else "0.0")
                for index in range(4)
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(
                '{"seed": 0, "fit_patients": ["patient00000", "patient00001"], '
                '"sel_patients": ["patient00002", "patient00003"]}\n',
                encoding="utf-8",
            )
            fit, sel = freeze_patient_split(frame, path, seed=0)
        self.assertEqual(set(fit["patient_id"]), {"patient00000", "patient00001"})
        self.assertEqual(set(sel["patient_id"]), {"patient00002", "patient00003"})

    def test_resolves_csv_prefix_to_local_train_folder(self):
        csv_path = "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "train" / "patient00001" / "study1" / "view1_frontal.jpg"
            image.parent.mkdir(parents=True)
            Image.new("L", (8, 8), 40).save(image)
            self.assertEqual(resolve_image(root, csv_path), image)
            frame = pd.DataFrame(
                {"Path": [csv_path, "CheXpert-v1.0/train/patient00009/study1/view1_frontal.jpg"],
                 "patient_id": ["patient00001", "patient00009"],
                 "label": [1, 0]}
            )
            with self.assertRaises(MissingFilmsError) as caught:
                attach_files(frame, root)
            self.assertEqual(len(caught.exception.missing), 1)

    def test_real_train_csv_when_present(self):
        csv_path = DEFAULT_CHEXPERT_ROOT / "train.csv"
        if not csv_path.is_file():
            self.skipTest("CheXpert train.csv is not on disk yet")
        filtered = load_filtered(DEFAULT_CHEXPERT_ROOT)
        self.assertTrue(set(filtered["label"].unique()) <= {0, 1})
        self.assertGreater(filtered["patient_id"].nunique(), 100)
        with tempfile.TemporaryDirectory() as tmp:
            fit, sel = freeze_patient_split(filtered, Path(tmp) / "split.json", seed=0)
        self.assertTrue(set(fit["patient_id"]).isdisjoint(sel["patient_id"]))
        self.assertEqual(
            set(fit["patient_id"]) | set(sel["patient_id"]),
            set(filtered["patient_id"]),
        )
        self.assertLess(sel["patient_id"].nunique(), fit["patient_id"].nunique())
        self.assertGreater(int((filtered["label"] == 1).sum()), 0)
        self.assertGreater(int((filtered["label"] == 0).sum()), 0)


def load_filtered_from_rows(rows: list[tuple[str, str, str]]):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train.csv").write_text(_csv(rows), encoding="utf-8")
        return load_filtered(root)


if __name__ == "__main__":
    unittest.main()
