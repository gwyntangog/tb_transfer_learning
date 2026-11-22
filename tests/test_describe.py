"""Dataset description on a tiny CheXpert CSV and a tiny TBX list summary."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tb_transfer.describe_data import describe_chexpert, list_summary


class DescribeTests(unittest.TestCase):
    def test_chexpert_funnel_counts_uncertain_and_blank(self):
        rows = [
            ("CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg", "Frontal", "1.0", "Female", "40", "PA"),
            ("CheXpert-v1.0/train/patient00002/study1/view1_frontal.jpg", "Frontal", "0.0", "Male", "50", "AP"),
            ("CheXpert-v1.0/train/patient00003/study1/view1_frontal.jpg", "Frontal", "-1.0", "Female", "60", "AP"),
            ("CheXpert-v1.0/train/patient00004/study1/view1_frontal.jpg", "Frontal", "", "Male", "", "PA"),
            ("CheXpert-v1.0/train/patient00001/study1/view1_lateral.jpg", "Lateral", "1.0", "Female", "40", ""),
        ]
        header = "Path,Frontal/Lateral,Pneumonia,Sex,Age,AP/PA,Lung Opacity"
        opacity = ["1.0", "0.0", "-1.0", "", "1.0"]
        lines = [header]
        for row, finding in zip(rows, opacity):
            lines.append(",".join(row) + f",{finding}")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
            image = root / "train" / "patient00001" / "study1" / "view1_frontal.jpg"
            image.parent.mkdir(parents=True)
            Image.new("L", (12, 8)).save(image)
            description = describe_chexpert(root, sizes=True)
        funnel = description["funnel"]
        self.assertEqual(funnel["n_rows"], 5)
        self.assertEqual(funnel["n_frontal"], 4)
        self.assertEqual(funnel["n_lateral"], 1)
        self.assertEqual(funnel["frontal_pneumonia_positive"], 1)
        self.assertEqual(funnel["frontal_pneumonia_negative"], 1)
        self.assertEqual(funnel["frontal_pneumonia_uncertain"], 1)
        self.assertEqual(funnel["frontal_pneumonia_blank"], 1)
        cohort = description["kept_cohort"]
        self.assertEqual(cohort["n_films"], 2)
        self.assertEqual(cohort["n_positive"], 1)
        self.assertEqual(cohort["n_negative"], 1)
        self.assertEqual(cohort["n_on_disk"], 1)
        self.assertEqual(cohort["n_missing"], 1)
        self.assertEqual(description["observations_on_kept_films"]["Pneumonia"]["positive"], 1)
        self.assertEqual(description["observations_on_kept_films"]["Lung Opacity"]["uncertain"], 0)
        self.assertEqual(description["image_sizes"]["n"], 1)
        self.assertEqual(description["image_sizes"]["width_min"], 12)
        self.assertEqual(description["image_sizes"]["height_min"], 8)

    def test_list_summary_counts_prevalence(self):
        rows = [
            ("imgs/tb/a.png", "tb", 1),
            ("imgs/health/b.png", "healthy", 0),
            ("imgs/sick/c.png", "sick", 0),
        ]
        summary = list_summary(rows)
        self.assertEqual(summary["tb"], 1)
        self.assertEqual(summary["n_negative"], 2)
        self.assertAlmostEqual(summary["prevalence"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
