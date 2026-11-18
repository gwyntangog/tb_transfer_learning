"""The gitignore has to keep datasets, checkpoints, and result dumps out of git."""

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GitignoreTests(unittest.TestCase):
    def test_data_and_checkpoints_are_ignored(self):
        ignored = [
            "data/train.csv",
            "data/imgs/tb/a.png",
            "outputs/stage_a/resnet50/best.pt",
            "outputs/stage_b_metrics.csv",
            "outputs/stage_b/tbx11k_holdout_seed0.json",
            "all_train.txt",
            "all_val.txt",
            "train.csv",
            "TBX11K/imgs/tb/x.png",
            "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg",
            "fetch_frontal_pneumonia.py",
            ".env",
        ]
        tracked = [
            "README.md",
            "requirements.txt",
            "tb_transfer/train_a.py",
            "tb_transfer/train_b.py",
            "tests/test_metrics.py",
            ".gitignore",
        ]
        for rel in ignored:
            result = subprocess.run(
                ["git", "check-ignore", "-q", rel],
                cwd=ROOT,
                check=False,
            )
            self.assertEqual(result.returncode, 0, rel)
        for rel in tracked:
            result = subprocess.run(
                ["git", "check-ignore", "-q", rel],
                cwd=ROOT,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, rel)


if __name__ == "__main__":
    unittest.main()
