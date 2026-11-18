"""Command-line help does not touch the datasets."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class HelpTests(unittest.TestCase):
    def test_stage_help(self):
        for module in ("tb_transfer.train_a", "tb_transfer.train_b"):
            result = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
