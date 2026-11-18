"""The epoch loop stops on the selection score and restores the best weights."""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from tb_transfer.engine import score_paths, train_run
from tb_transfer.metrics import average_precision


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, 1)

    def forward(self, images):
        return self.fc(images.flatten(1))


class EngineTests(unittest.TestCase):
    def test_patience_one_stops_when_the_score_does_not_rise(self):
        info = self._run(patience=1, max_epochs=5)
        self.assertEqual(info["epochs_ran"], 2)
        self.assertEqual(info["epoch"], 1)

    def test_patience_three_allows_three_misses(self):
        info = self._run(patience=3, max_epochs=20)
        self.assertEqual(info["epochs_ran"], 4)
        self.assertEqual(info["epoch"], 1)

    def test_checkpoint_matches_a_fresh_selection_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, labels = _films(root)
            model = Tiny()
            checkpoint = root / "tiny.pt"
            info, restored = train_run(
                model,
                paths,
                labels,
                paths,
                labels,
                lr=0.0,
                patience=1,
                max_epochs=2,
                batch_size=2,
                seed=0,
                device=torch.device("cpu"),
                checkpoint_path=checkpoint,
                run_name="tiny",
            )
            got_labels, scores = score_paths(
                restored, paths, labels, batch_size=2, device=torch.device("cpu")
            )
        self.assertAlmostEqual(average_precision(got_labels, scores), info["selection_auprc"])

    def _run(self, patience: int, max_epochs: int) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, labels = _films(root)
            info, _model = train_run(
                Tiny(),
                paths,
                labels,
                paths,
                labels,
                lr=0.0,
                patience=patience,
                max_epochs=max_epochs,
                batch_size=2,
                seed=0,
                device=torch.device("cpu"),
                checkpoint_path=root / "tiny.pt",
                run_name="tiny",
            )
        return info


def _films(root: Path):
    images = []
    for name, color in (("neg.png", 20), ("pos.png", 220)):
        path = root / name
        Image.new("RGB", (8, 8), color).save(path)
        images.append(path)
    return images, [0.0, 1.0]


if __name__ == "__main__":
    unittest.main()
