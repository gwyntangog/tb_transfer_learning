"""Grayscale films become 3-channel 224 tensors."""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from tb_transfer.config import IMAGENET_MEAN, IMAGENET_STD
from tb_transfer.dataset import load_image


class DatasetTests(unittest.TestCase):
    def test_grayscale_is_copied_to_three_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gray.png"
            Image.new("L", (8, 8), 128).save(path)
            tensor = load_image(path)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        restored = tensor * std + mean
        self.assertTrue(torch.allclose(restored[0], restored[1], atol=1e-5))
        self.assertTrue(torch.allclose(restored[1], restored[2], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
