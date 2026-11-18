"""Backbones, the new head, and Stage A weight loading."""

import tempfile
import unittest
from pathlib import Path

import torch

from tb_transfer.config import ARCHS
from tb_transfer.io_utils import atomic_torch_save
from tb_transfer.models import build_model, load_stage_a_backbone
from tb_transfer.seed import set_seed


class ModelTests(unittest.TestCase):
    def test_each_architecture_has_one_logit(self):
        for name in ARCHS:
            model = build_model(name, pretrained=False)
            output = model(torch.zeros(1, 3, 224, 224))
            self.assertEqual(tuple(output.shape), (1, 1), name)
            del model

    def test_same_seed_rebuilds_the_same_head(self):
        set_seed(0)
        first = build_model("resnet50", pretrained=False)
        set_seed(0)
        second = build_model("resnet50", pretrained=False)
        set_seed(1)
        third = build_model("resnet50", pretrained=False)
        self.assertTrue(torch.equal(first.fc.weight, second.fc.weight))
        self.assertFalse(torch.equal(first.fc.weight, third.fc.weight))

    def test_stage_a_load_keeps_the_new_head(self):
        torch.manual_seed(0)
        model = build_model("resnet50", pretrained=False)
        head = model.fc.weight.detach().clone()
        torch.manual_seed(1)
        donor = build_model("resnet50", pretrained=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "donor.pt"
            atomic_torch_save({"model": donor.state_dict()}, path)
            load_stage_a_backbone(model, path, "resnet50")
        self.assertTrue(torch.equal(model.fc.weight, head))
        self.assertFalse(torch.equal(model.fc.weight, donor.fc.weight))
        self.assertTrue(torch.equal(model.conv1.weight, donor.conv1.weight))


if __name__ == "__main__":
    unittest.main()
