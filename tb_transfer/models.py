"""ImageNet backbones with a new one-logit head."""

from __future__ import annotations

from pathlib import Path

from torch import nn
from torchvision import models

from tb_transfer.config import ARCHS
from tb_transfer.io_utils import load_checkpoint
from tb_transfer.seed import set_seed

_HEAD_PREFIX = {
    "resnet50": "fc.",
    "densenet121": "classifier.",
    "vit_b_16": "heads.",
}


def build_model(name: str, pretrained: bool) -> nn.Module:
    if name not in ARCHS:
        raise ValueError(f"Unknown architecture {name}.")
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        return model
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, 1)
    return model


def make_model(name: str, seed: int, stage_a_checkpoint: Path | None = None) -> nn.Module:
    """New head every time. A Stage A file replaces the backbone only."""
    set_seed(seed)
    model = build_model(name, pretrained=stage_a_checkpoint is None)
    if stage_a_checkpoint is not None:
        load_stage_a_backbone(model, stage_a_checkpoint, name)
    return model


def load_stage_a_backbone(model: nn.Module, checkpoint: Path, name: str) -> None:
    prefix = _HEAD_PREFIX[name]
    current_head = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(prefix)
    }
    if not current_head:
        raise RuntimeError(f"No head parameters with prefix {prefix!r} on {name}.")
    saved = load_checkpoint(checkpoint)
    donor = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
    filtered = {key: value for key, value in donor.items() if not key.startswith(prefix)}
    incompatible = model.load_state_dict(filtered, strict=False)
    model.load_state_dict(current_head, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith(prefix)]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match {name}. missing={missing} unexpected={unexpected}"
        )
