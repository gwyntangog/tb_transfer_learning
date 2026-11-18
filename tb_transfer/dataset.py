"""Load a chest film as a 3-channel 224x224 ImageNet tensor."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from tb_transfer.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

# Full-resolution CheXpert films can exceed Pillow's default pixel cap.
Image.MAX_IMAGE_PIXELS = None

_MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)


def load_image(path: Path) -> torch.Tensor:
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            array = np.array(rgb, dtype=np.float32, copy=True) / 255.0
    except OSError as exc:
        raise OSError(f"Could not read {path}") from exc
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - _MEAN) / _STD


class FilmDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[float]):
        if len(paths) != len(labels):
            raise ValueError("paths and labels differ in length.")
        self.paths = list(paths)
        self.labels = [float(label) for label in labels]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = load_image(self.paths[index])
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, label


def make_loader(
    paths: list[Path],
    labels: list[float],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = FilmDataset(paths, labels)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs["generator"] = generator
    if num_workers > 0:
        kwargs["worker_init_fn"] = _seed_worker
    return DataLoader(dataset, **kwargs)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
