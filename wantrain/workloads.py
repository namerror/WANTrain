from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


@dataclass(frozen=True)
class Workload:
    model: nn.Module
    train_loader: DataLoader
    validation_loader: DataLoader
    num_classes: int


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))


def build_workload(config: dict[str, Any], seed: int) -> Workload:
    name = str(config["name"])
    if name == "synthetic_classification":
        return _build_synthetic_workload(config, seed)
    if name == "cifar10_resnet18":
        return _build_cifar10_workload(config, seed)
    raise ValueError(f"unknown workload: {name}")


def _build_synthetic_workload(config: dict[str, Any], seed: int) -> Workload:
    generator = torch.Generator().manual_seed(seed)
    num_examples = int(config.get("num_examples", 256))
    input_dim = int(config.get("input_dim", 16))
    num_classes = int(config.get("num_classes", 4))

    features = torch.randn(num_examples, input_dim, generator=generator)
    weights = torch.randn(input_dim, num_classes, generator=generator)
    labels = torch.argmax(features @ weights, dim=1)
    dataset = TensorDataset(features, labels)
    train_dataset, validation_dataset = split_dataset(
        dataset, validation_split=float(config["validation_split"]), seed=seed
    )
    return Workload(
        model=TinyClassifier(input_dim=input_dim, num_classes=num_classes),
        train_loader=_loader(train_dataset, config, shuffle=True, seed=seed),
        validation_loader=_loader(validation_dataset, config, shuffle=False, seed=seed),
        num_classes=num_classes,
    )


def _build_cifar10_workload(config: dict[str, Any], seed: int) -> Workload:
    from torchvision import datasets, models, transforms

    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    data_dir = str(config["data_dir"])
    download = bool(config["download"])
    train_full = datasets.CIFAR10(data_dir, train=True, download=download, transform=transform)
    validation_full = datasets.CIFAR10(
        data_dir, train=True, download=False, transform=validation_transform
    )
    train_subset, _unused_augmented_validation = split_dataset(
        train_full, validation_split=float(config["validation_split"]), seed=seed
    )
    _unused_train, validation_indices = split_dataset(
        validation_full, validation_split=float(config["validation_split"]), seed=seed
    )

    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return Workload(
        model=model,
        train_loader=_loader(train_subset, config, shuffle=True, seed=seed),
        validation_loader=_loader(validation_indices, config, shuffle=False, seed=seed),
        num_classes=10,
    )


def split_dataset(
    dataset: Dataset[Any], *, validation_split: float, seed: int
) -> tuple[Dataset[Any], Dataset[Any]]:
    validation_size = max(1, int(len(dataset) * validation_split))
    train_size = len(dataset) - validation_size
    if train_size < 1:
        raise ValueError("dataset is too small for the requested validation split")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, validation_size], generator=generator)


def _loader(dataset: Dataset[Any], config: dict[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config.get("pin_memory", False)),
        generator=generator,
    )
