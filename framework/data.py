"""
datasets and dataset building stuff
handles loading datasets, applying transforms, building dataloaders

includes standard datasets and allows for adding more
"""

import torch
import torchvision.transforms.v2 as transforms
from jsonargparse import Namespace
from torchvision import datasets


def build_datasets(configs: Namespace) -> dict:
    """Builds datasets and returns them in a dict"""

    match configs.data.name.lower():
        case "mnist":
            return _build_mnist_datasets(configs)
        case "cifar10":
            return _build_cifar10_datasets(configs)
        case _:
            raise ValueError(f"Dataset {configs.data.name} not supported")

        # import additional datasets and add them here as needed

        # from XXX import _build_xxx_datasets
        # case "xxx":
        #     return _build_xxx_datasets(configs)


def _build_mnist_datasets(configs: Namespace) -> dict:
    """Builds MNIST datasets"""

    transform = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    )

    train_dataset = datasets.MNIST(
        root=configs.data.root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=configs.data.root, train=False, download=True, transform=transform
    )

    return {"train": train_dataset, "test": test_dataset, "num_classes": 10}


def _build_cifar10_datasets(configs: Namespace) -> dict:
    """Builds CIFAR-10 datasets"""

    transform = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    )

    train_dataset = datasets.CIFAR10(
        root=configs.data.root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=configs.data.root, train=False, download=True, transform=transform
    )

    return {"train": train_dataset, "test": test_dataset, "num_classes": 10}
