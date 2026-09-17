"""Fashion-MNIST data loading.

Swap point for later: replace `get_dataloaders` with a DeepFashion Dataset
once the AE/VAE pipeline is validated on Fashion-MNIST (see docs/DECISIONS.md).
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def get_dataloaders(data_dir=DEFAULT_DATA_DIR, batch_size=128, num_workers=2):
    transform = transforms.ToTensor()  # scales pixels to [0, 1]

    train_set = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"train batches: {len(train_loader)}, test batches: {len(test_loader)}")
    print(f"batch shape: {tuple(images.shape)}, dtype: {images.dtype}")
    print(f"pixel range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"sample labels: {[CLASS_NAMES[l] for l in labels[:8].tolist()]}")
