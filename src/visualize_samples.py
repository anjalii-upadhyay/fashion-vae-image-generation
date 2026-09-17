"""Visualize a grid of sample Fashion-MNIST images with their labels."""

import matplotlib.pyplot as plt

from data import CLASS_NAMES, get_dataloaders


def main():
    train_loader, _ = get_dataloaders(batch_size=16)
    images, labels = next(iter(train_loader))

    fig, axes = plt.subplots(2, 8, figsize=(14, 4.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0], cmap="gray")
        ax.set_title(CLASS_NAMES[labels[i]], fontsize=9)
        ax.axis("off")

    fig.suptitle("Fashion-MNIST samples")
    fig.subplots_adjust(hspace=0.5, top=0.85)
    fig.savefig("../outputs/sample_images.png", dpi=150)
    print("Saved outputs/sample_images.png")


if __name__ == "__main__":
    main()
