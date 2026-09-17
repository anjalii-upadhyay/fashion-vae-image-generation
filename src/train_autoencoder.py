"""Train the baseline Autoencoder and plot reconstructions vs originals."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from autoencoder import Autoencoder
from data import get_dataloaders

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def train(epochs=10, latent_dim=32, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(batch_size=256)

    model = Autoencoder(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            recon, _ = model(images)
            loss = criterion(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        history.append(avg_loss)
        print(f"epoch {epoch}/{epochs}  train MSE: {avg_loss:.5f}")

    return model, history, test_loader, device


def plot_reconstructions(model, test_loader, device, n=8):
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:n].to(device)
    with torch.no_grad():
        recon, _ = model(images)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 4))
    for i in range(n):
        axes[0, i].imshow(images[i, 0].cpu(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0].cpu(), cmap="gray")
        axes[1, i].axis("off")

    fig.text(0.02, 0.7, "original", va="center", rotation="vertical", fontsize=10)
    fig.text(0.02, 0.28, "reconstructed", va="center", rotation="vertical", fontsize=10)
    fig.suptitle("Autoencoder: original vs reconstructed")
    fig.subplots_adjust(hspace=0.15, top=0.85, left=0.1)
    save_path = OUTPUTS_DIR / "ae_reconstructions.png"
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def plot_loss_curve(history):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title("Autoencoder training loss")
    plt.tight_layout()
    save_path = OUTPUTS_DIR / "ae_loss_curve.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    model, history, test_loader, device = train(epochs=10)
    plot_loss_curve(history)
    plot_reconstructions(model, test_loader, device)
    torch.save(model.state_dict(), OUTPUTS_DIR / "ae_weights.pt")
    print(f"Saved model weights to {OUTPUTS_DIR / 'ae_weights.pt'}")
