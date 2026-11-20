"""Train the VAE and compare it against the baseline Autoencoder."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from autoencoder import Autoencoder
from data import get_dataloaders
from vae import VAE, vae_loss

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def train(epochs=15, latent_dim=32, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(batch_size=256)

    model = VAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"total": [], "recon": [], "kld": []}
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"total": 0.0, "recon": 0.0, "kld": 0.0}
        for images, _ in train_loader:
            images = images.to(device)
            recon, mu, logvar = model(images)
            loss, recon_loss, kld = vae_loss(recon, images, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals["total"] += loss.item() * images.size(0)
            totals["recon"] += recon_loss.item() * images.size(0)
            totals["kld"] += kld.item() * images.size(0)

        n = len(train_loader.dataset)
        for k in history:
            history[k].append(totals[k] / n)
        print(
            f"epoch {epoch}/{epochs}  total: {history['total'][-1]:.3f}  "
            f"recon: {history['recon'][-1]:.3f}  kld: {history['kld'][-1]:.3f}"
        )

    return model, history, test_loader, device


def plot_loss_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in zip(
        axes, ["total", "recon", "kld"],
        ["Total loss (ELBO)", "Reconstruction loss", "KL divergence"],
    ):
        ax.plot(range(1, len(history[key]) + 1), history[key], marker="o")
        ax.set_xlabel("epoch")
        ax.set_title(title)
    fig.tight_layout()
    save_path = OUTPUTS_DIR / "vae_loss_curves.png"
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def plot_ae_vs_vae_reconstructions(vae_model, test_loader, device, n=8):
    ae_model = Autoencoder(latent_dim=32).to(device)
    ae_model.load_state_dict(torch.load(OUTPUTS_DIR / "ae_weights.pt", map_location=device))
    ae_model.eval()
    vae_model.eval()

    images, _ = next(iter(test_loader))
    images = images[:n].to(device)
    with torch.no_grad():
        ae_recon, _ = ae_model(images)
        vae_recon, _, _ = vae_model(images)

    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 5.5))
    rows = [images, ae_recon, vae_recon]
    row_labels = ["original", "AE recon", "VAE recon"]
    for r, row_images in enumerate(rows):
        for i in range(n):
            axes[r, i].imshow(row_images[i, 0].cpu(), cmap="gray")
            axes[r, i].axis("off")
        fig.text(0.02, 0.85 - r * 0.3, row_labels[r], va="center", rotation="vertical", fontsize=10)

    fig.suptitle("AE vs VAE reconstructions")
    fig.subplots_adjust(hspace=0.15, top=0.88, left=0.1)
    save_path = OUTPUTS_DIR / "ae_vs_vae_reconstructions.png"
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    model, history, test_loader, device = train(epochs=15)
    plot_loss_curves(history)
    plot_ae_vs_vae_reconstructions(model, test_loader, device)
    torch.save(model.state_dict(), OUTPUTS_DIR / "vae_weights.pt")
    print(f"Saved model weights to {OUTPUTS_DIR / 'vae_weights.pt'}")
