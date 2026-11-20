"""Step 8 — generation: sample brand-new clothing designs from the VAE's
prior, and interpolate between two real items in latent space."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import get_dataloaders
from vae import VAE

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def load_vae(latent_dim=32, device="cpu"):
    model = VAE(latent_dim).to(device)
    model.load_state_dict(torch.load(OUTPUTS_DIR / "vae_weights.pt", map_location=device))
    model.eval()
    return model


def plot_random_samples(model, device, n=16, save_name="vae_generated_samples.png"):
    with torch.no_grad():
        images = model.sample(n, device)

    rows, cols = 2, n // 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0].cpu(), cmap="gray")
        ax.axis("off")
    fig.suptitle("VAE: new clothing designs sampled from the latent prior")
    fig.subplots_adjust(top=0.88, hspace=0.1)
    save_path = OUTPUTS_DIR / save_name
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def plot_interpolation(model, test_loader, device, steps=10, save_name="vae_latent_interpolation.png"):
    images, labels = next(iter(test_loader))
    # pick two images from different classes for a visible transition
    idx_a = 0
    idx_b = next(i for i in range(len(labels)) if labels[i] != labels[idx_a])

    img_a = images[idx_a : idx_a + 1].to(device)
    img_b = images[idx_b : idx_b + 1].to(device)

    with torch.no_grad():
        mu_a, _ = model.encoder(img_a)
        mu_b, _ = model.encoder(img_b)

        alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
        z_interp = (1 - alphas) * mu_a + alphas * mu_b
        decoded = model.decoder(z_interp)

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.5, 2))
    for i, ax in enumerate(axes):
        ax.imshow(decoded[i, 0].cpu(), cmap="gray")
        ax.axis("off")
    fig.suptitle("VAE latent-space interpolation between two garments")
    fig.subplots_adjust(top=0.78)
    save_path = OUTPUTS_DIR / save_name
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_vae(device=device)

    plot_random_samples(model, device)

    _, test_loader = get_dataloaders(batch_size=256)
    plot_interpolation(model, test_loader, device)
