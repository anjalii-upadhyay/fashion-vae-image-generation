"""Latent-space analysis: project the AE's latent vectors to 2D and color by
clothing category, to verify the encoder learned meaningful visual structure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from autoencoder import Autoencoder
from data import CLASS_NAMES, get_dataloaders

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def extract_latents(model, loader, device, max_samples=3000):
    model.eval()
    latents, labels = [], []
    n = 0
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            z = model.encoder(images)
            latents.append(z.cpu().numpy())
            labels.append(batch_labels.numpy())
            n += images.size(0)
            if n >= max_samples:
                break
    return np.concatenate(latents)[:max_samples], np.concatenate(labels)[:max_samples]


def plot_projection(coords, labels, title, save_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7
    )
    handles, _ = scatter.legend_elements(num=10)
    ax.legend(handles, CLASS_NAMES, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    save_path = OUTPUTS_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Autoencoder(latent_dim=32).to(device)
    model.load_state_dict(torch.load(OUTPUTS_DIR / "ae_weights.pt", map_location=device))

    _, test_loader = get_dataloaders(batch_size=256)
    latents, labels = extract_latents(model, test_loader, device, max_samples=3000)
    print(f"Extracted {latents.shape[0]} latent vectors of dim {latents.shape[1]}")

    pca_coords = PCA(n_components=2).fit_transform(latents)
    plot_projection(pca_coords, labels, "AE latent space (PCA)", "ae_latent_pca.png")

    tsne_coords = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(latents)
    plot_projection(tsne_coords, labels, "AE latent space (t-SNE)", "ae_latent_tsne.png")
