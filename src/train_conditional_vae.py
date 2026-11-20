"""Train the Conditional VAE and demonstrate on-demand, per-category generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from conditional_vae import NUM_CLASSES, ConditionalVAE
from data import CLASS_NAMES, get_dataloaders
from vae import vae_loss

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def train(epochs=15, latent_dim=32, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(batch_size=256)

    model = ConditionalVAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"total": [], "recon": [], "kld": []}
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"total": 0.0, "recon": 0.0, "kld": 0.0}
        for images, labels in train_loader:
            images = images.to(device)
            y_onehot = F.one_hot(labels, NUM_CLASSES).float().to(device)

            recon, mu, logvar = model(images, y_onehot)
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

    return model, history, device


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
    save_path = OUTPUTS_DIR / "cvae_loss_curves.png"
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def plot_class_conditioned_samples(model, device, classes, n_per_class=6):
    model.eval()
    fig, axes = plt.subplots(
        len(classes), n_per_class,
        figsize=(n_per_class * 1.4, len(classes) * 1.6),
    )
    with torch.no_grad():
        for r, class_idx in enumerate(classes):
            images = model.sample(class_idx, n_per_class, device)
            for c in range(n_per_class):
                axes[r, c].imshow(images[c, 0].cpu(), cmap="gray")
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
            axes[r, 0].set_ylabel(CLASS_NAMES[class_idx], rotation=0, ha="right", va="center", fontsize=9)

    fig.suptitle("Conditional VAE: generate a specific clothing category on demand")
    fig.subplots_adjust(top=0.9, left=0.15, hspace=0.3, wspace=0.1)
    save_path = OUTPUTS_DIR / "cvae_class_conditioned_samples.png"
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    model, history, device = train(epochs=15)
    plot_loss_curves(history)

    # "new sneakers" vs "new shirts" vs a couple more, per the brief's example
    demo_classes = [7, 6, 8, 1]  # Sneaker, Shirt, Bag, Trouser
    plot_class_conditioned_samples(model, device, demo_classes)

    torch.save(model.state_dict(), OUTPUTS_DIR / "cvae_weights.pt")
    print(f"Saved model weights to {OUTPUTS_DIR / 'cvae_weights.pt'}")
