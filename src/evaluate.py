"""Step 10 — quantitative evaluation of the AE and VAE on the full
Fashion-MNIST test set (10k images), to complement the qualitative
reconstruction/generation figures from the earlier steps with real
held-out numbers.

Metrics:
  - MSE  : mean squared pixel error, per image, averaged over the test set.
  - SSIM : structural similarity (skimage.metrics.structural_similarity),
           per image, averaged over the test set.

Usage (from inside src/, after both models have been trained):
    python3 evaluate.py
"""

import csv
from pathlib import Path

import torch
from skimage.metrics import structural_similarity as ssim

from autoencoder import Autoencoder
from data import get_dataloaders
from vae import VAE

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


@torch.no_grad()
def evaluate_model(model, test_loader, device, model_kind):
    """Average per-image MSE and SSIM over every image in test_loader.

    model_kind selects how to unpack forward()'s return value:
      "ae"  -> Autoencoder.forward returns (recon, z)
      "vae" -> VAE.forward returns (recon, mu, logvar), using the same
               stochastic reparameterized forward pass used everywhere
               else in this repo (e.g. train_vae.py's
               plot_ae_vs_vae_reconstructions), not a mu-only decode.
    """
    model.eval()
    mse_sum = 0.0
    ssim_sum = 0.0
    n_images = 0

    for images, _ in test_loader:
        images = images.to(device)
        if model_kind == "ae":
            recon, _ = model(images)
        else:
            recon, _, _ = model(images)

        # Per-image MSE: mean over (channel, H, W), then sum across the batch
        # so the running total divides into a true per-image average.
        batch_mse = torch.mean((recon - images) ** 2, dim=[1, 2, 3])
        mse_sum += batch_mse.sum().item()

        images_np = images.cpu().numpy()
        recon_np = recon.cpu().numpy()
        for i in range(images_np.shape[0]):
            ssim_sum += ssim(images_np[i, 0], recon_np[i, 0], data_range=1.0)

        n_images += images_np.shape[0]

    return mse_sum / n_images, ssim_sum / n_images


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader = get_dataloaders(batch_size=256)
    n_test = len(test_loader.dataset)

    ae = Autoencoder(latent_dim=32).to(device)
    ae.load_state_dict(torch.load(OUTPUTS_DIR / "ae_weights.pt", map_location=device))

    vae = VAE(latent_dim=32).to(device)
    vae.load_state_dict(torch.load(OUTPUTS_DIR / "vae_weights.pt", map_location=device))

    print(f"Evaluating on the full test set ({n_test} images)...")

    rows = []
    for name, model, kind in [("AE", ae, "ae"), ("VAE", vae, "vae")]:
        mse, ssim_score = evaluate_model(model, test_loader, device, kind)
        print(f"{name:>4}  MSE: {mse:.5f}   SSIM: {ssim_score:.4f}")
        rows.append({"model": name, "mse": f"{mse:.5f}", "ssim": f"{ssim_score:.4f}"})

    save_path = OUTPUTS_DIR / "eval_metrics.csv"
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "mse", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
