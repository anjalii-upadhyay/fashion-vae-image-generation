"""Conditional VAE — generate a specific clothing category on demand
(e.g. 'new sneakers' vs 'new shirts'), matching the digital-design use case
in the project brief. Same backbone as the plain VAE, but both encoder and
decoder are additionally conditioned on the one-hot class label."""

import torch
import torch.nn as nn

NUM_CLASSES = 10


class CVAEEncoder(nn.Module):
    def __init__(self, latent_dim=32, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7 + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7 + num_classes, latent_dim)

    def forward(self, x, y_onehot):
        h = self.conv(x).flatten(1)
        h = torch.cat([h, y_onehot], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class CVAEDecoder(nn.Module):
    def __init__(self, latent_dim=32, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(latent_dim + num_classes, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 14->28
            nn.Sigmoid(),
        )

    def forward(self, z, y_onehot):
        h = torch.cat([z, y_onehot], dim=1)
        h = self.fc(h).view(-1, 64, 7, 7)
        return self.deconv(h)


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=32, num_classes=NUM_CLASSES):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = CVAEEncoder(latent_dim, num_classes)
        self.decoder = CVAEDecoder(latent_dim, num_classes)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y_onehot):
        mu, logvar = self.encoder(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, y_onehot)
        return recon, mu, logvar

    def sample(self, class_idx, n, device):
        """Generate n new images of a specific clothing category on demand."""
        y_onehot = torch.zeros(n, self.num_classes, device=device)
        y_onehot[:, class_idx] = 1.0
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z, y_onehot)
