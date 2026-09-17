# Fashion VAE Image Generation

Industry-level fashion image generation system using Autoencoder + VAE — learns visual characteristics of clothing images and generates new, realistic-looking clothing designs.

## Project structure

```
fashion-vae-image-generation/
├── README.md
├── requirements.txt
├── data/                  # downloaded datasets (gitignored)
├── src/                   # model + training code
├── notebooks/             # exploration notebooks
├── outputs/                # sample reconstructions / generations (gitignored)
└── docs/
    ├── DECISIONS.md        # why we chose each dataset/tool/design
    └── TROUBLESHOOTING.md  # every mistake hit and how it was fixed
```

## Stack

- **Framework:** PyTorch
- **Dataset (current):** Fashion-MNIST — will switch to DeepFashion once the
  AE/VAE pipeline is validated (see `docs/DECISIONS.md`).

## Setup

```bash
pip install -r requirements.txt
```
