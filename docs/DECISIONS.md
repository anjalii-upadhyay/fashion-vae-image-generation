# Decision Log

Every non-trivial choice made on this project, with the reasoning behind it.
Updated as soon as a decision is made — not batched at the end.

Format per entry: **Date — Decision** / Options considered / Why this one / Reversibility.

---

## 2026-09-17 — Project topic: Autoencoder + VAE fashion image generation

**Decision:** Build a fashion image generation & representation system using
an Autoencoder (representation) followed by a VAE (generation), per IA
Activity 1, Topic 1.

**Why:** The brief explicitly asks for (a) learning visual characteristics
(representation) and (b) generating new realistic designs. An AE naturally
gives (a) via its latent bottleneck; a VAE's continuous, samplable latent
space is what makes (b) possible — a plain AE's latent space has "holes"
that don't decode to anything realistic.

---

## 2026-09-17 — Framework: PyTorch over TensorFlow/Keras

**Options considered:** PyTorch, TensorFlow/Keras.

**Why PyTorch:** Lighter weight to install and run, more transparent/custom
control over the VAE loss (reparameterization trick + KL term is easier to
write explicitly in PyTorch than to fit into Keras's high-level `fit()`
loop), and it's the framework the user is more comfortable maintaining.
TensorFlow was judged "heavy" for this project's scope — no need for its
production/serving tooling here.

**Reversibility:** Easy to swap later if needed; model logic is simple
enough to port, but not planned unless a hard requirement emerges.

---

## 2026-09-17 — Dataset: Fashion-MNIST first, DeepFashion later

**Options considered:** Fashion-MNIST only, DeepFashion only, Fashion-MNIST
→ DeepFashion staged approach.

**Why Fashion-MNIST first:** 28×28 grayscale, loads directly via
`torchvision.datasets.FashionMNIST` with no manual download/cleaning, small
enough to iterate fast on architecture and training loop bugs. Lets us
validate the AE → VAE pipeline cheaply before investing in a heavier
dataset.

**Why switch to DeepFashion later:** Higher-resolution RGB real fashion
photos give more "industry-level" / realistic generated output, matching
the business scenario (fashion-tech company, marketing prototypes) better
than Fashion-MNIST's simple icons.

**Trigger to switch:** Once the AE and VAE are training correctly and
producing coherent reconstructions/generations on Fashion-MNIST (end of
Step 8 in the project plan), we swap the `Dataset`/`DataLoader` for
DeepFashion and likely deepen the conv architecture for the higher
resolution. Same training loop, same loss functions.

---

## 2026-09-17 — Repository: `fashion-vae-image-generation`, public

**Decision:** New public GitHub repo, separate from the `daa` coursework
repo, named `fashion-vae-image-generation`.

**Why:** Keeps this project's history, issues, and README independent and
shareable (e.g. in a portfolio) rather than mixed into an unrelated repo.
Public so it's easy to link in the course submission.

---

## 2026-09-17 — Baseline AE architecture: conv encoder/decoder, latent_dim=32, MSE loss

**Options considered:** Fully-connected AE vs. convolutional AE; latent
dimension (8 / 16 / 32 / 64); MSE vs. BCE reconstruction loss.

**Why convolutional:** Clothing images have local spatial structure (edges,
textures) that conv layers capture far more efficiently than flattening to a
dense layer — fewer parameters, better reconstructions.

**Why latent_dim=32:** Small enough to force real compression (28×28=784
pixels → 32 numbers, ~24x reduction) so the latent space reflects learned
visual characteristics rather than near-lossless copying, but large enough
for Fashion-MNIST's 10 categories to be separable. Will revisit if the
downstream VAE needs more capacity.

**Why MSE over BCE:** Pixels are continuous grayscale intensities (not
binary), and MSE is simpler to reason about jointly with the VAE's KL term
later. BCE is common for Fashion-MNIST AEs too but was not chosen to keep
the loss function consistent and interpretable across AE and VAE.

**Result:** Train MSE dropped from 0.037 → 0.009 over 10 epochs (smooth,
monotonic — see `outputs/ae_loss_curve.png`). Reconstructions
(`outputs/ae_reconstructions.png`) preserve garment shape/silhouette
correctly; fine texture/text details are blurred, which is expected AE
behavior and motivates the move to a VAE next for cleaner generative
sampling.

---

<!-- New entries go above this line. Append, don't rewrite history. -->
