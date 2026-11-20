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

## 2026-11-20 — Latent space analysis: both PCA and t-SNE, 3000 test samples

**Options considered:** PCA only, t-SNE only, or both.

**Why both:** PCA is linear and fast, giving an honest (if coarser) view of
global structure; t-SNE is non-linear and better at revealing local cluster
structure but can distort global distances. Showing both avoids
over-trusting either one's artifacts, and is cheap to generate.

**Why 3000 samples (not the full 10k test set):** t-SNE is O(n^2)-ish and
slow beyond a few thousand points; 3000 is enough to see clear clustering
while keeping the plot fast to generate and readable (no overplotting).

**Result:** Both projections show real structure — Trousers, Bags, and the
footwear group (Sneaker/Sandal/Ankle boot) form clearly separated clusters
in both PCA and t-SNE. Upper-body garments (Pullover, Coat, Shirt, T-shirt)
overlap heavily, which matches how visually similar those classes are at
28x28 resolution rather than indicating an encoder problem. This confirms
the AE's 32-dim latent space captures meaningful visual characteristics,
validating it as the representation-learning half of the brief before
moving to the VAE.

---

## 2026-11-20 — VAE architecture, loss formulation, and training

**Architecture:** Same conv encoder/decoder backbone as the AE (so the
comparison is fair), but the encoder now outputs `mu` and `logvar` instead
of a single vector, with the reparameterization trick
`z = mu + eps * std` making sampling differentiable. Same `latent_dim=32`
as the AE for direct comparability.

**Loss formulation:** `reconstruction (MSE, sum reduction) + KL divergence
to N(0, I)`, both divided by batch size (standard ELBO form, beta=1 — no
beta-VAE weighting for this baseline). Sum reduction (not mean) is
necessary so the reconstruction term and KL term are on comparable scales;
mean reduction would make KL dominate and collapse the latent space.

**Why MSE again (not BCE):** Consistency with the AE's loss choice (see
earlier entry) — MSE corresponds to a Gaussian decoder assumption, which is
simpler to reason about than treating pixel intensities as Bernoulli
probabilities (BCE's implicit assumption).

**Epochs:** Trained for 15 (vs. the AE's 10) since the VAE optimizes a
harder joint objective (reconstruction + KL) and benefits from a bit more
time to balance the two terms.

**Result:** Total ELBO loss: 48.4 -> 25.7 over 15 epochs, with both the
reconstruction and KL terms decreasing smoothly (no collapse — KL settled
around 8.5, not 0, meaning the latent space is actually being used, not
ignored). See `outputs/vae_loss_curves.png`.

**AE vs VAE comparison** (`outputs/ae_vs_vae_reconstructions.png`): VAE
reconstructions are visibly slightly blurrier than the AE's (e.g. the "Lee"
logo text and plaid shirt pattern lose detail) — this is the expected
trade-off for a smooth, continuous, samplable latent space instead of pure
reconstruction fidelity, and is exactly what makes sampling new designs
from the VAE's prior work in the next step.

---

## 2026-11-20 — Generation: prior sampling + linear interpolation in mu-space

**Random sampling:** `z ~ N(0, I)` fed directly to the decoder (no encoder
involved) — this is the actual "generate a new design never seen in
training" capability the business brief asks for. Implemented as
`VAE.sample()` on the model itself since it only needs the decoder + the
known prior shape.

**Interpolation:** Linear interpolation between the `mu` (not a sampled
`z`) of two *different-class* test images, decoded at each step. Using
`mu` instead of a noisy sample gives a cleaner, more deterministic
interpolation path for the figure. Picking two different classes (rather
than two random images, which might land in the same class) makes the
transition visually informative.

**Result:** `outputs/vae_generated_samples.png` — 16 freshly sampled
designs; several are clearly recognizable garment types (shoes, trousers,
sweaters, boots) with some blobbier/ambiguous ones, which is expected for a
simple (non-conditional) VAE at 15 epochs on 28x28 grayscale.
`outputs/vae_latent_interpolation.png` — a 10-step interpolation from an
ankle boot to a pullover shows a smooth, gradual morph with no jarring
jumps, confirming the latent space is continuous and well-structured
(exactly what a plain AE's latent space would not reliably give us).

---

<!-- New entries go above this line. Append, don't rewrite history. -->
