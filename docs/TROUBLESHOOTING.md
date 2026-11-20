# Troubleshooting Log

Every mistake, blocker, and error hit on this project, with root cause and
the fix. Updated the moment a problem is resolved — not batched.

Format per entry: **Date — Problem** / Symptom / Root cause / Fix.

---

## 2026-09-17 — GitHub App couldn't create the repository

**Symptom:** Calling the repo-creation tool returned
`403 Resource not accessible by integration` when trying to create
`fashion-vae-image-generation` directly from the assistant.

**Root cause:** The Claude GitHub App installation didn't have
account-level "create repository" permission — it's normally scoped to
push/read on repos it's explicitly given access to, not open-ended repo
creation.

**Fix:** Created the repo manually via github.com/new, then attached it to
the session so the assistant could clone/push to it.

---

## 2026-09-17 — Push to the new repo was refused (403)

**Symptom:** `git push -u origin main` failed with:
`remote: Claude doesn't have GitHub access to
anjalii-upadhyay/fashion-vae-image-generation for your organization` /
`403`.

**Root cause:** The Claude GitHub App's repository access list didn't
automatically include the newly created repo — GitHub Apps only get access
to repos they're explicitly granted ("Only select repositories") unless
"All repositories" is chosen.

**Fix:** Added `fashion-vae-image-generation` to the Claude GitHub App's
repository access at
`github.com/apps/claude/installations/select_target`. Retried the push —
succeeded (verified with `git ls-remote`).

---

## 2026-09-17 — Fashion-MNIST re-downloaded into the wrong folder

**Symptom:** Running `src/visualize_samples.py` from inside `src/` created a
second copy of the dataset at `src/data/FashionMNIST` instead of reusing the
one already at `data/FashionMNIST`.

**Root cause:** `get_dataloaders()` used a relative default path
(`data_dir="data"`), which resolves against the current working directory,
not the project root — so running scripts from different folders re-downloaded
the dataset each time.

**Fix:** Changed the default in `src/data.py` to an absolute path computed
from `Path(__file__).resolve().parent.parent / "data"`, so it always
resolves to the project's `data/` folder regardless of the caller's cwd.
Removed the duplicate `src/data/` folder.

---

## 2026-09-17 — Sample grid image had overlapping labels

**Symptom:** `outputs/sample_images.png` (2 rows x 8 columns of sample
images) had the second row's title text overlapping the bottom of the first
row's images.

**Root cause:** `fig.tight_layout()` didn't add enough vertical space
between subplot rows for the title text at this figure size.

**Fix:** Replaced `tight_layout()` with explicit
`fig.subplots_adjust(hspace=0.5, top=0.85)` and increased figure height
slightly (4 → 4.5). Verified visually — labels no longer overlap.

---

## 2026-11-20 — `pip3 install` landed in the wrong Python's site-packages

**Symptom:** After a fresh environment reset, `pip3 install torch
torchvision matplotlib scikit-learn ...` reported success, but
`python3 train_autoencoder.py` failed with `ModuleNotFoundError: No module
named 'matplotlib'`.

**Root cause:** The environment had two Python installs — `python3` resolved
to 3.11, but `pip3` resolved to a separate Python 3.13's pip
(`pip 24.0 ... (python 3.13)`), so packages were installed into 3.13's
site-packages while the interpreter actually running the scripts was 3.11.

**Fix:** Reinstalled using `python3 -m pip install ...` instead of bare
`pip3`, which guarantees packages land in the site-packages of whichever
interpreter will import them.

---

## 2026-11-20 — matplotlib import failed on a broken system Pillow

**Symptom:** Even after fixing the pip/python mismatch above, importing
matplotlib raised `ImportError: cannot import name '_imaging' from 'PIL'`.

**Root cause:** A pre-existing Pillow install at the OS-level
`/usr/lib/python3/dist-packages` (visible to `python3 -m pip show Pillow`,
so pip considered the dependency "already satisfied") was missing its
compiled `_imaging` extension — likely built for a different Python ABI —
and that broken copy shadowed/conflicted with a working one.

**Fix:** `python3 -m pip install --force-reinstall --no-deps Pillow` to put
a matching, complete Pillow build in the interpreter's own site-packages.
Verified with `python3 -c "from PIL import Image; import
matplotlib.pyplot"`.

---

## 2026-11-20 — Session's local clone vanished between sessions

**Symptom:** Returning to the project in a new session, `/home/user/fashion-vae-image-generation`
no longer existed on disk (only `/home/user/daa` was present), even though
`add_repo` reported the repo as "already attached".

**Root cause:** Each session runs in an ephemeral container; local files
(including git clones and installed pip packages) do not persist across
sessions — only what's pushed to GitHub survives.

**Fix:** Re-ran `git clone` for the repo and reinstalled dependencies.
No work was lost since every completed step had already been committed and
pushed. Lesson: never treat the local clone or installed packages as
durable storage — only the pushed commits are.

---

## 2026-11-20 — Every fresh session forced a full retrain from scratch

**Symptom:** Each new session (fresh container, or a second clone for a
parallel subagent) had no trained model weights available, so every
latent-space analysis, generation, or evaluation step had to retrain the AE
(10 epochs) and VAE (15 epochs) first — several minutes of repeated,
avoidable work. User flagged this directly: "why didn't you save the
weights somewhere for our reference?"

**Root cause:** `.gitignore` blanket-excluded `*.pt`/`*.pth` from the
initial scaffold (step 2), on the generic "don't commit model checkpoints"
habit — applied without checking actual file size. The AE and VAE weights
are under 1.5MB each; there was no real bloat concern at this scale.

**Fix:** Changed `.gitignore` to explicitly track `outputs/*.pt` (removed
the blanket checkpoint-ignore rule). Committed the already-trained
`ae_weights.pt` and `vae_weights.pt` directly. Future sessions can
`git pull` and load weights immediately — no retraining needed unless the
architecture or hyperparameters change.

**Flagged for later:** This won't scale to DeepFashion — larger
images/deeper models will produce much bigger checkpoint files. Revisit
with Git LFS or GitHub Releases at that point rather than plain git.

---

<!-- New entries go above this line. Append, don't rewrite history. -->
