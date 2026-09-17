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

<!-- New entries go above this line. Append, don't rewrite history. -->
