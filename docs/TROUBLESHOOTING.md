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

<!-- New entries go above this line. Append, don't rewrite history. -->
