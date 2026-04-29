# v1.0 Public Release Checklist

Local prep is complete in this commit. The remaining steps require
GitHub UI / web access and are listed here so they are not lost.

## Before flipping the repo to public

1. **Rotate the GitHub PAT in `.git/config`.**
   The current `origin` remote URL inlines a personal access token
   (`ghp_…`). Even though `.git/config` is never pushed to the remote,
   any screen-share, copy-paste of `git remote -v`, or shared shell
   recording leaks a long-lived token. Rotate at
   <https://github.com/settings/tokens>, then re-set the remote
   without the token in the URL:

   ```bash
   git remote set-url origin https://github.com/kunwarshivam/agentops-bench.git
   git config --global credential.helper osxkeychain   # or another helper
   ```

   Re-authenticate on the next push; the keychain will hold the
   credential instead of the URL.

2. **Verify no `.env*` ever made it to history.**
   ```bash
   git log --all --source --remotes -- .env .env.local
   ```
   Should be empty. (Confirmed locally on 2026-04-28.)

3. **Push the v1.0 tag and feature branches** if not already pushed:
   ```bash
   git push origin main --tags
   git push origin paper/v1-arxiv-ready
   git push origin release/v1-public-prep
   ```

## Creating the v1.0 GitHub release

1. Go to <https://github.com/kunwarshivam/agentops-bench/releases/new>.
2. Tag: `v1.0` (already created locally).
3. Title: `AgentOps-Bench v1.0`.
4. Description: lift the abstract from `paper/body.tex` (the one ending
   `Apache-2.0`); add a one-line link to the preprint PDF in the repo.
5. Attach the trace bundle as an asset:
   `dist/agentops-bench-v1.0-pilot-traces.tar.gz` (14 MB).
6. Publish. Once the release is published, GitHub fires the webhook
   that mints a Zenodo deposit using `.zenodo.json`.

## Repo settings to set in the GitHub UI

In **Settings → General**:
- Description: `Reproducible benchmark for operational evaluation of
  tool-using LLM agents (six axes, three conditions, 7,200-run pilot).`
- Website: link to the arXiv PDF once it has an ID.
- Topics: `llm-agents`, `agent-evaluation`, `benchmark`,
  `prompt-injection`, `tool-use`, `ai-safety`, `reliability`.
- Features: Issues ✓, Discussions ✓, Wiki ✗, Projects ✗.

In **Settings → Branches**:
- Default branch: `main`.
- Add a branch protection rule for `main`: require PR before merging,
  require linear history, require status checks if/when CI is added.

In **Settings → Code security**:
- Enable Dependabot alerts.
- Enable secret scanning + push protection (catches future PAT leaks).

In **Settings → Pages**: leave disabled unless we add a project page.

## Zenodo

- v1.0 DOI: <https://doi.org/10.5281/zenodo.19875693>.
- The Zenodo record includes both the source archive and
  `agentops-bench-v1.0-pilot-traces.tar.gz`.
- README.md and CITATION.cff should use DOI `10.5281/zenodo.19875693`.

## arXiv

- Build both PDFs:
  ```bash
  cd paper
  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
  pdflatex main_neurips.tex && bibtex main_neurips && \
    pdflatex main_neurips.tex && pdflatex main_neurips.tex
  ```
- Submit `main.tex` (with `body.tex`, `references.bib`, `figs/`) to
  arXiv `cs.AI` with cross-listings `cs.CR`, `cs.LG`.
- Add the resulting arXiv URL to: README badges, CITATION.cff, the
  GitHub release description, the Zenodo deposit
  (`related_identifiers`).
