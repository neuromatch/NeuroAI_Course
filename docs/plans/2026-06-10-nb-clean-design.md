# nb-clean GitHub Action Design

**Date:** 2026-06-10  
**Status:** Approved

## Overview

A GitHub Action workflow that automatically cleans Jupyter notebooks when pull requests are opened against the `staging` branch. The workflow uses `nb-clean` to strip cell execution counts, metadata, and outputs from notebooks in `tutorials/` and `projects/`.

The workflow handles two cases differently based on whether the PR comes from a branch on the main repo or from a fork.

## Trigger

- **Workflow file:** `.github/workflows/nb-clean.yml`
- **Trigger event:** `pull_request` to `staging` branch
- **Path filter:** `tutorials/**/*.ipynb`, `projects/**/*.ipynb`
- **Skip condition:** If no `.ipynb` files changed in the PR, the workflow skips entirely
- **Skip condition:** If all changed notebooks are already clean, the workflow passes silently

## Main Repo Branch Flow

When the PR source branch is on the main repo (`github.event.pull_request.head.repo.full_name == github.repository`):

1. Checkout the PR head branch
2. Run `nb-clean check` on all `.ipynb` files in `tutorials/` and `projects/`
3. If dirty, run `nb-clean clean` to fix in place
4. Push a cleanup commit back to the same branch
5. Fail the CI check and comment on the PR with brief instructions

Instructions comment:
> Notebooks have been auto-cleaned with a cleanup commit above. For future PRs, run `uvx nb-clean clean tutorials/ projects/` locally before pushing.

## Fork Branch Flow

When the PR comes from a fork:

1. Checkout the PR head branch (from the fork)
2. Run `nb-clean check` on all `.ipynb` files in `tutorials/` and `projects/`
3. If dirty, run `nb-clean clean` to fix in place
4. Use `peter-evans/create-pull-request` action with `push-to-fork` option to push a cleanup branch to the fork and create a PR
5. Comment on the original PR with a link to the cleanup PR

**Fallback:** If the action cannot push to the fork (e.g., "Allow edits from maintainers" is disabled), post a comment with a GitHub gist containing the cleaned notebooks and manual instructions.

## Error Handling

- If `nb-clean` is not installed or fails, the workflow fails gracefully with a clear error message
- If git push fails, post a comment explaining the issue and providing manual instructions
- If the PR is already clean, the workflow passes silently
- If no `.ipynb` files changed, the workflow is skipped

## Edge Cases

- **Multiple commits:** Workflow re-runs on each push, re-applies cleanup
- **Rebase conflicts:** If contributor rebases after cleanup commit, git will fail and comment with instructions
- **Large PRs:** `nb-clean check` runs against all changed `.ipynb` files; protected by GitHub's 6-hour job limit

## Architecture

### Files

```
.github/
  workflows/
    nb-clean.yml          # Main workflow file
  scripts/
    nb-clean-check.sh     # Bash script: runs nb-clean check, returns exit code
    nb-clean-cleanup.sh   # Bash script: runs nb-clean clean, commits cleanup
```

### Workflow Structure

The workflow uses both `pull_request` and `pull_request_target` triggers. `pull_request` handles the main repo case; `pull_request_target` ensures fork PRs also get cleanup PRs with write access to the fork.

Permissions:
- `contents: write` — for pushing cleanup commits
- `pull-requests: write` — for creating cleanup PRs

### Key Steps

1. Checkout PR head branch
2. Set up Python 3.12
3. Install `nb-clean` via pip
4. Detect PR source (main repo vs fork) using `github.event.pull_request.head.repo.full_name`
5. Run `nb-clean check` on changed notebooks
6. Branch based on source:
   - Main repo: run cleanup script, push commit
   - Fork: use `peter-evans/create-pull-request@v8` with `push-to-fork`

## Tools & Dependencies

- **nb-clean** — Python package for cleaning Jupyter notebooks (`pip install nb-clean`)
- **peter-evans/create-pull-request@v8** — GitHub Action for creating/updating PRs, with `push-to-fork` support
- **actions/checkout@v6** — Check out the PR head branch
- **actions/setup-python@v6** — Set up Python 3.12
