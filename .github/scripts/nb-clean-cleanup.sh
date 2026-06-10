#!/usr/bin/env bash
set -euo pipefail

# Clean all notebooks in tutorials/ and projects/
find tutorials/ projects/ -name '*.ipynb' -type f -print0 | while IFS= read -r -d '' notebook; do
  echo "Cleaning: $notebook"
  nb-clean clean "$notebook"
done

# Configure git user for the cleanup commit
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

# Commit the cleaned notebooks
git add tutorials/ projects/
git commit -m "chore: clean notebooks with nb-clean" || echo "No changes to commit"

# Push the cleanup commit back to the PR branch
git push
