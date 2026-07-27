#!/usr/bin/env bash
set -euo pipefail

DIRTY=0

# Find all .ipynb files in tutorials/ and projects/
while IFS= read -r -d '' notebook; do
  if ! nb-clean check "$notebook"; then
    echo "DIRTY: $notebook"
    DIRTY=1
  fi
done < <(find tutorials/ projects/ -name '*.ipynb' -type f -print0 2>/dev/null)

# Export result for GitHub Actions
echo "dirty=$DIRTY" >> "$GITHUB_OUTPUT"

if [ "$DIRTY" -eq 1 ]; then
  echo "::error::Some notebooks are not clean. Run 'nb-clean clean' to fix them."
  exit 1
else
  echo "All notebooks are clean."
  exit 0
fi
