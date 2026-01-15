# Local Testing Guide for NeuroAI Course CI/CD

This guide explains how to test workflows and CI scripts locally before pushing to GitHub.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Testing Strategies](#testing-strategies)
- [Safety Guidelines](#safety-guidelines)
- [Troubleshooting](#troubleshooting)
- [Understanding act](#understanding-act)

---

## Prerequisites

### Required Tools

1. **Docker** (version 20.10+)
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Start Docker daemon
   sudo systemctl start docker
   sudo systemctl enable docker

   # Add user to docker group (logout/login required)
   sudo usermod -aG docker $USER
   ```

2. **act** (GitHub Actions local runner)
   ```bash
   # Install act
   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

   # Verify installation
   act --version
   ```

3. **Python 3.9+** with pip
   ```bash
   python3.9 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

### Verify Setup

Run the verification command:
```bash
make verify-setup
```

Expected output:
```
✅ All prerequisites satisfied

Docker version: Docker version 24.0.7
act version: act version 0.2.56
```

---

## Quick Start

### Safe Testing (Recommended)

The safest way to test workflows is using **dry-run mode**, which validates syntax and structure without executing:

```bash
# Validate all workflows
make test-workflows-dryrun

# Validate composite actions
make test-composite-dryrun
```

### Full Workflow Testing

⚠️ **CRITICAL**: Always commit changes before running full workflow tests!

```bash
# 1. Check if it's safe to test
make safety-check

# 2. Commit your changes
git add -A
git commit -m "WIP: testing workflows"

# 3. Test workflows
make test-workflow-pr      # Test PR workflow
make test-workflow-publish # Test publish workflow
```

### Local Book Building

Build and view the book locally without Docker:

```bash
# Build book
make build-local

# Serve at http://localhost:8000
make serve-book
```

---

## Testing Strategies

### 1. Dry-Run Validation (Fastest, Safest)

**Use for**: Quick syntax validation, composite action testing

```bash
make test-workflows-dryrun
```

**Advantages**:
- ⚡ Fast (seconds vs minutes)
- ✅ No git manipulation
- ✅ No Docker containers created
- ✅ Safe to run on uncommitted changes

**Limitations**:
- Doesn't actually execute steps
- Won't catch runtime errors

---

### 2. Script Testing (Safe, Fast)

**Use for**: Testing CI scripts without workflows

```bash
make test-scripts
```

**Advantages**:
- ✅ No git manipulation
- ✅ Tests actual Python logic
- ✅ Fast execution

**Requirements**:
- Create `tests/test_ci_scripts.py` with pytest tests

**Example test**:
```python
# tests/test_ci_scripts.py
import pytest

def test_generate_book_imports():
    """Test that generate_book.py can be imported."""
    import sys
    sys.path.insert(0, 'ci')
    import generate_book
    assert hasattr(generate_book, 'generate_toc')
```

---

### 3. Local Book Build (Safe, Slow)

**Use for**: End-to-end book building without GitHub Actions

```bash
make build-local
make serve-book
```

**Advantages**:
- ✅ No git manipulation
- ✅ Tests actual book building
- ✅ Can inspect HTML output

**Use case**: Testing book structure, TOC changes, theme updates

---

### 4. Full Workflow Testing with act (Advanced)

**Use for**: Testing complete workflow execution including environment setup

⚠️ **REQUIRES**: Clean git working tree (commit first!)

```bash
# ALWAYS run safety check first
make safety-check

# Test specific workflow
make test-workflow-pr
```

**Advantages**:
- ✅ Most realistic testing
- ✅ Tests full environment setup
- ✅ Validates composite actions in context

**Disadvantages**:
- ⚠️ Slow (5-15 minutes)
- ⚠️ **CAN CORRUPT GIT REPOSITORY IF NOT COMMITTED**
- ⚠️ Creates large Docker containers (~2GB)

---

## Safety Guidelines

### ⚠️ Critical: act and Git Manipulation

**The Problem**: When `act` tests `pull_request` events on uncommitted changes, it:

1. Creates a fake merge commit
2. Checks out a detached HEAD state
3. May change file ownership (Docker runs as root)
4. Can lose uncommitted work

**Real Example of What Went Wrong**:
```bash
$ make test-workflow-pr  # WITHOUT committing first

# Result:
# - Lost all composite action files
# - Repository in detached HEAD state
# - File ownership changed to root
# - Had to recreate everything
```

### ✅ Safe Testing Workflow

**Always follow this order**:

```bash
# 1. Check git status
make safety-check

# 2. If uncommitted changes, commit them
git add -A
git commit -m "WIP: testing workflows"

# 3. Now safe to test
make test-workflow-pr

# 4. After testing, verify git status
git status

# 5. If needed, reset to before test commit
git reset --soft HEAD~1
```

### ✅ Alternative: Use Dry-Run Mode

Skip the git risk entirely:

```bash
# No commit needed - completely safe
make test-workflows-dryrun
```

### ⚠️ If Git Gets Corrupted

If act corrupts your repository:

```bash
# 1. Fix file permissions
sudo chown -R $USER:$USER .

# 2. Check git status
git status

# 3. If detached HEAD, return to branch
git checkout main  # or your branch name

# 4. If you lost work, check reflog
git reflog
git reset --hard HEAD@{n}  # where n is before corruption
```

---

## Understanding act

### What is act?

`act` runs GitHub Actions workflows locally using Docker containers. It:
- Reads `.github/workflows/*.yml` files
- Creates Docker containers matching GitHub Actions runners
- Executes workflow steps inside containers
- Simulates GitHub event payloads (pull_request, push, etc.)

### How act Works

```
Local Machine          Docker Container (ubuntu-latest)
┌──────────────┐      ┌─────────────────────────────────┐
│              │      │                                 │
│  Workflow    │─────▶│  1. Checkout code               │
│  .yml file   │      │  2. Setup Python                │
│              │      │  3. Run composite actions       │
│              │      │  4. Execute scripts             │
│              │      │                                 │
│  .actrc      │─────▶│  Uses catthehacker/ubuntu image │
│  config      │      │                                 │
└──────────────┘      └─────────────────────────────────┘
                                   │
                                   ▼
                      Writes back to local filesystem
                      (via Docker bind mount)
```

### act Configuration (.actrc)

```ini
# Use GitHub Actions compatible Docker image
-P ubuntu-latest=catthehacker/ubuntu:act-latest

# Set architecture
--container-architecture linux/amd64

# Store artifacts locally
--artifact-server-path /tmp/artifacts
```

### act Event Simulation

When you run `act pull_request`, it:

1. Creates a fake pull request event JSON
2. Simulates a merge commit (combines your branch with target)
3. Checks out the merge commit
4. Runs workflow steps

**This is why uncommitted changes get lost!**

---

## Troubleshooting

### Error: "permission denied" when running Docker

**Cause**: User not in docker group

**Fix**:
```bash
sudo usermod -aG docker $USER
newgrp docker  # Or logout/login
```

---

### Error: "Cannot connect to Docker daemon"

**Cause**: Docker daemon not running

**Fix**:
```bash
sudo systemctl start docker
sudo systemctl enable docker  # Auto-start on boot
```

---

### Error: "failed to read 'action.yml' from action"

**Cause**: Composite action paths not resolved by act

**Solution**: Use dry-run mode to validate:
```bash
make test-composite-dryrun
```

---

### Error: "fatal: detected dubious ownership"

**Cause**: Docker created files as root

**Fix**:
```bash
sudo chown -R $USER:$USER .
```

---

### Error: "HEAD detached at pull/%!f(<nil>)/merge"

**Cause**: act simulated pull request event

**Fix**:
```bash
# Return to your branch
git checkout jupyter-book-2-migration

# Check for lost work
git reflog
```

---

### Workflow runs forever / hangs

**Cause**: Step waiting for input, or resource constraint

**Fix**:
```bash
# Kill act process
Ctrl+C

# Check Docker containers
docker ps
docker kill $(docker ps -q)  # Kill all running containers

# Check Docker resources
docker system df
docker system prune  # Clean up space
```

---

## Testing Workflow

### Recommended Development Cycle

```bash
# 1. Make changes to workflows or composite actions
vim .github/workflows/notebook-pr.yaml

# 2. Quick validation (5 seconds)
make test-workflows-dryrun

# 3. If validation passes, test scripts locally
make test-scripts

# 4. Build book locally to verify end-to-end
make build-local
make serve-book  # Review at localhost:8000

# 5. Commit changes
git add -A
git commit -m "feat: improve workflow caching"

# 6. Optional: Full workflow test with act
make safety-check
make test-workflow-pr  # Only if needed

# 7. Push and let GitHub Actions do final test
git push
```

### When to Use Each Method

| Test Method | When to Use | Time | Risk |
|-------------|-------------|------|------|
| `test-workflows-dryrun` | Every change | 5s | None |
| `test-scripts` | Script changes | 10s | None |
| `build-local` | Book changes | 2-5m | None |
| `test-workflow-pr` (act) | Major workflow changes | 10-15m | High if uncommitted |
| Push to GitHub | Final validation | 20-30m | None |

---

## Advanced Usage

### Test Specific Workflow Job

```bash
# Test only specific job from workflow
act pull_request -W .github/workflows/notebook-pr.yaml -j process-notebooks
```

### Use Different Event Types

```bash
# Test push event instead of pull_request
act push -W .github/workflows/publish-book.yml

# Test workflow_dispatch with inputs
act workflow_dispatch -W .github/workflows/publish-book.yml
```

### Debug Mode

```bash
# Verbose output
act pull_request -W .github/workflows/notebook-pr.yaml -v

# Even more verbose
act pull_request -W .github/workflows/notebook-pr.yaml -v -v
```

### Use Secrets

```bash
# Create .secrets file
echo "GITHUB_TOKEN=your_token" > .secrets

# Use secrets in act
act pull_request -W .github/workflows/notebook-pr.yaml --secret-file .secrets
```

---

## FAQ

### Q: Do I need to use act?

**A**: No! act is optional. You can:
- Use dry-run mode for validation
- Build locally with `make build-local`
- Test scripts with pytest
- Push to GitHub and test there

act is most useful for testing complex workflows, but adds complexity.

---

### Q: Why does act take so long?

**A**: First run downloads Docker images (~2GB). Subsequent runs are faster due to:
- Docker image caching
- pip caching (if workflow uses it)
- nmaci tools caching

Expect: 10-15 min first run, 3-5 min subsequent runs.

---

### Q: Can I test without Docker?

**A**: Yes, for book building:
```bash
make build-local  # No Docker needed
```

For workflow testing, Docker is required by act.

---

### Q: How do I avoid losing work with act?

**A**: Three options:

1. **Always commit before testing** (recommended)
2. **Use dry-run mode** (safe, faster)
3. **Test in separate repo clone**:
   ```bash
   git clone . ../neuroai-test
   cd ../neuroai-test
   make test-workflow-pr  # Safe, won't affect main repo
   ```

---

### Q: What if GitHub Actions work but act fails?

**A**: act is not 100% compatible with GitHub Actions. Differences:
- Networking (some external services blocked)
- Docker-in-Docker limitations
- Missing GitHub-specific environment variables

**Solution**: If act fails but you're confident the workflow is correct, push to GitHub and test there.

---

## Resources

- **act documentation**: https://github.com/nektos/act
- **GitHub Actions docs**: https://docs.github.com/en/actions
- **Makefile commands**: Run `make help`
- **CI/CD optimization plan**: See `CI_CD_OPTIMIZATION_PLAN.md`

---

## Summary

**For most testing needs**:
```bash
# Quick validation
make test-workflows-dryrun

# Local building
make build-local
```

**For comprehensive workflow testing**:
```bash
# ALWAYS commit first!
git add -A && git commit -m "WIP"
make safety-check
make test-workflow-pr
```

**Remember**: act is powerful but dangerous with uncommitted changes. When in doubt, use dry-run mode or build locally.
