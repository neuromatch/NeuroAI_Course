.PHONY: help verify-setup test-workflows test-scripts build-local clean safety-check

help:
	@echo "NeuroAI Course - Local Testing Commands"
	@echo ""
	@echo "⚠️  CRITICAL: ALWAYS commit your changes before running act tests!"
	@echo "⚠️  act can manipulate your git repository when testing pull_request events"
	@echo "⚠️  Use 'make safety-check' to verify git status before testing"
	@echo ""
	@echo "Setup:"
	@echo "  make verify-setup           - Check prerequisites (Docker, act)"
	@echo "  make safety-check           - Verify git status is safe for testing"
	@echo ""
	@echo "Safe Testing (Recommended):"
	@echo "  make test-workflows-dryrun  - Validate workflows without execution (SAFE)"
	@echo "  make test-composite-dryrun  - Validate composite actions (SAFE)"
	@echo ""
	@echo "Full Workflow Testing (REQUIRES COMMIT FIRST):"
	@echo "  make test-workflow-publish  - Test publish-book workflow"
	@echo "  make test-workflow-pr       - Test notebook-pr workflow"
	@echo ""
	@echo "Script Testing (No git manipulation):"
	@echo "  make test-scripts           - Test CI scripts with pytest"
	@echo ""
	@echo "Book Building:"
	@echo "  make build-local            - Build book locally with JB 1.x"
	@echo "  make serve-book             - Serve book at localhost:8000"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                  - Remove build artifacts"
	@echo ""

verify-setup:
	@echo "Verifying prerequisites..."
	@which docker > /dev/null || (echo "❌ Docker not found. Install: https://docs.docker.com/get-docker/" && exit 1)
	@which act > /dev/null || (echo "❌ act not found. Install: curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash" && exit 1)
	@docker ps > /dev/null 2>&1 || (echo "❌ Docker daemon not running. Start with: sudo systemctl start docker" && exit 1)
	@echo "✅ All prerequisites satisfied"
	@echo ""
	@echo "Docker version: $$(docker --version)"
	@echo "act version: $$(act --version)"

safety-check:
	@echo "Checking git status..."
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo ""; \
		echo "⚠️  WARNING: You have uncommitted changes!"; \
		echo "⚠️  act can corrupt your git repository when testing pull_request events"; \
		echo "⚠️  STRONGLY RECOMMENDED: Commit your changes first"; \
		echo ""; \
		git status --short; \
		echo ""; \
		echo "Options:"; \
		echo "  1. Commit: git add -A && git commit -m 'WIP: testing'"; \
		echo "  2. Use dry-run mode: make test-workflows-dryrun"; \
		echo "  3. Proceed anyway (NOT RECOMMENDED): make test-workflow-pr-unsafe"; \
		echo ""; \
		exit 1; \
	else \
		echo "✅ Working tree is clean - safe to run act"; \
	fi

# Safe testing - validates without execution
test-workflows-dryrun:
	@echo "🔍 Validating workflows (dry-run mode - SAFE)..."
	@echo ""
	@echo "Testing notebook-pr workflow..."
	act pull_request -W .github/workflows/notebook-pr.yaml --dryrun
	@echo ""
	@echo "Testing publish-book workflow..."
	act workflow_dispatch -W .github/workflows/publish-book.yml --dryrun
	@echo ""
	@echo "✅ All workflows validated successfully"

test-composite-dryrun:
	@echo "🔍 Validating composite actions (dry-run mode - SAFE)..."
	act pull_request -W .github/workflows/notebook-pr.yaml --dryrun
	@echo "✅ Composite actions validated"

# Full workflow testing - REQUIRES COMMIT FIRST
test-workflow-publish: safety-check
	@echo "🧪 Testing publish-book workflow..."
	@echo "⚠️  This will create Docker containers and may take several minutes"
	act workflow_dispatch -W .github/workflows/publish-book.yml

test-workflow-pr: safety-check
	@echo "🧪 Testing notebook-pr workflow..."
	@echo "⚠️  This will create Docker containers and may take several minutes"
	act pull_request -W .github/workflows/notebook-pr.yaml

# Unsafe versions that bypass safety check (NOT RECOMMENDED)
test-workflow-pr-unsafe:
	@echo "⚠️⚠️⚠️  RUNNING WITHOUT SAFETY CHECK - YOUR GIT REPO MAY BE CORRUPTED ⚠️⚠️⚠️"
	@sleep 3
	act pull_request -W .github/workflows/notebook-pr.yaml

test-workflow-publish-unsafe:
	@echo "⚠️⚠️⚠️  RUNNING WITHOUT SAFETY CHECK - YOUR GIT REPO MAY BE CORRUPTED ⚠️⚠️⚠️"
	@sleep 3
	act workflow_dispatch -W .github/workflows/publish-book.yml

# Script testing - safe, doesn't manipulate git
test-scripts:
	@echo "🧪 Testing CI scripts..."
	@if [ -d "tests" ]; then \
		pytest tests/ -v; \
	else \
		echo "⚠️  No tests directory found"; \
		echo "Create tests/test_ci_scripts.py for script testing"; \
	fi

# Local book building
build-local:
	@echo "📚 Building book locally..."
	@if [ ! -d "ci" ]; then \
		echo "Downloading nmaci tools..."; \
		wget -q https://github.com/neuromatch/nmaci/archive/refs/heads/main.tar.gz; \
		tar -xzf main.tar.gz; \
		pip install -r nmaci-main/requirements.txt > /dev/null; \
		mv nmaci-main/scripts/ ci/; \
		rm -r nmaci-main main.tar.gz; \
	fi
	@python ci/generate_book.py student
	@cd book && ln -sf ../tutorials tutorials && ln -sf ../projects projects && cd ..
	@jupyter-book build book
	@echo "✅ Book built successfully"
	@echo "📖 Open: book/_build/html/index.html"

serve-book:
	@echo "🌐 Serving book at http://localhost:8000"
	@cd book/_build/html && python -m http.server 8000

# Cleanup
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf book/_build book/.jupyter-cache
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"

clean-all: clean
	@echo "🧹 Removing ci directory..."
	@rm -rf ci/
	@echo "✅ Deep cleanup complete"
