.PHONY: format check test lint ty pyright typecheck bandit deptry pip-audit markdown-check verify-doc-examples integration-tests redis-tests redis-standalone-tests redis-cluster-tests bun-test property-tests stress-tests verify-mailbox verify-formal verify-formal-fast verify-formal-persist verify-all clean-extracted setup setup-tlaplus setup-redis demo demo-podman demo-claude-agent sync-docs all clean biome biome-fix slides slides-pdf

# =============================================================================
# Code Formatting
# =============================================================================

# Format code with ruff
format:
	@uv run ruff format -q .

# Check formatting without making changes
format-check:
	@uv run --all-extras python check.py -q format

# Run ruff linter
lint:
	@uv run --all-extras python check.py -q lint

# Run ruff linter with fixes
lint-fix:
	@uv run ruff check --fix -q .

# =============================================================================
# Frontend Linting (Biome)
# =============================================================================

# Run Biome linter and formatter check on frontend static files
biome:
	@if [ ! -d node_modules ]; then npm install --silent; fi
	@npx biome check src/weakincentives/cli/static/

# Run Biome with auto-fix
biome-fix:
	@if [ ! -d node_modules ]; then npm install --silent; fi
	@npx biome check --write src/weakincentives/cli/static/

# =============================================================================
# Security & Dependency Checks
# =============================================================================

# Run Bandit security scanner
bandit:
	@uv run --all-extras python check.py -q bandit

# Check for unused or missing dependencies with deptry
deptry:
	@uv run --all-extras python check.py -q deptry

# Run pip-audit for dependency vulnerabilities
pip-audit:
	@uv run --all-extras python check.py -q pip-audit

# =============================================================================
# Documentation Checks
# =============================================================================

# Validate Markdown formatting
markdown-check:
	@uv run --all-extras python check.py -q markdown

# Verify Python code examples in documentation
verify-doc-examples:
	@uv run --all-extras python check.py -q docs

# =============================================================================
# Presentation Slides
# =============================================================================

# Build reveal.js HTML slides with pandoc
slides:
	@echo "Building WINK presentation slides..."
	@mkdir -p docs/presentation/output
	@pandoc docs/presentation/wink-slides.md \
		-t revealjs \
		-s \
		--slide-level=2 \
		-V revealjs-url=https://unpkg.com/reveal.js@5.1.0 \
		-o docs/presentation/output/wink-slides.html
	@echo "Slides built: docs/presentation/output/wink-slides.html"

# Build PDF slides with pandoc (requires LaTeX)
slides-pdf:
	@echo "Building WINK presentation PDF..."
	@mkdir -p docs/presentation/output
	@pandoc docs/presentation/wink-slides.md \
		-t beamer \
		--slide-level=2 \
		-V theme:metropolis \
		-o docs/presentation/output/wink-slides.pdf
	@echo "PDF built: docs/presentation/output/wink-slides.pdf"

# =============================================================================
# Type Checking
# =============================================================================

# Run ty type checker (src only, consistent with pyright scope)
ty:
	@uv run --all-extras ty check --error-on-warning -qq src || \
			(echo "ty check failed; rerunning with verbose output..." >&2; \
			uv run --all-extras ty check --error-on-warning src)

# Run pyright type checker
pyright:
	@uv run --all-extras pyright --project pyproject.toml || \
		(echo "pyright failed; rerunning with verbose output..." >&2; \
		uv run --all-extras pyright --project pyproject.toml --verbose)

# Run all type checkers
typecheck:
	@uv run --all-extras python check.py -q typecheck

# =============================================================================
# Testing
# =============================================================================

# Run tests with coverage (100% minimum) and 10s per-test timeout
test: bun-test
	@uv run --all-extras python check.py -q test

# Run integration tests (tests skip automatically when API keys are not set)
integration-tests:
	@uv run --all-extras pytest --no-cov --strict-config --strict-markers -vv --maxfail=1 integration-tests

# Run all Redis integration tests (standalone + cluster)
redis-tests:
	@uv run --all-extras pytest --no-cov --strict-config --strict-markers -vv -m redis integration-tests

# Run Redis standalone tests only
redis-standalone-tests:
	@uv run --all-extras pytest --no-cov --strict-config --strict-markers -vv -m redis_standalone integration-tests

# Run Redis cluster tests only
redis-cluster-tests:
	@uv run --all-extras pytest --no-cov --strict-config --strict-markers -vv -m redis_cluster integration-tests

# Run JavaScript tests with Bun
bun-test:
	@if command -v bun >/dev/null 2>&1; then \
		bun test tests/js/; \
	else \
		echo "Bun not installed, skipping JS tests. Install with: curl -fsSL https://bun.sh/install | bash"; \
	fi

# =============================================================================
# Setup
# =============================================================================

# Download and install TLA+ tools (requires Java)
setup-tlaplus:
	@echo "Setting up TLA+ tools..."
	@if [ -f /usr/local/lib/tla2tools.jar ]; then \
		echo "TLA+ tools already installed at /usr/local/lib/tla2tools.jar"; \
	else \
		echo "Downloading TLA+ tools..."; \
		curl -sL -o /tmp/tla2tools.jar \
			https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar; \
		echo "Installing to /usr/local/lib/..."; \
		sudo mkdir -p /usr/local/lib; \
		sudo mv /tmp/tla2tools.jar /usr/local/lib/; \
		echo "Creating tlc wrapper script..."; \
		sudo mkdir -p /usr/local/bin; \
		echo '#!/bin/bash' | sudo tee /usr/local/bin/tlc > /dev/null; \
		echo 'exec java -XX:+UseParallelGC -jar /usr/local/lib/tla2tools.jar "$$@"' | sudo tee -a /usr/local/bin/tlc > /dev/null; \
		sudo chmod +x /usr/local/bin/tlc; \
		echo "TLA+ tools installed successfully"; \
	fi

# Start Redis server for testing (if not running)
setup-redis:
	@echo "Setting up Redis..."
	@if redis-cli ping >/dev/null 2>&1; then \
		echo "Redis already running"; \
	elif command -v redis-server >/dev/null 2>&1; then \
		echo "Starting Redis server..."; \
		redis-server --daemonize yes; \
		sleep 1; \
		redis-cli ping; \
	else \
		echo "Redis not installed. Install with: apt install redis-server"; \
		exit 1; \
	fi

# Setup all verification dependencies
setup: setup-tlaplus setup-redis
	@echo "Syncing Python dependencies..."
	@uv sync --all-extras
	@echo "Setup complete!"

# =============================================================================
# Formal Verification
# =============================================================================

# Run Hypothesis property-based tests
property-tests:
	@uv run --all-extras pytest tests/contrib/mailbox/test_redis_mailbox_properties.py \
		tests/contrib/mailbox/test_redis_mailbox_invariants.py \
		--no-cov -v --hypothesis-show-statistics

# Run concurrent stress tests
stress-tests:
	@uv run --all-extras pytest tests/contrib/mailbox/test_redis_mailbox_stress.py \
		--no-cov -v -m slow --timeout=120

# Run all mailbox verification
verify-mailbox: verify-formal property-tests
	@echo "All mailbox verification checks passed"

# =============================================================================
# Embedded TLA+ Specifications
# =============================================================================

# Full formal verification with TLC model checking (~30s)
# Uses temp directory (no filesystem pollution)
verify-formal:
	@uv run --all-extras pytest formal-tests/ --no-cov -q --timeout=240

# Fast extraction only (development, ~1s)
# Skips model checking - use only for rapid iteration
verify-formal-fast:
	@uv run --all-extras pytest formal-tests/ --no-cov -q --skip-model-check

# Full verification + persist specs to specs/tla/extracted/
verify-formal-persist:
	@uv run --all-extras pytest formal-tests/ --no-cov -v --persist-specs

# Run all formal verification (embedded specs + property tests)
verify-all: verify-formal property-tests
	@echo "All formal verification passed:"
	@echo "  - Embedded TLA+ specs with TLC model checking"
	@echo "  - Property-based tests (Hypothesis)"

# Remove extracted TLA+ specs
clean-extracted:
	@echo "Cleaning extracted TLA+ specs..."
	@rm -rf specs/tla/extracted/
	@echo "Cleaned specs/tla/extracted/"

# =============================================================================
# Demos
# =============================================================================

# Launch the interactive code reviewer demo
demo:
	@uv run --all-extras python code_reviewer_example.py

# Launch the interactive code reviewer demo using the Podman sandbox (if available)
demo-podman:
	@uv run --all-extras python code_reviewer_example.py --podman

# Launch the interactive code reviewer demo with Claude Agent SDK
# Supports both Anthropic API (ANTHROPIC_API_KEY) and AWS Bedrock (CLAUDE_CODE_USE_BEDROCK=1 + AWS_REGION)
# For Bedrock, credentials are resolved via AWS SDK credential chain (~/.aws/config, SSO cache, env vars)
demo-claude-agent:
	@if [ -z "$$ANTHROPIC_API_KEY" ] && [ "$$CLAUDE_CODE_USE_BEDROCK" != "1" -o -z "$$AWS_REGION" ]; then \
		echo "No authentication configured. Set either:" >&2; \
		echo "  - ANTHROPIC_API_KEY for Anthropic API" >&2; \
		echo "  - CLAUDE_CODE_USE_BEDROCK=1 and AWS_REGION for AWS Bedrock" >&2; \
		exit 1; \
	fi
	@uv run --all-extras python code_reviewer_example.py --claude-agent

# =============================================================================
# Main Check Target
# =============================================================================

# Run all checks (format, lint, typecheck, security, dependencies, architecture, docs, tests)
check: format-check lint typecheck bandit deptry pip-audit markdown-check biome bun-test test
	@uv run --all-extras python check.py -q architecture docs

# Synchronize documentation files into package
sync-docs:
	@mkdir -p src/weakincentives/docs/specs
	@mkdir -p src/weakincentives/docs/guides
	@cp llms.md src/weakincentives/docs/
	@cp CHANGELOG.md src/weakincentives/docs/
	@cp code_reviewer_example.py src/weakincentives/docs/
	@cp specs/*.md src/weakincentives/docs/specs/
	@cp guides/*.md src/weakincentives/docs/guides/
	@touch src/weakincentives/docs/__init__.py

# Run all checks and fixes
all: format lint-fix bandit deptry pip-audit typecheck test

# Clean cache files
clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	rm -rf specs/tla/extracted/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
