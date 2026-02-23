.PHONY: format check test lint ty pyright typecheck bandit deptry pip-audit markdown-check verify-doc-examples integration-tests redis-tests redis-standalone-tests redis-cluster-tests bun-test property-tests stress-tests verify-mailbox verify-formal verify-formal-fast verify-formal-persist verify-all clean-extracted setup setup-tlaplus setup-redis demo demo-gemini demo-opencode sync-docs all clean biome biome-fix test-group-1 test-group-2 test-group-3 test-group-4 test-group-5 test-group-6 test-parallel

# =============================================================================
# Code Formatting
# =============================================================================

# Format code with ruff
format:
	@uv run ruff format -q .

# Check formatting without making changes
format-check:
	@uv run --quiet --all-extras python check.py -q format

# Run ruff linter
lint:
	@uv run --quiet --all-extras python check.py -q lint

# Run ruff linter with fixes
lint-fix:
	@uv run ruff check --fix -q .

# =============================================================================
# Frontend Linting (Biome)
# =============================================================================

# Run Biome linter and formatter check on frontend static files
biome:
	@if [ ! -d node_modules ]; then npm install --silent; fi
	@output=$$(npx biome check src/weakincentives/cli/static/ 2>&1) || \
		{ echo "$$output"; exit 1; }

# Run Biome with auto-fix
biome-fix:
	@if [ ! -d node_modules ]; then npm install --silent; fi
	@npx biome check --write src/weakincentives/cli/static/

# =============================================================================
# Security & Dependency Checks
# =============================================================================

# Run Bandit security scanner
bandit:
	@uv run --quiet --all-extras python check.py -q bandit

# Check for unused or missing dependencies with deptry
deptry:
	@uv run --quiet --all-extras python check.py -q deptry

# Run pip-audit for dependency vulnerabilities
pip-audit:
	@uv run --quiet --all-extras python check.py -q pip-audit

# =============================================================================
# Documentation Checks
# =============================================================================

# Validate Markdown formatting
markdown-check:
	@uv run --quiet --all-extras python check.py -q markdown

# Verify Python code examples in documentation
verify-doc-examples:
	@uv run --all-extras python check.py -q docs

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
	@uv run --quiet --all-extras python check.py -q typecheck

# =============================================================================
# Testing
# =============================================================================

# Run tests. In CI: full coverage (100% required). Locally: only tests affected by changes.
# Local mode uses testmon coverage database for fast iteration. First run builds the
# database, subsequent runs skip tests unaffected by changes.
test:
	@if [ -n "$$CI" ]; then \
		uv run --quiet --all-extras python check.py -q test; \
	else \
		uv run --quiet --all-extras pytest -p no:cov -o addopts= --testmon --strict-config --strict-markers --timeout=10 --timeout-method=thread --tb=short --no-header --reruns=2 --reruns-delay=0.5 tests; \
	fi

# =============================================================================
# Parallel Test Groups (for CI)
# =============================================================================
# These targets split the test suite into 6 groups for parallel execution.
# Each group saves coverage data with a unique suffix for later combination.
# Use `make test` for local development (runs all tests sequentially).

# Disable fail-under for individual groups; coverage is checked after combining
PYTEST_COMMON = uv run --all-extras pytest --strict-config --strict-markers \
	--timeout=10 --timeout-method=thread --tb=short --no-header \
	--cov=src/weakincentives --cov=toolchain --cov-report= --cov-fail-under=0

# Group 1: Adapters tests (~220 tests)
test-group-1:
	@$(PYTEST_COMMON) tests/adapters --cov-report=
	@mv .coverage .coverage.1

# Group 2: CLI + Contrib tests (~150 tests)
test-group-2:
	@$(PYTEST_COMMON) tests/cli tests/contrib --cov-report=
	@mv .coverage .coverage.2

# Group 3: Evals + Serde tests (~280 tests)
test-group-3:
	@$(PYTEST_COMMON) tests/evals tests/serde --cov-report=
	@mv .coverage .coverage.3

# Group 4: Prompt + Prompts tests (~400 tests)
test-group-4:
	@$(PYTEST_COMMON) tests/prompt tests/prompts --cov-report=
	@mv .coverage .coverage.4

# Group 5: Runtime tests (~290 tests)
test-group-5:
	@$(PYTEST_COMMON) tests/runtime --cov-report=
	@mv .coverage .coverage.5

# Group 6: Tools + Root tests + Misc (~540 tests)
test-group-6:
	@$(PYTEST_COMMON) tests/tools tests/filesystem tests/resources tests/skills \
		tests/optimizers tests/toolchain tests/debug tests/formal formal-tests \
		tests/test_*.py --cov-report=
	@mv .coverage .coverage.6

# Run all test groups locally and combine coverage
test-parallel: test-group-1 test-group-2 test-group-3 test-group-4 test-group-5 test-group-6
	@uv run coverage combine .coverage.*
	@uv run coverage report --fail-under=100

# Run integration tests (tests skip automatically when API keys are not set)
integration-tests:
	@uv run --all-extras pytest --no-cov --strict-config --strict-markers -vv --maxfail=1 --timeout=300 integration-tests

# Run all Redis integration tests (standalone + cluster)
redis-tests:
	@uv run --all-extras pytest --no-cov --strict-config --strict-markers -vv --timeout=300 -m redis integration-tests

# Run Redis standalone tests only
redis-standalone-tests:
	@uv run --all-extras pytest --no-cov --strict-config --strict-markers -vv --timeout=300 -m redis_standalone integration-tests

# Run Redis cluster tests only
redis-cluster-tests:
	@uv run --all-extras pytest --no-cov --strict-config --strict-markers -vv --timeout=300 -m redis_cluster integration-tests

# Run JavaScript tests with Bun (via toolchain for consistent output)
bun-test:
	@uv run --quiet --all-extras python check.py -q bun-test

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

# Code reviewer demo
# Usage: make demo-claude    [PROJECT=...] [FOCUS="..."]
#        make demo-codex     [PROJECT=...] [FOCUS="..."]
#        make demo-gemini    [PROJECT=...] [FOCUS="..."] [MODEL=...]
#        make demo-opencode  [PROJECT=...] [FOCUS="..."] [MODEL=...]
PROJECT ?= test-repositories/sunfish
FOCUS ?= Review how the UCI implementation is handled via the packaging scripts
MODEL ?= openai/gpt-5.3-codex
demo-claude:
	@uv run --all-extras python code_reviewer_example.py --adapter claude "$(PROJECT)" "$(FOCUS)"
demo-codex:
	@uv run --all-extras python code_reviewer_example.py --adapter codex "$(PROJECT)" "$(FOCUS)"
demo-gemini:
	@uv run --all-extras python code_reviewer_example.py --adapter gemini "$(PROJECT)" "$(FOCUS)"
demo-opencode:
	@uv run --all-extras python code_reviewer_example.py --adapter opencode --model "$(MODEL)" "$(PROJECT)" "$(FOCUS)"
demo: demo-claude

# =============================================================================
# Main Check Target
# =============================================================================

# Run all checks (format, lint, typecheck, security, dependencies, architecture, docs, tests)
# In CI: full test coverage required. Locally: only tests affected by changes (via testmon).
check: format-check lint typecheck bandit deptry pip-audit markdown-check biome bun-test test
	@uv run --quiet --all-extras python check.py -q architecture code-length docs
	@echo "✓ All checks passed"

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
