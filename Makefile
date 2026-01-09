.PHONY: format check test lint ty pyright typecheck type-coverage bandit vulture deptry pip-audit markdown-check verify-doc-examples integration-tests redis-tests redis-standalone-tests redis-cluster-tests validate-integration-tests property-tests stress-tests verify-mailbox verify-formal verify-formal-fast verify-formal-persist verify-all clean-extracted setup setup-tlaplus setup-redis demo demo-podman demo-claude-agent sync-docs check-core-imports validate-modules all clean

# Format code with ruff
format:
	@uv run ruff format -q .

# Check formatting without making changes
format-check:
	@uv run ruff format -q --check .

# Run ruff linter
lint:
	@uv run ruff check --preview -q .

# Run ruff linter with fixes
lint-fix:
	@uv run ruff check --fix -q .

# Run Bandit security scanner
bandit:
	@uv run python build/run_bandit.py -q -r src/weakincentives

# Find unused code with vulture
vulture:
	@uv run vulture

# Check for unused or missing dependencies with deptry
deptry:
	@uv run python build/run_deptry.py

# Check that core modules don't import from contrib
check-core-imports:
	@uv run python build/check_core_imports.py

# Validate module boundaries and import patterns
validate-modules:
	@uv run python scripts/validate_module_boundaries.py

# Run pip-audit for dependency vulnerabilities
pip-audit:
	@uv run python build/run_pip_audit.py

# Validate Markdown formatting and local links
markdown-check:
	@uv run python build/run_mdformat.py
	@uv run python build/check_md_links.py

# Verify Python code examples in documentation
verify-doc-examples:
	@uv run --all-extras python build/verify_doc_examples.py -q

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
typecheck: ty pyright

# Check type coverage (100% completeness required)
type-coverage:
	@uv run --all-extras python build/run_type_coverage.py -q

# Run tests with coverage (100% minimum)
test:
	@uv run --all-extras python build/run_pytest.py --strict-config --strict-markers --maxfail=1 --cov-fail-under=100 -q --no-header --cov-report= tests

# Run OpenAI integration tests
integration-tests:
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "OPENAI_API_KEY is not set; export it to run integration tests." >&2; \
		exit 1; \
	fi
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

# Validate integration tests (typecheck without running)
validate-integration-tests:
	@uv run --all-extras python build/validate_integration_tests.py -q

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
		echo "✓ TLA+ tools installed successfully"; \
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
	@echo "✓ All formal verification passed:"
	@echo "  - Embedded TLA+ specs with TLC model checking"
	@echo "  - Property-based tests (Hypothesis)"

# Remove extracted TLA+ specs
clean-extracted:
	@echo "Cleaning extracted TLA+ specs..."
	@rm -rf specs/tla/extracted/
	@echo "✓ Cleaned specs/tla/extracted/"

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
demo-claude-agent:
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "ANTHROPIC_API_KEY is not set; export it to use Claude Agent SDK." >&2; \
		exit 1; \
	fi
	@uv run --all-extras python code_reviewer_example.py --claude-agent

# Run all checks (format check, lint, typecheck, type-coverage, bandit, vulture, deptry, check-core-imports, validate-modules, pip-audit, markdown, doc-examples, validate-integration-tests, test)
# Note: validate-modules is commented out pending fixes for existing violations
check: format-check lint typecheck type-coverage bandit vulture deptry check-core-imports pip-audit markdown-check verify-doc-examples validate-integration-tests test
# Uncomment after fixing module boundary violations:
# check: format-check lint typecheck type-coverage bandit vulture deptry check-core-imports validate-modules pip-audit markdown-check verify-doc-examples validate-integration-tests test

# Synchronize documentation files into package
sync-docs:
	@mkdir -p src/weakincentives/docs/specs
	@cp llms.md src/weakincentives/docs/
	@cp WINK_GUIDE.md src/weakincentives/docs/
	@cp CHANGELOG.md src/weakincentives/docs/
	@cp specs/*.md src/weakincentives/docs/specs/
	@touch src/weakincentives/docs/__init__.py

# Run all checks and fixes
all: format lint-fix bandit deptry pip-audit typecheck test

# Clean cache files
clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	rm -rf specs/tla/extracted/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
