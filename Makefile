.PHONY: format check test lint ty pyright typecheck type-coverage bandit vulture deptry pip-audit markdown-check integration-tests redis-tests redis-standalone-tests redis-cluster-tests validate-integration-tests mutation-test mutation-check tlaplus-check tlaplus-check-exhaustive property-tests stress-tests verify-mailbox extract-tla check-tla check-tla-fast verify-embedded verify-all compare-specs clean-extracted setup setup-tlaplus setup-redis demo demo-podman demo-claude-agent sync-docs all clean

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

# Run pip-audit for dependency vulnerabilities
pip-audit:
	@uv run python build/run_pip_audit.py

# Validate Markdown formatting and local links
markdown-check:
	@uv run python build/run_mdformat.py
	@uv run python build/check_md_links.py

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

# Run mutation tests with mutmut
mutation-test:
	@uv run --all-extras python build/run_mutmut.py

# Enforce the configured mutation score gate (for CI)
mutation-check:
	@uv run --all-extras python build/run_mutmut.py --check

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

# Download TLA+ tools (requires Java)
setup-tlaplus:
	@echo "Setting up TLA+ tools..."
	@mkdir -p tools
	@if [ ! -f tools/tla2tools.jar ]; then \
		echo "Downloading TLA+ tools..."; \
		curl -sL -o tools/tla2tools.jar \
			https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar; \
	else \
		echo "TLA+ tools already installed"; \
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

# Run TLC model checker in simulation mode (fast, for CI - ~2 seconds)
# Checks 1000+ random traces of depth 50, verifying invariants at each state
tlaplus-check:
	@echo "Running TLC model checker (simulation mode)..."
	@if [ ! -f tools/tla2tools.jar ]; then \
		echo "TLC not installed. Run: make setup-tlaplus"; \
		exit 1; \
	fi
	@cd specs/tla && java -XX:+UseParallelGC -jar ../../tools/tla2tools.jar \
		-simulate num=1000 -depth 50 \
		-config RedisMailboxMC-ci.cfg RedisMailbox.tla -workers auto

# Run exhaustive TLC model checking (slow, for thorough verification)
# Note: May take 10+ minutes depending on configuration
tlaplus-check-exhaustive:
	@echo "Running TLC model checker (exhaustive mode)..."
	@if [ ! -f tools/tla2tools.jar ]; then \
		echo "TLC not installed. Run: make setup-tlaplus"; \
		exit 1; \
	fi
	@cd specs/tla && java -XX:+UseParallelGC -jar ../../tools/tla2tools.jar \
		-config RedisMailboxMC.cfg RedisMailbox.tla -workers auto

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
verify-mailbox: tlaplus-check property-tests
	@echo "All mailbox verification checks passed"

# =============================================================================
# Embedded TLA+ Specifications
# =============================================================================

# Extract TLA+ specs from @formal_spec decorators
extract-tla:
	@echo "Extracting embedded TLA+ specifications..."
	@uv run pytest --extract-tla -v
	@echo "✓ Specs extracted to specs/tla/extracted/"

# Extract and model check embedded TLA+ specs
check-tla:
	@echo "Extracting and validating embedded TLA+ specifications..."
	@uv run pytest --check-tla -v
	@echo "✓ All embedded specs passed model checking"

# Extract embedded specs without model checking (fast)
check-tla-fast:
	@echo "Extracting embedded TLA+ specifications (no model checking)..."
	@uv run pytest --extract-tla -q
	@echo "✓ Specs extracted (skipped model checking)"

# Alias for check-tla
verify-embedded: check-tla
	@echo "✓ Embedded formal verification complete"

# Run all formal verification (legacy + embedded + property tests)
verify-all: tlaplus-check check-tla property-tests
	@echo "✓ All formal verification passed:"
	@echo "  - Legacy TLA+ specs (specs/tla/*.tla)"
	@echo "  - Embedded TLA+ specs (@formal_spec decorators)"
	@echo "  - Property-based tests (Hypothesis)"

# Compare legacy and embedded TLA+ specs
compare-specs:
	@echo "Comparing legacy and embedded specifications..."
	@uv run pytest --extract-tla -q
	@echo ""
	@echo "Legacy spec vs Embedded spec:"
	@if [ -f specs/tla/RedisMailbox.tla ] && [ -f specs/tla/extracted/RedisMailbox.tla ]; then \
		diff -u specs/tla/RedisMailbox.tla specs/tla/extracted/RedisMailbox.tla || true; \
	else \
		echo "One or both spec files not found"; \
	fi

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

# Run all checks (format check, lint, typecheck, type-coverage, bandit, vulture, deptry, pip-audit, markdown, validate-integration-tests, test)
check: format-check lint typecheck type-coverage bandit vulture deptry pip-audit markdown-check validate-integration-tests test

# Synchronize documentation files into package
sync-docs:
	@mkdir -p src/weakincentives/docs/specs
	@cp llms.md src/weakincentives/docs/
	@cp WINK_GUIDE.md src/weakincentives/docs/
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
