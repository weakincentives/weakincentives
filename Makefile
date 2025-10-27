.PHONY: format check test lint typecheck bandit pip-audit all clean

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
	@uv run python tools/run_bandit.py -q -r src/weakincentives

# Run pip-audit for dependency vulnerabilities
pip-audit:
	@uv run python tools/run_pip_audit.py

# Run type checker
typecheck:
	@uv run ty check --error-on-warning -qq .

# Run tests with coverage (100% minimum)
test:
	@uv run python tools/run_pytest.py --strict-config --strict-markers --maxfail=1 --cov-fail-under=100 -q --no-header --no-summary --cov-report=

# Run all checks (format check, lint, typecheck, bandit, pip-audit, test)
check: format-check lint typecheck bandit pip-audit test

# Run all checks and fixes
all: format lint-fix bandit pip-audit typecheck test

# Clean cache files
clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
