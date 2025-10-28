.PHONY: format format-check lint lint-fix bandit markdown-check deptry pip-audit typecheck test check all clean

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

# Run mdformat in check mode across Markdown files
markdown-check:
	@uv run python tools/run_mdformat.py

# Check for unused or missing dependencies with deptry
deptry:
	@uv run python tools/run_deptry.py

# Run pip-audit for dependency vulnerabilities
pip-audit:
	@uv run python tools/run_pip_audit.py

# Run type checker
typecheck:
	@uv run --all-extras ty check --error-on-warning -qq . || { \
	echo "ty check failed; rerunning with verbose output..." >&2; \
	uv run --all-extras ty check --error-on-warning .; }

# Run tests with coverage (100% minimum)
test:
	@uv run --all-extras python tools/run_pytest.py --strict-config --strict-markers --maxfail=1 --cov-fail-under=100 -q --no-header --no-summary --cov-report=

# Run all checks (format check, lint, typecheck, bandit, markdown, deptry, pip-audit, test)
check: format-check lint typecheck bandit markdown-check deptry pip-audit test

# Run all checks and fixes
all: format lint-fix bandit deptry pip-audit typecheck test

# Clean cache files
clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
