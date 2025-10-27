.PHONY: format check test lint typecheck all clean

# Format code with ruff
format:
	uv run ruff format .

# Check formatting without making changes
format-check:
	uv run ruff format --check .

# Run ruff linter
lint:
	uv run ruff check --preview --no-cache .

# Run ruff linter with fixes
lint-fix:
	uv run ruff check --fix .

# Run type checker
typecheck:
	uv run ty check --error-on-warning .

# Run tests with coverage (100% minimum)
test:
	uv run pytest --strict-config --strict-markers --maxfail=1 --cov-fail-under=100

# Run all checks (format check, lint, typecheck, test)
check: format-check lint typecheck test

# Run all checks and fixes
all: format lint-fix typecheck test

# Clean cache files
clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
