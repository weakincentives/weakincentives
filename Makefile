.PHONY: format check test lint typecheck all clean

# Format code with ruff
format:
	uv run ruff format .

# Check formatting without making changes
format-check:
	uv run ruff format --check .

# Run ruff linter
lint:
	uv run ruff check .

# Run ruff linter with fixes
lint-fix:
	uv run ruff check --fix .

# Run type checker
typecheck:
	uv run ty check .

# Run tests with coverage (80% minimum)
test:
	uv run pytest

# Run all checks (format check, lint, typecheck, test)
check: format-check lint typecheck test

# Run all checks and fixes
all: format lint-fix typecheck test

# Clean cache files
clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
