.PHONY: check format format-check lint typecheck test markdown markdown-fix all clean

all: check

check: format-check lint typecheck markdown test
	@echo "✓ All checks passed"

format:
	@uv run ruff format -q .

format-check:
	@uv run ruff format --check .

lint:
	@uv run ruff check .

lint-fix:
	@uv run ruff check --fix .

typecheck:
	@uv run pyright
	@uv run ty check src tests

test:
	@uv run pytest

markdown:
	@uv run mdformat --check $$(find . -name '*.md' -not -path './.venv/*')

markdown-fix:
	@uv run mdformat $$(find . -name '*.md' -not -path './.venv/*')

clean:
	@rm -rf .pytest_cache .ruff_cache .coverage dist build
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
