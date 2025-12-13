.PHONY: format check test lint ty pyright typecheck bandit vulture deptry pip-audit markdown-check integration-tests mutation-test mutation-check demo demo-podman demo-claude-agent all clean

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

# Validate Markdown formatting
markdown-check:
	@uv run python build/run_mdformat.py

# Run ty type checker
ty:
	@uv run --all-extras ty check --error-on-warning -qq \
		--exclude 'test-repositories/**' --exclude '.git/**' \
		--exclude '.uv-cache/**' --exclude '.venv/**' --exclude 'dist/**' . || \
			(echo "ty check failed; rerunning with verbose output..." >&2; \
			uv run --all-extras ty check --error-on-warning \
			--exclude 'test-repositories/**' --exclude '.git/**' \
			--exclude '.uv-cache/**' --exclude '.venv/**' --exclude 'dist/**' .)

# Run pyright type checker
pyright:
	@uv run --all-extras pyright --project pyproject.toml || \
		(echo "pyright failed; rerunning with verbose output..." >&2; \
		uv run --all-extras pyright --project pyproject.toml --verbose)

# Run all type checkers
typecheck: ty pyright

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

# Run all checks (format check, lint, typecheck, bandit, vulture, deptry, pip-audit, markdown, test)
check: format-check lint typecheck bandit vulture deptry pip-audit markdown-check test

# Run all checks and fixes
all: format lint-fix bandit deptry pip-audit typecheck test

# Clean cache files
clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
