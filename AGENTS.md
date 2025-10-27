# AGENTS.md

This handbook is the primary source of truth for autonomous or assisted agents working in the `weakincentives` repository. WARP and other entrypoints should defer to this document.

## Project Snapshot
- **Goal**: Build tooling for "side effect free" background agents; current codebase is intentionally minimal and pre-alpha.
- **Language**: Python 3.14 (see `.python-version`).
- **Package manager**: [`uv`](https://github.com/astral-sh/uv) orchestrates dependency management and task execution.
- **Build backend**: `hatchling` (configured in `pyproject.toml`).

## Repository Tour
- `src/weakincentives/`: Library source. Presently only exposes `hello()` as a placeholder entry point to keep packaging sane.
- `tests/`: Pytest suite. `tests/test_example.py` demonstrates the expected structure and ensures the package imports.
- `specs/`: Design docs and product specifications. `PROMPTS.md` defines the prompt abstraction requirements—read this before adding prompt-related code.
- `Makefile`: Canonical task surface (formatting, linting, typing, tests, aggregate checks, clean-up).
- `uv.lock`: Dependency lockfile maintained by `uv`.
- `hooks/` & `install-hooks.sh`: Symlink-based Git hook installer. Hooks expect `make check` to pass before commits land.
- `bump-version.sh`: Helper to increment the version in `pyproject.toml`; supports `major|minor|patch` or an explicit semantic version.

## Environment & Setup
1. Install Python 3.14 (pyenv users can run `pyenv install 3.14.0` if needed).
2. Install `uv` locally.
3. Sync dependencies and development tooling:
   ```bash
   uv sync
   ./install-hooks.sh
   ```
   The hook installer wires every script in `hooks/` into `.git/hooks/` via symlinks.

## Day-to-Day Commands
All commands are defined in `Makefile` and should be executed with `uv` to ensure a consistent virtualenv.

- `make format`: Auto-format via `ruff format`.
- `make format-check`: Formatting audit without changes.
- `make lint` / `make lint-fix`: Static analysis with Ruff (`--fix` enables autofixes).
- `make typecheck`: Runs `ty check .` (targeting Python 3.14).
- `make test`: Executes pytest with coverage thresholds enforced (`--cov-fail-under=80`).
- `make check`: Aggregates format-check, lint, typecheck, and tests.
- `make all`: Runs format, lint-fix, typecheck, and tests for fully automated cleanup.
- `make clean`: Purges caches (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `*.pyc`).

Prefer `make check` before every commit; git hooks will call the same pipeline.

## Testing & Quality Expectations
- Pytest is configured in `pyproject.toml` to collect coverage on `src/weakincentives` and fail if coverage dips below 80%.
- Type hints are part of the public contract (`py.typed`). Keep new code fully typed and run `make typecheck` when touching typing-sensitive areas.
- Ruff is both the formatter and the linter. Obey the default line length of 88 and the Python target version `py314`.
- Lint runs enable `I`, `UP`, `B`, `SIM`, `C4`, `ANN`, `RET`, `RSE`, `PTH`, and `ISC` rule families; fix or explicitly justify any violations when contributing.

## Release & Versioning Notes
- The package version lives in `pyproject.toml`. Use `./bump-version.sh [major|minor|patch|X.Y.Z]` to update it; the script prints next steps for tagging and pushing.
- No publishing automation exists yet—coordinate manual releases outside this repository.

## Adding Features
- Mirror the structure in `tests/` when adding new modules—write tests alongside new code and keep them deterministic.
- Follow test-driven development best practices while iterating; commit to the red→green→refactor loop so features stay focused and covered.
- Reference `specs/PROMPTS.md` before modifying or introducing prompt-related functionality.
- Write commit messages with useful descriptions that summarize the change set; avoid placeholder or empty messages.
- Keep the `README.md` concise; extended onboarding belongs here in `AGENTS.md`.
- When introducing new tooling, extend the `Makefile` so agents can discover it without hunting through scripts.

## Quick Facts
- `README.md` offers a public-facing overview; this document captures agent-facing operational details.
- The codebase is intentionally small; expect to create new modules under `src/weakincentives/` rather than modifying many existing files.
- All documentation should be ASCII unless the surrounding file already uses other characters.

Use this document as the authoritative onboarding and execution guide for automated agents. Update it whenever workflows or tooling change.
