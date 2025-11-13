# AGENTS.md

This handbook is the primary source of truth for autonomous or assisted agents working in the `weakincentives` repository. WARP and other entrypoints should defer to this document.

## Project Snapshot

- **Goal**: Build tooling for "side effect free" background agents; current codebase is intentionally minimal and pre-alpha.
- **Language**: Python 3.12 (see `.python-version`).
- **Package manager**: [`uv`](https://github.com/astral-sh/uv) orchestrates dependency management and task execution.
- **Build backend**: `hatchling` (configured in `pyproject.toml`).

## Repository Tour

- `src/weakincentives/`: Library source. Currently contains the prompt scaffolding and supporting modules.
- `tests/`: Pytest suite. `tests/test_example.py` demonstrates the expected structure and ensures the package imports.
- `specs/`: Design docs and product specifications. `PROMPTS.md` defines the prompt abstraction requirements—read this before adding prompt-related code.
- `Makefile`: Canonical task surface (formatting, linting, typing, tests, aggregate checks, clean-up).
- `uv.lock`: Dependency lockfile maintained by `uv`.
- `hooks/` & `install-hooks.sh`: Symlink-based Git hook installer. Hooks expect `make check` to pass before commits land.
- Version tags (`vX.Y.Z`, the `v` prefix is required) in git control the published package version; cut them manually when releasing.

## Environment & Setup

1. Install Python 3.12 (pyenv users can run `pyenv install 3.12.0` if needed).
1. Install `uv` locally.
1. Sync dependencies and development tooling:
   ```bash
   uv sync
   ./install-hooks.sh
   ```
   The hook installer wires every script in `hooks/` into `.git/hooks/` via symlinks.

## Day-to-Day Commands

All commands are defined in `Makefile` and should be executed with `uv` to ensure a consistent virtualenv. The targets are configured to run in quiet mode—by default they suppress success chatter and surface only warnings or errors. Mirror this style when invoking tools manually so downstream agents process as few tokens as possible.

- `make format`: Auto-format via `ruff format -q`.
- `make format-check`: Formatting audit without changes (`ruff format -q --check`).
- `make lint` / `make lint-fix`: Static analysis with Ruff in quiet mode (`ruff check --preview -q`; add `--fix` when autofixing).
- `make bandit`: Security linting via `build/run_bandit.py`, which patches Python 3.14 AST regressions before invoking Bandit.
- `make deptry`: Dependency graph audit on `src/weakincentives` that surfaces unused or missing requirements while staying quiet on success.
- `make pip-audit`: Dependency vulnerability audit that stays silent on success via `build/run_pip_audit.py`.
- `make markdown-check`: Runs `mdformat --check` through `build/run_mdformat.py` to ensure Markdown stays consistently formatted. This command now runs inside `make check`.
- `make typecheck`: Runs `ty check -qq --error-on-warning` for silent success.
- `make test`: Executes pytest through `build/run_pytest.py`, only emitting failures or warnings while enforcing coverage (`--cov-fail-under=80`).
- `make check`: Aggregates format-check, lint, typecheck, Bandit, deptry, pip-audit, markdown-check, and tests with minimal output.
- `make all`: Runs format, lint-fix, Bandit, pip-audit, typecheck, and tests while staying quiet on success.
- `make clean`: Purges caches (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `*.pyc`).

Prefer `make check` before every commit; git hooks will call the same pipeline.

**You MUST run `make check` before committing any changes.**

## Optional Dependencies

- `asteval` powers the sandboxed evaluation tool. Install it with
  `uv sync --extra asteval` (or `pip install weakincentives[asteval]`) when you
  need the `AstevalSection`; core workflows remain stdlib-only without it.
- `openai` is exposed as an extra. Install it with `uv sync --extra openai` during development or `pip install weakincentives[openai]` for consumers. Adapter modules guard imports and raise clear guidance when the extra is missing, so the core package stays lean.
- `make test` (and thus `make check` / `make all`) automatically run with all extras enabled via `uv run --all-extras`, ensuring adapter integrations stay validated.

## Testing & Quality Expectations

- Pytest is configured in `pyproject.toml` to collect coverage on `src/weakincentives` and fail if coverage dips below 80%.
- Threaded regression tests use a `threadstress` plugin that repeats marked tests with randomized `ThreadPoolExecutor` sizes.\
  Use `uv run pytest --threadstress-iterations=<n>` to increase the number of runs and
  `--threadstress-max-workers=<n>` to clamp the worker range when debugging or cranking up CI stress.
- Type hints are part of the public contract (`py.typed`). Keep new code fully typed and run `make typecheck` when touching typing-sensitive areas.
- Lean on the type checker and existing type annotations—prefer static guarantees over defensive `isinstance` checks or attribute existence guards unless they are semantically required.
- Ruff is both the formatter and the linter. Obey the default line length of 88 and the Python target version `py314`.
- Lint runs enable `I`, `UP`, `B`, `SIM`, `C4`, `ANN`, `RET`, `RSE`, `PTH`, and `ISC` rule families; fix or explicitly justify any violations when contributing.
- Tool handlers must populate the new `ToolResult.success` flag—set `success=False` and leave `value=None` (or a structured error payload) when a handler fails so adapters and reducers can reason about the outcome.

## Design by Contract Framework

- Read `specs/DBC.md` before adding or modifying library code that encodes behavioural expectations.
- Use the helpers in `weakincentives.dbc` to capture preconditions, postconditions, invariants, and purity requirements whenever they clarify intent.
- `@require`/`@ensure` should guard public functions that rely on specific argument shapes or return guarantees.
- `@ensure` adds `result` or `exception` keyword arguments to the predicate scope so decorators can reason about outcomes without reshaping the positional signature.
- Wrap stateful classes such as reducers or session managers with `@invariant` and mark helper methods with `skip_invariant` when those checks would add noise.
- Mark utility functions that should remain side-effect free with `@pure`; the pytest plugin enables enforcement automatically during `make test` / `make check`.
- When contributing new modules, prefer adding the relevant decorators from the outset so downstream agents inherit strong contracts without needing refactors.

## TDD Workflow Recipe

- Read the relevant spec in `specs/` and any prior plans to anchor scope, capture target module paths, and list the concrete behaviors you will validate.
- Break the work into thin iterations that each pair a specific test module or case (for example `tests/prompts/test_text_section.py`) with the minimal production changes required in `src/weakincentives/`.
- For every iteration, author the failing test first, run it directly with `uv run pytest tests/path_to_test.py -k target_case`, and confirm it asserts the desired API, errors, and context payloads.
- Implement only the code needed to satisfy the new test (including exports and defaults), then rerun the targeted test and expand to `make test` once the local loop is green.
- Refactor immediately after the test passes: deduplicate helpers, thread depth or placeholder metadata, and update docs so the next iteration starts from a clean slate.
- Repeat the loop until the feature-level acceptance scenario is covered, finish with `make check`, and capture the final iteration summary in the commit message or handoff notes.

## Release & Versioning Notes

- Package releases read their version from the latest git tag (`vX.Y.Z`, must include the `v`). Use `git tag vX.Y.Z` followed by `git push origin vX.Y.Z` when you’re ready to release.
- No publishing automation exists yet—coordinate manual releases outside this repository.

## Adding Features

- Mirror the structure in `tests/` when adding new modules—write tests alongside new code and keep them deterministic.
- Follow test-driven development best practices while iterating; commit to the red→green→refactor loop so features stay focused and covered.
- Reference `specs/PROMPTS.md` before modifying or introducing prompt-related functionality.
- Keep `CHANGELOG.md` current—document noteworthy changes as you make them so release prep stays trivial.
- Write commit messages with useful descriptions that summarize the change set; avoid placeholder or empty messages.
- Keep the `README.md` concise; extended onboarding belongs here in `AGENTS.md`.
- When introducing new tooling, extend the `Makefile` so agents can discover it without hunting through scripts.

## Quick Facts

- `README.md` offers a public-facing overview; this document captures agent-facing operational details.
- The codebase is intentionally small; expect to create new modules under `src/weakincentives/` rather than modifying many existing files.
- All documentation should be ASCII unless the surrounding file already uses other characters.

Use this document as the authoritative onboarding and execution guide for automated agents. Update it whenever workflows or tooling change.
