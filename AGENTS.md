# AGENTS.md

This handbook is the canonical reference for agents contributing to the
`weakincentives` repository. Follow it when planning, implementing, and
reviewing work—other entry points (WARP scripts, READMEs, etc.) should defer to
this document.

When making changes, simplify and reason from first principles rather than
layering extra indirection.

## Snapshot

- **Goal**: Build a rigorously tested, design-by-contract oriented platform for
  orchestrating "side effect free" background agents.
- **Primary language**: Python 3.12.
- **Packaging & tasks**: [`uv`](https://github.com/astral-sh/uv) drives
  dependency resolution and command execution. `hatchling` is the build backend.
- **Style & linting**: Ruff handles formatting and linting with Python 3.14
  syntax enabled. The repository is formatted using `ruff format` (88 char line
  limit).

## Repository Tour

```
.
├── build/                  # Wrapper scripts for CI-friendly tooling
├── hooks/                  # Symlink-friendly git hooks (installed via script)
├── integration-tests/      # OpenAI adapter smoke tests (API key required)
├── specs/                  # Product and design-by-contract specifications
├── src/weakincentives/     # Library source code
│   ├── adapters/           # Model & tool adapter implementations
│   ├── cli/                # Command line entrypoints (e.g., `wink` demo)
│   ├── dbc/                # Contract decorators & enforcement helpers
│   ├── prompt/             # Prompt abstraction (sections, chapters, tools)
│   ├── runtime/            # Session orchestration and event loop helpers
│   ├── serde/              # Serialization helpers and codecs
│   └── tools/              # Built-in tool handlers (planning, VFS, asteval)
├── tests/                  # Pytest suite (unit + contract verification)
│   └── plugins/            # Custom pytest plugins (DbC, threadstress, serde)
└── test-repositories/      # Fixtures used by serde and tool contract tests
```

## Environment & Setup

- **Step 1**: Install Python 3.12 (via pyenv or your preferred manager).

- **Step 2**: Install `uv`.

- **Step 3**: Bootstrap the workspace:

  ```bash
  uv sync
  ./install-hooks.sh
  ```

  The hook installer symlinks everything under `hooks/` into `.git/hooks/`.

> **Contract**: Do not commit without running `make check`. The pre-commit hook
> expects a clean run.

## Day-to-Day Commands (Always via `uv run`)

All targets live in the `Makefile` and run silently on success. Mirror that
behavior when invoking tools manually so downstream agents do not waste tokens
parsing noise.

- `make format`: Apply `ruff format -q .`.
- `make format-check`: Audit formatting without modifying files.
- `make lint` / `make lint-fix`: Run `ruff check --preview -q` (add `--fix` when
  autofixing).
- `make bandit`: Security scan through `build/run_bandit.py`.
- `make deptry`: Detect unused or missing dependencies via `build/run_deptry.py`.
- `make pip-audit`: Audit dependency vulnerabilities with
  `build/run_pip_audit.py`.
- `make markdown-check`: Enforce Markdown formatting via
  `build/run_mdformat.py`.
- `make typecheck`: Execute `ty check --error-on-warning -qq` followed by
  `pyright` (reruns verbosely on failure).
- `make test`: Run pytest through `build/run_pytest.py` with strict config,
  strict markers, `--maxfail=1`, and a **100% coverage floor**.
- `make integration-tests`: Exercise adapter integration scenarios (requires
  `OPENAI_API_KEY`).
- `make check`: Aggregate format-check, lint, typecheck, security, dependency,
  Markdown, and test stages.
- `make all`: Run fixers plus the full verification suite (`format`,
  `lint-fix`, etc.).
- `make clean`: Remove caches (`__pycache__`, `.ruff_cache`, `.pytest_cache`).

## Testing Doctrine

- **Coverage**: The test suite must remain at 100% coverage on
  `src/weakincentives`. Any gap is a regression. `build/run_pytest.py` only emits
  output when failures or warnings occur; investigate any noise immediately.
- **Strict Pytest**: Configuration uses `--strict-config` and `--strict-markers`.
  Add new markers to `pyproject.toml` if needed.
- **Retries**: `pytest-rerunfailures` reruns each failing test twice (0.5 s delay)
  so transient threadstress flakes cannot mask persistent regressions—if a test
  fails after the retries it is a real failure.
- **Plugins**:
  - `tests/plugins/dbc.py` enforces pre/post/invariant contracts during tests.
  - `tests/plugins/threadstress.py` repeats marked tests across randomized
    thread pool sizes (`--threadstress-*` flags control intensity).
  - `tests/plugins/dataclass_serde.py` ensures serde helpers follow the documented
    dataclass contract.
- **Extras**: `uv run --all-extras` is wired into `make test` and related targets
  so optional integrations stay exercised.
- **Integration Tests**: Located in `integration-tests/` and excluded from the
  default pipeline. Run them before releasing or when touching OpenAI adapters.

## Design by Contract Expectations

- Read `specs/DBC.md` before editing any module covered by DbC rules.
- Decorate public APIs with `@require`, `@ensure`, `@invariant`, and `@pure`
  (from `weakincentives.dbc`). Contracts are part of the runtime behavior—tests
  and plugins assert they fire.
- Contract predicates may return `bool` or `(bool, message)` tuples. Raise clear
  `AssertionError`s when contracts fail. Avoid catching these errors except in
  tests that intentionally assert on them.
- State-managing classes should maintain invariants via `@invariant`. Use
  `skip_invariant` on helper methods to limit noise without weakening checks.
- When contracts need fixtures or factories, place them under `tests/helpers/`
  so they can be shared across suites.

## TDD Workflow (Non-Negotiable)

- **Step 1**: Ground yourself in the relevant spec (see `specs/`) and prior
  plans.
- **Step 2**: Start every iteration by writing a failing test.
  - Target the narrowest scope possible (single test module/case).
  - Run it directly: `uv run pytest path/to/test.py -k name`.
- **Step 3**: Implement the minimum production change to make the test pass.
- **Step 4**: Rerun the focused test. Once green, escalate to `make test`, then
  `make check`.
- **Step 5**: Refactor immediately after each green iteration. Keep contracts,
  typing, and docs aligned with the new behavior.
- **Step 6**: Update `CHANGELOG.md` whenever you introduce user-visible
  behavior.

## Source Code Conventions

- The package exports types via `src/weakincentives/__init__.py`; maintain the
  public surface thoughtfully and keep it typed (`py.typed` is present).
- Keep module-level side effects minimal; `tests/test_thread_safety.py` relies on
  predictable initialization.
- Prefer composition and small helpers—large features usually introduce new
  modules rather than bloating existing ones.
- When adding tools or adapters, also update the relevant registries and ensure
  serde and runtime tests cover the behavior end-to-end.
- Treat type annotations as the single source of truth for data validation.
  Model inputs precisely (using `TypeVar` bounds, `TypedDict`s, etc.) instead of
  layering `isinstance` checks, normalization shims, or redundant guards—assume
  the type checker runs for every caller.

## Release & Versioning

- Releases are manual. Tag the commit with `vX.Y.Z` (the `v` prefix is required)
  and push the tag.
- `CHANGELOG.md` should always describe unreleased changes under an "Unreleased"
  section so tagging is painless.

## Quick Reference

- `README.md`: Public-facing overview.
- `ROADMAP.md`, `WARP.md`: Strategy and automation guides.
- `GLOSSARY.md`: Terminology canon.
- `specs/PROMPTS.md`: Deep dive into the prompt abstraction; required reading for
  prompt work.
- `specs/DBC.md`: Contract patterns and rationale.

Use this document as the authoritative guide. If tooling, architecture, or
expectations shift, update this file in the same change set.
