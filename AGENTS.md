# AGENTS.md

Canonical guidance for agents working in the `weakincentives` repository. All
work must comply with this file.

## Stability Notice

This project is alpha-quality software. All APIs may change in backwards
incompatible ways without notice, and maintaining backward compatibility is not
a goal at any time.

## Architectural Overview

- **Domain**: A design-by-contract platform for orchestrating side-effect-free
  background agents.
- **Runtime core** (`src/weakincentives/runtime/`): Session orchestration,
  concurrency helpers, and event loops.
- **Prompt system** (`src/weakincentives/prompt/`): Sections and tool
  wiring for model-facing prompts.
- **Adapters** (`src/weakincentives/adapters/`): Model and tool adapters, plus
  structured-output integration.
- **Tools** (`src/weakincentives/tools/`): Built-in planning, VFS, and asteval
  handlers.
- **Contracts** (`src/weakincentives/dbc/`): Design-by-contract decorators and
  enforcement utilities that gate public APIs.
- **Serde** (`src/weakincentives/serde/`): Serialization helpers used by tools
  and adapters.
- **CLI** (`src/weakincentives/cli/`): User entrypoints such as the `wink`
  demonstration command.

## Workflow & Environment

1. Install Python 3.12 and `uv`, then bootstrap with `uv sync` and
   `./install-hooks.sh`.
1. Use `uv run` with the `Makefile` targets. Silence is success.
1. Run **`make check` before every commit**; hooks expect a clean run, and a
   request is only complete once `make check` has succeeded regardless of the
   changes involved.

### Key Commands

- `make format` / `make format-check`: `ruff format` (88-char lines).
- `make lint` / `make lint-fix`: `ruff check --preview` (autofix optional).
- `make bandit`, `make deptry`, `make pip-audit`, `make markdown-check`:
  security, dependency, and Markdown enforcement.
- `make typecheck`: `ty check --error-on-warning -qq` then `pyright`.
- `make test`: Pytest via `build/run_pytest.py`, strict markers, coverage floor
  at 100% for `src/weakincentives`.
- `make integration-tests`: OpenAI adapter scenarios (requires `OPENAI_API_KEY`).
- `make check`: Aggregates format-check, lint, typecheck, security, dependency,
  Markdown, and tests.

## Testing & Contracts

- Uphold design-by-contract. Decorate public APIs with `@require`, `@ensure`,
  `@invariant`, and `@pure` from `weakincentives.dbc`.
- Coverage for `src/weakincentives` must remain 100%. Focused test runs may hit
  the coverage floor; finish with `make test`.
- Pytest is strict (`--strict-config`, `--strict-markers`) and retries flakes
  twice. Add fixtures to `tests/helpers/` when contracts need support.

## Code Conventions

- Keep module side effects minimal; thread-safety tests rely on predictable
  initialization.
- Favor composition and small helpers; prefer adding modules over bloating
  existing ones.
- Treat type annotations as the source of truth; model inputs precisely rather
  than layering runtime guards.
- Update registries and tests when adding adapters or tools; keep public exports
  curated via `src/weakincentives/__init__.py`.
- Use shared helpers for cross-cutting concerns (logging, deadlines, rollback,
  locking) instead of re-implementing them per call site.

## Spec Index (when to consult)

- **ADAPTERS.md**: Provider adapters, structured output, and throttling—before
  adding or modifying adapters.
- **DATACLASSES.md**: Serde utilities and frozen dataclass patterns—when adding
  dataclass models or serialization.
- **DBC.md**: Design-by-contract patterns—mandatory before editing DbC-covered
  modules.
- **EVALS.md**: Evaluation framework—when implementing evals, datasets,
  evaluators, or metrics aggregation.
- **EXHAUSTIVENESS.md**: Union type totality and `assert_never` patterns—when
  adding union variants or match statements on events and operations.
- **FILESYSTEM.md**: Filesystem protocol, backend implementations, and
  ToolContext integration—when adding or modifying filesystem operations.
- **LOGGING.md**: Logging expectations—when adjusting logging surfaces.
- **POLICIES_OVER_WORKFLOWS.md**: Design philosophy—preferring declarative
  policies over rigid workflows for unattended agent autonomy.
- **PROMPTS.md**: Prompt abstraction, structured output, composition, and
  progressive disclosure—required for prompt system work.
- **PROMPT_OPTIMIZATION.md**: Override system and optimizer abstraction—when
  altering prompt overrides or optimization.
- **SESSIONS.md**: Session lifecycle, events, deadlines, and budgets—when
  modifying runtime session code.
- **SLICES.md**: Slice storage protocol, factory configuration, and JSONL
  persistence—when modifying slice storage or adding persistence backends.
- **THREAD_SAFETY.md**: Thread safety guarantees—when touching concurrency
  helpers or shared state.
- **TOOLS.md**: Tool catalog, planning tools, and strategies—before adding or
  modifying tools.
- **TOOL_POLICIES.md**: Sequential tool dependencies (read-before-write, keyed
  constraints)—when adding policy enforcement to tools.
- **FEEDBACK_PROVIDERS.md**: Ongoing progress feedback for unattended
  agents—stall detection, drift monitoring, and context injection.
- **FORMAL_VERIFICATION.md**: Embedding TLA+ specifications in Python using
  `@formal_spec` decorator, running TLC model checker, and verification
  testing—when adding formal specs or modifying verified code.
- **VERIFICATION.md**: Redis mailbox detailed specification with invariants
  and property-based tests—when modifying mailbox implementations.
- **WORKSPACE.md**: VFS, Podman sandbox, asteval, and workspace digest—when
  editing workspace tooling.

## Guides Index (how-to material)

- **guides/code-review-agent.md**: End-to-end walkthrough of building a code
  review agent using `MainLoop`, workspace tools, and planning.

Guides cover quickstarts, patterns, recipes, and best practices. Design specs
(what the system guarantees) live in `specs/`.

## Quick Reference

- `README.md`: Public overview.
- `ROADMAP.md`, `WARP.md`: Strategy and automation.
- `GLOSSARY.md`: Terminology canon.
- `CHANGELOG.md`: Track user-visible behavior under “Unreleased”.

Use this file as the authoritative guide; keep it updated when processes,
architecture, or expectations change.
