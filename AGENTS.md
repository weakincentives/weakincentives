# AGENTS.md

Canonical guidance for agents working in the `weakincentives` repository. All
work must comply with this file.

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
- Mutation testing protects behavioral intent. When you add tests for areas
  covered by `mutation.toml` (runtime session reducers and serde hotspots), run
  `make mutation-test` locally to ensure new assertions kill the generated
  mutants. Survivors indicate missing checks; extend the test to exercise the
  mutated behavior before relying on the mutation gate in CI.

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

- **ADAPTERS.md**: Adapter responsibilities and registration rules—before adding
  or modifying adapters.
- **ASTEVAL.md**: Asteval tool behavior—when touching expression evaluation or
  sandboxing.
- **DATACLASS_SERDE.md**: Dataclass serialization contract—when adding serde
  helpers or dataclass models.
- **DBC.md**: Design-by-contract patterns—mandatory before editing DbC-covered
  modules.
- **DEADLINES.md**: Deadline propagation—when adding timeouts or scheduling
  logic.
- **EVENTS.md**: Event model—when emitting or handling runtime events.
- **LITE_LLM_ADAPTER.md**: Lite LLM adapter rules—before touching lightweight
  adapter integrations.
- **LOGGING.md**: Logging expectations—when adjusting logging surfaces.
- **NATIVE_OPENAI_STRUCTURED_OUTPUTS.md**: Native structured outputs for OpenAI
  models—when implementing or updating OpenAI structured outputs.
- **OPENAI_RESPONSES_API.md**: OpenAI responses API behaviors—when modifying
  OpenAI adapter responses.
- **PLANNING_STRATEGIES.md**: Planning approach inventory—when changing planner
  logic.
- **PLANNING_TOOL.md**: Planning tool contract—when editing the planning tool or
  its prompts.
- **PODMAN_SANDBOX.md**: Podman-based sandboxing—when integrating or altering
  sandbox execution.
- **PROMPTS.md**: Prompt abstraction deep dive—required for prompt system work.
- **PROMPTS_COMPOSITION.md**: Prompt composition rules—when combining prompt
  components.
- **PROMPT_OVERRIDES.md**: Override mechanics—when altering runtime prompt
  override behavior.
- **SESSIONS.md**: Session lifecycle and orchestration—when modifying runtime
  session code.
- **STRUCTURED_OUTPUT.md**: Structured output handling—when adjusting schema or
  parser logic.
- **SUBAGENTS.md**: Sub-agent orchestration—when coordinating nested agents.
- **THREAD_SAFETY.md**: Thread safety guarantees—when touching concurrency
  helpers or shared state.
- **THROTTLING.md**: Throttling rules—when adjusting rate
  limiting or throttling paths.
- **TOOLS.md**: Tool catalog and contracts—before adding or modifying tools.
- **VFS_TOOLS.md**: Virtual file system tools—when editing VFS behavior.
- **WINK_CLI.md**: `wink` CLI design—when changing CLI surfaces.
- **WORKSPACE_DIGEST.md**: Workspace digest mechanism—when updating digest
  calculations or usage.

## Quick Reference

- `README.md`: Public overview.
- `ROADMAP.md`, `WARP.md`: Strategy and automation.
- `GLOSSARY.md`: Terminology canon.
- `CHANGELOG.md`: Track user-visible behavior under “Unreleased”.

Use this file as the authoritative guide; keep it updated when processes,
architecture, or expectations change.
