# CLAUDE.md

Quick-reference for AI assistants working in the `weakincentives` repository.

## What this repo is

A small, modular Python toolkit for building reliable AI agents. The base
install ships only the **spine** (`src/weakincentives/core/`):

- `errors.py` — stable `WinkError` hierarchy.
- `prompt.py` — `Section[ParamsT]`, `MarkdownSection`, `Prompt`,
  `RenderedPrompt`.
- `tool.py` — `Tool`, `ToolResult`, `ToolContext`, `ToolHandler`, `ToolPolicy`.
- `session.py` — `Session`, `SliceAccessor`, `Replace`, `Append`, `@reducer`,
  spine-emitted `PromptRendered` / `ToolInvoked` / `PromptExecuted` events.
- `snapshot.py` — `Snapshot` (with `to_json` / `from_json`), `TypeRegistry`,
  `Snapshotable`, `capture`, `restore`.
- `transactions.py` — `tool_transaction`, `execute_tool`, `PendingToolTracker`.
- `protocols.py` — `ProviderAdapter`, `EventListener`, `ResourceProvider`,
  `Deadline`, `Budget`, `Usage`, `ToolCall`, `PromptResponse`.

Layered extras (all stdlib-only) on top of the spine:

**Layer 1 — foundations:**

- `weakincentives.clock` — clocks, `Deadline`, `Budget`, `BudgetTracker`.
- `weakincentives.serde` — `parse`/`dump` for nested dataclasses with
  constraints and polymorphic unions.
- `weakincentives.dbc` — `@require`, `@ensure`, `@invariant`.

**Layer 2 — state and IO:**

- `weakincentives.filesystem` — `Filesystem` + snapshotable
  `InMemoryFilesystem`.
- `weakincentives.resources` — scoped DI container.

**Layer 3 — orchestration:**

- `weakincentives.transcript` — append-only event log.
- `weakincentives.runtime` — `AgentLoop` driving a `ProviderAdapter`.

**Layer 4 — quality and observability:**

- `weakincentives.evals` — datasets, evaluators, runner.
- `weakincentives.debug` — `DebugBundle` with JSON round-trip.

**Layer 5 — provider integrations:**

- `weakincentives.adapters.noop` — scripted test adapter.
- Real adapters (`openai`, `claude`, `litellm`, `acp`, `codex`) arrive
  behind their own pip extras as they are rebuilt.

Each subpackage imports only the layers strictly below it. The layered
rules and migration plan live in `specs/ARCHITECTURE.md`; the spine is
documented in `specs/SPINE.md`.

## Definition of Done

No work is considered complete until `make check` passes:

```bash
make check  # ruff format + lint, pyright strict, ty, markdown, pytest
```

This runs:

- `ruff format --check` and `ruff check` (lint)
- `pyright` in strict mode + `ty` on `src` and `tests`
- `mdformat --check` on every `*.md`
- `pytest` with 100% line and branch coverage required

If any check fails, fix it. Do not lower the bar.

## Commands

```bash
uv sync           # install dependencies
make format       # auto-format with ruff
make lint         # ruff check
make typecheck    # pyright + ty
make test         # full pytest with coverage
make markdown-fix # format markdown files
make check        # everything
```

## Style

- 88-character lines, double quotes, ruff-format.
- Strict pyright + `ty`. Annotations are the source of truth.
- `@dataclass(frozen=True, slots=True)` for value types.
- Use `@reducer(on=Event)` on frozen dataclasses for state slices; install
  them with `session.install(SliceClass)`.
- Public names listed in `__all__`. Anything not exported is private.
- Errors must subclass `WinkError`; choose the right category in
  `core/errors.py`.

## Spine invariants

These are enforced by tests and reviewed at every change:

- `weakincentives.core` imports nothing outside the standard library and
  itself.
- Top-level `weakincentives.__all__` matches `weakincentives.core.__all__`
  exactly.
- Every reachable code path is covered (`pytest --cov-fail-under=100`).
- Every module is under 720 lines; every function under 120.

## Where to read first

| Topic | File |
| --- | --- |
| Layered package architecture | `specs/ARCHITECTURE.md` |
| Spine design and rationale | `specs/SPINE.md` |
| Philosophy | `specs/POLICIES_OVER_WORKFLOWS.md` |
| Public surface (spine) | `src/weakincentives/core/__init__.py` |
| Error hierarchy | `src/weakincentives/core/errors.py` |
| Extension points | `src/weakincentives/core/protocols.py` |
