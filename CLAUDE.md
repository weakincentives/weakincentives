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

Higher-level features (provider adapters, evals, debug bundles, formal
verification, skills, …) will land as opt-in extras as they are rebuilt.
Each must depend only on `weakincentives.core` and other extras.

The full design is in `specs/SPINE.md`.

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
| Spine design and rationale | `specs/SPINE.md` |
| Philosophy | `specs/POLICIES_OVER_WORKFLOWS.md` |
| Public surface | `src/weakincentives/core/__init__.py` |
| Error hierarchy | `src/weakincentives/core/errors.py` |
| Extension points | `src/weakincentives/core/protocols.py` |
