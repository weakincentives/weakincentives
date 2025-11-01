# ASTEVAL Evaluation Tool Specification

## Overview

This document defines how to embed the [asteval](https://github.com/lmfit/asteval)
interpreter inside the weakincentives tool stack so language models can evaluate
small Python expressions. The tool executes inside the agent runtime and
operates entirely on the in-memory Virtual File System (VFS) snapshot described
in `specs/VFS_TOOLS.md`. It exposes a single read–eval–print surface that
accepts an expression string, optional helper definitions, and an optional list
of read/write file operations to run before or after evaluation.

## Goals

- **Deterministic Sandbox** – Provide a constrained Python expression evaluator
  with predictable globals, no network or host filesystem access, and bounded
  runtime.
- **VFS Bridge** – Let expressions interact with the session-scoped VFS through
  strongly typed helper hooks (read, write, list), making file manipulation part
  of a single tool call.
- **Prompt-Friendly Contract** – Model parameters and results with dataclasses
  so prompt sections can advertise the tool with structured JSON schemas.
- **Traceability** – Capture stdout, stderr, and VFS mutations so orchestrators
  can relay comprehensive results back to the LLM and persist them in session
  state reducers.

## Non-Goals

- Running arbitrary Python modules, packages, or long-running scripts.
- Supporting asynchronous execution or yielding partial results.
- Allowing expressions to access the host filesystem directly.

## Module Layout

- Implementation lives in `weakincentives.tools.asteval`.
- The module exports `AstevalSection` as its primary entry point, mirroring the
  `PlanningToolsSection` and `VfsToolsSection` patterns. The section is
  responsible for instantiating the tool, registering it with the prompt, and
  surfacing human-readable documentation of capabilities and limits.
- Tests reside in `tests/tools/test_asteval_tool.py` and use the public tool
  contract, not internal helpers.
- Supporting utilities (e.g. stdout capture) may live in
  `weakincentives.tools._shared` if they are reused by other tools.

## External Dependency

Add `asteval>=1.0.6` to `pyproject.toml` and sync it through `uv`. The module
must import from `asteval` lazily inside the handler to avoid import costs when
unused. Document the dependency in `CHANGELOG.md` under the "Unreleased"
section.

## Tool Contract

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from weakincentives.prompt import Tool, ToolResult
from weakincentives.tools.vfs import VfsPath

ExpressionMode = Literal["expr", "statements"]


@dataclass(slots=True, frozen=True)
class EvalFileRead:
    path: VfsPath


@dataclass(slots=True, frozen=True)
class EvalFileWrite:
    path: VfsPath
    content: str
    mode: Literal["create", "overwrite", "append"] = "create"


@dataclass(slots=True, frozen=True)
class EvalParams:
    code: str
    mode: ExpressionMode = "expr"
    globals: dict[str, str] = field(default_factory=dict)
    reads: tuple[EvalFileRead, ...] = field(default_factory=tuple)
    writes: tuple[EvalFileWrite, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class EvalResult:
    value_repr: str | None
    stdout: str
    stderr: str
    globals: dict[str, str]
    reads: tuple[EvalFileRead, ...]
    writes: tuple[EvalFileWrite, ...]


eval_tool = Tool[EvalParams, EvalResult](
    name="evaluate_python",
    description="Evaluate a short Python expression in a sandboxed environment with optional VFS access.",
    handler=run_asteval,
)
```

`run_asteval` is a synchronous handler that returns a `ToolResult` instance
directly; no asynchronous facade is provided.

### Parameter Semantics

- `code` – Python source limited to ≤ 2_000 characters. Validation rejects
  control characters outside tab/newline.
- `mode` – `"expr"` returns the expression result and disallows assignments;
  `"statements"` allows multi-line statements and returns the last expression
  value if present.
- `globals` – Optional dictionary of variable names to JSON-serializable string
  representations that bootstrap the interpreter state. They are parsed inside
  the handler; invalid JSON should raise a `ToolValidationError` with context.
- `reads` – A sequence of file reads to inject into `globals`. Each read loads a
  VFS file and exposes its text under `globals[path_alias]`. The alias is the
  joined POSIX path string.
- `writes` – Files to write back to the VFS after evaluation. Content can
  include template variables referencing keys from the evaluator globals. The
  handler resolves these templates using `str.format_map` against the final
  globals snapshot.

### Result Semantics

- `value_repr` – `repr` of the evaluation result or `None` if no value was
  produced.
- `stdout` / `stderr` – Captured streams produced during evaluation. Both are
  bounded to 4_096 characters; longer streams are truncated with an ellipsis.
- `globals` – Final global variables (stringified) for transparency. Only values
  with JSON-safe primitive types (`str`, `int`, `float`, `bool`, `None`) are
  returned; others fall back to their `repr` prefixed with `"!repr:"` to
  communicate lossy conversion.
- `reads` / `writes` – Echoed input descriptors, potentially augmented with the
  final content that was written.

## Runtime Behaviour

### Interpreter Setup

- Instantiate `asteval.Interpreter(use_numpy=False, minimal=True)` for a minimal
  environment.
- Replace the interpreter's `symtable` with a new dictionary seeded with:
  - Safe math functions and constants (`math`, `statistics` subsets).
  - A `read_text(path: str) -> str` helper that resolves against the current VFS
    snapshot (detailed below).
  - A `write_text(path: str, content: str, mode: str = "overwrite")` helper that
    queues VFS writes for the reducer.
- Disable attributes listed in `asteval.ALL_DISALLOWED` and clear `node_handlers`
  for callables that are not pure (e.g. `exec`, `eval`).
- Set `Interpreter.error` to an empty list before each run; capture errors after
  execution and map them into `stderr`.

### Execution Flow

1. Validate parameters, including code length, ASCII check, deduplicated paths,
   and disallowing overlapping read/write targets.
1. Load requested `reads` from the VFS snapshot (see below) and insert each file
   under its POSIX path key in the globals mapping.
1. Merge caller-provided `globals` by parsing their JSON and injecting values
   into the interpreter symtable.
1. Capture stdout/stderr using `contextlib.redirect_stdout` /
   `redirect_stderr` into `io.StringIO` buffers.
1. If `mode == "expr"`, call `interpreter.eval(code)`. For statements, use
   `interpreter(text=code, parse=True)` and return the last expression result if
   present.
1. On any `asteval` error or Python exception, return a failed `ToolResult` with
   `stderr` populated and `value_repr=None`. The tool should not raise unless a
   non-recoverable configuration error occurs.
1. Apply queued writes using the VFS reducer (next section) and populate the
   result payload.

### Timeout Guard

- Wrap the synchronous handler in a blocking timeout guard that fails fast after
  5 seconds. Prefer `signal.setitimer(signal.ITIMER_REAL, 5.0)` on Unix; fall
  back to a worker thread with `join(timeout=5.0)` on platforms without signal
  support.
- On timeout, cancel execution and return `stderr="Execution timed out."` while
  discarding any queued writes.

## VFS Integration

The tool operates against the session's `VirtualFileSystem` snapshot:

- Accept `VirtualFileSystem` via dependency injection (e.g. handler closure or
  section factory) so the tool stays pure with respect to session state.
- Reads resolve via `select_latest(session, VirtualFileSystem)`; missing files
  raise `ToolValidationError` with a helpful message.
- The entire VFS namespace is available to evaluation code—no additional
  whitelisting or host mount mirrors are required.
- Writes execute through the existing reducer from `VfsToolsSection`. The
  handler enqueues `WriteFile` operations and dispatches them via
  `session.reduce(replace_latest, VirtualFileSystem, updated_snapshot)`.
- Writes obey the same guards as native VFS tools: ASCII-only, max 48_000
  characters, depth ≤ 16, segment length ≤ 80.
- When `write_text` helper is used inside code, it reuses the same queueing path
  to ensure atomic updates after evaluation completes.

## Prompt Integration

- Provide a dedicated section class, `AstevalSection`, in
  `weakincentives.tools.asteval`. The section is the public entry point: it
  registers `eval_tool` with the prompt and emits markdown that summarizes key
  capabilities, limits, timeout behaviour, and safety rules.
- The section accepts a `Session` and the shared VFS handle only; it should note
  that all files currently tracked in the VFS are available to reads and writes.
- Sections should describe the helper functions (`read_text`, `write_text`) and
  remind models to keep code short to avoid timeouts.

## Telemetry & Logging

- Emit debug logs (structured dictionaries) for each evaluation run containing:
  `{"event": "asteval.run", "mode": mode, "stdout_len": len(stdout), "stderr_len": len(stderr), "write_count": len(writes)}`.
- Log truncated `code` (first 200 characters) for observability while avoiding
  leaking full scripts.

## Testing Strategy

- Unit tests cover validation failures (bad paths, oversized code, invalid JSON
  globals, overlapping writes).
- Functional tests run both expression and statement modes, verifying stdout,
  stderr, globals persistence, and VFS read/write behaviour.
- Include a regression test demonstrating timeout handling by evaluating an
  infinite loop and confirming the tool returns a timeout error without side
  effects.

## Documentation

- Update `docs/` with a short guide referencing this spec once the tool ships.
- Add the tool to `specs/TOOLS.md` examples when ready so new contributors see
  how custom tools integrate into prompts.
