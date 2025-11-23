# ASTEVAL Evaluation Tool Specification

## Overview

This document defines how to embed the [asteval](https://github.com/lmfit/asteval)
interpreter inside the weakincentives tool stack so language models can evaluate
small Python expressions. The tool executes inside the agent runtime and
operates entirely on the in-memory Virtual File System (VFS) snapshot described
in `specs/VFS_TOOLS.md`. It exposes a single read–eval–print surface that
accepts an expression string, optional helper definitions, and an optional list
of read/write file operations to run before or after evaluation.

## Rationale and Scope

### Why the tool exists

- Provide a deterministic, resource-bounded way to let agents run short Python
  snippets without breaking the "side effect free" contract. The implementation
  lives in `src/weakincentives/tools/asteval.py` and injects into prompt flows
  through `AstevalSection`, mirroring the other built-in tool sections so it can
  be advertised consistently to LLMs.
- Keep Python optional for downstream consumers by parking the dependency
  behind the `asteval` extra in `pyproject.toml`; lazy imports raise a targeted
  error instructing users to install `weakincentives[asteval]` when the module
  is absent.

### Guiding principles

- **Sandbox first** – Strip privileged nodes (import/exec) and expose only a
  whitelisted symtable of math/statistics helpers, custom `read_text`/`write_text`
  hooks, and a constrained `print`, keeping runtime behavior transparent.
- **Short, synchronous runs** – Enforce 2,000-character code limits, 4,096-char
  stream caps, and a 5-second timeout so the tool behaves predictably within the
  agent event loop instead of turning into a background worker.
- **VFS-only side effects** – Route all reads/writes through the in-memory VFS,
  validating relative ASCII paths, deduplicating targets, and templating writes
  against the final globals snapshot so edits remain explicit and reversible.
- **Contractable surface** – Model parameters and results with frozen
  dataclasses to keep prompt schemas stable and to simplify serialization inside
  `ToolResult` payloads and reducers.

### Scope boundaries and present-state caveats

- The timeout guard is implemented with a daemon thread/event rather than the
  signal-based approach envisioned in early drafts, so long-running code is
  interrupted cooperatively only when the worker thread returns.
- The packaged dependency currently targets `asteval>=1.0.7`, which supersedes
  the `>=1.0.6` minimum described in earlier planning notes. The handler still
  raises the same runtime message when the optional dependency is missing.
- The symtable includes a minimal set of builtins plus math/statistics, but it
  also exposes the raw `print` replacement used for stdout capture; additional
  helpers (like custom reducers) remain intentionally out of scope to preserve
  a narrow attack surface.
- The tool executes multi-line scripts via `interpreter.eval` and captures the
  repr of the last expression when no errors occurred; if stderr is populated
  and no value is produced, any staged writes are discarded and summarized in
  the returned message, making error handling explicit to prompt authors.

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

Expose `asteval>=1.0.6` behind a dedicated optional extra so the default
installation remains stdlib-only. Update `pyproject.toml` with an
`[project.optional-dependencies].asteval` entry, keep imports lazy inside the
handler, and raise a descriptive runtime error that instructs users to install
`weakincentives[asteval]` when the extra is missing. Document the packaging
change in `CHANGELOG.md` under the "Unreleased" section.

## Tool Contract

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from weakincentives.prompt.tools import Tool, ToolResult
from weakincentives.tools.vfs import VfsPath

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

- `code` – Python source limited to ≤ 2_000 characters. The interpreter executes
  the payload as a multi-line script and returns the repr of the final
  expression when present. Validation rejects control characters outside
  tab/newline.
- `globals` – Optional dictionary of variable names to JSON strings. Each value
  is decoded with `json.loads` before evaluation so the interpreter receives the
  underlying primitive (e.g. `int`, `float`, `bool`, `None`, `str`) or nested
  structures composed of those primitives. Payloads that fail to decode or
  resolve to unsupported types raise a `ToolValidationError` with the offending
  key name so prompt authors receive immediate feedback.
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
- `globals` – Final global variables (stringified) for transparency. Values that
  originated from decoded input globals reappear here as their JSON-safe
  primitive equivalents (`str`, `int`, `float`, `bool`, `None`) or nested
  containers composed of those primitives. If evaluation produces values that
  cannot be losslessly serialized, the handler falls back to their `repr`
  prefixed with `"!repr:"` to communicate the conversion boundary.
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

1. Validate parameters, including code length, UTF-8 check, deduplicated paths,
   and disallowing overlapping read/write targets.
1. Load requested `reads` from the VFS snapshot (see below) and insert each file
   under its POSIX path key in the globals mapping.
1. Merge caller-provided `globals` by parsing their JSON and injecting values
   into the interpreter symtable.
1. Capture stdout/stderr using `contextlib.redirect_stdout` /
   `redirect_stderr` into `io.StringIO` buffers.
1. Execute the payload with `interpreter.eval(code)` so the interpreter parses
   it as a multi-line script and returns the last expression result when
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
- Writes obey the same guards as native VFS tools: UTF-8 text only, max 48_000
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
- Include concrete examples in the section copy so agents know how to invoke the
  registered tool and what the JSON payload looks like.

### Example Tool Calls

The section documentation should embed an example payload so consumers can see
the full shape the dispatcher expects:

```json
{
  "name": "evaluate_python",
  "arguments": {
    "code": "total = 0\nfor value in range(5):\n    total += value\nprint(total)\ntotal",
    "globals": {},
    "reads": [],
    "writes": []
  }
}
```

Accompany the example with copy explaining that stdout/stderr are captured and
that the final expression result, when present, is surfaced through
`value_repr`.

## Telemetry & Logging

- Emit debug logs (structured dictionaries) for each evaluation run containing:
  `{"event": "asteval.run", "stdout_len": len(stdout), "stderr_len": len(stderr), "write_count": len(writes)}`.
- Log truncated `code` (first 200 characters) for observability while avoiding
  leaking full scripts.

## Testing Strategy

- Unit tests cover validation failures (bad paths, oversized code, invalid JSON
  globals, overlapping writes).
- Functional tests exercise single-line expressions and multi-line scripts,
  verifying stdout, stderr, globals persistence, and VFS read/write behaviour.
- Include a regression test demonstrating timeout handling by evaluating an
  infinite loop and confirming the tool returns a timeout error without side
  effects.

## Documentation

- Update `docs/` with a short guide referencing this spec once the tool ships.
- Add the tool to `specs/TOOLS.md` examples when ready so new contributors see
  how custom tools integrate into prompts.
