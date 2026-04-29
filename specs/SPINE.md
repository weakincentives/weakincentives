# SPINE Specification

## Purpose

The **spine** is the load-bearing core of `weakincentives`: the smallest set
of abstractions every higher-level feature must depend on, frozen behind a
stable public API. The base `pip install weakincentives` ships only the
spine. Provider adapters, resources, evals, debug bundles, formal
verification, skills, and similar layers will return as optional extras that
import from the spine and never the other way around.

## Principles

- **Inverted dependencies.** The spine has zero dependencies on anything
  outside the standard library. Extras depend on the spine, not vice versa.
- **Stdlib-only at the spine.** Provider SDKs, YAML, Redis, websockets, and
  any other third-party library are extras concerns.
- **Frozen API, semver discipline.** Every public symbol is listed in
  `weakincentives.core.__all__` (mirrored at `weakincentives.__all__`).
  Breaking changes to anything in `__all__` require a major-version bump.
- **Tight cluster, not single file.** Bottle-style cohesion at scale is a
  small focused folder. Each spine file does one job and stays under the
  720-line cap.
- **Thread-safe by construction.** `Session` owns an `RLock`. Callers do not
  need to know about locks to be correct under concurrent use.
- **Persistent by design.** `Snapshot.to_json` / `from_json` round-trip
  session state through JSON; resources participate via `Snapshotable`.
- **Extension via protocols, not subclassing.** Adapters, listeners,
  resource providers, and tool policies plug in by satisfying narrow
  `Protocol`s defined in `core.protocols`.

## Package Layout

```
src/weakincentives/
├── __init__.py              # public re-exports + __all__ (mirrors core)
└── core/                    # the spine — stdlib-only, frozen API
    ├── __init__.py          # public re-exports + __all__
    ├── errors.py            # WinkError hierarchy
    ├── prompt.py            # Section[ParamsT], MarkdownSection, Prompt
    ├── protocols.py         # ProviderAdapter, EventListener, Deadline, Budget …
    ├── session.py           # Session, SliceAccessor, Replace, Append, reducer
    ├── snapshot.py          # Snapshot, Snapshotable, TypeRegistry, capture/restore
    ├── tool.py              # Tool, ToolResult, ToolContext, ToolPolicy
    └── transactions.py      # tool_transaction, execute_tool, PendingToolTracker
```

Each spine file has one job. Future extras land alongside `core/` in their
own subpackages and ship behind pip extras.

## Components

### `core/errors.py`

A flat hierarchy callers pattern-match on:

```
WinkError                       (base)
├── PromptError
├── ToolError
├── SessionError
├── SnapshotError
├── TransactionError
└── ContractError
    ├── DeadlineExceededError
    ├── BudgetExceededError
    └── PolicyDeniedError
```

Every error has a stable qualified name
(`weakincentives.core.errors.WinkError`, …).

### `core/prompt.py`

- `Section[ParamsT]` — base class parameterized by a frozen-dataclass
  parameter type. Fields: `title`, `key`, `params_type`, `default_params`,
  `children`, `tools`, `enabled`. Concrete sections override `render_body`.
- `MarkdownSection[ParamsT]` — `string.Template` body. Validates at
  construction that every `$placeholder` is a field on `params_type`.
- `Prompt` — `ns`, `key`, `sections`, `params`. Validates duplicate tool
  names, identifier shape, and produces `RenderedPrompt` via depth-first
  walk with deterministic numbered headings (`## 1. …`, `### 1.1. …`).
  `Prompt.bind(**params)` returns a new `Prompt` with merged section
  parameters.
- `RenderedPrompt` — `text`, `tools`.

### `core/tool.py`

- `Tool[ParamsT, ResultT]` — typed handler container. Validates name shape
  (`^[a-z0-9_-]{1,64}$`) and 1-200 char description.
- `ToolHandler[ParamsT, ResultT]` (Protocol) — `(params, context) -> ToolResult[ResultT]`.
- `ToolResult[ResultT]` — `ok(value, message="")` / `error(message)`.
  Frozen, slotted.
- `ToolContext` — `session`, `prompt`, `rendered`, optional `deadline`,
  optional `resources` (`ResourceProvider`). Exposes
  `get_resource(Protocol)`.
- `ToolPolicy` (Protocol) — `before_invoke(tool, context)` (raise
  `PolicyDeniedError` to block) and `on_result(tool, result)`. The spine
  ships zero concrete policies.

### `core/session.py`

- `Session` — thread-safe via internal `RLock`. Public methods all acquire
  the lock; reducers run inside it. Methods: `__getitem__`, `dispatch`,
  `register`, `install`, `subscribe`, `publish`, `reset`.
- `SliceAccessor[T]` — atomic `all()`, `latest()`, `where(predicate)`,
  `seed(value)`, `append(value)`, `clear()`.
- `Replace[T]`, `Append[T]` — frozen `SliceOp` dataclasses.
- `@reducer(on=Event)` — marks methods on a frozen-dataclass slice.
  `Session.install(SliceCls, *, initial=None)` registers every decorated
  method.
- Spine-emitted observability events, all frozen dataclasses:
  `PromptRendered`, `ToolInvoked`, `PromptExecuted`.

### `core/snapshot.py`

- `Snapshot` — frozen mapping of slice type → tuple of items, plus a UTC
  `created_at` timestamp and `schema_version`. `to_json(*, registry)` and
  `from_json(raw, *, registry)` round-trip.
- `TypeRegistry` — maps stable identifiers
  (`{module}.{qualname}`) to frozen-dataclass classes.
- `Snapshotable[StateT]` (Protocol) — `snapshot() -> StateT`,
  `restore(state)`. Resources participating in transactions implement this.
- `capture(session)` / `restore(session, snapshot)` — atomic helpers.

### `core/transactions.py`

- `tool_transaction(session, *, snapshotables=())` — context manager that
  captures session + resource state on entry and restores it on any
  exception.
- `execute_tool(prompt, session, name, params, *, snapshotables=(), policies=())` — runs the named tool with full transactional rollback,
  invokes any `ToolPolicy` chain, emits `PromptRendered` and `ToolInvoked`.
- `PendingToolTracker` — for callback-driven adapters; `begin(call_id, tool_name)`, `end(call_id, *, success)`, `abort(call_id)`.

### `core/protocols.py`

`runtime_checkable` `Protocol`s plus value types adapters share:

- `Deadline` — `remaining() -> timedelta`, `expired() -> bool`.
- `Budget` — `record(usage)`, `check()`, `consumed -> Usage`.
- `Usage` — frozen dataclass, `input_tokens`/`output_tokens`/`total_tokens`.
- `ToolCall` — name + arguments + optional call id.
- `PromptResponse` — `text`, `tool_calls`, `usage`, `finish_reason`.
- `EventListener` — `on_event(event)`.
- `ResourceProvider` — `get(protocol_type) -> T`.
- `ProviderAdapter` — `evaluate(rendered, session, *, deadline=None) -> PromptResponse`. The single entry point for extras integrating models.

## Stability Contract

- Every public symbol is listed in `weakincentives/core/__init__.py`'s
  `__all__`. The top-level `weakincentives/__init__.py` mirrors it. A test
  asserts equality of the two sets.
- After v1.0.0, breaking changes to `__all__` require a major bump; new
  symbols are minor; bug fixes are patches.
- Error class identity is stable. Subclasses may be added; never removed or
  reparented.
- New extension points are additive; existing protocols never gain required
  methods.

## Thread Safety

- `Session` acquires an `RLock` for every public method, including
  `SliceAccessor` operations and `dispatch`.
- `Snapshot` is immutable.
- `Snapshotable` implementations are responsible for their own locking.
- `EventListener.on_event` is called synchronously on the publishing
  thread. Listeners that need queuing implement that themselves.
- `ProviderAdapter.evaluate` may be called from any thread; adapters own
  their concurrency.

## Persistent Snapshots

- Slice values must be JSON-friendly: primitives, `None`, tuples, lists, or
  frozen dataclasses whose fields recursively obey the same rule.
- Polymorphic encoding: dataclass instances become objects with a
  `__type__` field whose value is the registered identifier.
- Schema version `"1"` is stamped into every payload and verified on
  `from_json`.

## Deadline / Budget Protocol Shape

- Spine treats `Deadline` and `Budget` as protocols, never concrete types.
- `RenderedPrompt` does not (yet) carry a deadline; tools receive an
  optional `Deadline` via `ToolContext`. Concrete implementations live in
  extras.

## Acceptance Criteria

A spine change is "done" when:

- `make check` passes (ruff format + lint, pyright strict, ty, mdformat,
  pytest with 100% line and branch coverage).
- Every file in `weakincentives/core/` is under 720 lines; every function
  is under 120.
- `weakincentives/core/` imports nothing outside `__future__`, the stdlib,
  and other modules in `weakincentives.core`.
- `tests/test_public_api.py` passes — `weakincentives.__all__` mirrors
  `weakincentives.core.__all__` exactly, every advertised symbol resolves,
  and no symbol leaks in from a forbidden subpackage.
- An integration test exercises a non-trivial prompt + tool + session flow
  using only `weakincentives.core` (no extras).
- Thread-safety tests dispatch concurrent events without state corruption.
- Snapshot round-trip tests write to JSON, parse back, and produce equal
  state.

## Non-Goals

The spine deliberately does not include:

- Concrete provider integrations (adapters extra).
- Concrete `Deadline` / `Budget` / `Clock` implementations.
- Dependency injection container.
- Filesystem sandbox.
- Debug bundle authoring or reading.
- Evaluation framework.
- Formal verification helpers.
- Design-by-contract decorators.
- Skills loading and validation.
- Structured output parsing, progressive disclosure, prompt overrides,
  feedback providers, task completion checkers.
- AgentLoop, mailbox, DLQ, watchdog, lease extender.
- CLI, web UI, query engine.
- Anything beyond the standard library.

If something is genuinely needed in core but currently lives in an extra,
the bar to promote it is: at least two unrelated extras need it, it has no
external dependencies, and it adds \<100 lines.

## Open Questions

- **Async support.** Should `ProviderAdapter.evaluate` have an async
  sibling? Lean: no; an async wrapper is an adapter concern.
- **Reducer registration scope.** Per-`Session` instance (current). Global
  registries risk cross-test pollution.
- **Event publication thread.** Synchronous on the publisher thread
  (current); queueing is an extras concern.
- **Snapshot encoding.** Hand-rolled JSON walker (current) keeps the spine
  stdlib-only. A richer codec adapter can ship in an extras module.
- **Deadline propagation.** Currently passed via `ToolContext`. May add to
  `RenderedPrompt` once a real adapter needs it.
