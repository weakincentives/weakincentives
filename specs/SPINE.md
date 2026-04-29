# SPINE Specification

## Purpose

The **spine** is the load-bearing core of weakincentives: the smallest set of
abstractions every other piece of the package must depend on, frozen behind a
stable public API. Everything else — provider adapters, resources, evals,
debug bundles, formal verification, skills, DBC, structured output, optimizers
— is an **extra** that imports from the spine and never the other way around.

The goal is industrial: a 1500-line spine that is portable, reviewable, and
versioned, with a 60k-line ecosystem layered on top of it.

This spec supersedes `specs/DISTILLED.md`. The distilled module remains as a
single-file teaching artifact; the spine is the production foundation it
graduates into.

## Principles

- **Inverted dependencies.** The spine has zero dependencies on the rest of the
  package. Extras depend on the spine, not vice versa. No "core helpers"
  reaching back into adapters or filesystem.
- **Stdlib-only.** The spine imports nothing outside the Python standard
  library. Optional features (YAML, Redis, provider SDKs) live in extras.
- **Frozen API, semver discipline.** Everything in `weakincentives.core.__all__`
  is the published surface. Breaking changes require a major version bump.
- **Tight cluster, not single file.** Bottle-style cohesion at scale is a
  small, focused folder of files — not one 2000-line module. Each file has
  one job and stays under the project's 720-line cap.
- **Thread-safe by construction.** The spine is correct under concurrent use
  without callers needing to know about locks.
- **Persistent by design.** Snapshots round-trip through JSON. State that
  matters is replayable.
- **Extension via protocols, not subclassing.** Provider adapters, event
  listeners, resources, and policies plug in by satisfying narrow `Protocol`s
  defined in the spine.

## Package Layout

```
src/weakincentives/
├── core/                    # the spine — stdlib-only, frozen API
│   ├── __init__.py          # public re-exports + __all__
│   ├── errors.py            # WinkError hierarchy
│   ├── prompt.py            # Section[ParamsT], Prompt, RenderedPrompt
│   ├── tool.py              # Tool, ToolResult, ToolContext, ToolPolicy
│   ├── session.py           # Session, SliceAccessor, Replace, Append, reducer
│   ├── snapshot.py          # Snapshot, Snapshotable, JSON round-trip
│   ├── transactions.py      # tool_transaction, execute_tool, PendingToolTracker
│   └── protocols.py         # ProviderAdapter, EventListener, Deadline, Budget
│
├── adapters/                # extras: OpenAI, LiteLLM, Claude SDK, Codex, ACP
├── resources/               # extras: ResourceRegistry, scoped DI
├── filesystem/              # extras: in-memory + host filesystems
├── debug/                   # extras: bundle writer/reader
├── evals/                   # extras: evaluation framework
├── formal/                  # extras: TLA+ integration
├── skills/                  # extras: agent skills
├── serde/                   # extras: dataclass serialization
├── dbc/                     # extras: design-by-contract decorators
├── runtime/                 # extras: AgentLoop, Mailbox, DLQ, Watchdog
├── prompt/                  # extras: feedback, overrides, progressive disclosure
├── budget.py, deadlines.py, clock.py   # extras: concrete impls of core protocols
└── distilled.py             # teaching artifact; thin re-export over core
```

Each spine file targets 200-450 lines. Hard cap 720 (project policy). If a
file approaches 600, split before it gets there.

## Components

### `errors.py`

A flat hierarchy callers can pattern-match on:

```
WinkError                   # base
├── PromptError             # construction or render failures
├── ToolError               # tool registration or invocation failures
├── SessionError            # reducer / slice / dispatch failures
├── SnapshotError           # serialization or restore failures
├── TransactionError        # rollback failures
└── ContractError           # base for budget/deadline/policy violations
    ├── DeadlineExceededError
    ├── BudgetExceededError
    └── PolicyDeniedError
```

Every error has a stable qualified name (`weakincentives.core.errors.X`) so
extras can catch by class without importing internals.

### `prompt.py`

- `Section[ParamsT]` — typed sections parameterized by a frozen dataclass.
  Fields: `title`, `key`, `template`, `children`, `tools`, `policies`,
  `default_params`, `enabled` (predicate). Validates template placeholders
  against `ParamsT` fields at construction.
- `MarkdownSection[ParamsT]` — concrete `string.Template`-based renderer.
  Open for other section kinds via subclass / protocol but `MarkdownSection`
  is the only one shipped in the spine.
- `Prompt` — `ns`, `key`, `sections`, `policies`, optional `default_params`.
  Validates duplicate tool names, missing required params, invalid identifiers
  at construction.
- `RenderedPrompt` — `text`, `tools`, `policies`, `output_type` (optional
  dataclass for structured output), `deadline` (optional).

Section parameters are looked up by walking the prompt's `default_params` map
plus per-section `default_params` plus an explicit `bind()` argument. Missing
required fields raise `PromptError` at render time, never silently render the
empty string.

### `tool.py`

- `Tool[ParamsT, ResultT]` — typed `name`, `description`, `handler`,
  `examples`, `accepts_overrides`. Handler signature:
  `(params: ParamsT, context: ToolContext) -> ToolResult[ResultT]`.
- `ToolResult[ResultT]` — `ok(value, message="")`, `error(message)`, with
  `success`, `value`, `message`. Frozen dataclass.
- `ToolContext` — `session`, `prompt`, `rendered`, `deadline` (optional),
  `event_bus` (optional), plus a generic `get_resource(Protocol)` method that
  delegates to whatever resource provider was wired into the session.
- `ToolPolicy` (Protocol) — `before_invoke(tool, context) -> None` (raise
  `PolicyDeniedError` to block); `on_result(tool, result) -> None`. The spine
  ships zero concrete policies; extras provide them.

### `session.py`

- `Session` — thread-safe via internal `RLock`. Public methods all acquire the
  lock; reducers run while holding it. Methods: `__getitem__`, `dispatch`,
  `register`, `install`, `snapshot`, `restore`, `clone`.
- `SliceAccessor[T]` — `all()`, `latest()`, `where(predicate)`, `seed(value)`,
  `append(value)`, `clear()`. Each method is atomic.
- `Replace[T]`, `Append[T]` — frozen `SliceOp` dataclasses returned by reducers.
- `@reducer(on=Event)` decorator — marks methods on frozen-dataclass slices.
- `install(slice_type, *, initial=None)` — registers all decorated methods.

Reducers must return a `SliceOp` and must be pure. The session enforces this
by treating the input slice tuple as immutable (it's a `tuple`, not a `list`).

### `snapshot.py`

- `Snapshot` — immutable mapping of `slice_type -> tuple[item, ...]` plus a
  schema version and a creation timestamp. `to_json()` and `from_json(raw)`
  use a registry-based polymorphic encoding.
- `TypeRegistry` — `register(cls)` records a stable identifier for a frozen
  dataclass; `resolve(identifier)` reverses it. Slice classes and event
  classes opt in by being passed to `Session.install()` (auto-registered) or
  via explicit `TypeRegistry.register()`.
- `Snapshotable[StateT]` (Protocol) — `snapshot() -> StateT`,
  `restore(state: StateT) -> None`. Filesystems, KV stores, and other
  resources implement this so transactions can roll back state outside the
  session.

### `transactions.py`

- `tool_transaction(session, *snapshotables)` — context manager that captures
  a `CompositeSnapshot` (session + each `Snapshotable`) on entry and restores
  it on any raised exception.
- `execute_tool(prompt, session, name, params, *, snapshotables=(), policies=())` —
  the canonical invocation entry point. Resolves the tool by name, runs the
  policy chain, takes a composite snapshot, runs the handler, restores on
  exception or `ToolResult.error`, fires `ToolInvoked` events.
- `PendingToolTracker` — for adapters using callback-style hook execution
  (Claude Agent SDK, ACP); manages snapshots across `pre_tool_use` /
  `post_tool_use` boundaries.

### `protocols.py`

The narrow extension surface, all `runtime_checkable` `Protocol`s:

- `ProviderAdapter` — `evaluate(prompt, session, *, deadline=None) -> PromptResponse`.
  The single method extras must implement to integrate a model.
- `PromptResponse` — frozen dataclass with `text`, `tool_calls`, `usage`,
  `finish_reason`. Stable schema.
- `EventListener` — `on_event(event: SessionEvent) -> None`. Spine emits
  `PromptRendered`, `ToolInvoked`, `PromptExecuted`. Extras (transcript,
  debug bundle, eval) subscribe.
- `Deadline` — `remaining() -> timedelta`, `expired() -> bool`. Spine takes
  this as an injected dependency; concrete `Deadline` lives in
  `weakincentives.deadlines`.
- `Budget` — `record(usage)`, `check() -> raises ContractError`,
  `consumed -> Usage`. Same arrangement as `Deadline`.
- `ResourceProvider` — `get(protocol_type) -> object`. Tools call
  `context.get_resource(MyProtocol)`. The spine ships `NullResourceProvider`;
  extras (DI registry) ship richer ones.

## Stability Contract

- Every public symbol is listed in `weakincentives/core/__init__.py`'s
  `__all__`. Anything not in `__all__` is private even if it doesn't start
  with `_`.
- The package follows semver from the day the spine ships. Breaking changes
  to anything in `__all__` require a major bump; new symbols are minor; bug
  fixes are patches.
- Deprecations get one minor cycle of `DeprecationWarning` before removal.
- New extension points are additive; existing protocols never gain required
  methods.
- Error class identity is stable. Subclasses may be added, never removed or
  reparented.

## Thread Safety

- `Session` owns an `RLock`. Every public method acquires it. Reducers and
  `SliceAccessor` operations all run inside that lock.
- `Snapshot` is an immutable frozen dataclass; safe to read from any thread
  once produced.
- `Snapshotable` implementations must serialize their own state (the spine
  cannot lock external resources for them).
- `EventListener.on_event` is called synchronously on the publishing thread.
  Listeners that need async dispatch implement that themselves.
- `ProviderAdapter.evaluate` may be called from any thread; adapters are
  responsible for their own concurrency.

## Persistent Snapshots

- `Snapshot.to_json()` produces a JSON string with:
  - `schema_version: "1"`
  - `created_at` (ISO 8601 UTC)
  - `slices: [{type: "...", items: [{...}, ...]}, ...]`
- `Snapshot.from_json(raw, *, registry)` reconstructs a `Snapshot`. The
  registry must contain every slice type referenced; unknown types raise
  `SnapshotError`.
- All slice values must be JSON-serializable frozen dataclasses. Nested
  dataclasses, primitives, and tuples thereof are supported. Tools are not
  part of the snapshot (they are part of the prompt definition).
- `CompositeSnapshot` (in `transactions.py`) extends this with resource
  snapshots keyed by resource type.

## Deadline / Budget Protocol Shape

- Spine knows about `Deadline` and `Budget` as protocols, not implementations.
- `RenderedPrompt.deadline` and `ToolContext.deadline` carry an optional
  `Deadline`. Tools call `context.deadline.expired()` if they care.
- `execute_tool` checks `deadline.expired()` before invoking the handler and
  raises `DeadlineExceededError` if so.
- `Budget` is queried by adapters between provider calls; the spine exposes
  the `record_usage` event listener hook so budgets can subscribe.
- Concrete `Deadline`/`Budget` implementations stay in extras. The spine
  works with whatever satisfies the protocol.

## Migration Plan

Phased and incremental. Each phase ends green on `make check`.

**Phase 0 — Spec freeze.** This document. No code changes.

**Phase 1 — Skeleton.** Create `src/weakincentives/core/` with the seven files
above, populated by lifting and upgrading `distilled.py`. Add full test
coverage. Define `__all__`. Land protocols (empty-bodied) for the four
extension points.

**Phase 2 — Thread safety + JSON snapshots + typed section params.** The
three big functional upgrades over `distilled.py`. New tests for race
conditions (using threading-stress markers) and snapshot round-trip.

**Phase 3 — Move existing implementations onto the spine.** Refactor
`weakincentives.prompt.*`, `weakincentives.runtime.session.*`,
`weakincentives.runtime.transactions` to import from `core` and become thin
extension layers. Adapters, evals, debug, formal stay where they are but
their imports flip to `core`.

**Phase 4 — Re-export at the top level.** Update
`src/weakincentives/__init__.py` so `from weakincentives import Prompt, Tool, Session` resolves to `core` symbols. Remove duplicated implementations.

**Phase 5 — Distilled retires.** `distilled.py` becomes a thin re-export
shim that imports from `core` (or is deleted). `specs/DISTILLED.md` updates
to point at `SPINE.md`.

**Phase 6 — Versioning.** Tag a v1.0.0 release that locks the spine API.

Each phase is one or more PRs. Phases 1-2 should not touch existing extras.

## Acceptance Criteria

A spine is "done" when:

- `make check` passes with zero errors.
- `weakincentives/core/` has 100% line and branch coverage.
- Every file in `weakincentives/core/` is under 720 lines and every function
  is under 120 lines.
- `weakincentives/core/` imports nothing outside `__future__`, the standard
  library, and other modules in `weakincentives.core`.
- A test asserts the import-only-stdlib invariant.
- An integration test exercises a non-trivial prompt + tool + session flow
  using only `weakincentives.core` (no extras).
- Thread-safety tests dispatch concurrent events from multiple threads
  without state corruption.
- Snapshot round-trip tests write to JSON, parse back, and produce equal
  state.
- Each existing extra (`adapters`, `evals`, `debug`, `formal`, `skills`,
  `runtime`, `prompt`, `resources`, `dbc`, `serde`) has been audited and
  imports only from `weakincentives.core` and other extras.
- An "API surface" test diffs the public symbol list against a checked-in
  baseline; new exports require updating the baseline.
- `pyright --strict` and `ty` both green on the spine.

## Non-Goals

The spine deliberately does **not** include:

- Concrete provider integrations (adapters extra).
- Concrete `Deadline` / `Budget` / `Clock` implementations (clock/budget
  extras).
- Dependency injection container (resources extra).
- Filesystem sandbox (filesystem extra).
- Debug bundle authoring or reading (debug extra).
- Evaluation framework (evals extra).
- Formal verification helpers (formal extra).
- Design-by-contract decorators (dbc extra).
- Skills loading and validation (skills extra).
- Structured output parsing, progressive disclosure, prompt overrides,
  feedback providers, task completion checkers (prompt extra).
- AgentLoop, Mailbox, DLQ, Watchdog, lease extender (runtime extra).
- CLI, web UI, query engine (cli extras).
- YAML, Redis, websockets — anything beyond the standard library.

If something is genuinely needed in core but currently lives in an extra,
the bar to promote it is: at least two unrelated extras need it, it has no
external dependencies, and it adds \<100 lines.

## Open Questions

These are decisions deferred until implementation forces them:

1. **Async support.** Should `ProviderAdapter.evaluate` have an async sibling
   in core? Lean: no. Async is an adapter concern and a thin async wrapper
   over `evaluate` lives in extras. Revisit if every shipped adapter ends up
   needing the same wrapper.
1. **Section parameter types.** Dataclasses only, or also `TypedDict` and
   plain protocols? Lean: dataclasses only, matching the rest of the package.
1. **Reducer registration scope.** Are reducers registered per-`Session`
   instance or globally? Lean: per-instance (current behavior); global
   registration has cross-test pollution risk.
1. **Event publication thread.** Synchronous on publisher thread (current),
   or queued? Lean: synchronous; queueing is an extra.
1. **Snapshot encoding.** Hand-rolled JSON walker, or reuse `weakincentives.serde`?
   Lean: hand-rolled in spine to keep stdlib-only; serde extra can offer a
   richer codec adapter.
1. **Tool name uniqueness scope.** Globally per-prompt (current), or also
   across mounted skills? Lean: per-prompt for now, revisit when skills move
   to spine (which they shouldn't).

## Estimated Scope

| Item | Lines | Risk |
| --- | --- | --- |
| Spine skeleton (Phase 1) | ~1500 across 7 files | Low |
| Thread safety tests (Phase 2) | ~300 | Medium |
| JSON snapshots (Phase 2) | ~400 in spine + ~200 in tests | Medium |
| Typed section params (Phase 2) | ~200 in spine + ~300 in tests | Medium |
| Refactor existing modules onto spine (Phase 3) | ~3000 lines moved/edited | High |
| Top-level re-export + extras audit (Phase 4) | ~500 | Medium |
| Versioning + baselines (Phase 6) | ~100 | Low |

Total: roughly two to three weeks of focused refactor for one engineer,
landing across ~10 PRs. Phases 1 and 2 are independently shippable; Phase 3
is the long pole.
