# Changelog

## Unreleased

### Foundational layer 1 modules

Five stdlib-only foundational extras on top of the spine, each with full
coverage:

- `weakincentives.clock` — clock protocols (`WallClock`, `MonotonicClock`,
  `Sleeper`, `AsyncSleeper`, `Clock`), `SystemClock`, deterministic
  `FakeClock`, concrete `Deadline.create`, `Budget`, and thread-safe
  `BudgetTracker`.
- `weakincentives.serde` — `parse(cls, data)` and `dump(value)` for nested
  frozen dataclasses with `Annotated` field constraints
  (`ge`/`le`/`gt`/`lt`/`min_length`/`max_length`/`pattern`) and polymorphic
  union resolution via a `__type__` discriminator.
- `weakincentives.dbc` — `@require`, `@ensure`, and `@invariant` decorators
  for design-by-contract style runtime checks.
- `weakincentives.filesystem` — `Filesystem` protocol and a thread-safe
  `InMemoryFilesystem` that implements `core.Snapshotable` so tool
  transactions roll back filesystem mutations.
- `weakincentives.resources` — `ResourceRegistry`, `Binding`, scoped
  lifetimes (`SINGLETON`, `TOOL_CALL`, `PROTOTYPE`), `ScopedResourceContext`
  that satisfies `core.ResourceProvider`, and `Closeable`/`PostConstruct`
  hooks.

`specs/ARCHITECTURE.md` documents the full layered package design, including
which extras land in subsequent phases (`runtime`, `transcript`, `evals`,
`debug`, adapters, …).

### Reset to spine

The package is rebooted around a small, modular spine
(`weakincentives.core`). Backwards compatibility with previous releases is not
preserved; existing imports will not resolve. See `specs/SPINE.md` for the
new design.

What landed:

- Hierarchical typed prompts (`Section[ParamsT]`, `MarkdownSection[ParamsT]`,
  `Prompt`) that bundle tools and validate template placeholders against a
  parameter dataclass at construction.
- Thread-safe event-sourced sessions with pure reducers and atomic
  `SliceAccessor[T]` operations.
- Snapshot capture/restore with JSON round-trip via `TypeRegistry` and
  schema-versioned encoding.
- Transactional tool execution (`tool_transaction`, `execute_tool`,
  `PendingToolTracker`) that rolls back session state and registered
  `Snapshotable` resources on any failure.
- Stable error hierarchy (`WinkError`, `PromptError`, `ToolError`,
  `SessionError`, `SnapshotError`, `TransactionError`, `ContractError` and
  subclasses) for pattern matching.
- Extension protocols (`ProviderAdapter`, `EventListener`, `ResourceProvider`,
  `ToolPolicy`, `Deadline`, `Budget`) so higher-level layers can plug in.
- Spine-emitted observability events: `PromptRendered`, `ToolInvoked`,
  `PromptExecuted`.

Removed: every previous subpackage (adapters, runtime, evals, debug, formal,
skills, dbc, resources, filesystem, contrib, cli, serde, prompt, types, the
top-level clock/budget/deadlines modules, the `wink` CLI, and all their
documentation, tests, and demo assets). Extras will return as opt-in
installs as they are rebuilt on top of the spine.
