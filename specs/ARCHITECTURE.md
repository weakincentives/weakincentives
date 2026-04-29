# Architecture

The package is a **layered, modular toolkit** organised so each layer
depends only on layers strictly below it. The base install ships only
Layer 0 (the spine); every other layer is an opt-in subpackage with its
own pip extra.

This document describes the boundaries, the dependency rules, and where
each future feature lands.

## First Principles

1. **Everything is layered.** A module at layer N can import from layers
   0..N-1 and from peer modules in layer N that it explicitly declares.
   Sideways and upward imports are forbidden.

1. **Core is protocol-first.** When two layers need to talk, the contract
   lives as a `Protocol` in `weakincentives.core.protocols` (or a
   subpackage's `protocols.py`). Concrete implementations live in extras.

1. **Single-purpose subpackages.** Each subpackage owns one cohesive
   capability. If a module starts hosting two unrelated concepts, it is
   a candidate for splitting.

1. **Stdlib-only at the spine.** Layers 1+ may add third-party
   dependencies but only behind their own pip extra group.

1. **Composition, not inheritance.** Capabilities are wired together via
   protocol-satisfying objects (`ProviderAdapter`, `EventListener`,
   `ResourceProvider`, `Snapshotable`, `ToolPolicy`), never by
   subclassing across layers.

1. **Each subpackage is independently testable.** No subpackage may
   depend on another's internal helpers; only the public `__all__` of a
   peer is fair game.

1. **Frozen public API per subpackage.** Every layer publishes its
   surface in `__all__` and follows semver discipline once it ships.

## Dependency Graph

```
Layer 6   cli          (wink CLI, debug viewer)
Layer 5   adapters/*   (openai, claude, litellm, acp, codex)
Layer 4   evals        debug         skills        formal
Layer 3   runtime      transcript
Layer 2   filesystem   resources
Layer 1   clock        serde         dbc
Layer 0   core         (spine — stdlib only)
```

Arrows point downward only. A module may reach across only within the
same layer when it makes architectural sense (e.g. `runtime` may use
`transcript`), but never up.

## Layers in Detail

### Layer 0 — `weakincentives.core` (the spine)

Already shipped. Stdlib-only. Defines:

- The error hierarchy (`WinkError` and family).
- `Section[ParamsT]`, `Prompt`, `RenderedPrompt`.
- `Tool`, `ToolResult`, `ToolContext`, `ToolPolicy`.
- `Session` (thread-safe), `SliceAccessor`, reducer machinery,
  observability events.
- `Snapshot` with JSON round-trip + `Snapshotable` protocol.
- Transactional `execute_tool` and `tool_transaction`.
- Extension protocols: `ProviderAdapter`, `EventListener`,
  `ResourceProvider`, `Deadline`, `Budget`.

See `specs/SPINE.md`.

### Layer 1 — Foundations (stdlib-only)

Single-file utilities every higher layer relies on. Each module imports
only `weakincentives.core` and the standard library.

#### `weakincentives.clock`

Concrete implementations of the `Deadline` and `Budget` protocols plus
the clock primitives that drive them:

- `WallClock` / `MonotonicClock` / `Sleeper` / `AsyncSleeper` protocols.
- `SystemClock` — production clock backed by `time` and `datetime`.
- `FakeClock` — controllable clock for deterministic tests.
- `Deadline` — frozen wall-clock cutoff with `remaining()` / `expired()`.
- `Budget`, `BudgetTracker` — token + time accounting.

#### `weakincentives.serde`

Dataclass serialization beyond what `core.snapshot` does:

- `parse(cls, data)` and `dump(obj)` for nested frozen dataclasses.
- Field constraints via `Annotated[type, {"ge": 0, "pattern": "..."}]`.
- Polymorphic union encoding via `__type__` discriminator.
- Stable type identifiers.

The spine's snapshot module uses a minimal in-house encoder; richer
features (constraints, unions, custom codecs) live here.

#### `weakincentives.dbc`

Design-by-contract decorators:

- `@require(predicate, message=...)` — preconditions.
- `@ensure(predicate, message=...)` — postconditions on `result`.
- `@invariant(predicate)` — class invariants.

### Layer 2 — State and IO

#### `weakincentives.filesystem`

Filesystem abstraction implementing `Snapshotable`:

- `Filesystem` protocol — read/write/exists/list.
- `InMemoryFilesystem` — fully snapshotable, deterministic for tests.
- `HostFilesystem` — backed by `pathlib.Path` for production.
- Optional path policies (read-only roots, allow/deny lists).

Tools register a filesystem as a snapshotable resource so transactional
tool execution rolls back filesystem mutations alongside session state.

#### `weakincentives.resources`

Dependency injection container that satisfies `core.ResourceProvider`:

- `Binding(protocol, factory, scope)` — declarative wiring.
- `ResourceRegistry` — collects bindings.
- Scopes: `SINGLETON`, `TOOL_CALL`, `PROTOTYPE`.
- `ScopedResourceContext` — context manager that owns the per-call cache
  and propagates `Snapshotable` resources for transaction rollback.
- Lifecycle protocols: `Closeable`, `PostConstruct`, `Snapshotable`.

### Layer 3 — Orchestration

#### `weakincentives.runtime`

Higher-level orchestration of prompt evaluation:

- `AgentLoop` — drives an adapter through render → call → tool → repeat,
  enforcing budgets, deadlines, and policies.
- `Mailbox` (in-memory) — request/response abstraction.
- `Watchdog` / `Heartbeat` — health monitoring.
- `Lifecycle` — graceful shutdown coordination.

Depends on `clock` (deadlines), `resources` (DI), and `core`.

#### `weakincentives.transcript`

Unified event-log format consumable by `debug` and `evals`:

- `TranscriptEntry` schema.
- `TranscriptListener` implementing `EventListener`.
- Adapters map their native events into transcript entries.

### Layer 4 — Quality and observability

- `weakincentives.evals` — datasets, evaluators, session-aware scoring.
- `weakincentives.debug` — debug bundles, replay, comparison.
- `weakincentives.skills` — Agent Skills loading (uses `pyyaml`).
- `weakincentives.formal` — TLA+ integration helpers.

These can subscribe to spine events without coupling to each other.

### Layer 5 — Provider integrations

One subpackage per provider. Each declares its own pip extra:

- `weakincentives.adapters.openai` — `pip install weakincentives[openai]`.
- `weakincentives.adapters.claude` — `[claude]`.
- `weakincentives.adapters.litellm` — `[litellm]`.
- `weakincentives.adapters.acp` — `[acp]`.
- `weakincentives.adapters.codex` — `[codex]`.

Each implements `core.ProviderAdapter`, depends on `runtime`, `clock`,
and `transcript`, plus the provider's third-party SDK.

### Layer 6 — User interfaces

- `weakincentives.cli` — `wink` command for inspecting bundles, listing
  prompts, etc. Optional web UI behind `[wink]` extra.

## Cross-cutting Conventions

### Public API per subpackage

Every subpackage re-exports its public symbols in `__all__` from its
top-level `__init__.py`. That tuple is the published contract; nothing
else is supported. A test asserts the surface against a frozen baseline
once the subpackage tags v1.

### Resources as the integration glue

Layered features compose through resources:

- A tool calls `context.get_resource(Filesystem)` instead of importing a
  filesystem implementation directly.
- The DI container hands back whatever satisfies the protocol.
- Adapters wire production resources; tests wire fakes.

This is how `filesystem`, `clock`, `transcript`, and any future capability
plug into a session without each subpackage knowing about the others.

### Events as the observation channel

Every higher-level feature observes the system through `EventListener`:

- `transcript` records spine events into a unified log.
- `debug` snapshots the transcript and session into a bundle.
- `evals` listens for `ToolInvoked` to score tool usage.
- adapters publish their own events alongside spine events.

Listeners may not mutate session state directly. To react with state
changes, register a reducer for the same event type.

### Snapshotable as the rollback channel

Anything that lives inside a tool transaction is a `Snapshotable`:
filesystems, key-value stores, scoped resources. Transactions roll all
of them back as a unit when a tool fails.

## Pip Extras

The base install requires only the standard library. Each non-stdlib
extra (or extra with non-trivial third-party deps) declares its own
optional dependency group:

```toml
[project.optional-dependencies]
yaml         = ["pyyaml>=6.0.3"]      # skills
openai       = ["openai>=1.0"]
claude       = ["claude-agent-sdk>=0.1"]
litellm      = ["litellm>=1.0"]
acp          = ["agent-client-protocol>=0.9", "mcp>=1.27", "uvicorn>=0.44"]
codex        = ["websockets>=13.0"]
wink         = ["fastapi>=0.135", "uvicorn>=0.44"]
all          = ["weakincentives[yaml,openai,claude,litellm,acp,codex,wink]"]
```

Stdlib-only extras (`clock`, `serde`, `dbc`, `filesystem`, `resources`,
`runtime`, `transcript`, `evals`, `debug`, `formal`) are always
importable; pip extras exist for optional third-party deps.

## Testing Conventions

- 100% line and branch coverage, package-wide.
- Every module under 720 lines; every function under 120 (project policy).
- Unit tests live in `tests/<subpackage>/` and target one subpackage.
- Each layer has integration tests that exercise the layer below it
  through its public API only.
- Thread-safety for any shared mutable state is enforced through stress
  tests with `threading.Barrier`.

## Migration Plan

Implementation is staged. Each phase ends green on `make check`.

| Phase | Adds | Status |
| --- | --- | --- |
| 0 | spine (`core/`) | shipped |
| 1 | `clock`, `serde`, `dbc` | shipped |
| 2 | `filesystem`, `resources` | shipped |
| 3 | `runtime`, `transcript` | shipped |
| 4 | `evals`, `debug`, `skills`, `formal` | shipped |
| 5 | `adapters/*` | `noop` + `openai_compatible` shipped |
| 6 | `cli` (the `wink` command) | shipped |

Each phase is independently shippable. Earlier phases never wait on
later ones.

## What Stays Out (Deliberately)

- Cross-layer helper modules. If two subpackages need the same helper,
  it gets promoted to a lower layer or stays duplicated; it does not
  become a sideways "shared" module.
- A single registry of all known tools / prompts. Each Prompt owns its
  tools; there is no global table.
- Implicit globals (clocks, registries, metrics) — every dependency is
  injected.
- Magic discovery via entry points. Loading is explicit.

## Open Questions

- Whether `transcript` should be its own layer-3 subpackage or live
  inside `runtime`. Lean: separate, because `evals` and `debug` both
  depend on it but not on `runtime`.
- Whether `skills` belongs at Layer 4 (alongside `evals`) or Layer 6
  (alongside `cli`). Lean: Layer 4 — the runtime mounts skills, the CLI
  doesn't.
- Whether the spine should grow a `Clock` protocol (separate from
  `Deadline`/`Budget`) so resources can inject one. Lean: yes when the
  first concrete need appears.
