# Changelog

## Unreleased

### Skills, formal, CLI, and an OpenAI-compatible adapter

- `weakincentives.skills` — `Skill`, `SkillMount`, `load_skill`,
  `load_skills`, `render_skill`. Skill files are markdown with TOML
  frontmatter (`+++ ... +++`); the loader uses only the standard library
  via `tomllib`.
- `weakincentives.formal` — `@formal_spec` decorator that attaches a
  `FormalSpec` metadata dataclass to functions/classes plus a
  module-level registry (`all_formal_specs`, `reset_registry`). Ships
  the metadata layer; downstream tooling consumes it.
- `weakincentives.cli` — the `wink` command, registered as a project
  script. Today's only subcommand is `wink debug <bundle.json>`, which
  pretty-prints a debug bundle's metadata, transcript entries, and
  slice contents without needing a `TypeRegistry`.
- `weakincentives.adapters.openai_compatible` —
  `OpenAICompatibleAdapter` against a structural `ChatClient` protocol.
  Works with the real OpenAI SDK, with Mistral/Together/Fireworks/etc.,
  or with hand-rolled fakes — the adapter never imports `openai`. Maps
  `RenderedPrompt` → chat messages, exposes the prompt's tools as
  function-calling specs, propagates deadlines as request timeouts, and
  decodes assistant tool calls back into `core.ToolCall`.

### Orchestration, observability, and a stub adapter

Layers 3, 4, and 5 (stub) of the architecture, all stdlib-only:

- `weakincentives.transcript` — append-only `Transcript` plus
  `TranscriptListener` (implements `core.EventListener`) that records
  every published event as a JSON-friendly `TranscriptEntry`.
- `weakincentives.runtime` — `AgentLoop` driving a `core.ProviderAdapter`
  through render → call → tool → repeat. Honors optional deadlines and
  budgets, fires `AgentLoopStarted` / `AgentIterationStarted` /
  `AgentIterationCompleted` / `AgentLoopFinished` lifecycle events, and
  enforces a configurable `max_iterations` ceiling
  (`MaxIterationsExceeded`).
- `weakincentives.adapters.noop` — `NoopAdapter` + `ScriptedResponse`,
  the deterministic test adapter that replays scripted replies. Provides
  the integration point real adapters (OpenAI, Claude, LiteLLM, ACP,
  Codex) will plug into in their own subpackages with their own pip
  extras.
- `weakincentives.evals` — `Dataset`, `EvalCase`, `EvalReport`, the
  `Evaluator` protocol, and built-in evaluators `ExactMatch`, `Contains`,
  `ToolCalled`, `AllToolsSucceeded`, plus a `run_evaluation` driver that
  builds a fresh session + transcript per case.
- `weakincentives.debug` — `DebugBundle` packaging a session snapshot, a
  recorded transcript, and a metadata dict, with JSON round-trip via
  `core.TypeRegistry` and a `from_session(...)` helper.

`AgentLoop` accepts the spine's `Budget` protocol, so concrete budgets
from `weakincentives.clock` plug in directly.

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
