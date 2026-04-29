# Changelog

## Unreleased

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
