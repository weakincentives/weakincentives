# Changelog

Release highlights for weakincentives.

## Unreleased

- _No changes yet._

## v0.6.0 - 2025-11-05

### Prompt & Delegation

- Added delegation prompt composition helpers, a `SubagentsSection`, and the
  `dispatch_subagents` tool so parent prompts can fan work out to subagents
  while reusing parent response formats and reducer wiring.
- Rebuilt the local prompt overrides store with structured logging, strict
  slug/tag validation, file-level locks, and atomic writes, and taught sections
  and tools to declare an `accepts_overrides` flag so override pipelines target
  only opted-in surfaces.

### Tool Runtime

- Introduced the typed `ToolContext` passed to every handler (rendered prompt,
  adapter, session, and bus) and updated planning, VFS, and ASTEval sections to
  pull session state from the context while validating handler signatures.
- Added configurable planning strategy templates and tighter reducer wiring for
  plan updates, aligning built-in planning tools with the new context and
  override controls.
- Made the ASTEval integration an optional extra and removed the signal-based
  timeout shim while keeping the tool quiet by default.

### Session & Adapters

- Hardened session concurrency with RLock-protected reducers, snapshot restores,
  and new thread-safety regression tests/specs while the event bus now emits
  structured logs for publish failures.
- Centralized adapter protocols and the conversation runner, enforcing that
  adapters always supply a session and event bus before executing prompts and
  improving tool invocation error reporting.

### Logging & Telemetry

- Added a structured logging facility used across sessions, event buses, and
  prompt overrides, alongside dedicated unit tests and README guidance for
  configuring INFO-level output in the code review example.

### Documentation, Examples & Tooling

- Replaced the multi-file code review demo with an updated
  `code_reviewer_example.py` that mounts repositories, tracks tool calls, and
  emits structured logs, removing the legacy example modules/tests.
- Expanded the specs portfolio with new documents for the CLI, logging schema,
  tool context, planning strategies, prompt composition, subagents, and thread
  safety, plus refreshed README sections.
- Added a `make demo` target, tightened the Bandit compatibility shim, and
  refreshed dependency locks.

## v0.5.0 - 2025-11-02

### Session & State

- Added session snapshot capture and rollback APIs to persist dataclass slices, with new helpers and regression coverage.

### Prompt Overrides

- Introduced `LocalPromptOverridesStore` for persisting prompt overrides on disk with strict validation and a README walkthrough.
- Renamed the prompt overrides protocol and exports to `PromptOverridesStore` so runtime and specs share consistent terminology.

### Tool Execution & Adapters

- Centralized adapter tool execution through a shared helper, removing redundant aliases and unifying reducer rollback handling.
- Tool handlers now emit structured JSON responses (including a `success` flag) and adapters treat failures as non-fatal session events.

### Events & Telemetry

- Event buses now return a `PublishResult` summary capturing handler failures and expose `raise_if_errors` for aggregated exceptions.

### Tooling & Quality

- Enabled Pyright strict mode and tightened type contracts across adapters, tool serialization, and session snapshot plumbing.

### Documentation

- Added specs covering session snapshots, local prompt overrides, prompt subagent dispatch, and tool error handling.
- Expanded ASTEval guidance with tool invocation examples and refreshed README tutorials with spec links and symbol search tooling.

## v0.4.0 - 2025-11-01

### Evaluation Tools

- Added `AstevalSection` to expose an `evaluate_python` tool that runs inside
  the sandbox, bridges the virtual filesystem, captures stdout/stderr,
  templates writes, and enforces timeouts.
- Declared `asteval>=1.0.6` as a runtime dependency and documented the
  synchronous handler contract in the ASTEval spec.

### Virtual Filesystem

- Extended the VFS to accept UTF-8 content for writes and host mounts,
  refreshed prompt guidance, and mounted the sunfish README to demonstrate
  multibyte data.

### Examples

- Added a code review agent example that exercises the VFS helpers safely and
  surfaces tool call history through the console session scaffold.
- Wired the ASTEval section into the code review prompt example so agents can
  invoke the `evaluate_python` tool during reviews.

### Typing & Tests

- Expanded type annotations across prompts, adapters, and examples, removed
  Ruff annotation ignores, and broadened the pytest suite to cover new
  behaviors and VFS regression cases while updating coverage configuration.

### Documentation

- Replaced the README quickstart with a step-by-step code review tutorial that
  contrasts Weak Incentives with LangGraph and DSPy.
- Expanded the ASTEval specification with the section entry point, full-VFS
  access guidance, and updated timeout expectations.
- Removed the legacy `docs/` pages now superseded by `specs/`.

## v0.3.0 - 2025-11-01

### Prompt & Rendering

- Renamed and reorganized the prompt authoring primitives (`MarkdownSection`,
  `SectionNode`, `Tool`, `ToolResult`, `parse_structured_output`, …) under the
  consolidated `weakincentives.prompt` surface.
- Prompts now require namespaces and explicit section keys so overrides line up with
  rendered content and structured response formats.
- Added tool-aware prompt version metadata and the `PromptVersionStore` override
  workflow to track section edits and tool changes across revisions.

### Session & State

- Introduced the `Session` container with typed reducers/selectors that capture prompt
  outputs and tool payloads directly from emitted events.
- Added helper reducers (`append`, `replace_latest`, `upsert_by`) and selectors
  (`select_latest`, `select_where`) to simplify downstream state management.

### Built-in Tools

- Shipped the planning tool suite (`PlanningToolsSection` plus typed plan dataclasses)
  for creating, updating, and tracking multi-step execution plans inside a session.
- Added the virtual filesystem tool suite (`VfsToolsSection`) with host-mount
  materialization, ASCII write limits, and reducers that maintain a versioned snapshot.

### Events & Telemetry

- Implemented the event bus with `ToolInvoked` and `PromptExecuted` payloads and wired
  adapters/examples to publish them for sessions or external observers.

### Adapters

- Added a LiteLLM adapter behind the `litellm` extra with tool execution parity and
  structured output parsing.
- Updated the OpenAI adapter to emit native JSON schema response formats, tighten
  `tool_choice` handling, avoid echoing tool payloads, and surface richer telemetry.

### Examples

- Rebuilt the OpenAI and LiteLLM demos as shared CLI entry points powered by the new
  code review agent scaffold, complete with planning and virtual filesystem sections.

### Tooling & Packaging

- Lowered the supported Python baseline to 3.12 (the repository now pins 3.14 for
  development) and curated package exports to match the reorganized modules.
- Added OpenAI integration tests and stabilized the tool execution loop used by the
  adapters.
- Raised the optional `litellm` extra to require the latest upstream release.

### Documentation

- Documented the planning and virtual filesystem tool suites, optional provider extras,
  and updated installation guidance.
- Refreshed the README and supporting docs to highlight the new prompt workflow,
  adapters, and development tooling expectations.

## v0.2.0 - 2025-10-29

### Highlights

- Launched the prompt composition system with typed `Prompt`, `Section`, and `TextSection` building blocks, structured rendering, and placeholder validation backed by comprehensive tests.
- Added tool orchestration primitives including the `Tool` dataclass, shared dataclass handling, duplicate detection, and prompt-level aggregation utilities.
- Delivered stdlib-only dataclass serde helpers (`parse`, `dump`, `clone`, `schema`) for lightweight validation and JSON serialization.

### Integrations

- Introduced an optional OpenAI adapter behind the `openai` extra that builds configured clients and provides friendly guidance when the dependency is missing.

### Developer Experience

- Tightened the quality gate with quiet wrappers for Ruff, Ty, pytest (100% coverage), Bandit, Deptry, and pip-audit, all wired through `make check`.
- Adopted Hatch VCS versioning, refreshed `pyproject.toml` metadata, and standardized automation scripts for releases.

### Documentation

- Replaced `WARP.md` with a comprehensive `AGENTS.md` handbook describing workflows, TDD guidance, and integration expectations.
- Added prompt and tool specifications under `specs/` and refreshed the README to highlight the new primitives and developer tooling.

## v0.1.0 - 2025-10-22

Initial repository bootstrap with the package scaffold, testing and linting toolchain, CI configuration, and contributor documentation.
