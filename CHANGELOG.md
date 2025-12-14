# Changelog

Release highlights for weakincentives.

## Unreleased

### Claude Agent SDK Adapter

A new adapter enables running WINK agents through Claude Code's agentic
infrastructure, with full CLI capabilities and hermetic execution:

```python
from weakincentives.adapters import ClaudeAgentSDKAdapter, ClaudeAgentSDKModelConfig

adapter = ClaudeAgentSDKAdapter(
    model=ClaudeAgentSDKModelConfig(model="claude-sonnet-4-20250514"),
)
response = adapter.evaluate(prompt, session=session)
```

Key capabilities:

- **Session synchronization**: Hook-based state sync keeps your WINK session
  as the source of truth while Claude Code executes tools
- **MCP tool bridging**: WINK tools are exposed to Claude Code via the Model
  Context Protocol, enabling native tool execution
- **Workspace mounting**: Mount directories into Claude Code's workspace with
  configurable read/write permissions
- **Isolation mode**: Run without affecting the host's `~/.claude` config
  using ephemeral home directories
- **Network policies**: Programmatic control over network access
  (`no_network`, `api_only`, `with_domains`)

### Declarative State Slices

State slices can now be defined declaratively with reducers co-located as
methods on the dataclass itself:

```python
from dataclasses import dataclass, replace
from weakincentives.runtime.session import state_slice, reducer

@dataclass(frozen=True)
class AddStep:
    step: str

@state_slice(initial=lambda: AgentPlan())
@dataclass(frozen=True)
class AgentPlan:
    steps: tuple[str, ...] = ()
    current_step: int = 0

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> "AgentPlan":
        return replace(self, steps=(*self.steps, event.step))

# Install once, then use naturally
session.install(AgentPlan)
session.mutate(AgentPlan).dispatch(AddStep(step="Research"))
session[AgentPlan].latest()  # Convenient indexing access
```

The `@state_slice` decorator extracts `@reducer`-decorated methods and
auto-registers them when `session.install()` is called. An optional `initial`
factory enables reducers to handle events even when no state exists yet.

Sessions also now support indexing for convenient query access:
`session[Plan].latest()` is equivalent to `session.query(Plan).latest()`.

### Session State Observers

Sessions now support reactive observation of state changes, eliminating the
need for polling:

```python
def on_plan_change(state: list[Plan]) -> None:
    print(f"Plan updated: {len(state)} items")

subscription = session.observe(Plan, on_plan_change)
# ... later
subscription.unsubscribe()
```

Observers are notified immediately after any mutation to their watched slice.

### Session-Managed Visibility Overrides

Visibility overrides are now managed through session state using Redux-style
events:

```python
from weakincentives.prompt import SetVisibilityOverride, VisibilityOverrides

session.mutate(VisibilityOverrides).dispatch(
    SetVisibilityOverride(path=("section",), visibility=SectionVisibility.FULL)
)
response = adapter.evaluate(prompt, session=session)
```

Visibility selectors and enabled predicates now accept an optional `session`
parameter for dynamic decisions based on session state:

```python
def dynamic_visibility(params: MyParams, *, session: Session) -> SectionVisibility:
    state = session.query(MyState).latest()
    return SectionVisibility.FULL if state and state.expanded else SectionVisibility.SUMMARY
```

Sessions automatically register visibility reducers—no manual setup required.

### Breaking Changes

- **Visibility overrides moved to Session state**: The `visibility_overrides`
  parameter has been removed from all adapter `evaluate()` methods. Use the
  `VisibilityOverrides` session state slice instead (see above).

- **Tools relocated to `weakincentives.contrib.tools`**: Planning, VFS,
  Asteval, Podman, and workspace digest tools now live in the `contrib`
  package. Update imports:

  ```python
  # Before
  from weakincentives.tools import PlanningToolsSection, VfsToolsSection

  # After
  from weakincentives.contrib.tools import PlanningToolsSection, VfsToolsSection
  ```

- **Optimizer relocated to `weakincentives.contrib.optimizers`**: The
  `WorkspaceDigestOptimizer` moved to the contrib package.

- **Error types now at root**: `DeadlineExceededError` and
  `ToolValidationError` are now exported directly from `weakincentives`.

### Architecture

The library is now organized as "core primitives" + "batteries for specific
agent styles":

- **Core** (`weakincentives.*`): Prompt composition, sessions, adapters, serde,
  design-by-contract—the minimal essential abstractions
- **Contrib** (`weakincentives.contrib.*`): Planning, VFS, Podman, asteval,
  workspace digest optimizer—optional domain-specific utilities

### Wink Debugger

- Slice list now displays class names instead of full module paths for
  improved readability

## v0.13.0 - 2025-12-07

### MainLoop Orchestration

- Added `MainLoop[UserRequestT, OutputT]` abstract orchestrator that standardizes
  agent workflow execution: receive request, build prompt, evaluate, handle
  visibility expansion, publish result. Implementations define only the
  domain-specific factories via `create_prompt()` and `create_session()`.
- Added event types for request/response routing: `MainLoopRequest[T]`,
  `MainLoopCompleted[T]`, and `MainLoopFailed`.
- Added `MainLoopConfig` for configuring default deadline and budget constraints.
- The MainLoop automatically handles `VisibilityExpansionRequired` exceptions,
  accumulating visibility overrides and retrying evaluation with a shared
  `BudgetTracker` across retries.

### Budget Abstraction

- Added `Budget` resource envelope combining time and token limits (`deadline`,
  `max_total_tokens`, `max_input_tokens`, `max_output_tokens`). At least one
  limit must be set.
- Added `BudgetTracker` for thread-safe cumulative token tracking across
  multiple evaluations against a Budget.
- Added `BudgetExceededError` exception raised when any budget dimension is
  breached, with typed `BudgetExceededDimension` indicating which limit was hit.

### Session Runtime

- Added `session.query(T)` unified selector API that consolidates all session
  state queries into a fluent interface with `.latest()`, `.where(predicate)`,
  and `.all()` methods.
- Added `session.mutate()` fluent API as the counterpart to `session.query()`.
  This provides a unified interface for all session state mutations:
  - `session.mutate(T).seed(values)` - Initialize/replace slice values
  - `session.mutate(T).clear(predicate?)` - Remove items from a slice
  - `session.mutate(T).dispatch(event)` - Event-driven mutation through reducers
  - `session.mutate(T).append(value)` - Shorthand for dispatch with default reducer
  - `session.mutate(T).register(E, reducer)` - Register reducer for event type
  - `session.mutate().reset()` - Clear all slices
  - `session.mutate().rollback(snapshot)` - Restore from snapshot

### Prompts & Templates

- Added generic `PromptTemplate[OutputT]` and `Prompt[OutputT]` abstractions that
  derive structured-output schema (object vs array, allow-extra-keys) and
  support parameterized binding.
- `PromptTemplate` is now immutable using `FrozenDataclass` with cached
  descriptor computation.
- Sections now include their path in rendered output for traceability.
- Added `SectionVisibility` enum (`FULL`, `SUMMARY`) with callable visibility
  selectors that can compute visibility from bound parameters.
- Added `summary` property to sections for use with `SUMMARY` visibility mode.
- Removed `parse_output` flag from `PromptTemplate` - parsing is now always
  performed by adapters.
- Removed `ResponseFormatSection` requirement - structured output instructions
  are now handled directly by adapters.

### Adapters

- Renamed `ConversationRunner` to `InnerLoop` with improved abstraction and
  maintainability.
- Added typed configuration objects: `LLMConfig`, `OpenAIClientConfig`,
  `OpenAIModelConfig`, `LiteLLMClientConfig`, `LiteLLMModelConfig`.
- Enabled parallel tool calls in OpenAI adapter via `parallel_tool_calls`
  config option.
- Simplified adapter method signatures to use `session` parameter only,
  removing separate `bus` parameter since session now owns its event bus.
- Added native web search tool integration spec for provider-executed tools.

### Error Handling

- Introduced `WinkError` as the root exception class for all library exceptions,
  allowing callers to catch all weakincentives errors with a single handler.
  Existing exceptions now inherit from `WinkError` while maintaining backward
  compatibility with their original base types (`ValueError`, `RuntimeError`).

### Tools

- Added `Tool.wrap` static helper that constructs `Tool` instances from handler
  annotations and docstrings, simplifying tool definition for common cases.
- Tools now support `None` for parameter and result types, including schema
  generation and prompt rendering for handlers that accept no input or return
  no structured output.

### Events & Telemetry

- Added `TokenUsage` tracking to `PromptExecuted` and `ToolInvoked` events,
  exposing prompt and completion token counts from provider responses.
- Added `unsubscribe(event_type, handler)` method to the `EventBus` protocol and
  `InProcessEventBus` implementation, allowing handlers to be removed after
  registration. The method returns `True` if the handler was found and removed,
  `False` otherwise.

### Dataclasses

- Added `FrozenDataclass` decorator that provides pre-init normalization hooks,
  `copy()` and `asdict()` helpers, and mapping utilities for immutable dataclass
  patterns. Exported from the package root.

### Wink Debugger

- Polished `wink debug` web interface with enhanced UX including keyboard
  navigation, improved layout, and better state visualization.
- Enhanced snapshot viewer with collapsible tree navigation, search filtering,
  and download capabilities.

### Tools & Sandboxes

- Podman sandbox containers now start with networking disabled (`network_mode=none`),
  and an integration test verifies they cannot reach external hosts.
- **Breaking**: Removed all subagent and prompt delegation code including
  `SubagentsSection`, `dispatch_subagents`, `DelegationPrompt`, and related
  composition helpers. This approach is not the right fit from a state and
  context management perspective.

### Documentation & Specs

- Added `MAIN_LOOP.md` spec documenting MainLoop orchestration and visibility
  override handling.
- Added `HOSTED_TOOLS.md` spec for provider-executed tools (web search, code
  interpreter).
- Added `TASK_SECTION.md` spec for task dataclass patterns in prompts.
- Added `WINK_OVERRIDES.md` spec documenting the `wink overrides` CLI command.
- Simplified and consolidated specification documents for clarity.

### Examples

- Refactored `code_reviewer_example.py` to use the new `MainLoop` abstraction,
  demonstrating the recommended pattern for agent workflow implementation.
- Code reviewer REPL now creates a fresh five-minute default deadline per
  request so long-running interactive sessions continue working without manual
  deadline overrides.

### Quality & Infrastructure

- Parallelized CI jobs for faster feedback.
- Added Vulture for dead code detection.
- Removed backward compatibility code and import shims.

## v0.12.0 - 2025-11-30

### Wink Debugger & Snapshots

- Snapshot viewer now renders markdown-rich values with a toggle between
  rendered HTML and raw markdown, refreshed styling, and separate state/event
  panels plus refresh controls.
- Snapshots persist session tags and parent/child relationships, surfacing tags
  in the debug UI and filtering out stray `.jsonl` bundles when listing
  snapshots.
- Debug app defaults and pagination were tightened, shared snapshot slice
  payloads are reused across the UI, and snapshot path handling is documented so
  missing or renamed files stay discoverable.

### Sessions

- Session locking was simplified and invariant/`skip_invariant` typing tightened
  to keep reducer mutations predictable and thread-safe.

### Prompts & Events

- Prompt render/execution events include the prompt descriptor, clone methods
  return `Self` for stricter typing, and `OptimizationScope` is now a `StrEnum`
  (update any comparisons against the enum values).
- Prompt rendering events tolerate payloads that omit the `value` field to keep
  logging resilient to adapter differences.
- Removed the prompt chapter abstraction; prompts now compose sections only and
  chapter-focused specs and tests have been removed.

### Adapters & API Surface

- Provider responses now flow through shared dataclasses, OpenAI response format
  handling is simplified, and adapter `__all__` exports are sorted for
  deterministic imports.
- Public API exports were clarified, legacy import shims and the package-level
  `__getattr__` fallback were removed, and schema/digest helpers were trimmed to
  reduce surface area.

### Tools, VFS & Sandboxes

- Host mount traversal now uses `Path.walk` for more reliable discovery in VFS
  and Podman-backed workspaces.
- Tool validation tightened with explicit `ToolExample` support and new examples
  covering VFS, planning, ASTEval, and Podman tools.
- Tool execution flow now relies on focused helpers for argument parsing,
  deadline checks, handler invocation, and result logging to reduce complexity
  while preserving structured events.
- Podman tool configuration helpers were refactored and workspace digest keys
  simplified to keep sandboxed runs and tests aligned.

### Serde & Schema

- Dataclass serde imports were consolidated into authoritative modules and
  shared schema helpers/constants were cleaned up to avoid duplication.

### Quality & Testing

- Added mutation-testing support and broader Ruff quality checks to catch issues
  earlier.
- Added unit coverage for tool execution success cases, validation failures,
  deadline expirations, and unexpected handler exceptions.

### Documentation & Specs

- AGENTS guidance now calls out the alpha/stability status explicitly, and
  throttling specs were aligned with the current implementation.

## v0.11.0 - 2025-11-23

### Wink Debugger & Snapshot Explorer

- Added a `wink debug` FastAPI UI for browsing session snapshot JSON files with
  reload/download actions, copy-to-clipboard, and a searchable tree viewer.
- Improved the viewer with keyword search, expand/collapse controls, compact
  array rendering, newline preservation, scrollable truncation boxes, and
  consistent spacing/padding so large payloads stay readable.

### Snapshot Persistence & Session State

- Session snapshots now log writes, skip empty payloads, and include runtime
  events while tolerating unknown slice types during replay.
- The code review example persists snapshots by default; snapshot writes are
  isolated in tests, and workspace digest optimization clones ASTEval tools to
  keep digest renders consistent.

### Adapters & Throttling

- Migrated the OpenAI adapter to the Responses API, updating tool negotiation,
  structured outputs, and retry behavior to match the new endpoint and tests.
- Added a shared throttling policy with jittered backoff and structured
  `ThrottleError`s across OpenAI and LiteLLM adapters, plus conversation-runner
  retry coverage.

### Documentation

- Added an OpenAI Responses API migration guide, a snapshot explorer walkthrough,
  and refreshed AGENTS/session hierarchy guidance.
- Clarified specs for throttling, deadlines, planning strategies/tools, prompts,
  overrides, adapters, logging, structured output, VFS, Podman
  sandboxing, and related design rationales.

## v0.10.0 - 2025-11-22

### Public API & Imports

- Curated exports now live in `api.py` modules across the root package and major
  subsystems; `__init__` files lazy-load those surfaces to keep import paths
  stable while trimming broad re-exports.

### Prompt Authoring & Overrides

- Rendered prompts now include hierarchical numbering (with punctuation) on
  section headings and descriptors so multi-section outputs remain readable and
  traceable.
- Prompts and sections have explicit clone contracts that rebind to new sessions
  and event buses, keeping reusable prompt trees isolated.
- The prompt overrides store protocol is unified and re-exported through the
  versioned overrides module so custom stores and adapters target the same
  interface.
- Structured output metadata now flows through the shared prompt protocols and
  rendered prompts, simplifying adapters that need container/dataclass details
  without provider-specific shims.

### Tools & Execution

- Provider tool calls run through a shared `tool_execution` context manager that
  standardizes argument parsing, deadline enforcement, logging, and publish/
  rollback handling; executor wrappers stay thin and fix prior mutable default
  edge cases.
- Added deadline coverage and clearer telemetry around tool execution to surface
  timeouts consistently.
- Introduced workspace digest helpers (`WorkspaceDigest`, `WorkspaceDigestSection`)
  so sessions can persist and render cached workspace summaries across runs.

### Session & Concurrency

- `Session` now owns an in-process event bus by default, removing boilerplate
  for common setups while preserving the ability to inject a custom bus.
- Added a reusable locking helper around session mutations to guard reducers and
  state updates without sprinkling ad hoc locks.

## v0.9.0 - 2025-11-17

### Podman Sandbox

- Added `PodmanSandboxSection` (renamed from `PodmanToolsSection`) as a
  Podman-backed workspace that mirrors the VFS
  contract, exposes `shell_execute`, `ls`/`read_file`/`write_file`/`rm`, and
  publishes `PodmanWorkspace` slices so reducers can inspect container state. The section sits behind the new `weakincentives[podman]`
  optional extra, is re-exported from `weakincentives.tools`, and ships with a
  spec plus pytest marker for Podman-only suites.
- `evaluate_python` is now available inside the Podman sandbox with the same
  return schema as ASTEval but without the 2,000-character cap, and file writes
  now run through `podman cp` to keep binary-safe edits consistent with the VFS.
- The `code_reviewer_example.py` prompt auto-detects local Podman connections,
  mounts repositories into the sandbox, and falls back to the in-memory VFS when
  Podman is unavailable so the workflow keeps functioning in both modes.

### Tooling & Runtime

- `PlanningToolsSection`, `VfsToolsSection`, and `AstevalSection` now require a
  live `Session` at construction time, register their reducers immediately, and
  verify that tool handlers execute within the same `Session`/event bus. Update
  custom prompts to pass the session you plan to route through `ToolContext`.
- Section registration reinstates strict validation for tool entries and tool
  generics: `Tool[...]` now enforces dataclass result types (or sequences of
  dataclasses), handler annotations must match the declared generics, and
  `ToolContext` exposes typed prompt/adapter protocols so invalid overrides fail
  fast during initialization.

### Prompt Runtime & Overrides

- `Prompt.render` accepts any overrides store implementing the new
  `PromptOverridesStoreProtocol`, and we now export `PromptProtocol`,
  `ProviderAdapterProtocol`, and `RenderedPromptProtocol` for adapters and tool
  authors that need structural typing without importing the concrete prompt
  classes.
- `Prompt` exposes a `structured_output` descriptor (and `RenderedPrompt` mirrors
  it) so runtimes can inspect the resolved dataclass/container contract directly
  instead of juggling separate `output_type`, `container`, and `allow_extra_keys`
  attributes.

### Serde, Logging & Session State

- Introduced canonical JSON typing helpers under `weakincentives.types` (also
  re-exported from the package root) and rewired structured logging to enforce
  JSON-friendly payloads, nested adapter support, and context-preserving
  `StructuredLogger.bind()` calls.
- Dataclass serialization/deserialization now keeps set outputs deterministic,
  respects `exclude_none` inside nested collections, and emits clearer errors
  when schema/constraint validators fail, while session snapshots gained typed
  slice aliases plus an `event_bus` accessor on `Session` for reducer helpers.

## v0.8.0 - 2025-11-15

### Prompt Runtime & Overrides

- `Prompt.render` now accepts override stores and tags directly, removing the helper
  functions by plumbing the parameters through provider adapters and the code review
  CLI so tagged overrides stay consistent end-to-end.
- The code reviewer example gained typed request/response dataclasses, a centralized
  `initialize_code_reviewer_runtime`, plan snapshot rendering, and event-bus driven
  prompt/tool logging so multi-turn runs stay deterministic and easy to trace.

### Built-in Tools

- The virtual filesystem suite was rewritten to match the DeepAgents contract: new
  `VfsPath`/`VirtualFileSystem` dataclasses, ASCII/UTF-8 guards, list/glob/grep/edit
  helpers, host mount previews, and refreshed exports/specs/tests keep VFS sessions
  deterministic.
- The ASTEval tool always runs multi-line inputs, dropping the expression-mode toggle
  while refreshing the section template, serde fixtures, and tests to reflect the
  simplified contract.
- Added pytest-powered audits that enforce documentation, slots, tuple defaults, and
  precise typing on every built-in tool dataclass.

### Runtime & Infrastructure

- `Deadline.remaining` now rejects naive datetime overrides and `configure_logging`
  differentiates explicit log levels from environment defaults to keep adapters'
  logging choices intact.
- Raised dependency minimums, added `pytest-rerunfailures` to the dev stack, refreshed
  CI workflow pins, and updated planning/serde tests for the stricter type checking.
- Thread-safety docs/tests were rewritten around the shared session plus overrides
  store story, retiring the legacy Sunfish session scaffolding and clarifying reducer
  expectations.

### Documentation

- `AGENTS.md` now mirrors the current repository layout, strict TDD workflow, and DbC
  expectations so contributors have a single source of truth for the process.
- The VFS and ASTEval specs were refreshed to describe the expanded file tooling and
  simplified execution contract, keeping the docs aligned with the new surfaces.

## v0.7.0 - 2025-11-13

### Sessions & Contracts

- Introduced `Session.reset()` to clear accumulated slices without removing
  reducer registrations, making long-lived interactive flows easier to manage.
- Added a design-by-contract module that exposes `require`, `ensure`,
  `invariant`, and `pure` decorators along with runtime toggles and a pytest
  plugin so projects can opt into contract enforcement when debugging.
- Wrapped the session container with invariants that validate UUID metadata and
  timezone-aware timestamps, wiring the DbC utilities into the public export
  surface to keep runtime guarantees consistent across adapters.

### Prompt Authoring

- Parameterless prompts and sections now accept zero-argument `enabled`
  callables, keeping declarative gating logic concise.
- Centralized structured output payload parsing into shared helpers used by the
  prompt runtime and adapters to keep response handling consistent.

### Events & Telemetry

- Replaced the `PromptStarted` lifecycle event with `PromptRendered`, removed
  wrapper dataclasses, and now emit the rendered prompt plus adapter metadata
  directly to reducers and subscribers.
- Assigned stable UUID identifiers to prompt and tool events while enforcing
  timezone-aware session creation for richer telemetry.

### Tooling & Quality

- Added the `pytest-randomly` plugin to shake out order-dependent test
  assumptions during development.

## v0.6.0 - 2025-11-05

### Prompt & Overrides

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
  tool context, planning strategies, prompt composition, and thread safety, plus
  refreshed README sections.
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

- Added specs covering session snapshots, local prompt overrides, and tool error handling.
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
