# Weak Incentives (WINK)

WINK is an open source toolkit for developing and optimizing side-effect-free
background agents (e.g., research, review, and coding agents) that translate
end-user instructions into deterministic actions. The library is designed for
human developers, but this README is written so automated coding agents (OpenAI
Codex web/CLI, Claude Code, Cursor background agents, etc.) can be productive
consumers of the API surface without depending on internals.

The public API centers on typed prompts, declarative tool contracts, replayable
sessions, and provider-agnostic adapters so you can keep determinism,
observability, and safety front and center while iterating on agent behaviors.

This document is the package README published to PyPI. It focuses on the
supported, public API surface you can depend on when wiring WINK into your own
agent or orchestration system.

## Installation

WINK targets Python 3.12+. Install the core library:

```bash
pip install weakincentives
```

Optional extras enable specific providers or tooling:

- `pip install "weakincentives[openai]"` for the OpenAI adapter.
- `pip install "weakincentives[litellm]"` for the LiteLLM adapter.
- `pip install "weakincentives[asteval]"` to enable the sandboxed Python eval
  tool.
- `pip install "weakincentives[podman]"` for Podman-based sandboxes.
- `pip install "weakincentives[wink]"` for the demo CLI (`wink`).

## Key Concepts

- **Prompts** (`weakincentives.prompt.Prompt`): Composable, dataclass-driven
  blueprints that render deterministic model inputs while exposing tool
  contracts.
- **Tools** (`weakincentives.prompt.Tool`): Declarative descriptions of
  capabilities the model can invoke. Tools surface type-checked handlers and
  renderable schemas.
- **Sessions** (`weakincentives.runtime.Session`): Redux-style ledgers that
  record every prompt render and tool invocation as immutable events. Reducers
  keep state deterministic and replayable.
- **Adapters** (`weakincentives.adapters.ProviderAdapter`): Bridges to model
  providers that negotiate tool calls and structured outputs without locking you
  into a single vendor.
- **Structured Output** (`parse_structured_output`): JSON-schema-backed parsing
  that turns model responses into typed dataclass instances.
- **Overrides** (`PromptOverride`, `LocalPromptOverridesStore`): Hash-based
  prompt overrides that let you refine prompt text safely in version control.

## Public API

- `weakincentives`: Curated entrypoints for building prompts, tools, and
  sessions.
  - Classes and functions:
    - `Deadline`: Immutable value object describing a wall-clock expiration.
    - `JSONValue`: Type alias for JSON-compatible primitives, objects, and arrays.
    - `MarkdownSection`: Render markdown content using `string.Template`.
    - `Prompt`: Coordinate prompt sections and their parameter bindings.
    - `PromptResponse`: Structured result emitted by an adapter evaluation.
    - `StructuredLogger`: Logger adapter enforcing a minimal structured event schema.
    - `SupportsDataclass`: Protocol satisfied by dataclass types and instances.
    - `Tool`: Describe a callable tool exposed by prompt sections.
    - `ToolContext`: Immutable container exposing prompt execution state to handlers.
    - `ToolHandler`: Callable protocol implemented by tool handlers.
    - `ToolResult`: Structured response emitted by a tool handler.
    - `configure_logging`: Configure the root logger with sensible defaults.
    - `get_logger`: Return a `StructuredLogger` scoped to a name.
    - `parse_structured_output`: Parse a model response into the structured output type declared by the prompt.
  - Modules: `adapters`, `cli`, `deadlines`, `debug`, `prompt`, `runtime`,
    `serde`, `tools`, `types`.
- `weakincentives.adapters`: Provider integrations and throttling primitives.
  - Constants: `LITELLM_ADAPTER_NAME`, `OPENAI_ADAPTER_NAME`.
  - Types:
    - `AdapterName`: Type alias for adapter names.
    - `PromptEvaluationError`: Raised when evaluation against a provider fails.
    - `PromptResponse`: Structured result emitted by an adapter evaluation.
    - `ProviderAdapter`: Abstract base class describing the synchronous adapter contract.
    - `SessionProtocol`: Protocol describing the session interface required by adapters.
    - `ThrottleError`: Raised when a throttle policy denies a request.
    - `ThrottlePolicy`: Protocol for rate limiting policies.
  - Factory: `new_throttle_policy`: Factory for creating throttle policies.
- `weakincentives.prompt`: Prompt authoring, rendering, and override helpers.
  - Authoring:
    - `MarkdownSection`: Render markdown content using `string.Template`.
    - `Prompt`: Coordinate prompt sections and their parameter bindings.
    - `RenderedPrompt`: Result of rendering a prompt.
    - `Section`: Base class for prompt sections.
    - `SectionPath`: Path to a section.
    - `Tool`: Describe a callable tool exposed by prompt sections.
    - `ToolContext`: Immutable container exposing prompt execution state to handlers.
    - `ToolExample`: Representative invocation for a tool documenting inputs and outputs.
    - `ToolHandler`: Callable protocol implemented by tool handlers.
    - `ToolRenderableResult`: Protocol for tool results that can be rendered.
    - `ToolResult`: Structured response emitted by a tool handler.
    - `SupportsDataclass`: Protocol satisfied by dataclass types and instances.
    - `SupportsToolResult`: Protocol for tool results.
    - `PromptProtocol`: Protocol for prompts.
    - `RenderedPromptProtocol`: Protocol for rendered prompts.
    - `ProviderAdapterProtocol`: Protocol for provider adapters.
  - Composition:
    - `DelegationParams`: Parameters for delegation prompts.
    - `DelegationPrompt`: Prompt for delegating tasks to sub-agents.
    - `DelegationSummarySection`: Section for summarizing delegation results.
    - `ParentPromptParams`: Parameters for parent prompts.
    - `ParentPromptSection`: Section for including parent prompt context.
    - `RecapParams`: Parameters for recap sections.
    - `RecapSection`: Section for recapping previous turns.
  - Overrides:
    - `LocalPromptOverridesStore`: Store for local prompt overrides.
    - `PromptDescriptor`: Descriptor for a prompt.
    - `PromptLike`: Protocol for objects that look like prompts.
    - `PromptOverride`: Override for a prompt.
    - `PromptOverridesError`: Raised when prompt overrides fail.
    - `PromptOverridesStore`: Protocol for prompt override stores.
    - `SectionDescriptor`: Descriptor for a section.
    - `SectionOverride`: Override for a section.
    - `ToolDescriptor`: Descriptor for a tool.
    - `ToolOverride`: Override for a tool.
    - `hash_json`: Hash a JSON value.
    - `hash_text`: Hash a text value.
  - Structured output and validation:
    - `OutputParseError`: Raised when structured output parsing fails.
    - `StructuredOutputConfig`: Configuration for structured output.
    - `parse_structured_output`: Parse a model response into the structured output type declared by the prompt.
    - `PromptError`: Base class for prompt errors.
    - `PromptRenderError`: Raised when prompt rendering fails.
    - `PromptValidationError`: Raised when prompt validation fails.
- `weakincentives.runtime`: Session and event primitives.
  - Logging:
    - `StructuredLogger`: Logger adapter enforcing a minimal structured event schema.
    - `configure_logging`: Configure the root logger with sensible defaults.
    - `get_logger`: Return a `StructuredLogger` scoped to a name.
  - Events:
    - `EventBus`: Interface for publishing events.
    - `HandlerFailure`: Event emitted when a handler fails.
    - `InProcessEventBus`: Simple in-process event bus.
    - `PromptExecuted`: Event emitted when a prompt is executed.
    - `PromptRendered`: Event emitted when a prompt is rendered.
    - `PublishResult`: Result of publishing an event.
    - `ToolInvoked`: Event emitted when a tool is invoked.
  - Session ledger:
    - `DataEvent`: Event carrying data.
    - `ReducerContext`: Context for reducers.
    - `ReducerContextProtocol`: Protocol for reducer context.
    - `ReducerEvent`: Base class for reducer events.
    - `ReducerEventWithValue`: Reducer event with a value.
    - `Session`: Immutable event ledger with Redux-like reducers.
    - `SessionProtocol`: Protocol for sessions.
    - `Snapshot`: Session snapshot.
    - `SnapshotProtocol`: Protocol for session snapshots.
    - `SnapshotRestoreError`: Raised when snapshot restoration fails.
    - `SnapshotSerializationError`: Raised when snapshot serialization fails.
    - `TypedReducer`: Reducer for typed state.
    - `append`: Append an event to a session.
    - `build_reducer_context`: Build a reducer context.
    - `iter_sessions_bottom_up`: Iterate over sessions bottom-up.
    - `replace_latest`: Replace the latest value in a session.
    - `replace_latest_by`: Replace the latest value in a session by key.
    - `select_all`: Select all values from a session.
    - `select_latest`: Select the latest value from a session.
    - `select_where`: Select values from a session matching a predicate.
    - `upsert_by`: Upsert a value in a session by key.
- `weakincentives.tools`: Built-in tool sections and dataclasses.
  - Planning:
    - `AddStep`: Add a step to a plan.
    - `ClearPlan`: Clear the plan.
    - `MarkStep`: Mark a step in a plan.
    - `NewPlanStep`: New step for a plan.
    - `Plan`: A plan.
    - `PlanStatus`: Status of a plan.
    - `PlanStep`: A step in a plan.
    - `PlanningStrategy`: Strategy for planning.
    - `PlanningToolsSection`: Section for planning tools.
    - `ReadPlan`: Read the plan.
    - `SetupPlan`: Setup a plan.
    - `StepStatus`: Status of a step.
    - `UpdateStep`: Update a step in a plan.
  - Sandboxes and VFS:
    - `AstevalSection`: Section for asteval tools.
    - `DispatchSubagentsParams`: Parameters for dispatching subagents.
    - `HostMount`: Host mount configuration.
    - `SubagentIsolationLevel`: Isolation level for subagents.
    - `SubagentResult`: Result of a subagent.
    - `SubagentsSection`: Section for subagents.
    - `VirtualFileSystem`: Virtual file system.
    - `WorkspaceDigest`: Digest of a workspace.
    - `WorkspaceDigestSection`: Section for workspace digest.
    - `VfsFile`: Virtual file system file.
    - `VfsPath`: Virtual file system path.
    - `VfsToolsSection`: Section for VFS tools.
    - `build_dispatch_subagents_tool`: Build a tool for dispatching subagents.
    - `clear_workspace_digest`: Clear the workspace digest.
    - `dispatch_subagents`: Dispatch subagents.
    - `latest_workspace_digest`: Get the latest workspace digest.
    - `set_workspace_digest`: Set the workspace digest.
  - File operations:
    - `DeleteEntry`: Delete a file or directory.
    - `EditFileParams`: Parameters for editing a file.
    - `EvalFileRead`: Read a file for evaluation.
    - `EvalFileWrite`: Write a file for evaluation.
    - `EvalParams`: Parameters for evaluation.
    - `EvalResult`: Result of evaluation.
    - `FileInfo`: Information about a file.
    - `GlobMatch`: Match for a glob pattern.
    - `GlobParams`: Parameters for globbing.
    - `GrepMatch`: Match for a grep pattern.
    - `GrepParams`: Parameters for grepping.
    - `ListDirectory`: List a directory.
    - `ListDirectoryParams`: Parameters for listing a directory.
    - `ListDirectoryResult`: Result of listing a directory.
    - `ReadFile`: Read a file.
    - `ReadFileParams`: Parameters for reading a file.
    - `ReadFileResult`: Result of reading a file.
    - `RemoveParams`: Parameters for removing a file.
    - `ToolValidationError`: Raised when tool validation fails.
    - `WriteFile`: Write a file.
    - `WriteFileParams`: Parameters for writing a file.
  - Podman extras (lazy-loaded):
    - `PodmanSandboxConfig`: Configuration for Podman sandbox.
    - `PodmanSandboxSection`: Section for Podman sandbox.
    - `PodmanShellParams`: Parameters for Podman shell.
    - `PodmanShellResult`: Result of Podman shell.
    - `PodmanWorkspace`: Podman workspace.
- `weakincentives.serde`: Dataclass serialization helpers.
  - `clone`: Clone a dataclass.
  - `dump`: Dump a dataclass to JSON-compatible types.
  - `parse`: Parse JSON-compatible types into a dataclass.
  - `schema`: Generate a JSON schema for a dataclass.
- `weakincentives.types`: JSON typing helpers.
  - `ContractResult`: Result of a contract check.
  - `JSONArray`: Type alias for JSON arrays.
  - `JSONArrayT`: Type variable for JSON arrays.
  - `JSONObject`: Type alias for JSON objects.
  - `JSONObjectT`: Type variable for JSON objects.
  - `JSONValue`: Type alias for JSON-compatible primitives, objects, and arrays.
  - `ParseableDataclassT`: Type variable for parseable dataclasses.
- `weakincentives.dbc`: Design-by-contract utilities.
  - `dbc_active`: Return `True` when DbC checks should run.
  - `dbc_enabled`: Context manager to temporarily enable DbC.
  - `disable_dbc`: Force DbC enforcement off.
  - `enable_dbc`: Force DbC enforcement on.
  - `ensure`: Validate postconditions once the callable returns or raises.
  - `invariant`: Enforce invariants before and after public method calls.
  - `pure`: Validate that the wrapped callable behaves like a pure function.
  - `require`: Validate preconditions before invoking the wrapped callable.
  - `skip_invariant`: Mark a method so invariants are not evaluated around it.
- `weakincentives.cli`: CLI entrypoints, notably the `wink` module.

## Agent-facing operational notes

- WINK does not run unattended background agents by itself. It provides
  deterministic primitives that research/review/coding agents (or humans) drive
  explicitly via prompts, tool handlers, and adapters.
- Rendering is side-effect-free: `Prompt.render()` produces a typed
  `RenderedPrompt` containing message content, declared tools, and any
  structured-output schema, but does not contact providers until you pass it to
  an adapter.
- Tool handlers are synchronous callables; use them to gate filesystem or
  network access and to enforce policy before applying patches. Handlers accept
  the typed params plus a keyword-only `context` and return `ToolResult`
  instances (`ToolResult(message=..., value=..., success=True/False)`) to keep
  session logs consistent.
- `PromptResponse` carries the prompt name, rendered text, and parsed output
  (when structured output is requested) so you can safely resume after partial
  failures or retries.
- Sessions are immutable ledgers: reducers consume `PromptRendered`,
  `PromptExecuted`, and `ToolInvoked` events that include `event_id`,
  `session_id`, timestamps, and provider metadata so you can join prompt and
  tool flows deterministically.

## Quickstart: Build a Coding Agent

The example below shows how to build a coding agent that takes user
instructions, renders a prompt with tool contracts, and executes tool calls in a
replayable session.

```python
from dataclasses import dataclass
from weakincentives import (
    MarkdownSection,
    Prompt,
    Tool,
    ToolContext,
    ToolResult,
    parse_structured_output,
)
from weakincentives.adapters import PromptEvaluationError
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import InProcessEventBus, Session, configure_logging

# 1) Define the structured plan you expect the model to return.
@dataclass
class Plan:
    summary: str
    steps: list[str]

# 2) Describe a tool the model can call to edit files.
@dataclass
class ApplyPatchArgs:
    path: str
    diff: str


@dataclass
class ApplyPatchResult:
    applied: bool


def apply_patch(params: ApplyPatchArgs, *, context: ToolContext) -> ToolResult[ApplyPatchResult]:
    patch = params.diff
    # ...apply the patch to the local filesystem...
    return ToolResult(message="applied", value=ApplyPatchResult(applied=True))


apply_patch_tool = Tool[ApplyPatchArgs, ApplyPatchResult](
    name="apply_patch",
    description="Apply a unified diff to the workspace.",
    handler=apply_patch,
)

# 3) Compose a prompt with Markdown sections and the tool.
user_goal = "Add a health-check endpoint to the FastAPI app"
prompt = Prompt[
    Plan
](
    ns="agent",  # Namespace for grouping related prompts.
    key="coding-agent",  # Unique identifier for overrides and session events.
    name="coding-agent",
    sections=[
        MarkdownSection(
            title="Goal",
            template=user_goal,
            key="goal",
        ),
        MarkdownSection(
            title="Instructions",
            template=(
                "Propose a plan, then call `apply_patch` with a diff that satisfies"
                " the goal."
            ),
            key="instructions",
            tools=(apply_patch_tool,),  # Register tools on sections, not prompts.
        ),
    ],
)

# 4) Run the prompt through an adapter and capture the response.
adapter = OpenAIAdapter(model="gpt-4o-mini")

configure_logging()
event_bus = InProcessEventBus()
session = Session(bus=event_bus)

try:
    response = adapter.evaluate(
        prompt,
        bus=event_bus,
        session=session,
    )
    plan = response.output or parse_structured_output(
        response.text or "", prompt.render()
    )
    print("Plan:", plan)
except PromptEvaluationError as exc:
    # Inspect exc.response for partial data or retry cues.
    raise
```

### What happens at runtime?

- The prompt renders Markdown plus a JSON-schema tool contract for
  `apply_patch`.
- `ProviderAdapter.evaluate` negotiates tool calls with the provider, emitting
  `PromptExecuted` and `ToolInvoked` events into the session ledger.
- `parse_structured_output` parses the model reply into the `Plan` dataclass.
- The session keeps every event immutable so you can snapshot, replay, or
  inspect them later.

## Prompt Authoring (`weakincentives.prompt`)

- **`Prompt`**: Construct with `ns`, `key`, optional `name`, ordered `sections`
  (typically `MarkdownSection`), and an optional structured-output specialization
  (e.g., `Prompt[Plan]`). Call `render()` to obtain a `RenderedPrompt` if you need
  to inspect the provider payload; adapters render internally when you call
  `evaluate`.
- **`MarkdownSection`**: Simple Markdown content built from a `title`,
  `template`, and unique `key`. Use multiple sections to keep prompt text modular
  and override-friendly. Pass `tools=(...)` to expose tool contracts within a
  section.
- **`Tool`**: Declarative tool description parameterized as `Tool[Params, Result]` where `Params` **must** be a dataclass type and `Result` is a dataclass (or sequence of dataclasses). Key fields:
  - `name`: Identifier exposed to the model.
  - `description`: Natural-language guidance (1-200 ASCII characters).
  - `handler`: Callable accepting `(params: Params, *, context: ToolContext)`
    and returning `ToolResult[Result]`.
- **`ToolContext`**: Passed to tool handlers. Provides `prompt`, optional
  `rendered_prompt`, `adapter`, `session`, `event_bus`, and optional
  `deadline`—use these to enforce policy and publish custom events.
- **`ToolResult`**: Structured handler response with `message`, `value`, and a
  `success` flag.
- **`StructuredOutputConfig` / `parse_structured_output`**: Attach a
  dataclass-driven schema to a prompt or parse the final model message into a
  concrete instance. `OutputParseError` surfaces validation issues.
- **`Section`**: Lower-level composition primitive for advanced prompt layouts.
- **Delegation helpers**: `DelegationPrompt`, `DelegationSummarySection`,
  `ParentPromptSection`, and related params let you compose supervising agents
  that delegate subtasks while maintaining typed contracts.
- **Overrides**: Use `PromptDescriptor`/`SectionDescriptor` and
  `LocalPromptOverridesStore` to persist hash-scoped overrides alongside your
  repo. Overrides are validated against the descriptor hashes to prevent drift.

## Adapter Layer (`weakincentives.adapters`)

- **`OpenAIAdapter` / `LiteLLMAdapter`**: Primary entrypoints for sending prompts
  to their respective providers. They handle tool negotiation, streaming, and
  structured-output contracts.
- **`PromptResponse`**: Carries the prompt name, rendered text, and parsed output
  (when you request structured output). Available even when
  `PromptEvaluationError` is raised so you can inspect partial results.
- **`SessionProtocol`**: Minimal interface adapters expect when you pass
  `session=` to `evaluate`. `runtime.Session` implements it out of the box.
- **Throttling**: Use `new_throttle_policy` or `ThrottlePolicy` to limit
  provider calls (e.g., rate limits). `ThrottleError` signals enforced delays.

## Runtime & Events (`weakincentives.runtime`)

- **`Session`**: Immutable event ledger with Redux-like reducers. Feed events in
  with `append(session, event)` or convenience selectors like `replace_latest`
  and `upsert_by`. `Snapshot`/`SnapshotProtocol` provide persistence helpers.
- **Reducers and selectors**: Use `TypedReducer` with `ReducerContext` to manage
  typed state slices. `select_all`, `select_latest`, and `select_where` help read
  accumulated data for analytics, auditing, or downstream actions.
- **Events**: `PromptExecuted` and `ToolInvoked` events capture every model
  exchange. `EventBus`/`InProcessEventBus` publish events to reducers and custom
  observers. `HandlerFailure` and `PublishResult` offer backpressure and error
  reporting controls.
- **Logging**: `configure_logging()` wires a structured logger; `get_logger`
  retrieves a module-level logger. `StructuredLogger` is a protocol you can
  implement for custom sinks.

## Session Snapshots for Debugging

Use `Session.snapshot()` to capture an immutable JSON bundle of the session's
state for offline debugging, replay, or handoff to another agent. Snapshots can
be stored, shared, or restored with `Snapshot.from_json(...)` and
`Session.rollback(...)` when you need to rehydrate session slices.

Snapshot payloads follow a stable, schema-versioned layout:

- **Envelope**: `version` (currently `"1"`), `created_at` (ISO timestamp), and
  optional `parent_id` plus `children_ids` when sessions are nested.
- **Slices**: Each entry captures one session slice. Fields include
  `slice_type` (module-qualified dataclass for the slice key), `item_type`
  (dataclass type stored within the slice), and `items` (a list of serialized
  dataclass mappings produced by `weakincentives.serde.dump`).
- **Deterministic ordering**: Slices and items are sorted during serialization
  so diffs stay stable in version control.

You can export snapshots as files, feed them into analysis tools, or inspect the
raw JSON to understand what prompts, tool invocations, and reducer-managed data
existed at a point in time—without needing live provider access.

## Deadlines (`weakincentives.Deadline`)

`Deadline` instances communicate time budgets across prompts, tools, and
provider calls. Pass them into prompts or adapters to ensure long-running tasks
fail fast and emit consistent timeout metadata.

## Data Types (`weakincentives.types`)

`JSONValue` represents JSON-compatible payloads frequently used in tool results
and structured outputs.

## Prompt Overrides in CI/CD

Ship prompts and overrides together:

1. Render a `PromptDescriptor` for your prompt and commit it.
1. Store edits in a `LocalPromptOverridesStore` scoped to your repo root.
1. Gate deployments by validating override hashes so prompts only change when
   descriptors do.

## CLI (`wink` extra)

Install the `wink` extra to experiment locally:

```bash
pip install "weakincentives[wink]"
wink --help
```

The CLI demonstrates prompt rendering, override validation, and adapter wiring
without writing additional code.

## Versioning & Stability

- Public APIs are the objects exported from `weakincentives` and the
  submodules documented above.
- Adapters are optional; include only the extras you need.
- Keep `StructuredOutputConfig`, tool schemas, and overrides in version control
  so your agents remain deterministic and auditable.

## License

Apache License 2.0. See `LICENSE` for details.
