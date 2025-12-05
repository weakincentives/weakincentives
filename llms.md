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

## Architecture Overview

An agent harness in WINK consists of these core components wired together:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Harness                            │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  PromptTemplate  │───▶│      Prompt      │                   │
│  │  (immutable)     │    │  (with bindings) │                   │
│  └──────────────────┘    └────────┬─────────┘                   │
│                                   │                             │
│                                   ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │    EventBus      │◀───│ ProviderAdapter  │                   │
│  │ (observability)  │    │ (OpenAI/LiteLLM) │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │     Session      │◀───│   ToolHandlers   │                   │
│  │ (Redux-like state│    │ (side effects)   │                   │
│  │   + event log)   │    └──────────────────┘                   │
│  └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow**: PromptTemplate → Prompt (with bound params) → ProviderAdapter.evaluate() → LLM response → Tool calls dispatched → ToolResult returned → Session updated → Events published

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
    - `PromptTemplate`: Immutable prompt blueprint with sections (import from here).
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

## Quickstart: Minimal Agent Harness

The simplest agent harness requires: a prompt template, an adapter, and a session.

```python
from dataclasses import dataclass
from weakincentives import MarkdownSection, Prompt
from weakincentives.prompt import PromptTemplate  # Note: PromptTemplate from .prompt
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import Session

# 1. Define the structured output the model should return
@dataclass(slots=True, frozen=True)
class TaskResponse:
    summary: str
    next_steps: list[str]

# 2. Build the prompt template with sections
template = PromptTemplate[TaskResponse](
    ns="myapp/tasks",
    key="task-agent",
    name="task-agent",
    sections=[
        MarkdownSection(
            title="Instructions",
            template="Analyze the following task and provide a summary with next steps.",
            key="instructions",
        ),
    ],
)

# 3. Create session and adapter (Session creates its own EventBus internally)
session = Session()
adapter = OpenAIAdapter(model="gpt-4o-mini")

# 4. Wrap template in Prompt and evaluate
prompt = Prompt(template)
response = adapter.evaluate(prompt, bus=session.event_bus, session=session)

# 5. Access the typed result
result: TaskResponse = response.output
print(f"Summary: {result.summary}")
```

## Complete Agent with Tools

This example shows a coding agent with tool handlers, proper error handling,
and session state access:

```python
from dataclasses import dataclass
from datetime import timedelta
from weakincentives import MarkdownSection, Prompt, Tool, ToolContext, ToolResult
from weakincentives.prompt import PromptTemplate
from weakincentives.adapters import PromptEvaluationError
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import Session, configure_logging

# 1. Define structured I/O types (frozen dataclasses recommended)
@dataclass(slots=True, frozen=True)
class Plan:
    summary: str
    steps: list[str]

@dataclass(slots=True, frozen=True)
class ApplyPatchArgs:
    path: str
    diff: str

@dataclass(slots=True, frozen=True)
class ApplyPatchResult:
    applied: bool
    message: str

# 2. Implement tool handler with ToolContext access
def apply_patch(params: ApplyPatchArgs, *, context: ToolContext) -> ToolResult[ApplyPatchResult]:
    """Tool handlers receive params + context, return ToolResult."""
    # Check deadline if time-bounded (use remaining() method)
    if context.deadline and context.deadline.remaining() <= timedelta(0):
        return ToolResult(
            message="Deadline exceeded",
            value=ApplyPatchResult(applied=False, message="timeout"),
            success=False,
        )

    # Perform the action (your implementation here)
    # ...apply the patch to the filesystem...

    return ToolResult(
        message="Patch applied successfully",
        value=ApplyPatchResult(applied=True, message="ok"),
        success=True,
    )

# 3. Create Tool with typed parameters
apply_patch_tool = Tool[ApplyPatchArgs, ApplyPatchResult](
    name="apply_patch",
    description="Apply a unified diff to the workspace.",
    handler=apply_patch,
)

# 4. Compose prompt with sections and tools
template = PromptTemplate[Plan](
    ns="agent",
    key="coding-agent",
    name="coding-agent",
    sections=[
        MarkdownSection(
            title="Goal",
            template="Add a health-check endpoint to the FastAPI app",
            key="goal",
        ),
        MarkdownSection(
            title="Instructions",
            template="Propose a plan, then call apply_patch with a diff that satisfies the goal.",
            key="instructions",
            tools=(apply_patch_tool,),  # Tools attach to sections
        ),
    ],
)

# 5. Set up runtime and evaluate
configure_logging()
session = Session()
adapter = OpenAIAdapter(model="gpt-4o-mini")

try:
    prompt = Prompt(template)
    response = adapter.evaluate(prompt, bus=session.event_bus, session=session)
    plan: Plan = response.output
    print(f"Plan: {plan.summary}")
except PromptEvaluationError as exc:
    # Access error details for debugging
    print(f"Phase: {exc.phase}")  # "request", "response", or "tool"
    print(f"Prompt: {exc.prompt_name}")
    raise
```

### What happens at runtime?

- The prompt renders Markdown plus a JSON-schema tool contract for
  `apply_patch`.
- `ProviderAdapter.evaluate` negotiates tool calls with the provider, emitting
  `PromptExecuted` and `ToolInvoked` events into the session ledger.
- The session keeps every event immutable so you can snapshot, replay, or
  inspect them later.

## Multi-Turn Agent Loop

For agents that run multiple turns (e.g., planning then executing):

```python
from weakincentives.runtime import select_latest, append

# Run planning phase
plan_response = adapter.evaluate(plan_prompt, bus=session.event_bus, session=session)
plan = plan_response.output

# Store plan in session for later access
session = append(session, plan)

# Run execution phase with plan context
for step in plan.steps:
    exec_prompt = Prompt(exec_template).bind(StepParams(step=step))
    exec_response = adapter.evaluate(exec_prompt, bus=session.event_bus, session=session)
    # Session accumulates all events across turns

# Later, retrieve stored state
stored_plan = select_latest(session, Plan)
```

## Prompt Authoring (`weakincentives.prompt`)

### PromptTemplate and Prompt

`PromptTemplate` is immutable and defines the structure. `Prompt` wraps it with
parameter bindings and overrides:

```python
from weakincentives import MarkdownSection, Prompt
from weakincentives.prompt import PromptTemplate

# Immutable template
template = PromptTemplate[OutputType](
    ns="myapp/agents",           # Namespace for organization
    key="my-agent",              # Unique identifier
    name="my-agent",             # Human-readable name
    sections=[...],              # Ordered sections
    inject_output_instructions=True,  # Auto-add JSON format instructions
)

# Wrap with bindings
prompt = Prompt(template)
bound = prompt.bind(MyParams(value="..."))  # Returns self for chaining
```

### MarkdownSection with Parameters

Use `string.Template` syntax (`${param}`) for dynamic content:

```python
@dataclass(slots=True, frozen=True)
class TaskParams:
    objective: str
    context: str = ""

section = MarkdownSection[TaskParams](
    title="Task",
    template="Objective: ${objective}\n\nContext: ${context}",
    key="task",
    default_params=TaskParams(objective="", context=""),  # Fallback values
)

# Bind params when evaluating
prompt = Prompt(template).bind(TaskParams(objective="Fix the bug", context="..."))
```

### Tool Definition

- **`Tool`**: Declarative tool description parameterized as `Tool[Params, Result]` where `Params` **must** be a dataclass type and `Result` is a dataclass (or sequence of dataclasses). Key fields:
  - `name`: Identifier exposed to the model.
  - `description`: Natural-language guidance (1-200 ASCII characters).
  - `handler`: Callable accepting `(params: Params, *, context: ToolContext)`
    and returning `ToolResult[Result]`.

### Tool.wrap Helper

Use `Tool.wrap` to create a Tool from a handler function. The function name
becomes the tool name, and the docstring becomes the description:

```python
from weakincentives import Tool, ToolContext, ToolResult

@dataclass
class SearchParams:
    query: str

@dataclass
class SearchResult:
    matches: list[str]

def search(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
    """Search for content in the workspace."""
    return ToolResult(
        message="Search complete",
        value=SearchResult(matches=["result1", "result2"]),
        success=True,
    )

# Wrap uses __name__ as tool name and docstring as description
search_tool = Tool.wrap(search)
# Equivalent to:
# Tool[SearchParams, SearchResult](name="search", description="Search for content in the workspace.", handler=search)
```

### ToolContext

Passed to tool handlers. Provides access to runtime state:

```python
def my_handler(params: MyParams, *, context: ToolContext) -> ToolResult[MyResult]:
    context.prompt          # The Prompt being evaluated
    context.rendered_prompt # RenderedPrompt (if available)
    context.adapter         # The ProviderAdapter
    context.session         # Current Session
    context.event_bus       # EventBus for publishing
    context.deadline        # Optional Deadline for timeouts
```

### ToolResult

Structured handler response:

```python
ToolResult(
    message="Human-readable feedback",  # Shown to model
    value=MyResult(...),                 # Typed payload
    success=True,                        # Success flag
    exclude_value_from_context=False,    # If True, value not sent to model
)
```

### Additional Prompt Components

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

### OpenAI Adapter

```python
from weakincentives.adapters.openai import OpenAIAdapter

# Basic usage (uses native JSON schema mode by default)
adapter = OpenAIAdapter(model="gpt-4o-mini")

# With custom client
from openai import OpenAI
client = OpenAI(api_key="sk-...")
adapter = OpenAIAdapter(model="gpt-4o", client=client)

# With client factory (lazy initialization)
adapter = OpenAIAdapter(
    model="gpt-4o",
    client_factory=lambda: OpenAI(api_key="..."),
)

# Disable native structured output if needed
adapter = OpenAIAdapter(
    model="gpt-4o",
    use_native_response_format=False,
)
```

### LiteLLM Adapter

```python
from weakincentives.adapters.litellm import LiteLLMAdapter

# Works with any LiteLLM-supported model
adapter = LiteLLMAdapter(model="claude-3-sonnet-20240229")
adapter = LiteLLMAdapter(model="gpt-3.5-turbo")

# With custom completion function
import litellm
adapter = LiteLLMAdapter(
    model="gpt-4",
    completion=litellm.completion,
)
```

### PromptResponse

Returned by `adapter.evaluate()`:

```python
response = adapter.evaluate(prompt, bus=bus, session=session)

response.output      # Parsed structured output (your dataclass)
response.text        # Raw model response text
response.prompt_name # Name of the prompt
response.usage       # TokenUsage(input_tokens, output_tokens, ...)
```

### Error Handling

```python
from weakincentives.adapters import PromptEvaluationError

try:
    response = adapter.evaluate(prompt, bus=session.event_bus, session=session)
except PromptEvaluationError as exc:
    exc.phase          # "request", "response", or "tool"
    exc.prompt_name    # Which prompt failed
    exc.provider_payload  # Raw provider request/response
    exc.__cause__      # Original exception
```

### Rate Limiting

```python
from weakincentives.adapters import new_throttle_policy, ThrottleError

policy = new_throttle_policy(requests_per_minute=60)

try:
    response = adapter.evaluate(prompt, bus=session.event_bus, session=session, throttle=policy)
except ThrottleError:
    # Wait and retry
    pass
```

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

## Built-in Tool Sections (`weakincentives.tools`)

WINK provides ready-to-use sections that expose common agent capabilities:

### VfsToolsSection - Sandboxed File Operations

```python
from pathlib import Path
from weakincentives.tools import VfsToolsSection, HostMount, VfsPath

# Create a sandboxed workspace with file tools
vfs_section = VfsToolsSection(
    session=session,
    mounts=(
        HostMount(
            host_path="./my-repo",
            mount_path=VfsPath(("workspace",)),
            include_glob=("*.py", "*.md", "*.json"),
            exclude_glob=("*.pyc", "__pycache__/**", ".git/**"),
            max_bytes=600_000,  # Limit total bytes loaded
        ),
    ),
    allowed_host_roots=(Path("."),),  # Restrict file access
)
```

Exposes tools: `read_file`, `write_file`, `list_directory`, `glob`, `grep`

### PlanningToolsSection - Multi-Step Planning

```python
from weakincentives.tools import PlanningToolsSection, PlanningStrategy

planning_section = PlanningToolsSection(
    session=session,
    strategy=PlanningStrategy.PLAN_ACT_REFLECT,
    # Or: PlanningStrategy.GOAL_DECOMPOSE_ROUTE_SYNTHESISE
)
```

Exposes tools: `setup_plan`, `read_plan`, `add_step`, `update_step`, `mark_step`

### SubagentsSection - Parallel Delegation

```python
from weakincentives.tools import SubagentsSection, SubagentIsolationLevel

subagents_section = SubagentsSection(
    isolation_level=SubagentIsolationLevel.FORK,  # Isolated session copy
)
# Note: SubagentsSection doesn't require a session parameter
```

Exposes: `dispatch_subagents` for parallel subtask execution

### WorkspaceDigestSection - Workspace Summary

```python
from weakincentives.tools import WorkspaceDigestSection

digest_section = WorkspaceDigestSection(session=session)
```

Renders a summary of the current workspace state for context.

### Combining Sections

```python
from weakincentives.prompt import PromptTemplate

template = PromptTemplate[Response](
    ns="myapp",
    key="full-agent",
    name="full-agent",
    sections=[
        MarkdownSection(title="Goal", template="...", key="goal"),
        planning_section,
        vfs_section,
        digest_section,
        subagents_section,
    ],
)
```

## Session State Management (Redux-like)

Sessions maintain immutable state with selectors for querying:

```python
from weakincentives.runtime import (
    Session,
    append,           # Add a value to session
    select_latest,    # Get most recent instance of type
    select_all,       # Get all instances of type
    select_where,     # Get instances matching predicate
    replace_latest,   # Replace most recent instance
    upsert_by,        # Update or insert by key function
)

# Append data to session
session = append(session, Plan(summary="...", steps=[...]))
session = append(session, PlanStep(title="Step 1", status="pending"))

# Query session state
latest_plan = select_latest(session, Plan)
all_steps = select_all(session, PlanStep)
completed = select_where(session, PlanStep, lambda s: s.status == "completed")

# Update existing data
session = replace_latest(session, Plan, updated_plan)
session = upsert_by(session, step, lambda s: s.id)  # Upsert by id field
```

### Accessing Session in Tool Handlers

```python
def my_tool_handler(params: MyParams, *, context: ToolContext) -> ToolResult[MyResult]:
    session = context.session

    # Read current plan
    plan = select_latest(session, Plan)
    if not plan:
        return ToolResult(message="No plan found", value=None, success=False)

    # Tool handlers can't mutate session directly, but adapters
    # automatically record ToolInvoked events
    return ToolResult(message="Done", value=MyResult(...), success=True)
```

## Event Subscription and Observability

Subscribe to events for logging, metrics, or custom behavior:

```python
from weakincentives.runtime import (
    Session,
    PromptRendered,
    PromptExecuted,
    ToolInvoked,
)

# Session creates its own EventBus internally
session = Session()

# Subscribe to prompt renders
def on_render(event: object) -> None:
    if isinstance(event, PromptRendered):
        print(f"Rendered: {event.rendered_prompt[:100]}...")

session.event_bus.subscribe(PromptRendered, on_render)

# Subscribe to tool invocations
def on_tool(event: object) -> None:
    if isinstance(event, ToolInvoked):
        print(f"Tool '{event.name}' called, success={event.result.success}")

session.event_bus.subscribe(ToolInvoked, on_tool)

# Subscribe to prompt completions (includes token usage)
def on_complete(event: object) -> None:
    if isinstance(event, PromptExecuted):
        usage = event.usage
        if usage:
            print(f"Tokens: {usage.input_tokens} in, {usage.output_tokens} out")

session.event_bus.subscribe(PromptExecuted, on_complete)

# Events are dispatched synchronously during adapter.evaluate()
response = adapter.evaluate(prompt, bus=session.event_bus, session=session)
```

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
provider calls. Pass them into adapters to ensure long-running tasks fail fast:

```python
from datetime import datetime, timedelta, UTC
from weakincentives import Deadline

# Create a deadline 5 minutes from now
deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))

# Check remaining time in tool handlers
if deadline.remaining() <= timedelta(0):
    # Handle timeout
    pass
```

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

## Common Patterns

### Hierarchical Sections

Sections can contain children for organized prompts:

```python
root = MarkdownSection(
    title="Agent Instructions",
    template="You are a helpful assistant.",
    key="root",
    children=[
        MarkdownSection(title="Guidelines", template="Be concise.", key="guidelines"),
        MarkdownSection(title="Constraints", template="No harmful content.", key="constraints"),
    ],
)
# Renders as nested markdown:
# ## 1. Agent Instructions
# You are a helpful assistant.
# ### 1.1. Guidelines
# Be concise.
# ### 1.2. Constraints
# No harmful content.
```

### Tool Examples for Better Model Understanding

```python
from weakincentives import Tool, ToolExample

search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search the codebase",
    handler=search_handler,
    examples=(
        ToolExample(
            description="Find all Python files with 'async'",
            input=SearchParams(query="async def", file_pattern="*.py"),
            output=SearchResult(matches=["src/api.py:10", "src/worker.py:25"]),
        ),
    ),
)
```

### Dataclass Serialization

```python
from weakincentives.serde import dump, parse, schema, clone

# Serialize dataclass to JSON-compatible dict
data = dump(my_dataclass)

# Parse dict back to dataclass
instance = parse(MyDataclass, data)

# Generate JSON schema for dataclass
json_schema = schema(MyDataclass)

# Deep clone a dataclass
copy = clone(my_dataclass)
```

### Design-by-Contract Validation

```python
from weakincentives.dbc import require, ensure, invariant

@require(lambda params: params.query, "Query must not be empty")
@ensure(lambda result: result.success or result.message, "Failed results must have message")
def my_tool_handler(params: Params, *, context: ToolContext) -> ToolResult[Result]:
    ...
```

## Full Production Harness Example

A complete, production-ready agent harness with all recommended patterns:

```python
"""Complete agent harness example with full observability."""
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from pathlib import Path

from weakincentives import (
    Deadline,
    MarkdownSection,
    Prompt,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.prompt import PromptTemplate
from weakincentives.adapters import PromptEvaluationError
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import (
    PromptExecuted,
    Session,
    ToolInvoked,
    append,
    configure_logging,
    get_logger,
    select_latest,
)
from weakincentives.tools import (
    HostMount,
    PlanningStrategy,
    PlanningToolsSection,
    VfsPath,
    VfsToolsSection,
    WorkspaceDigestSection,
)


# ============================================================
# 1. STRUCTURED TYPES
# ============================================================

@dataclass(slots=True, frozen=True)
class AgentConfig:
    """Configuration for the agent."""
    workspace_path: str
    timeout_seconds: float = 300.0


@dataclass(slots=True, frozen=True)
class TaskParams:
    """Parameters for the task section."""
    objective: str
    context: str = ""


@dataclass(slots=True, frozen=True)
class TaskResponse:
    """Structured response from the agent."""
    summary: str
    files_modified: list[str]
    next_steps: list[str]


@dataclass(slots=True, frozen=True)
class ExecuteCommandParams:
    """Parameters for command execution tool."""
    command: str
    working_dir: str = "."


@dataclass(slots=True, frozen=True)
class ExecuteCommandResult:
    """Result of command execution."""
    stdout: str
    stderr: str
    exit_code: int


# ============================================================
# 2. TOOL HANDLERS
# ============================================================

def execute_command(
    params: ExecuteCommandParams, *, context: ToolContext
) -> ToolResult[ExecuteCommandResult]:
    """Execute a shell command in the workspace."""
    # Check deadline using remaining() method
    if context.deadline and context.deadline.remaining() <= timedelta(0):
        return ToolResult(
            message="Deadline exceeded before command execution",
            value=ExecuteCommandResult(stdout="", stderr="timeout", exit_code=-1),
            success=False,
        )

    import subprocess
    try:
        result = subprocess.run(
            params.command,
            shell=True,
            cwd=params.working_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return ToolResult(
            message=f"Command completed with exit code {result.returncode}",
            value=ExecuteCommandResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            ),
            success=result.returncode == 0,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            message="Command timed out",
            value=ExecuteCommandResult(stdout="", stderr="timeout", exit_code=-1),
            success=False,
        )


execute_tool = Tool[ExecuteCommandParams, ExecuteCommandResult](
    name="execute_command",
    description="Execute a shell command in the workspace",
    handler=execute_command,
)


# ============================================================
# 3. BUILD PROMPT TEMPLATE
# ============================================================

def build_agent_template(session: Session, config: AgentConfig) -> PromptTemplate[TaskResponse]:
    """Build the agent's prompt template with all sections."""
    return PromptTemplate[TaskResponse](
        ns="myapp/agents",
        key="task-agent",
        name="task-agent",
        sections=[
            MarkdownSection(
                title="Role",
                template="You are a skilled software engineer assistant.",
                key="role",
            ),
            MarkdownSection[TaskParams](
                title="Task",
                template="Objective: ${objective}\n\nAdditional context: ${context}",
                key="task",
                default_params=TaskParams(objective=""),
            ),
            PlanningToolsSection(
                session=session,
                strategy=PlanningStrategy.PLAN_ACT_REFLECT,
            ),
            VfsToolsSection(
                session=session,
                mounts=(
                    HostMount(
                        host_path=config.workspace_path,
                        mount_path=VfsPath(("workspace",)),
                        include_glob=("*.py", "*.md", "*.json", "*.yaml", "*.toml"),
                        exclude_glob=("*.pyc", "__pycache__/**", ".git/**"),
                        max_bytes=500_000,
                    ),
                ),
                allowed_host_roots=(Path(config.workspace_path).resolve(),),
            ),
            WorkspaceDigestSection(session=session),
            MarkdownSection(
                title="Commands",
                template="Use execute_command for shell operations when needed.",
                key="commands",
                tools=(execute_tool,),
            ),
        ],
    )


# ============================================================
# 4. AGENT HARNESS
# ============================================================

class AgentHarness:
    """Production agent harness with full observability."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = get_logger("agent")

        # Session creates its own EventBus internally
        self.session = Session()
        self.adapter = OpenAIAdapter(model="gpt-4o")

        # Build template
        self.template = build_agent_template(self.session, config)

        # Set up observability
        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        """Subscribe to events for logging and metrics."""

        def on_tool_invoked(event: object) -> None:
            if isinstance(event, ToolInvoked):
                self.logger.info(
                    "Tool invoked",
                    event="tool.invoked",
                    tool=event.name,
                    success=event.result.success,
                )

        def on_prompt_executed(event: object) -> None:
            if isinstance(event, PromptExecuted):
                usage = event.usage
                if usage:
                    self.logger.info(
                        "Prompt executed",
                        event="prompt.executed",
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    )

        self.session.event_bus.subscribe(ToolInvoked, on_tool_invoked)
        self.session.event_bus.subscribe(PromptExecuted, on_prompt_executed)

    def run(self, objective: str, context: str = "") -> TaskResponse:
        """Execute the agent with the given objective."""
        configure_logging()

        # Create deadline
        deadline = Deadline(
            expires_at=datetime.now(UTC) + timedelta(seconds=self.config.timeout_seconds)
        )

        # Bind parameters
        prompt = Prompt(self.template).bind(
            TaskParams(objective=objective, context=context)
        )

        try:
            response = self.adapter.evaluate(
                prompt,
                bus=self.session.event_bus,
                session=self.session,
                deadline=deadline,
            )

            # Store result in session for later retrieval
            if response.output:
                self.session = append(self.session, response.output)

            return response.output

        except PromptEvaluationError as exc:
            self.logger.error(
                "Agent evaluation failed",
                event="agent.error",
                phase=exc.phase,
                prompt=exc.prompt_name,
            )
            raise

    def get_last_result(self) -> TaskResponse | None:
        """Retrieve the last task result from session."""
        return select_latest(self.session, TaskResponse)


# ============================================================
# 5. USAGE
# ============================================================

if __name__ == "__main__":
    config = AgentConfig(workspace_path="./my-project")
    harness = AgentHarness(config)

    result = harness.run(
        objective="Add input validation to the user registration endpoint",
        context="The endpoint is in src/api/users.py",
    )

    print(f"Summary: {result.summary}")
    print(f"Modified: {result.files_modified}")
    print(f"Next steps: {result.next_steps}")
```

## Versioning & Stability

- Public APIs are the objects exported from `weakincentives` and the
  submodules documented above.
- Adapters are optional; include only the extras you need.
- Keep `StructuredOutputConfig`, tool schemas, and overrides in version control
  so your agents remain deterministic and auditable.

## License

Apache License 2.0. See `LICENSE` for details.
