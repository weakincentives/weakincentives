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

An agent harness in WINK wires together five core components:

1. **PromptTemplate** (`weakincentives.prompt.PromptTemplate`): Immutable blueprint defining sections, tools, and structured output schema. Import from `weakincentives.prompt`.

1. **Prompt** (`weakincentives.Prompt`): Wraps a template with parameter bindings and optional overrides. Call `.bind()` to attach runtime params before evaluation.

1. **Session** (`weakincentives.runtime.Session`): Redux-like event ledger that records all prompt renders, tool invocations, and custom state. Creates its own `EventBus` internally (access via `session.event_bus`).

1. **ProviderAdapter** (`OpenAIAdapter`, `LiteLLMAdapter`): Bridges prompts to LLM providers. Call `adapter.evaluate(prompt, bus=session.event_bus, session=session)` to execute.

1. **Tool handlers**: Functions with signature `(params: ParamsT, *, context: ToolContext) -> ToolResult[ResultT]` that implement side effects when the model requests tool calls.

**Data flow**: PromptTemplate → Prompt (with bound params) → `adapter.evaluate()` → LLM response → Tool calls dispatched → ToolResult returned → Session updated → Events published

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
    - `Budget`: Resource envelope combining time and token limits.
    - `BudgetExceededError`: Exception raised when a budget limit is breached.
    - `BudgetTracker`: Thread-safe tracker for cumulative token usage against a Budget.
    - `Deadline`: Immutable value object describing a wall-clock expiration.
    - `FrozenDataclass`: Decorator providing immutable dataclass utilities (copy, asdict, normalization).
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
    - `WinkError`: Base class for all weakincentives exceptions.
    - `configure_logging`: Configure the root logger with sensible defaults.
    - `get_logger`: Return a `StructuredLogger` scoped to a name.
    - `parse_structured_output`: Parse a model response into the structured output type declared by the prompt.
  - Modules: `adapters`, `cli`, `deadlines`, `debug`, `optimizers`, `prompt`,
    `runtime`, `serde`, `tools`, `types`.
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
    - `SectionNode`: Node in section tree.
    - `SectionPath`: Path to a section.
    - `SectionVisibility`: Enum controlling how a section is rendered (`FULL`, `SUMMARY`).
    - `Tool`: Describe a callable tool exposed by prompt sections.
    - `ToolContext`: Immutable container exposing prompt execution state to handlers.
    - `ToolExample`: Representative invocation for a tool documenting inputs and outputs.
    - `ToolHandler`: Callable protocol implemented by tool handlers.
    - `ToolRenderableResult`: Protocol for tool results that can be rendered.
    - `ToolResult`: Structured response emitted by a tool handler.
    - `SupportsDataclass`: Protocol satisfied by dataclass types and instances.
    - `SupportsDataclassOrNone`: Protocol for dataclass types or None.
    - `SupportsToolResult`: Protocol for tool results.
    - `PromptProtocol`: Protocol for prompts.
    - `PromptTemplateProtocol`: Protocol for prompt templates.
    - `RenderedPromptProtocol`: Protocol for rendered prompts.
    - `ProviderAdapterProtocol`: Protocol for provider adapters.
  - Composition:
    - `DelegationParams`: Parameters for delegation prompts.
    - `DelegationPrompt`: Prompt for delegating tasks to sub-agents.
    - `DelegationSummarySection`: Section for summarizing delegation results.
    - `OpenSectionsParams`: Parameters for progressive disclosure of sections.
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
    - `VisibilityExpansionRequired`: Raised when model requests expansion of summarized sections.
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
- `weakincentives.optimizers`: Prompt optimization algorithms and utilities.
  - Protocol and base classes:
    - `PromptOptimizer`: Protocol for prompt optimization algorithms.
    - `BasePromptOptimizer`: Abstract base class for prompt optimizers.
    - `OptimizerConfig`: Base configuration dataclass with `accepts_overrides` field.
  - Context and results:
    - `OptimizationContext`: Immutable context bundle with adapter, event bus, deadline, and overrides.
    - `OptimizationResult`: Generic result container with response, artifact, and metadata.
    - `WorkspaceDigestResult`: Result of workspace digest optimization.
    - `PersistenceScope`: Enum for artifact storage location (`SESSION`, `GLOBAL`).
  - Concrete implementations:
    - `WorkspaceDigestOptimizer`: Optimizer for generating task-agnostic workspace summaries.
  - Events:
    - `OptimizationStarted`: Event emitted when optimizer begins work.
    - `OptimizationCompleted`: Event emitted on successful completion.
    - `OptimizationFailed`: Event emitted when optimization raises exception.
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

## Quickstart Snippets

### Minimal harness setup

```python
from weakincentives import MarkdownSection, Prompt
from weakincentives.prompt import PromptTemplate
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import Session

@dataclass(slots=True, frozen=True)
class TaskResponse:
    summary: str
    next_steps: list[str]

template = PromptTemplate[TaskResponse](
    ns="myapp/tasks", key="task-agent", name="task-agent",
    sections=[MarkdownSection(title="Instructions", template="...", key="instructions")],
)

session = Session()
adapter = OpenAIAdapter(model="gpt-4o-mini")
response = adapter.evaluate(Prompt(template), bus=session.event_bus, session=session)
result: TaskResponse = response.output
```

### Defining a tool handler

```python
from weakincentives import Tool, ToolContext, ToolResult

@dataclass(slots=True, frozen=True)
class PatchArgs:
    path: str
    diff: str

@dataclass(slots=True, frozen=True)
class PatchResult:
    applied: bool

def apply_patch(params: PatchArgs, *, context: ToolContext) -> ToolResult[PatchResult]:
    # context.session, context.deadline, context.event_bus available
    return ToolResult(message="Applied", value=PatchResult(applied=True), success=True)

patch_tool = Tool[PatchArgs, PatchResult](
    name="apply_patch", description="Apply a unified diff.", handler=apply_patch
)
```

### Attaching tools to sections

```python
section = MarkdownSection(
    title="Instructions", template="Use apply_patch to edit files.",
    key="instructions", tools=(patch_tool,),
)
```

### Multi-turn with session state

```python
from weakincentives.runtime import append, select_latest

response = adapter.evaluate(prompt, bus=session.event_bus, session=session)
session = append(session, response.output)  # Store result
later = select_latest(session, TaskResponse)  # Retrieve later
```

### Error handling

```python
from weakincentives.adapters import PromptEvaluationError

try:
    response = adapter.evaluate(prompt, bus=session.event_bus, session=session)
except PromptEvaluationError as exc:
    print(exc.phase, exc.prompt_name)  # "request"/"response"/"tool"
```

## Prompt Authoring (`weakincentives.prompt`)

### PromptTemplate and Prompt

```python
from weakincentives.prompt import PromptTemplate

template = PromptTemplate[OutputType](
    ns="myapp/agents", key="my-agent", name="my-agent",
    sections=[...], inject_output_instructions=True,
)
prompt = Prompt(template).bind(MyParams(value="..."))  # Bind returns self
```

### MarkdownSection with parameters

Use `${param}` syntax for dynamic content:

```python
section = MarkdownSection[TaskParams](
    title="Task", template="Objective: ${objective}", key="task",
    default_params=TaskParams(objective=""),
)
```

### Tool.wrap helper

Creates a Tool using the function's `__name__` and docstring:

```python
def search(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
    """Search for content."""  # Becomes tool description
    return ToolResult(message="Done", value=SearchResult(...), success=True)

search_tool = Tool.wrap(search)  # name="search", description="Search for content."
```

### ToolContext fields

Available in tool handlers via `context`:

- `context.session` - Current Session
- `context.deadline` - Optional Deadline (check with `deadline.remaining()`)
- `context.event_bus` - EventBus for publishing
- `context.prompt` / `context.rendered_prompt` / `context.adapter`

### ToolResult fields

```python
ToolResult(message="...", value=MyResult(...), success=True, exclude_value_from_context=False)
```

### Additional components

- **`parse_structured_output`**: Parse model response into typed dataclass
- **Delegation**: `DelegationPrompt`, `DelegationSummarySection`, `ParentPromptSection`
- **Overrides**: `LocalPromptOverridesStore` for hash-scoped prompt refinements

## Adapter Layer (`weakincentives.adapters`)

### OpenAI and LiteLLM adapters

```python
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.adapters.litellm import LiteLLMAdapter

adapter = OpenAIAdapter(model="gpt-4o-mini")  # Native JSON schema by default
adapter = OpenAIAdapter(model="gpt-4o", client=custom_client)  # Custom client
adapter = LiteLLMAdapter(model="claude-3-sonnet-20240229")  # Any LiteLLM model
```

### PromptResponse fields

```python
response = adapter.evaluate(prompt, bus=session.event_bus, session=session)
response.output       # Parsed dataclass
response.text         # Raw text
response.usage        # TokenUsage(input_tokens, output_tokens, ...)
```

### Rate limiting

```python
from weakincentives.adapters import new_throttle_policy
policy = new_throttle_policy(requests_per_minute=60)
response = adapter.evaluate(..., throttle=policy)
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

### VfsToolsSection - Sandboxed file operations

```python
from weakincentives.tools import VfsToolsSection, HostMount, VfsPath
vfs = VfsToolsSection(
    session=session,
    mounts=(HostMount(host_path="./repo", mount_path=VfsPath(("workspace",)),
                      include_glob=("*.py",), exclude_glob=("*.pyc",), max_bytes=600_000),),
    allowed_host_roots=(Path("."),),
)
```

Tools: `read_file`, `write_file`, `list_directory`, `glob`, `grep`

### PlanningToolsSection - Multi-step planning

```python
from weakincentives.tools import PlanningToolsSection, PlanningStrategy
planning = PlanningToolsSection(session=session, strategy=PlanningStrategy.PLAN_ACT_REFLECT)
```

Tools: `setup_plan`, `read_plan`, `add_step`, `update_step`, `mark_step`

### SubagentsSection - Parallel delegation

```python
from weakincentives.tools import SubagentsSection, SubagentIsolationLevel
subagents = SubagentsSection(isolation_level=SubagentIsolationLevel.FORK)
```

Tools: `dispatch_subagents`

### WorkspaceDigestSection

```python
digest = WorkspaceDigestSection(session=session)  # Renders workspace summary
```

## Session State Management

### Selectors

```python
from weakincentives.runtime import append, select_latest, select_all, select_where, replace_latest

session = append(session, my_data)
latest = select_latest(session, MyType)
all_items = select_all(session, MyType)
filtered = select_where(session, MyType, lambda x: x.status == "done")
session = replace_latest(session, MyType, updated)
```

### In tool handlers

```python
def handler(params, *, context: ToolContext) -> ToolResult:
    plan = select_latest(context.session, Plan)
    # Tool handlers can't mutate session; adapters record ToolInvoked events
```

## Event Subscription

```python
from weakincentives.runtime import PromptRendered, PromptExecuted, ToolInvoked

session = Session()
session.event_bus.subscribe(ToolInvoked, lambda e: print(e.name))
session.event_bus.subscribe(PromptExecuted, lambda e: print(e.usage))
```

## Session Snapshots

Use `Session.snapshot()` to capture session state for debugging or replay.
Restore with `Snapshot.from_json(...)` and `Session.rollback(...)`.

## Deadlines

```python
from datetime import datetime, timedelta, UTC
from weakincentives import Deadline

deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
# In handlers: if deadline.remaining() <= timedelta(0): ...
```

## Budgets

Budgets combine time and token limits into a single resource envelope:

```python
from datetime import datetime, timedelta, UTC
from weakincentives import Budget, BudgetTracker, BudgetExceededError, Deadline

# Create a budget with deadline and token limits
budget = Budget(
    deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=10)),
    max_total_tokens=100_000,
    max_input_tokens=80_000,
    max_output_tokens=20_000,
)

# Track usage across evaluations
tracker = BudgetTracker(budget=budget)
tracker.record_cumulative("eval-1", usage)  # Record TokenUsage from response
tracker.check()  # Raises BudgetExceededError if any limit breached
```

## Serialization (`weakincentives.serde`)

```python
from weakincentives.serde import dump, parse, schema, clone
data = dump(my_dataclass)           # To JSON-compatible dict
obj = parse(MyDataclass, data)      # From dict
json_schema = schema(MyDataclass)   # JSON schema
copy = clone(my_dataclass)          # Deep clone
```

## Additional Patterns

### Hierarchical sections

```python
root = MarkdownSection(title="Root", template="...", key="root", children=[
    MarkdownSection(title="Child", template="...", key="child"),
])
```

### Tool examples

```python
tool = Tool[P, R](name="search", description="...", handler=h, examples=(
    ToolExample(description="Find X", input=P(...), output=R(...)),
))
```

### Design-by-contract

```python
from weakincentives.dbc import require, ensure
@require(lambda p: p.query, "Query required")
def handler(params, *, context): ...
```

### Section visibility (progressive disclosure)

```python
from weakincentives.prompt import MarkdownSection, SectionVisibility

# Section with summary for progressive disclosure
section = MarkdownSection(
    title="Details",
    template="Full detailed content...",
    key="details",
    summary="Brief summary of the section",
    visibility=SectionVisibility.SUMMARY,  # Show summary by default
)
```

### Prompt optimizers

```python
from weakincentives.optimizers import (
    OptimizationContext,
    PersistenceScope,
    WorkspaceDigestOptimizer,
)

context = OptimizationContext(
    adapter=adapter,
    event_bus=session.event_bus,
    overrides_store=overrides_store,
)
optimizer = WorkspaceDigestOptimizer(context, store_scope=PersistenceScope.SESSION)
result = optimizer.optimize(prompt, session=session)
# result.digest contains the workspace summary
```

## CLI

```bash
pip install "weakincentives[wink]"
wink --help
```

## Example

See `code_reviewer_example.py` in the repository for a complete production
harness demonstrating all patterns: structured types, tool handlers, built-in
sections (VFS, Planning, Subagents), event subscription, and prompt overrides.

## Versioning & Stability

- Public APIs are the objects exported from `weakincentives` and the
  submodules documented above.
- Adapters are optional; include only the extras you need.
- Keep `StructuredOutputConfig`, tool schemas, and overrides in version control
  so your agents remain deterministic and auditable.

## License

Apache License 2.0. See `LICENSE` for details.
