# WINK (Weak Incentives) - Agent Reference

Dense technical guide for AI coding agents. WINK is a Python 3.12+ library for
building deterministic, side-effect-free background agents with typed prompts,
immutable sessions, and provider-agnostic adapters.

## Core Philosophy

**The prompt is the agent.** Prompts are hierarchical documents where sections
bundle instructions and tools together. No separate tool registry; capabilities
live in the prompt definition.

**Event-driven state.** All mutations flow through pure reducers processing
typed events. State is immutable and inspectable via snapshots.

**Provider-agnostic.** Same agent definition works across OpenAI, LiteLLM, and
Claude Agent SDK via adapter abstraction.

---

## Module Map

```
weakincentives                    # Top-level exports
weakincentives.prompt             # Prompt authoring, sections, tools
weakincentives.prompt.overrides   # Hash-based prompt iteration
weakincentives.runtime            # Session, events, lifecycle, mailbox
weakincentives.runtime.session    # Slice ops, reducers, snapshots
weakincentives.runtime.events     # Dispatcher, event types
weakincentives.runtime.mailbox    # Message queues
weakincentives.adapters           # Provider base, config, throttling
weakincentives.adapters.openai    # OpenAIAdapter
weakincentives.adapters.litellm   # LiteLLMAdapter
weakincentives.adapters.claude_agent_sdk  # ClaudeAgentSDKAdapter
weakincentives.contrib.tools      # VFS, planning, asteval, podman
weakincentives.contrib.optimizers # WorkspaceDigestOptimizer
weakincentives.contrib.mailbox    # RedisMailbox
weakincentives.resources          # Dependency injection
weakincentives.filesystem         # Filesystem protocol
weakincentives.evals              # Evaluation framework
weakincentives.serde              # Dataclass serialization
weakincentives.dbc                # Design-by-contract decorators
weakincentives.formal             # TLA+ specification embedding
weakincentives.skills             # Agent Skills support
weakincentives.types              # JSON type aliases
```

---

## Import Cheatsheet

### Essential Imports

```python
# Prompt authoring
from weakincentives import Prompt, MarkdownSection, Tool, ToolContext, ToolResult
from weakincentives.prompt import PromptTemplate, SectionVisibility

# Runtime
from weakincentives.runtime import Session, MainLoop, MainLoopConfig
from weakincentives.runtime import InProcessDispatcher, PromptExecuted, ToolInvoked

# Adapters
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.adapters.litellm import LiteLLMAdapter
from weakincentives.adapters import PromptResponse, PromptEvaluationError

# Constraints
from weakincentives import Budget, BudgetTracker, Deadline

# Contrib tools
from weakincentives.contrib.tools import (
    PlanningToolsSection, Plan, PlanStep,
    VfsToolsSection, HostMount, VfsPath,
    WorkspaceDigestSection, WorkspaceDigest,
    AstevalSection,
    PodmanSandboxSection, PodmanSandboxConfig,
)

# Serde
from weakincentives.serde import dump, parse, schema, clone

# Resources
from weakincentives.resources import Binding, Scope, ResourceRegistry
```

### Claude Agent SDK

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    HostMount,  # Different from contrib.tools.HostMount
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)
```

---

## Minimal Working Example

```python
from dataclasses import dataclass
from weakincentives import Prompt, MarkdownSection
from weakincentives.prompt import PromptTemplate
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import Session

@dataclass(slots=True, frozen=True)
class TaskParams:
    objective: str

@dataclass(slots=True, frozen=True)
class TaskResult:
    summary: str
    steps: list[str]

template = PromptTemplate[TaskResult](
    ns="myapp",
    key="task-agent",
    name="task_agent",
    sections=(
        MarkdownSection[TaskParams](
            title="Task",
            key="task",
            template="Complete: ${objective}",
        ),
    ),
)

session = Session()
adapter = OpenAIAdapter(model="gpt-4o-mini")
prompt = Prompt(template).bind(TaskParams(objective="Review the auth module"))
response = adapter.evaluate(prompt, session=session)
result: TaskResult = response.output
```

---

## Core Patterns

### 1. Prompt Construction

**PromptTemplate** is the immutable blueprint. **Prompt** wraps it with bindings.

```python
from weakincentives.prompt import PromptTemplate, MarkdownSection, Prompt

@dataclass(slots=True, frozen=True)
class OutputType:
    answer: str

# Template: defines structure, typed output
template = PromptTemplate[OutputType](
    ns="namespace",       # Required: grouping
    key="prompt-key",     # Required: unique identifier
    name="prompt_name",   # Optional: display name
    sections=(...),       # Ordered sections
)

# Prompt: wraps template, binds parameters
prompt = Prompt(template).bind(MyParams(...))

# Render to inspect
rendered = prompt.render()
print(rendered.text)       # Markdown content
print(rendered.tools)      # Tool tuple
print(rendered.output_type)  # OutputType class
```

### 2. Sections

All sections inherit from `Section`. Most common: `MarkdownSection`.

```python
from weakincentives.prompt import MarkdownSection, SectionVisibility

@dataclass(slots=True, frozen=True)
class ReviewParams:
    focus: str

section = MarkdownSection[ReviewParams](
    title="Review Guidelines",        # Rendered as heading
    key="review-guidelines",          # Unique within prompt
    template="Focus on: ${focus}",    # Template.substitute syntax
    default_params=ReviewParams(focus="correctness"),
    tools=(my_tool,),                 # Tools attached to section
    children=(...),                   # Nested sections
    visibility=SectionVisibility.FULL,  # Or SUMMARY for progressive disclosure
    summary="Guidelines available.",  # Shown when visibility=SUMMARY
    enabled=lambda p: p.focus != "",  # Conditional enablement
    accepts_overrides=True,           # Allow prompt overrides
)
```

**Placeholder syntax**: `${field_name}` from the params dataclass.

**Children**: Nest sections for hierarchy. Heading levels auto-increment.

### 3. Tools

Tools are typed with params and result dataclasses.

```python
from weakincentives import Tool, ToolContext, ToolResult
from dataclasses import dataclass, field

@dataclass(slots=True, frozen=True)
class SearchParams:
    query: str = field(metadata={"description": "Search query"})
    limit: int = field(default=10, metadata={"description": "Max results"})

@dataclass(slots=True, frozen=True)
class SearchResult:
    matches: list[str]

def search_handler(
    params: SearchParams,
    *,
    context: ToolContext,
) -> ToolResult[SearchResult]:
    # Access session state
    plan = context.session[Plan].latest()

    # Access resources
    fs = context.filesystem  # Shorthand for context.resources.get(Filesystem)

    # Check deadline
    if context.deadline and context.deadline.remaining().total_seconds() < 5:
        return ToolResult.error("Deadline too close")

    # Do work...
    matches = ["result1", "result2"]

    # Return typed result
    return ToolResult.ok(SearchResult(matches=matches), message="Found 2 matches")

# Create tool
search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search for content",
    handler=search_handler,
)

# Alternative: wrap function (uses __name__ and docstring)
search_tool = Tool.wrap(search_handler)

# Attach to section
section = MarkdownSection(
    title="Search",
    key="search",
    template="Use search tool when needed.",
    tools=(search_tool,),
)
```

**ToolResult constructors**:
- `ToolResult.ok(value, message="...")` - Success with typed value
- `ToolResult.error("message")` - Failure, value=None

### 4. Sessions

Redux-style immutable state container with typed slices.

```python
from weakincentives.runtime import Session, InProcessDispatcher
from weakincentives.runtime.session import replace_latest, append_all

# Create session
session = Session()  # Creates InProcessDispatcher internally
# Or with explicit dispatcher:
dispatcher = InProcessDispatcher()
session = Session(dispatcher=dispatcher)

# Query state
plan = session[Plan].latest()           # Most recent or None
all_plans = session[Plan].all()         # All values as tuple
active = session[Plan].where(lambda p: p.status == "active")
exists = session[Plan].exists()         # Boolean

# Mutations (all dispatch events internally)
session[Plan].seed(initial_plan)        # Initialize/replace slice
session[Plan].append(new_plan)          # Append via default reducer
session[Plan].clear()                   # Clear all
session[Plan].clear(lambda p: p.done)   # Clear matching predicate

# Register custom reducer
def my_reducer(values: tuple[Plan, ...], event: UpdatePlan, *, context) -> tuple[Plan, ...]:
    # Return new tuple
    return values + (updated_plan,)

session[Plan].register(UpdatePlan, my_reducer)

# Dispatch events
session.dispatch(UpdatePlan(step_id=1, status="done"))

# Snapshots
snapshot = session.snapshot()
json_str = snapshot.to_json()
session.restore(Snapshot.from_json(json_str))

# Session hierarchy
child = Session(dispatcher=dispatcher, parent=session)
```

**Built-in reducers** (from `weakincentives.runtime.session`):
- `append_all` - Always append (default)
- `replace_latest` - Keep only most recent
- `upsert_by(key_fn)` - Replace by key
- `replace_latest_by(key_fn)` - Latest per key

### 5. Declarative State Slices

Co-locate reducers with state using `@reducer` decorator:

```python
from dataclasses import dataclass, replace
from weakincentives.runtime.session import reducer

@dataclass(frozen=True)
class AddStep:
    step: str

@dataclass(frozen=True)
class CompleteStep:
    step_id: int

@dataclass(frozen=True)
class AgentPlan:
    steps: tuple[str, ...] = ()
    current: int = 0

    @reducer(on=AddStep)
    def add(self, event: AddStep) -> "AgentPlan":
        return replace(self, steps=self.steps + (event.step,))

    @reducer(on=CompleteStep)
    def complete(self, event: CompleteStep) -> "AgentPlan":
        return replace(self, current=self.current + 1)

# Install on session
session.install(AgentPlan)
session[AgentPlan].seed(AgentPlan())
session.dispatch(AddStep(step="Research"))
```

### 6. Adapters

Provider-agnostic evaluation interface.

```python
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.adapters.litellm import LiteLLMAdapter
from weakincentives.adapters import OpenAIModelConfig, ThrottleError

# Basic
adapter = OpenAIAdapter(model="gpt-4o")

# With config
adapter = OpenAIAdapter(
    model="gpt-4o",
    model_config=OpenAIModelConfig(
        temperature=0.7,
        max_tokens=4096,
    ),
)

# LiteLLM (multi-provider)
adapter = LiteLLMAdapter(model="claude-3-sonnet-20240229")

# Evaluate
response = adapter.evaluate(
    prompt,
    session=session,
    deadline=deadline,        # Optional
    budget=budget,            # Optional
    budget_tracker=tracker,   # Optional
)

# Response fields
response.output       # Parsed dataclass (OutputType)
response.text         # Raw text
response.prompt_name  # Prompt identifier

# Error handling
try:
    response = adapter.evaluate(prompt, session=session)
except PromptEvaluationError as e:
    # e.phase: "request" | "response" | "tool" | "budget" | "deadline"
    # e.prompt_name: str
    print(f"Failed at {e.phase}: {e}")
except ThrottleError as e:
    # e.retry_after: float
    print(f"Throttled, retry in {e.retry_after}s")
```

### 7. Claude Agent SDK Adapter

Native Claude Code capabilities with hermetic isolation.

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    HostMount,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)

# Create workspace (materializes files to temp dir)
workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(
        HostMount(
            host_path="/path/to/project",
            mount_path="project",
            include_glob=("*.py", "*.md"),
            exclude_glob=("__pycache__/*",),
            max_bytes=5_000_000,
        ),
    ),
    allowed_host_roots=("/path/to",),
)

# Configure isolation
adapter = ClaudeAgentSDKAdapter(
    model="claude-sonnet-4-5-20250929",
    client_config=ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",  # Auto-approve tools
        cwd=str(workspace.temp_dir),
        isolation=IsolationConfig(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            network_policy=NetworkPolicy.no_network(),  # API only
            sandbox=SandboxConfig(enabled=True),
        ),
    ),
)

# Use and cleanup
response = adapter.evaluate(prompt, session=session)
workspace.cleanup()
```

**Isolation modes**:
- `NetworkPolicy.no_network()` - API access only
- `NetworkPolicy(allowed_domains=("docs.python.org",))` - Specific domains
- `SandboxConfig(enabled=True)` - OS-level sandboxing

### 8. MainLoop Orchestration

Standardized request/response workflow with visibility expansion handling.

```python
from weakincentives.runtime import MainLoop, MainLoopConfig, MainLoopRequest, Session
from weakincentives.runtime import InMemoryMailbox
from weakincentives.prompt import Prompt

class ReviewLoop(MainLoop[ReviewRequest, ReviewResult]):
    def __init__(self, adapter, requests):
        super().__init__(
            adapter=adapter,
            requests=requests,
            config=MainLoopConfig(
                budget=Budget(max_total_tokens=50000),
            ),
        )
        self._template = build_template()

    def prepare(self, request: ReviewRequest) -> tuple[Prompt[ReviewResult], Session]:
        prompt = Prompt(self._template).bind(request)
        session = Session()
        return prompt, session

# Direct execution
loop = ReviewLoop(adapter=adapter, requests=mailbox)
response, session = loop.execute(ReviewRequest(...))

# Mailbox-driven
requests = InMemoryMailbox(name="requests")
responses = InMemoryMailbox(name="responses")

requests.send(
    MainLoopRequest(request=ReviewRequest(...)),
    reply_to=responses,
)

# Run loop (blocks, processes from mailbox)
loop.run(max_iterations=None)

# Graceful shutdown
loop.shutdown(timeout=5.0)
```

### 9. Contrib Tools

**VFS (Virtual Filesystem)**:

```python
from weakincentives.contrib.tools import VfsToolsSection, HostMount, VfsPath

vfs = VfsToolsSection(
    session=session,
    mounts=(
        HostMount(
            host_path="./repo",
            mount_path=VfsPath(("workspace",)),
            include_glob=("*.py",),
            exclude_glob=("*.pyc",),
            max_bytes=600_000,
        ),
    ),
    allowed_host_roots=(Path("."),),
)
# Tools: ls, read_file, write_file, edit_file, glob, grep, rm
```

**Planning**:

```python
from weakincentives.contrib.tools import (
    PlanningToolsSection, PlanningStrategy, Plan, PlanStep
)

planning = PlanningToolsSection(
    session=session,
    strategy=PlanningStrategy.PLAN_ACT_REFLECT,
)
# Tools: planning_setup_plan, planning_add_step, planning_update_step, planning_read_plan

# Query plan state
plan = session[Plan].latest()
for step in plan.steps:
    print(f"[{step.status}] {step.title}")
```

**Workspace Digest**:

```python
from weakincentives.contrib.tools import WorkspaceDigestSection, WorkspaceDigest

digest_section = WorkspaceDigestSection(session=session)
# Auto-renders workspace summary from session[WorkspaceDigest]
```

### 10. Resources

Dependency injection with scoped lifecycles.

```python
from weakincentives.resources import Binding, Scope, ResourceRegistry

# Build registry
registry = ResourceRegistry.of(
    Binding(Config, lambda r: Config.from_env()),
    Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
    Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),  # Fresh per call
)

# Use with prompt
prompt = Prompt(template).bind(
    params,
    resources={HTTPClient: http_client},  # Direct instance
)
# Or with bindings
prompt = Prompt(template).bind(
    params,
    resources={
        Config: Binding(Config, lambda r: Config()),
    },
)

# Lifecycle management
with prompt.resources:
    http = prompt.resources.get(HTTPClient)
    response = adapter.evaluate(prompt, session=session)
# Cleaned up automatically

# In tool handlers
def handler(params, *, context: ToolContext) -> ToolResult:
    http = context.resources.get(HTTPClient)
    fs = context.filesystem  # Shorthand
```

**Scopes**:
- `Scope.SINGLETON` - Once per context (default)
- `Scope.TOOL_CALL` - Fresh per tool invocation
- `Scope.PROTOTYPE` - Fresh every resolution

### 11. Serialization

```python
from weakincentives.serde import dump, parse, schema, clone

# Serialize dataclass to dict
data = dump(my_dataclass)

# Parse dict to dataclass
obj = parse(MyDataclass, data)

# JSON schema
json_schema = schema(MyDataclass)

# Deep clone
copy = clone(my_dataclass)
```

### 12. Design-by-Contract

```python
from weakincentives.dbc import require, ensure, invariant, pure

@require(lambda x: x > 0, "x must be positive")
@ensure(lambda result: result >= 0, "result non-negative")
def compute(x: int) -> int:
    return x * 2

@invariant(lambda self: self.count >= 0)
class Counter:
    count: int = 0

@pure  # Validates no side effects
def hash_value(x: str) -> int:
    return hash(x)
```

---

## Decision Trees

### Which Adapter?

```
Need Claude Code native tools? → ClaudeAgentSDKAdapter
Need multi-provider support?   → LiteLLMAdapter
OpenAI only?                   → OpenAIAdapter
```

### Which Workspace Tool?

```
Claude Agent SDK mode?         → ClaudeAgentWorkspaceSection
Need shell execution?          → PodmanSandboxSection
Standard file ops only?        → VfsToolsSection
```

### Which Reducer?

```
Recording every event?         → append_all (default)
Only latest value matters?     → replace_latest
Keyed upsert (like cache)?     → upsert_by(key_fn)
Complex state transitions?     → @reducer decorator on dataclass
```

### Session vs Resource State?

```
Agent state (plans, results)?  → Session slices
Runtime deps (HTTP, DB)?       → ResourceRegistry
Filesystem state?              → Filesystem via resources
```

---

## Common Pitfalls

1. **Forgetting `slots=True, frozen=True`** on dataclasses - breaks serde
2. **Missing `${}` in templates** - use `${field}` not `{field}`
3. **Tool handler signature** - must be `(params, *, context: ToolContext)`
4. **ToolResult return** - use `.ok()` or `.error()`, not raw constructor
5. **Session mutations** - all go through `dispatch()`, use accessor methods
6. **Resource access outside context** - use `with prompt.resources:` block
7. **Duplicate tool names** - raises `PromptValidationError`
8. **Hash mismatch in overrides** - stale overrides silently filtered

---

## Event Types

```python
from weakincentives.runtime import (
    PromptRendered,    # After render, before provider call
    PromptExecuted,    # After all tools and parsing
    ToolInvoked,       # After each tool handler
    TokenUsage,        # Token consumption data
)

# Subscribe
session.dispatcher.subscribe(ToolInvoked, lambda e: print(e.name))
```

---

## Error Hierarchy

```
WinkError                       # Base for all WINK errors
├── DeadlineExceededError       # Wall-clock limit hit
├── BudgetExceededError         # Token limit breached
├── ToolValidationError         # Tool params invalid
├── PromptError                 # Prompt system errors
│   ├── PromptValidationError   # Construction failures
│   ├── PromptRenderError       # Render failures
│   ├── OutputParseError        # Structured output invalid
│   └── VisibilityExpansionRequired  # Progressive disclosure request
├── SnapshotRestoreError        # Snapshot restore failed
└── TransactionError            # Transaction failed
```

---

## Development Commands

```bash
uv sync && ./install-hooks.sh   # Setup

make format      # ruff format (88-char)
make lint        # ruff check --preview
make typecheck   # ty + pyright strict
make test        # pytest, 100% coverage required
make check       # ALL checks - run before commit

make bandit      # Security scan
make deptry      # Dependency analysis
make pip-audit   # Vulnerability scan
```

---

## File Layout

```
src/weakincentives/
├── adapters/           # OpenAI, LiteLLM, Claude Agent SDK
│   └── claude_agent_sdk/
├── cli/                # wink CLI
├── contrib/
│   ├── tools/          # Planning, VFS, asteval, podman, workspace digest
│   ├── optimizers/     # WorkspaceDigestOptimizer
│   └── mailbox/        # RedisMailbox
├── dataclasses/        # FrozenDataclass utilities
├── dbc/                # @require, @ensure, @invariant, @pure
├── debug/              # Log collector, session inspection
├── evals/              # Evaluation framework
├── filesystem/         # Filesystem protocol
├── formal/             # TLA+ embedding
├── optimizers/         # Optimizer framework
├── prompt/             # Sections, tools, rendering, overrides
│   └── overrides/      # LocalPromptOverridesStore
├── resources/          # DI with Binding, Scope
├── runtime/
│   ├── events/         # Dispatcher, event types
│   ├── mailbox/        # Message queue protocol
│   └── session/        # Session, slices, reducers
│       └── slices/     # MemorySlice, JsonlSlice
├── serde/              # dump, parse, schema, clone
├── skills/             # Agent Skills support
└── types/              # JSONValue, type aliases
```

---

## Key Specs

Read before modifying related code:

| Spec | Topic |
|------|-------|
| `specs/PROMPTS.md` | Prompt system, composition, overrides |
| `specs/SESSIONS.md` | Session lifecycle, events, budgets |
| `specs/TOOLS.md` | Tool registration, planning tools |
| `specs/ADAPTERS.md` | Provider adapters, throttling |
| `specs/CLAUDE_AGENT_SDK.md` | SDK adapter, isolation, MCP |
| `specs/WORKSPACE.md` | VFS, Podman, asteval |
| `specs/DBC.md` | Design-by-contract patterns |
| `specs/RESOURCE_REGISTRY.md` | Dependency injection |
| `specs/MAIN_LOOP.md` | MainLoop orchestration |
| `specs/MAILBOX.md` | Message queue abstraction |

---

## Quick Reference

### PromptTemplate

```python
PromptTemplate[OutputT](
    ns: str,                    # Namespace (required)
    key: str,                   # Unique key (required)
    name: str | None,           # Display name
    sections: tuple[Section],   # Ordered sections
    resources: ResourceRegistry,
)
```

### MarkdownSection

```python
MarkdownSection[ParamsT](
    title: str,                 # Heading text
    key: str,                   # Unique key
    template: str,              # ${field} syntax
    default_params: ParamsT,
    tools: tuple[Tool],
    children: tuple[Section],
    visibility: SectionVisibility,
    summary: str,               # For SUMMARY visibility
    enabled: Callable[[ParamsT], bool],
    accepts_overrides: bool,
)
```

### Tool

```python
Tool[ParamsT, ResultT](
    name: str,                  # ^[a-z0-9_-]{1,64}$
    description: str,           # 1-200 chars
    handler: ToolHandler,
    examples: tuple[ToolExample],
    accepts_overrides: bool,
)

# Handler signature
def handler(params: ParamsT, *, context: ToolContext) -> ToolResult[ResultT]
```

### ToolContext

```python
context.session           # Session
context.deadline          # Deadline | None
context.budget_tracker    # BudgetTracker | None
context.resources         # ScopedResourceContext (from prompt)
context.filesystem        # Filesystem | None (shorthand)
context.prompt            # PromptProtocol
context.rendered_prompt   # RenderedPromptProtocol | None
context.adapter           # ProviderAdapterProtocol
```

### Session

```python
session[T].latest()       # T | None
session[T].all()          # tuple[T, ...]
session[T].where(pred)    # tuple[T, ...]
session[T].exists()       # bool
session[T].seed(value)    # Initialize slice
session[T].append(value)  # Dispatch to reducers
session[T].clear()        # Clear slice
session[T].register(E, reducer)  # Register reducer
session.dispatch(event)   # Broadcast dispatch
session.snapshot()        # Snapshot
session.restore(snap)     # Restore from snapshot
```

### Budget

```python
Budget(
    deadline: Deadline | None,
    max_total_tokens: int | None,
    max_input_tokens: int | None,
    max_output_tokens: int | None,
)

tracker = BudgetTracker(budget)
tracker.record_cumulative(eval_id, usage)
tracker.check()  # Raises BudgetExceededError
```

---

## Example: Complete Agent

See `code_reviewer_example.py` for production patterns:
- Structured output types
- VFS/Planning tool sections
- MainLoop implementation
- Event subscription
- Prompt overrides
- Claude Agent SDK mode

---

## Alpha Status

All APIs may change without backward compatibility. No deprecation warnings;
unused code is deleted completely.

---

## License

Apache License 2.0
