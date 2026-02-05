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

**Provider-agnostic.** Same agent definition works across agentic harnesses via
adapter abstraction. WINK integrates with execution harnesses like Claude Agent SDK.

______________________________________________________________________

## Guiding Principles

### Definition vs Harness

WINK separates what you own from what the runtime provides:

**Agent Definition (you own and iterate):**

- **Prompt** - A structured decision procedure, not a loose string
- **Tools** - The capability surface; the only place side effects occur
- **Policies** - Enforceable invariants constraining tool use and state
- **Feedback** - "Are we done?" checks preventing premature termination

**Execution Harness (runtime-owned):**

- Planning/act loop driving tool calls
- Sandboxing and permissions (filesystem, shell, network)
- Retries, throttling, crash recovery
- Deadlines, budgets, operational guardrails

The harness keeps changing (and increasingly comes from vendor runtimes), but
your agent definition should not. WINK makes the definition a first-class
artifact you can version, review, test, and port across runtimes.

### The Prompt is the Agent

Most frameworks treat prompts as afterthoughts—templates glued to separately
registered tool lists. WINK inverts this: you define an agent as a single
hierarchical document where each section bundles its own instructions and tools.

```text
PromptTemplate[ReviewResponse]
├── MarkdownSection (guidance)
├── WorkspaceDigestSection     ← cached codebase summary
├── MarkdownSection (reference docs, progressive disclosure)
├── MarkdownSection (task instructions)
└── MarkdownSection (user request)
```

**Note:** Tool sections for filesystem, planning, and shell execution are
provided by the execution harness (e.g., Claude Agent SDK) rather than
defined in the prompt. This keeps agent definitions portable across runtimes.

**Why this matters:**

1. **Co-location** - Instructions and tools live together. The section that
   explains filesystem navigation provides the `read_file` tool. Documentation
   can't drift from implementation.

1. **Progressive disclosure** - Nest child sections to reveal advanced
   capabilities when relevant. The LLM sees numbered, hierarchical headings.

1. **Dynamic scoping** - Each section has an `enabled` predicate. Disable a
   section and its entire subtree—tools included—disappears from the prompt.

1. **Typed all the way down** - Sections are parameterized with dataclasses.
   Placeholders are validated at construction time. Tools declare typed params
   and results.

### Policies Over Workflows

**Prefer declarative policies over prescriptive workflows.**

A workflow encodes _how_ to accomplish a goal—a predetermined sequence that
fractures when encountering unexpected situations. A policy encodes _what_ the
goal requires—constraints the agent must satisfy while remaining free to find
any valid path.

```text
Workflow (brittle):              Policy (flexible):
1. Read the file                 - File must be read before overwriting
2. Parse the AST                 - Tests must pass before deployment
3. Generate patch                - Sensitive ops require confirmation
4. Write file
5. Run tests
```

When workflow step 3 fails, the agent is stuck. When a policy check fails, the
agent can reason about alternatives that satisfy the constraint.

**Key policy characteristics:**

- **Declarative** - State what must be true, not how to make it true
- **Composable** - Policies combine via conjunction (all must allow)
- **Fail-closed** - When uncertain, deny; let the agent adapt
- **Observable** - Explain denials to enable self-correction

### Transactional Tools

Tool calls are atomic transactions. When a tool fails:

1. Session state rolls back to pre-call state
1. Filesystem changes revert
1. Error result returned to LLM with guidance

Failed tools don't leave partial state. This enables aggressive retry and
recovery strategies.

### One Sentence Summary

> "You write the agent definition (prompt, tools, policies, feedback); the
> runtime owns the harness (planning loop, sandboxing, orchestration). WINK
> keeps the definition portable while runtimes evolve."

______________________________________________________________________

## Accessing Documentation

After installing WINK, use `wink docs` to access bundled documentation:

```bash
wink docs --reference   # This file (API reference)
wink docs --guide       # Usage guide with tutorials
wink docs --specs       # All specification documents
wink docs --changelog   # Release history

# Combine flags for multiple sections
wink docs --reference --specs

# Pipe to clipboard or other tools
wink docs --specs | pbcopy
wink docs --guide | llm "Summarize key concepts"
```

**Available documentation:**

- `--reference` - Dense API reference (this file)
- `--guide` - Step-by-step usage guide with examples
- `--specs` - Design specifications (adapters, sessions, tools, etc.)
- `--changelog` - Version history and breaking changes

______________________________________________________________________

## Module Map

```text
weakincentives                    # Top-level exports
weakincentives.prompt             # Prompt authoring, sections, tools
weakincentives.prompt.overrides   # Hash-based prompt iteration
weakincentives.runtime            # Session, events, lifecycle, mailbox
weakincentives.runtime.session    # Slice ops, reducers, snapshots
weakincentives.runtime.events     # Dispatcher, event types
weakincentives.runtime.mailbox    # Message queues
weakincentives.adapters           # Provider base, config, throttling
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

______________________________________________________________________

## Import Cheatsheet

### Essential Imports

```python
# Top-level exports
from weakincentives import (
    Prompt,
    MarkdownSection,
    Tool,
    ToolContext,
    ToolResult,
    Budget,
    BudgetTracker,
    Deadline,
)

# Prompt system
from weakincentives.prompt import (
    PromptTemplate,
    Section,
    SectionVisibility,
    ToolExample,
    RenderedPrompt,
)

# Runtime
from weakincentives.runtime import (
    Session,
    InProcessDispatcher,
    AgentLoop,
    AgentLoopConfig,
    AgentLoopRequest,
    PromptExecuted,
    ToolInvoked,
    Snapshot,
    append_all,
    replace_latest,
    upsert_by,
)

# Adapters
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from weakincentives.adapters import PromptResponse

# Contrib tools
from weakincentives.contrib.tools import (
    WorkspaceDigestSection,
    WorkspaceDigest,
    InMemoryFilesystem,
    set_workspace_digest,
    clear_workspace_digest,
    latest_workspace_digest,
)

# Serde
from weakincentives.serde import dump, parse, schema, clone

# Resources
from weakincentives.resources import Binding, Scope, ResourceRegistry

# Tool policies
from weakincentives.prompt import (
    ReadBeforeWritePolicy,
    SequentialDependencyPolicy,
    PolicyDecision,
    PolicyState,
    ToolPolicy,
)

# Feedback providers
from weakincentives.prompt import (
    DeadlineFeedback,
    Feedback,
    FeedbackProvider,
    FeedbackProviderConfig,
    FeedbackTrigger,
)
```

### Claude Agent SDK

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    HostMount,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
    # Task completion
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
    PlanBasedChecker,
    CompositeChecker,
)
```

______________________________________________________________________

## Minimal Working Example

```python
from dataclasses import dataclass

from weakincentives import Prompt, MarkdownSection
from weakincentives.prompt import PromptTemplate
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
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
adapter = ClaudeAgentSDKAdapter()
prompt = Prompt(template).bind(TaskParams(objective="Review the auth module"))
response = adapter.evaluate(prompt, session=session)
result: TaskResult = response.output
```

______________________________________________________________________

## Core Patterns

### 1. Prompt Construction

**PromptTemplate** is the immutable blueprint. **Prompt** wraps it with
bindings.

```python
from dataclasses import dataclass

from weakincentives import Prompt
from weakincentives.prompt import PromptTemplate, MarkdownSection


@dataclass(slots=True, frozen=True)
class OutputType:
    answer: str


# Template: defines structure, typed output
template = PromptTemplate[OutputType](
    ns="namespace",  # Required: grouping
    key="prompt-key",  # Required: unique identifier
    name="prompt_name",  # Optional: display name
    sections=(MarkdownSection(title="Task", key="task", template="Do something"),),
)

# Prompt: wraps template, binds parameters
prompt = Prompt(template)

# Render to inspect
rendered = prompt.render()
print(rendered.text)  # Markdown content
print(rendered.tools)  # Tool tuple
```

### 2. Sections

All sections inherit from `Section`. Most common: `MarkdownSection`.

```python
from dataclasses import dataclass

from weakincentives.prompt import MarkdownSection, SectionVisibility


@dataclass(slots=True, frozen=True)
class ReviewParams:
    focus: str


section = MarkdownSection[ReviewParams](
    title="Review Guidelines",  # Rendered as heading
    key="review-guidelines",  # Unique within prompt
    template="Focus on: ${focus}",  # Template.substitute syntax
    default_params=ReviewParams(focus="correctness"),
    tools=(),  # Tools attached to section
    children=(),  # Nested sections
    visibility=SectionVisibility.FULL,  # Or SUMMARY for progressive disclosure
    summary="Guidelines available.",  # Shown when visibility=SUMMARY
    accepts_overrides=True,  # Allow prompt overrides
)
```

**Placeholder syntax**: `${field_name}` from the params dataclass.

**Children**: Nest sections for hierarchy. Heading levels auto-increment.

### 3. Tools

Tools are typed with params and result dataclasses.

```python nocheck
from dataclasses import dataclass, field

from weakincentives import Tool, ToolContext, ToolResult


@dataclass(slots=True, frozen=True)
class SearchParams:
    query: str = field(metadata={"description": "Search query"})
    limit: int = field(default=10, metadata={"description": "Max results"})


@dataclass(slots=True, frozen=True)
class SearchResult:
    snippets: tuple[str, ...]

    def render(self) -> str:
        return "\n".join(f"- {s}" for s in self.snippets)


def search_handler(
    params: SearchParams,
    *,
    context: ToolContext,
) -> ToolResult[SearchResult]:
    # Access session state
    # plan = context.session[Plan].latest()

    # Access resources
    fs = context.filesystem  # Shorthand for context.resources.get_optional(Filesystem)

    # Check deadline
    if context.deadline and context.deadline.remaining().total_seconds() < 5:
        return ToolResult.error("Deadline too close")

    # Do work...
    snippets = ("result1", "result2")

    # Return typed result
    return ToolResult.ok(SearchResult(snippets=snippets), message="Found 2 results")


# Create tool
search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search for content",
    handler=search_handler,
)
```

**ToolResult constructors**:

- `ToolResult.ok(value, message="...")` - Success with typed value
- `ToolResult.error("message")` - Failure, value=None

### 4. Tool Examples

Provide representative invocations for documentation and few-shot learning.

```python
from dataclasses import dataclass, field

from weakincentives import Tool, ToolContext, ToolResult
from weakincentives.prompt import ToolExample


@dataclass(slots=True, frozen=True)
class LookupParams:
    entity_id: str = field(metadata={"description": "ID to fetch"})


@dataclass(slots=True, frozen=True)
class LookupResult:
    entity_id: str
    url: str


def lookup_handler(
    params: LookupParams, *, context: ToolContext
) -> ToolResult[LookupResult]:
    result = LookupResult(entity_id=params.entity_id, url="https://example.com/...")
    return ToolResult.ok(result, message=f"Fetched {result.entity_id}")


lookup_tool = Tool[LookupParams, LookupResult](
    name="lookup_entity",
    description="Fetch information for an entity ID.",
    handler=lookup_handler,
    examples=(
        ToolExample(
            description="Basic lookup",
            input=LookupParams(entity_id="abc-123"),
            output=LookupResult(entity_id="abc-123", url="https://example.com/abc-123"),
        ),
    ),
)
```

### 5. Sessions

Redux-style immutable state container with typed slices.

```python
from dataclasses import dataclass
from weakincentives.runtime import Session, InProcessDispatcher, Snapshot
from weakincentives.runtime import replace_latest, append_all

@dataclass(frozen=True, slots=True)
class TaskState:
    """Example state type for demonstration."""
    name: str
    status: str = "pending"

# Create session
session = Session()  # Creates InProcessDispatcher internally

# Query state
task = session[TaskState].latest()  # Most recent or None
all_tasks = session[TaskState].all()  # All values as tuple
active = session[TaskState].where(lambda t: t.status == "active")
exists = session[TaskState].exists()  # Boolean

# Mutations (all dispatch events internally)
# session[TaskState].seed(initial_task)  # Initialize/replace slice
# session[TaskState].append(new_task)    # Append via default reducer
session[TaskState].clear()  # Clear all

# Snapshots
snapshot = session.snapshot()
session.restore(snapshot)
```

**Built-in reducers** (from `weakincentives.runtime`):

- `append_all` - Always append (default)
- `replace_latest` - Keep only most recent
- `upsert_by(key_fn)` - Replace by key
- `replace_latest_by(key_fn)` - Latest per key

### 6. Adapters

Provider-agnostic evaluation interface.

```python nocheck
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from weakincentives.adapters import PromptResponse
from weakincentives.errors import DeadlineExceededError

# Claude Agent SDK (recommended)
adapter = ClaudeAgentSDKAdapter()

# Evaluate
response = adapter.evaluate(
    prompt,
    session=session,
    deadline=deadline,  # Optional
    budget=budget,  # Optional
    budget_tracker=tracker,  # Optional
)

# Response fields
response.output  # Parsed dataclass (OutputType)
response.text  # Raw text
response.prompt_name  # Prompt identifier
```

### 7. Claude Agent SDK Adapter

Native Claude Code capabilities with hermetic isolation.

```python
import os

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    HostMount,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)
from weakincentives.runtime import Session

session = Session()

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
    model="claude-opus-4-6",
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
# response = adapter.evaluate(prompt, session=session)
workspace.cleanup()
```

**Isolation modes**:

- `NetworkPolicy.no_network()` - API access only
- `NetworkPolicy(allowed_domains=("docs.python.org",))` - Specific domains
- `SandboxConfig(enabled=True)` - OS-level sandboxing

### 8. Tool Policies

Enforce sequential dependencies between tool invocations.

```python
from weakincentives.prompt import (
    ReadBeforeWritePolicy,
    SequentialDependencyPolicy,
    PolicyDecision,
)

# Read-before-write: must read existing files before overwriting
rbw_policy = ReadBeforeWritePolicy(
    read_tools=frozenset({"read_file"}),
    write_tools=frozenset({"write_file", "edit_file"}),
)

# Sequential dependency: enforce tool ordering
seq_policy = SequentialDependencyPolicy(
    dependencies={
        "deploy": frozenset({"test", "build"}),  # deploy requires test AND build
        "build": frozenset({"lint"}),  # build requires lint
    }
)
```

**Policy behavior**:

- `ReadBeforeWritePolicy`: New files can be created freely; existing files must
  be read first. Tracks read paths in session `PolicyState` slice.
- `SequentialDependencyPolicy`: Tool B requires tool A to have succeeded.
  Tracks invoked tools in session `PolicyState` slice.

### 9. Feedback Providers

Deliver ongoing progress feedback during unattended execution.

```python
from weakincentives.prompt import (
    DeadlineFeedback,
    FeedbackProviderConfig,
    FeedbackTrigger,
)

# Built-in: deadline feedback (warns about remaining time)
deadline_config = FeedbackProviderConfig(
    provider=DeadlineFeedback(warning_threshold_seconds=120),
    trigger=FeedbackTrigger(every_n_seconds=30),  # Check every 30 seconds
)
```

**Trigger conditions** (OR'd together):

- `every_n_calls` - Run after N tool calls since last feedback
- `every_n_seconds` - Run after N seconds elapsed

### 10. Task Completion Checkers

Verify agents complete all tasks before stopping. Critical for unattended
agents.

```python
from dataclasses import dataclass
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    PlanBasedChecker,
    CompositeChecker,
)

# Define a plan type that the checker can inspect
@dataclass(frozen=True, slots=True)
class TaskPlan:
    objective: str
    steps: tuple  # Should have items with .status attribute

# Plan-based: ensure all plan steps are "done"
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        task_completion_checker=PlanBasedChecker(plan_type=TaskPlan),
    ),
)
```

**Hook integration** (Claude Agent SDK):

1. **PostToolUse Hook**: After `StructuredOutput`, checker verifies completion.
   If incomplete, adds feedback to encourage continuation.
1. **Stop Hook**: Before allowing stop, checker verifies. If incomplete, signals
   `needsMoreTurns: True`.

### 11. Contrib Tools

**Workspace Digest**: Caching layer for workspace summaries.

```python
from weakincentives.contrib.tools import (
    WorkspaceDigestSection,
    WorkspaceDigest,
    set_workspace_digest,
    latest_workspace_digest,
)
from weakincentives.runtime import Session

session = Session()

# Create digest section in prompt
digest_section = WorkspaceDigestSection(session=session)

# Populate digest (typically done by exploration agent)
set_workspace_digest(
    session,
    section_key="workspace-digest",
    body="Full project analysis with dependencies, structure...",
    summary="Python web app with FastAPI backend.",
)

# Query digest
digest = latest_workspace_digest(session, "workspace-digest")
if digest:
    print(digest.summary)
```

**In-Memory Filesystem**: Session-scoped filesystem for testing.

```python
from weakincentives.contrib.tools import InMemoryFilesystem
from weakincentives.filesystem import ReadResult

fs = InMemoryFilesystem()
fs.write("test.txt", "Hello, world!")
read_result: ReadResult = fs.read("test.txt")
print(read_result.content)  # "Hello, world!"
```

**Note:** Tool sections for filesystem operations, planning, and shell execution
are provided by the execution harness (e.g., Claude Agent SDK) rather than
defined in WINK. This keeps agent definitions portable across runtimes.

### 12. Resources

Dependency injection with scoped lifecycles.

```python
from weakincentives.resources import Binding, Scope, ResourceRegistry


class Config:
    pass


class HTTPClient:
    pass


# Build registry
registry = ResourceRegistry.of(
    Binding(Config, lambda r: Config()),
    Binding(HTTPClient, lambda r: HTTPClient()),
)
```

**Scopes**:

- `Scope.SINGLETON` - Once per context (default)
- `Scope.TOOL_CALL` - Fresh per tool invocation
- `Scope.PROTOTYPE` - Fresh every resolution

### 13. Serialization

```python
from dataclasses import dataclass

from weakincentives.serde import dump, parse, schema, clone


@dataclass(frozen=True)
class MyData:
    value: int


data = MyData(value=42)

# Serialize dataclass to dict
d = dump(data)

# Parse dict to dataclass
obj = parse(MyData, d)

# JSON schema
json_schema = schema(MyData)

# Deep clone
copy = clone(data)
```

### 14. Design-by-Contract

```python
from weakincentives.dbc import require, ensure, invariant, pure


@require(lambda x: x > 0)
@ensure(lambda result: result >= 0)
def compute(x: int) -> int:
    return x * 2


@invariant(lambda self: self.count >= 0)
class Counter:
    count: int = 0


@pure  # Validates no side effects
def hash_value(x: str) -> int:
    return hash(x)
```

______________________________________________________________________

## Best Practices

### Agent Design

1. **Define clear policies**: Use policy sections to specify constraints
1. **Verify completion**: Enable `TaskCompletionChecker` for unattended agents
1. **Set budgets**: Always configure `Budget` with token limits
1. **Use deadlines**: Set wall-clock limits via `Deadline`
1. **Provide feedback**: Configure `FeedbackProvider` for long-running tasks

### Tool Implementation

1. **Type everything**: Use `@dataclass(slots=True, frozen=True)` for
   params/results
1. **Document params**: Add `metadata={"description": "..."}` to all fields
1. **Handle failures gracefully**: Return `ToolResult.error()`, don't raise
1. **Check deadlines**: Early-exit if `context.deadline.remaining()` is low
1. **Access resources properly**: Use `context.resources.get(Protocol)`

### Session Management

1. **Snapshot before risky operations**: `session.snapshot()` enables rollback
1. **Use typed slices**: Query via `session[Type].latest()`, not raw access
1. **Dispatch events**: Never mutate state directly; use `session.dispatch()`
1. **Register reducers early**: Call `session[Type].register()` before
   dispatching

### Prompt Authoring

1. **Keep sections focused**: One concern per section
1. **Use progressive disclosure**: Set `visibility=SUMMARY` for verbose content
1. **Attach tools to relevant sections**: Tools should be near their
   instructions
1. **Apply policies at appropriate level**: Section-level for local constraints,
   prompt-level for global ones

______________________________________________________________________

## Decision Trees

### Which Adapter?

```text
Need agentic harness?          → ClaudeAgentSDKAdapter (recommended)
```

### Which Workspace Tool?

```text
Claude Agent SDK mode?         → ClaudeAgentWorkspaceSection
Testing/evaluation?            → InMemoryFilesystem
```

**Note:** Filesystem and shell execution tools are provided by the execution
harness (e.g., Claude Agent SDK) rather than defined in WINK prompts.

### Which Reducer?

```text
Recording every event?         → append_all (default)
Only latest value matters?     → replace_latest
Keyed upsert (like cache)?     → upsert_by(key_fn)
Complex state transitions?     → @reducer decorator on dataclass
```

### Session vs Resource State?

```text
Agent state (plans, results)?  → Session slices
Runtime deps (HTTP, DB)?       → ResourceRegistry
Filesystem state?              → Filesystem via resources
```

______________________________________________________________________

## Common Pitfalls

1. **Forgetting `slots=True, frozen=True`** on dataclasses - breaks serde
1. **Missing `${}` in templates** - use `${field}` not `{field}`
1. **Tool handler signature** - must be `(params, *, context: ToolContext)`
1. **ToolResult return** - use `.ok()` or `.error()`, not raw constructor
1. **Session mutations** - all go through `dispatch()`, use accessor methods
1. **Resource access outside context** - use `with prompt.resources:` block
1. **Duplicate tool names** - raises `PromptValidationError`
1. **Hash mismatch in overrides** - stale overrides silently filtered

______________________________________________________________________

## Event Types

```python
from weakincentives.runtime import (
    PromptRendered,  # After render, before provider call
    PromptExecuted,  # After all tools and parsing
    ToolInvoked,  # After each tool handler
    TokenUsage,  # Token consumption data
)
```

______________________________________________________________________

## Error Hierarchy

```text
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

______________________________________________________________________

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

______________________________________________________________________

## File Layout

```text
src/weakincentives/
├── adapters/           # Claude Agent SDK
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

______________________________________________________________________

## Key Specs

Read before modifying related code:

| Spec | Topic |
| --------------------------- | ---------------------------------------- |
| `specs/PROMPTS.md` | Prompt system, composition, overrides |
| `specs/SESSIONS.md` | Session lifecycle, events, budgets |
| `specs/TOOLS.md` | Tool registration, planning tools |
| `specs/GUARDRAILS.md` | Tool policies, feedback providers, task completion |
| `specs/ADAPTERS.md` | Provider adapters, throttling |
| `specs/CLAUDE_AGENT_SDK.md` | SDK adapter, isolation, MCP |
| `specs/WORKSPACE.md` | VFS, Podman, asteval |
| `specs/DBC.md` | Design-by-contract patterns |
| `specs/RESOURCE_REGISTRY.md` | Dependency injection |
| `specs/AGENT_LOOP.md` | AgentLoop orchestration |
| `specs/MAILBOX.md` | Message queue abstraction |

______________________________________________________________________

## Quick Reference

### PromptTemplate

```text
PromptTemplate[OutputT](
    ns: str,                    # Namespace (required)
    key: str,                   # Unique key (required)
    name: str | None,           # Display name
    sections: tuple[Section],   # Ordered sections
)
```

### MarkdownSection

```text
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

```text
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

```text
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

```text
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

```text
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

______________________________________________________________________

## Example: Complete Agent

See `code_reviewer_example.py` for production patterns:

- Structured output types
- VFS/Planning tool sections
- AgentLoop implementation
- Event subscription
- Prompt overrides
- Claude Agent SDK mode

______________________________________________________________________

## Alpha Status

All APIs may change without backward compatibility. No deprecation warnings;
unused code is deleted completely.

______________________________________________________________________

## License

Apache License 2.0
