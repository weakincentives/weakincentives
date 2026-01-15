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
├── WorkspaceDigestSection     ← auto-generated codebase summary
├── MarkdownSection (reference docs, progressive disclosure)
├── PlanningToolsSection       ← contributes planning_* tools
│   └── (nested planning docs)
├── VfsToolsSection            ← contributes ls/read_file/write_file/...
│   └── (nested filesystem docs)
└── MarkdownSection (user request)
```

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

A workflow encodes *how* to accomplish a goal—a predetermined sequence that
fractures when encountering unexpected situations. A policy encodes *what* the
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

______________________________________________________________________

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

# Tool policies
from weakincentives.prompt import (
    ReadBeforeWritePolicy,
    SequentialDependencyPolicy,
    PolicyDecision,
)

# Feedback providers
from weakincentives.prompt import (
    DeadlineFeedback,
    Feedback,
    FeedbackProvider,
    FeedbackProviderConfig,
    FeedbackTrigger,
)

# Task completion (Claude Agent SDK)
from weakincentives.adapters.claude_agent_sdk import (
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
    PlanBasedChecker,
    CompositeChecker,
)
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

______________________________________________________________________

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

______________________________________________________________________

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

### 13. Tool Policies

Enforce sequential dependencies between tool invocations. Policies gate tool
calls and track state in session slices.

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
        "build": frozenset({"lint"}),            # build requires lint
    }
)

# Attach policies to sections
section = MarkdownSection(
    title="Filesystem",
    key="filesystem",
    template="Read and write files.",
    tools=(read_file, write_file, edit_file),
    policies=(rbw_policy,),  # Section-level policy
)

# Or attach to entire prompt
template = PromptTemplate(
    ns="my-agent",
    key="main",
    sections=[...],
    policies=(seq_policy,),  # Prompt-level policy (all tools)
)
```

**Policy behavior**:

- `ReadBeforeWritePolicy`: New files can be created freely; existing files must
  be read first. Tracks read paths in session `PolicyState` slice.
- `SequentialDependencyPolicy`: Tool B requires tool A to have succeeded.
  Tracks invoked tools in session `PolicyState` slice.

**Custom policies** implement the `ToolPolicy` protocol:

```python
from weakincentives.prompt import PolicyDecision

class ApprovalRequiredPolicy:
    """Require approval tool before destructive operations."""

    @property
    def name(self) -> str:
        return "approval_required"

    def check(self, tool, params, *, context) -> PolicyDecision:
        if tool.name not in {"delete_file", "deploy"}:
            return PolicyDecision.allow()

        state = context.session[PolicyState].latest()
        if state and "approve" in state.invoked_tools:
            return PolicyDecision.allow()
        return PolicyDecision.deny("Call 'approve' tool first")

    def on_result(self, tool, params, result, *, context) -> None:
        if result.success and tool.name == "approve":
            # Record approval in session state
            state = context.session[PolicyState].latest() or PolicyState(
                policy_name=self.name
            )
            new_state = PolicyState(
                policy_name=self.name,
                invoked_tools=state.invoked_tools | {"approve"},
                invoked_keys=state.invoked_keys,
            )
            context.session[PolicyState].seed(new_state)
```

### 14. Feedback Providers

Deliver ongoing progress feedback during unattended execution. Providers
analyze patterns and inject guidance into the agent's context.

```python
from weakincentives.prompt import (
    DeadlineFeedback,
    Feedback,
    FeedbackProvider,
    FeedbackProviderConfig,
    FeedbackTrigger,
)

# Built-in: deadline feedback (warns about remaining time)
deadline_config = FeedbackProviderConfig(
    provider=DeadlineFeedback(warning_threshold_seconds=120),
    trigger=FeedbackTrigger(every_n_seconds=30),  # Check every 30 seconds
)

# Attach to prompt template
template = PromptTemplate(
    ns="my-agent",
    key="main",
    sections=[...],
    feedback_providers=(deadline_config,),
)
```

**Trigger conditions** (OR'd together):

- `every_n_calls` - Run after N tool calls since last feedback
- `every_n_seconds` - Run after N seconds elapsed

**Custom feedback provider**:

```python
@dataclass(frozen=True)
class ToolUsageMonitor:
    """Warn when too many tool calls without apparent progress."""

    max_calls_without_progress: int = 20

    @property
    def name(self) -> str:
        return "ToolUsageMonitor"

    def should_run(self, *, context) -> bool:
        return True  # Always run when triggered

    def provide(self, *, context) -> Feedback:
        count = context.tool_call_count
        if count > self.max_calls_without_progress:
            return Feedback(
                provider_name=self.name,
                summary=f"You have made {count} tool calls.",
                suggestions=(
                    "Review what you've accomplished so far.",
                    "Check if you're making progress toward the goal.",
                ),
                severity="caution",
            )
        return Feedback(
            provider_name=self.name,
            summary=f"Progress check: {count} tool calls made.",
            severity="info",
        )

# Register with trigger
config = FeedbackProviderConfig(
    provider=ToolUsageMonitor(max_calls_without_progress=15),
    trigger=FeedbackTrigger(every_n_calls=10),
)
```

**Feedback delivery**: Injected immediately after tool execution via adapter
hooks. First matching provider wins.

### 15. Task Completion Checkers

Verify agents complete all tasks before stopping. Critical for ensuring agents
don't prematurely terminate while work remains incomplete.

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    PlanBasedChecker,
    CompositeChecker,
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)
from weakincentives.contrib.tools.planning import Plan

# Plan-based: ensure all plan steps are "done"
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        task_completion_checker=PlanBasedChecker(plan_type=Plan),
    ),
)

# Composite: combine multiple checkers (all must pass)
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        task_completion_checker=CompositeChecker(
            checkers=(PlanBasedChecker(plan_type=Plan), TestPassingChecker()),
            all_must_pass=True,  # Both must pass
        ),
    ),
)
```

**Custom completion checker**:

```python
class TestPassingChecker:
    """Ensure all tests pass before allowing completion."""

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        test_results = context.session[TestResult].latest()

        if test_results is None:
            return TaskCompletionResult.incomplete(
                "No test results found. Please run the test suite."
            )

        if test_results.failed > 0:
            return TaskCompletionResult.incomplete(
                f"{test_results.failed} tests failing. Fix before completing."
            )

        return TaskCompletionResult.ok(f"All {test_results.passed} tests passing.")


class FileExistsChecker:
    """Ensure required output files exist."""

    def __init__(self, required_files: tuple[str, ...]) -> None:
        self._required = required_files

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        if context.filesystem is None:
            return TaskCompletionResult.ok("No filesystem to check.")

        missing = [f for f in self._required if not context.filesystem.exists(f)]

        if missing:
            return TaskCompletionResult.incomplete(
                f"Missing required files: {', '.join(missing)}"
            )

        return TaskCompletionResult.ok("All required files exist.")
```

**Hook integration** (Claude Agent SDK):

1. **PostToolUse Hook**: After `StructuredOutput`, checker verifies completion.
   If incomplete, adds feedback to encourage continuation.
1. **Stop Hook**: Before allowing stop, checker verifies. If incomplete, signals
   `needsMoreTurns: True`.
1. **Final verification**: After SDK completes, raises `PromptEvaluationError`
   if tasks remain incomplete.

### 16. Tool Examples

Provide representative invocations for documentation and few-shot learning.

```python
from weakincentives import Tool
from weakincentives.prompt import ToolExample

@dataclass(slots=True, frozen=True)
class LookupParams:
    entity_id: str = field(metadata={"description": "ID to fetch"})

@dataclass(slots=True, frozen=True)
class LookupResult:
    entity_id: str
    url: str

def lookup_handler(params: LookupParams, *, context: ToolContext) -> ToolResult[LookupResult]:
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
        ToolExample(
            description="Lookup with special characters",
            input=LookupParams(entity_id="user@domain"),
            output=LookupResult(entity_id="user@domain", url="https://example.com/user%40domain"),
        ),
    ),
)
```

**Example constraints**:

- `description` must be ≤ 200 characters
- `input` must be an instance of the params dataclass
- `output` must be an instance of the result dataclass
- Examples are validated during prompt rendering

______________________________________________________________________

## Best Practices

### Agent Design

1. **Plan first**: Use `PlanningToolsSection` to structure work before acting
1. **Verify completion**: Enable `TaskCompletionChecker` for unattended agents
1. **Set budgets**: Always configure `Budget` with token limits
1. **Use deadlines**: Set wall-clock limits via `Deadline`
1. **Provide feedback**: Configure `FeedbackProvider` for long-running tasks

### Tool Implementation

1. **Type everything**: Use `@dataclass(slots=True, frozen=True)` for params/results
1. **Document params**: Add `metadata={"description": "..."}` to all fields
1. **Handle failures gracefully**: Return `ToolResult.error()`, don't raise
1. **Check deadlines**: Early-exit if `context.deadline.remaining()` is low
1. **Access resources properly**: Use `context.resources.get(Protocol)`

### Session Management

1. **Snapshot before risky operations**: `session.snapshot()` enables rollback
1. **Use typed slices**: Query via `session[Type].latest()`, not raw access
1. **Dispatch events**: Never mutate state directly; use `session.dispatch()`
1. **Register reducers early**: Call `session[Type].register()` before dispatching

### Prompt Authoring

1. **Keep sections focused**: One concern per section
1. **Use progressive disclosure**: Set `visibility=SUMMARY` for verbose content
1. **Attach tools to relevant sections**: Tools should be near their instructions
1. **Apply policies at appropriate level**: Section-level for local constraints,
   prompt-level for global ones

______________________________________________________________________

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
    PromptRendered,    # After render, before provider call
    PromptExecuted,    # After all tools and parsing
    ToolInvoked,       # After each tool handler
    TokenUsage,        # Token consumption data
)

# Subscribe
session.dispatcher.subscribe(ToolInvoked, lambda e: print(e.name))
```

______________________________________________________________________

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

______________________________________________________________________

## Key Specs

Read before modifying related code:

| Spec | Topic |
|------|-------|
| `specs/PROMPTS.md` | Prompt system, composition, overrides |
| `specs/SESSIONS.md` | Session lifecycle, events, budgets |
| `specs/TOOLS.md` | Tool registration, planning tools |
| `specs/TOOL_POLICIES.md` | Sequential dependencies, read-before-write |
| `specs/FEEDBACK_PROVIDERS.md` | Trajectory feedback, stall detection |
| `specs/TASK_COMPLETION.md` | Task completion verification |
| `specs/ADAPTERS.md` | Provider adapters, throttling |
| `specs/CLAUDE_AGENT_SDK.md` | SDK adapter, isolation, MCP |
| `specs/WORKSPACE.md` | VFS, Podman, asteval |
| `specs/DBC.md` | Design-by-contract patterns |
| `specs/RESOURCE_REGISTRY.md` | Dependency injection |
| `specs/MAIN_LOOP.md` | MainLoop orchestration |
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
    resources: ResourceRegistry,
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
- MainLoop implementation
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
