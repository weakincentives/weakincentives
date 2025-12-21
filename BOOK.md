# The WINK Book

A comprehensive guide to building deterministic, observable background agents
with Weak Incentives.

---

## Table of Contents

1. [Philosophy](#1-philosophy)
2. [Core Concepts](#2-core-concepts)
3. [Getting Started](#3-getting-started)
4. [The Prompt System](#4-the-prompt-system)
5. [Session and State Management](#5-session-and-state-management)
6. [Tools and Workspaces](#6-tools-and-workspaces)
7. [Provider Adapters](#7-provider-adapters)
8. [Design-by-Contract](#8-design-by-contract)
9. [Advanced Patterns](#9-advanced-patterns)
10. [Testing and Debugging](#10-testing-and-debugging)
11. [API Reference](#11-api-reference)

---

## 1. Philosophy

### The Shift in AI Engineering

The reasoning loop is moving model-side. Complex nested workflows with heavy
orchestration may work today, but they won't age well—models are absorbing that
scaffolding. What remains constant:

- **Tools** — giving models capabilities to act
- **Retrieval** — providing relevant context
- **Context Engineering** — the art of structuring information for optimal reasoning

Context engineering is a genuinely new discipline. What's relevant now? What to
summarize versus preserve? How do you structure information so models reason
over it well? There's no clean precedent from traditional software engineering.
Builders who master it early win.

Some orchestration stays—for auditability, cost control, and hard constraints.
WINK focuses there: typed prompts as the primary artifact, observable session
state, sandboxed tools. The framework handles scaffolding that must remain; you
focus on what you're feeding the model.

### The Prompt Is the Agent

Most agent frameworks treat prompts as an afterthought—templates glued to
separately registered tool lists. WINK inverts this: **the prompt *is* the
agent**.

You define an agent as a single hierarchical document where each section
bundles its own instructions and tools together. The prompt fully determines
what the agent can think and do.

```
PromptTemplate[ReviewResponse]
├── MarkdownSection (guidance)
├── WorkspaceDigestSection          ← auto-generated codebase summary
├── MarkdownSection (reference)     ← progressive disclosure
├── PlanningToolsSection            ← contributes planning_* tools
│   └── (nested planning docs)
├── VfsToolsSection                 ← contributes ls/read_file/write_file
│   └── (nested filesystem docs)
└── MarkdownSection (user request)
```

### Five Guiding Principles

1. **Prompt-First Design** — Centralize behavior in a single hierarchical
   prompt structure, not scattered across templates, tool registries, and
   routing logic.

2. **Type Safety Throughout** — From dataclass parameters to structured outputs
   to tool schemas, types are the source of truth. Mismatches surface early,
   before reaching LLMs.

3. **Determinism and Observability** — All state transitions flow through pure
   reducers. Tool calls, prompt evaluations, and custom state produce a
   replayable ledger—perfect for debugging, auditing, and testing.

4. **Minimal Dependencies** — No Pydantic, no heavyweight stacks. Custom serde
   modules provide validation without sprawling dependency trees.

5. **Design by Contract for Robustness** — Public APIs use `@require`,
   `@ensure`, `@invariant`, and `@pure` decorators as internal safety nets.

---

## 2. Core Concepts

### The Five-Part Agent Architecture

Every WINK agent wires together five core components:

```
PromptTemplate (Blueprint)
    ↓
Prompt (Bound Parameters)
    ↓
Session (Event Ledger + State)
    ↓
ProviderAdapter (OpenAI/LiteLLM/Claude Agent SDK)
    ↓
Tool Handlers (Side Effects)
```

### PromptTemplate and Prompt

A **PromptTemplate** is an immutable blueprint defining namespace, key,
sections, tools, and structured output schema. It's the source of truth for
what an agent can do.

A **Prompt** wraps a template and binds runtime parameters. Multiple bindings
per type are allowed (last one wins); rendering validates completeness.

```python
from weakincentives.prompt import PromptTemplate, Prompt, MarkdownSection

@dataclass(slots=True, frozen=True)
class ReviewParams:
    request: str

@dataclass(slots=True, frozen=True)
class ReviewResponse:
    summary: str
    issues: list[str]

# Blueprint: defines structure and capabilities
template = PromptTemplate[ReviewResponse](
    ns="my-app",
    key="code-review",
    name="code_review_agent",
    sections=(
        MarkdownSection[ReviewParams](
            title="Instructions",
            template="Review the following: ${request}",
            key="instructions",
        ),
    ),
)

# Instance: bind runtime parameters
prompt = Prompt(template).bind(ReviewParams(request="Check main.py"))
rendered = prompt.render()  # Validates and produces final text
```

### Sections

Sections are typed building blocks that:

- Render markdown content
- Contribute tools the LLM can invoke
- Nest child sections for hierarchy
- Enable/disable based on runtime state

```python
MarkdownSection[ParamsType](
    title="Section Title",           # Rendered as heading
    template="Content with ${var}",  # string.Template format
    key="section-key",               # Must match ^[a-z0-9][a-z0-9._-]{0,63}$
    default_params=ParamsType(...),  # Optional defaults
    tools=(my_tool,),                # Tools this section provides
    children=(child_section,),       # Nested sections
    enabled=lambda ctx: True,        # Dynamic enable/disable
    visibility=SectionVisibility.FULL,  # FULL or SUMMARY
    summary="Brief version",         # Shown when visibility=SUMMARY
)
```

### Session

Sessions are immutable event ledgers. All state changes flow through pure
reducers:

```python
from weakincentives.runtime import Session

session = Session(bus=event_bus)

# Query state via indexing
session[Plan].latest()                    # Most recent value
session[Plan].all()                       # All values in slice
session[Plan].where(lambda p: p.active)   # Filter by predicate

# Dispatch events (through reducers)
session.broadcast(AddStep(...))           # Broadcast to all reducers

# Direct mutations via indexing
session[Plan].seed(initial_plan)          # Initialize slice
session[Plan].register(AddStep, reducer)  # Register reducer
session[Plan].clear()                     # Clear slice
```

### Provider Adapters

Adapters bridge prompts to specific LLM providers:

```python
from weakincentives.adapters.openai import OpenAIAdapter

adapter = OpenAIAdapter(model="gpt-4o")
response = adapter.evaluate(prompt, session=session, deadline=deadline)

output: ReviewResponse = response.output  # Typed result
```

### Tools

Tools are declared, not discovered. Each tool lives within a section:

```python
from weakincentives.prompt import Tool, ToolContext, ToolResult

@dataclass(slots=True, frozen=True)
class SearchParams:
    query: str

@dataclass(slots=True, frozen=True)
class SearchResult:
    matches: list[str]

def search_handler(
    params: SearchParams,
    *,
    context: ToolContext,
) -> ToolResult[SearchResult]:
    # context.session, context.deadline, context.event_bus available
    matches = do_search(params.query)
    return ToolResult(
        message=f"Found {len(matches)} results",
        value=SearchResult(matches=matches),
        success=True,
    )

search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search the codebase",
    handler=search_handler,
)
```

---

## 3. Getting Started

### Installation

```bash
uv add weakincentives

# Optional extras
uv add "weakincentives[openai]"           # OpenAI adapter
uv add "weakincentives[litellm]"          # LiteLLM adapter
uv add "weakincentives[claude-agent-sdk]" # Claude Agent SDK adapter
uv add "weakincentives[asteval]"          # Safe Python evaluation
uv add "weakincentives[podman]"           # Podman sandbox
uv add "weakincentives[wink]"             # Debug UI
```

### Hello World: A Simple Agent

```python
from dataclasses import dataclass
from weakincentives.prompt import PromptTemplate, Prompt, MarkdownSection
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import Session
from weakincentives.runtime.events import InProcessEventBus

# 1. Define output type
@dataclass(slots=True, frozen=True)
class Greeting:
    message: str
    language: str

# 2. Create session and event bus
bus = InProcessEventBus()
session = Session(bus=bus)

# 3. Build prompt template
template = PromptTemplate[Greeting](
    ns="hello",
    key="greeter",
    name="greeting_agent",
    sections=(
        MarkdownSection(
            title="Instructions",
            template="Generate a friendly greeting in any language.",
            key="instructions",
        ),
    ),
)

# 4. Bind and evaluate
adapter = OpenAIAdapter(model="gpt-4o")
prompt = Prompt(template)
response = adapter.evaluate(prompt, session=session)

# 5. Use typed result
greeting: Greeting = response.output
print(f"{greeting.message} (in {greeting.language})")
```

### A Complete Example: Code Review Agent

Here's a realistic agent that reviews code with planning, file access, and
structured output:

```python
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from weakincentives.prompt import PromptTemplate, Prompt, MarkdownSection
from weakincentives.contrib.tools import (
    PlanningToolsSection,
    VfsToolsSection,
    WorkspaceDigestSection,
    HostMount,
    VfsPath,
    PlanningStrategy,
)
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import Session, MainLoop
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.deadlines import Deadline

# Output structure
@dataclass(slots=True, frozen=True)
class ReviewResponse:
    summary: str
    issues: list[str]
    next_steps: list[str]

# Input parameters
@dataclass(slots=True, frozen=True)
class ReviewRequest:
    request: str = field(metadata={"description": "User review request"})

# MainLoop subclass for multi-turn execution
class CodeReviewLoop(MainLoop[ReviewRequest, ReviewResponse]):
    def __init__(self, adapter, bus):
        super().__init__(adapter=adapter, bus=bus)
        self._session = Session(bus=bus)
        self._template = self._build_template()

    def create_prompt(self, request: ReviewRequest) -> Prompt[ReviewResponse]:
        return Prompt(self._template).bind(request)

    def create_session(self) -> Session:
        return self._session

    def _build_template(self) -> PromptTemplate[ReviewResponse]:
        return PromptTemplate[ReviewResponse](
            ns="examples/code-review",
            key="code-review-session",
            name="code_review_agent",
            sections=(
                MarkdownSection(
                    title="Code Review Brief",
                    template="""
                    You are a code review assistant. Use the available tools
                    to explore the codebase and identify issues.

                    Respond with JSON containing:
                    - summary: One paragraph describing your findings
                    - issues: List of concrete issues found
                    - next_steps: Actionable recommendations
                    """,
                    key="guidance",
                ),
                WorkspaceDigestSection(session=self._session),
                PlanningToolsSection(
                    session=self._session,
                    strategy=PlanningStrategy.PLAN_ACT_REFLECT,
                ),
                VfsToolsSection(
                    session=self._session,
                    mounts=(
                        HostMount(
                            host_path="./repo",
                            mount_path=VfsPath(("repo",)),
                            include_glob=("*.py", "*.md"),
                            max_bytes=500_000,
                        ),
                    ),
                    allowed_host_roots=(Path("."),),
                ),
                MarkdownSection[ReviewRequest](
                    title="Review Request",
                    template="${request}",
                    key="request",
                ),
            ),
        )

# Run the agent
bus = InProcessEventBus()
adapter = OpenAIAdapter(model="gpt-4o")
loop = CodeReviewLoop(adapter, bus)

deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
response, session = loop.execute(
    ReviewRequest(request="Review the authentication module"),
    deadline=deadline,
)

print(f"Summary: {response.output.summary}")
for issue in response.output.issues:
    print(f"  - {issue}")
```

---

## 4. The Prompt System

The prompt system is WINK's core innovation. Understanding it deeply unlocks
the library's full power.

### Why Prompts Are Different in WINK

Traditional approach:
```
Template file → Separate tool registry → Routing logic → Ad-hoc state
```

WINK approach:
```
PromptTemplate (unified structure with embedded tools and typed parameters)
    ↓
Self-describing agent (prompt fully determines capabilities)
```

Benefits:

1. **Co-location** — Instructions and tools live together. The section
   explaining filesystem navigation provides the `read_file` tool.
   Documentation can't drift from implementation.

2. **Progressive disclosure** — Nest child sections to reveal capabilities
   when relevant. The LLM sees hierarchical headings mirroring code structure.

3. **Dynamic scoping** — Each section has an `enabled` predicate. Disable a
   section and its entire subtree—tools included—disappears.

4. **Type safety** — Sections use dataclass parameters. Placeholders are
   validated at construction. Tools declare typed params and results.

### MarkdownSection in Depth

`MarkdownSection` is the primary section type:

```python
from weakincentives.prompt import MarkdownSection, SectionVisibility

@dataclass(slots=True, frozen=True)
class ProjectParams:
    name: str
    version: str = "1.0"

section = MarkdownSection[ProjectParams](
    # Required: unique identifier
    key="project-info",

    # Required: heading text
    title="Project Information",

    # Template uses string.Template syntax
    template="""
    Project: ${name}
    Version: ${version}

    Follow the project conventions when making changes.
    """,

    # Default parameter values
    default_params=ProjectParams(name="unnamed"),

    # Summary for progressive disclosure
    summary="Project info available on request.",

    # Visibility: FULL (default) or SUMMARY
    visibility=SectionVisibility.SUMMARY,

    # Tools this section provides
    tools=(info_tool, config_tool),

    # Nested sections
    children=(conventions_section, style_guide_section),

    # Dynamic enable/disable
    enabled=lambda ctx: ctx.get("show_project_info", True),

    # Whether overrides can modify this section
    accepts_overrides=True,
)
```

### Section Composition and Hierarchy

Sections nest to create structured prompts:

```python
template = PromptTemplate[Output](
    ns="myapp",
    key="main",
    sections=(
        MarkdownSection(
            title="Overview",
            template="High-level guidance...",
            key="overview",
            children=(
                MarkdownSection(
                    title="Code Style",
                    template="Style rules...",
                    key="style",
                ),
                MarkdownSection(
                    title="Testing",
                    template="Testing guidelines...",
                    key="testing",
                ),
            ),
        ),
        VfsToolsSection(session=session),  # Adds file tools
        PlanningToolsSection(session=session),  # Adds planning tools
    ),
)
```

Renders as:

```markdown
## 1. Overview
High-level guidance...

### 1.1. Code Style
Style rules...

### 1.2. Testing
Testing guidelines...

## 2. Workspace Tools
[VFS section content with tool descriptions]

## 3. Planning
[Planning section content with tool descriptions]
```

### Progressive Disclosure

Sections can start summarized and expand on demand:

```python
reference_section = MarkdownSection(
    title="Reference Documentation",
    template="""
    ## Architecture
    Detailed architecture documentation...

    ## API Reference
    Complete API documentation...
    """,
    summary="Documentation available. Request expansion if needed.",
    key="reference",
    visibility=SectionVisibility.SUMMARY,
)
```

When visibility is `SUMMARY`, the model sees only the summary. The `MainLoop`
handles expansion requests automatically:

1. Model calls `open_sections` tool
2. `MainLoop` catches `VisibilityExpansionRequired`
3. Visibility override applied to session
4. Prompt re-rendered with full content
5. Evaluation retried

### Structured Output

Prompts are generic over their output type:

```python
@dataclass(slots=True, frozen=True)
class Analysis:
    findings: list[str]
    risk_level: str
    recommendations: list[str]

template = PromptTemplate[Analysis](...)  # Output type specified
prompt = Prompt(template)
response = adapter.evaluate(prompt, session=session)

# response.output is typed as Analysis
analysis: Analysis = response.output
```

The adapter automatically:
1. Generates JSON schema from the dataclass
2. Instructs the model to return JSON
3. Parses and validates the response
4. Returns typed result in `response.output`

### Namespace and Key Conventions

Prompts are identified by `(namespace, key)` pairs:

```python
template = PromptTemplate[Output](
    ns="examples/code-review",      # Namespace (slash-separated path)
    key="code-review-session",      # Key (kebab-case identifier)
    name="code_review_agent",       # Display name
    ...
)
```

These identifiers enable:
- Version control of prompt overrides
- Telemetry and logging correlation
- Cache keying and memoization

---

## 5. Session and State Management

Sessions provide Redux-style state management with immutable event ledgers.

### Creating Sessions

```python
from weakincentives.runtime import Session
from weakincentives.runtime.events import InProcessEventBus

# Create event bus
bus = InProcessEventBus()

# Create session with event bus
session = Session(bus=bus)

# Sessions can have parents for hierarchical state
child_session = Session(bus=bus, parent=session)
```

### Typed State Slices

Sessions store typed state in "slices" indexed by type:

```python
from weakincentives.contrib.tools import Plan

# Query the latest value
plan = session[Plan].latest()

# Query all values (ledger-style)
all_plans = session[Plan].all()

# Filter by predicate
active_plans = session[Plan].where(lambda p: p.status == "active")

# Initialize a slice
session[Plan].seed(initial_plan)

# Clear a slice
session[Plan].clear()
```

### Reducers

State changes flow through pure reducers:

```python
from dataclasses import dataclass
from weakincentives.runtime import ReducerContext

@dataclass(slots=True, frozen=True)
class ItemAdded:
    item: str

@dataclass(slots=True, frozen=True)
class ItemList:
    items: tuple[str, ...]

def item_reducer(
    current: tuple[ItemList, ...],
    event: ItemAdded,
    *,
    context: ReducerContext,
) -> tuple[ItemList, ...]:
    """Pure reducer: returns new state, never mutates."""
    existing = current[-1].items if current else ()
    new_items = existing + (event.item,)
    return (ItemList(items=new_items),)

# Register reducer
session[ItemList].register(ItemAdded, item_reducer)

# Dispatch event (triggers reducer)
session.broadcast(ItemAdded(item="new item"))

# Query updated state
items = session[ItemList].latest()
```

### Built-in Reducers

WINK provides common reducer patterns:

```python
from weakincentives.runtime import append_all, replace_latest, upsert_by

# append_all: Ledger-style, keeps all values
session[Event].register(NewEvent, append_all)

# replace_latest: Keep only the most recent value
session[Config].register(UpdateConfig, replace_latest)

# upsert_by: Update or insert based on key
session[Item].register(UpsertItem, upsert_by(lambda x: x.id))
```

### Event Bus

Sessions connect to event buses for telemetry:

```python
from weakincentives.runtime import PromptRendered, PromptExecuted, ToolInvoked

# Subscribe to events
bus.subscribe(PromptRendered, lambda e: print(f"Rendered: {e.prompt_name}"))
bus.subscribe(ToolInvoked, lambda e: print(f"Tool: {e.name}"))
bus.subscribe(PromptExecuted, lambda e: print(f"Tokens: {e.usage.total_tokens}"))
```

### Snapshots

Sessions support full state capture and restore:

```python
# Capture current state
snapshot = session.snapshot()

# Restore from snapshot
session.restore(snapshot)

# Serialize for storage
json_data = snapshot.to_json()
restored = Snapshot.from_json(json_data)
```

This enables:
- Transactional tool execution (rollback on failure)
- Debugging (replay from specific points)
- Testing (deterministic state setup)

### Observers

Subscribe to slice changes:

```python
def on_plan_change(old: Plan | None, new: Plan) -> None:
    print(f"Plan updated: {new.steps}")

session.observe(Plan, on_plan_change)
```

---

## 6. Tools and Workspaces

WINK provides sandboxed tools for file access, code execution, and planning.

### Tool Definition

Tools are typed and declarative:

```python
from weakincentives.prompt import Tool, ToolContext, ToolResult

@dataclass(slots=True, frozen=True)
class CountParams:
    text: str

@dataclass(slots=True, frozen=True)
class CountResult:
    words: int
    characters: int

def count_handler(
    params: CountParams,
    *,
    context: ToolContext,
) -> ToolResult[CountResult]:
    words = len(params.text.split())
    chars = len(params.text)
    return ToolResult(
        message=f"Counted {words} words and {chars} characters",
        value=CountResult(words=words, characters=chars),
        success=True,
    )

count_tool = Tool[CountParams, CountResult](
    name="count_text",
    description="Count words and characters in text",
    handler=count_handler,
)
```

### ToolContext

Handlers receive immutable context:

```python
def my_handler(params: Params, *, context: ToolContext) -> ToolResult[Result]:
    # Access session state
    current_plan = context.session[Plan].latest()

    # Access resources (filesystem, etc.)
    filesystem = context.resources.get(Filesystem)

    # Check deadline
    if context.deadline.remaining() <= timedelta(0):
        return ToolResult(message="Timeout", success=False)

    # Publish events
    context.event_bus.publish(MyEvent(...))

    return ToolResult(message="Done", value=Result(...), success=True)
```

### Virtual Filesystem (VFS)

The VFS provides sandboxed file access:

```python
from weakincentives.contrib.tools import VfsToolsSection, HostMount, VfsPath

vfs_section = VfsToolsSection(
    session=session,
    mounts=(
        HostMount(
            host_path="./src",                    # Host directory
            mount_path=VfsPath(("src",)),         # Virtual path
            include_glob=("*.py", "*.md"),        # Include patterns
            exclude_glob=("**/__pycache__/**",),  # Exclude patterns
            max_bytes=1_000_000,                  # Size limit
        ),
    ),
    allowed_host_roots=(Path("."),),  # Security: limit host access
)
```

VFS tools provided:
- `ls` — List directory contents
- `read_file` — Read file contents
- `write_file` — Write file (to VFS, not host)
- `mkdir` — Create directory

Files mounted from host are read-only. Writes create VFS-local copies.

### Planning Tools

Planning tools help agents organize multi-step work:

```python
from weakincentives.contrib.tools import PlanningToolsSection, PlanningStrategy

planning_section = PlanningToolsSection(
    session=session,
    strategy=PlanningStrategy.PLAN_ACT_REFLECT,
)
```

Available strategies:

- `PLAN_ACT_REFLECT` — Full cycle: plan, execute, review
- `PLAN_ACT` — Planning and execution without reflection
- `MINIMAL` — Lightweight tracking only

Tools provided:
- `planning_set_plan` — Create or replace the plan
- `planning_update_step` — Update a step's status
- `planning_add_step` — Add a new step
- `planning_get_plan` — Retrieve current plan

### Workspace Digest

Auto-generate codebase summaries:

```python
from weakincentives.contrib.tools import WorkspaceDigestSection

digest_section = WorkspaceDigestSection(session=session)
```

On first use, the `WorkspaceDigestOptimizer` analyzes mounted files and
generates a summary that's stored in the session for subsequent requests.

### Asteval Section

Safe Python expression evaluation:

```python
from weakincentives.contrib.tools import AstevalSection

asteval_section = AstevalSection(
    session=session,
    globals={"math": __import__("math")},  # Whitelist safe modules
)
```

Features:
- Sandboxed execution (no file/network access)
- Captures stdout/stderr
- Records VFS mutations
- Time/memory limits

### Podman Sandbox

Full container isolation for shell commands:

```python
from weakincentives.contrib.tools import PodmanSandboxSection, PodmanSandboxConfig

podman_section = PodmanSandboxSection(
    session=session,
    config=PodmanSandboxConfig(
        mounts=mounts,
        allowed_host_roots=(Path("."),),
        # Network disabled by default
    ),
)
```

Provides `shell_execute` tool for running commands in isolated containers.

---

## 7. Provider Adapters

Adapters bridge prompts to LLM providers with a uniform interface.

### OpenAI Adapter

```python
from weakincentives.adapters.openai import (
    OpenAIAdapter,
    OpenAIClientConfig,
    OpenAIModelConfig,
)

adapter = OpenAIAdapter(
    model="gpt-4o",
    client_config=OpenAIClientConfig(
        api_key="sk-...",       # Or from OPENAI_API_KEY env
        base_url=None,          # Custom base URL
        timeout=60.0,           # Request timeout
    ),
    model_config=OpenAIModelConfig(
        temperature=0.7,
        max_tokens=4096,
    ),
)

response = adapter.evaluate(
    prompt,
    session=session,
    deadline=deadline,
    budget=budget_tracker,
)
```

### LiteLLM Adapter

Access 100+ providers through LiteLLM:

```python
from weakincentives.adapters.litellm import LiteLLMAdapter, LiteLLMModelConfig

adapter = LiteLLMAdapter(
    model="anthropic/claude-3-sonnet",  # LiteLLM model format
    model_config=LiteLLMModelConfig(
        temperature=0.7,
    ),
)
```

### Claude Agent SDK Adapter

Full agentic capabilities with Claude:

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

# Create workspace with mounted code
workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(
        HostMount(host_path="./src", mount_path="src"),
    ),
    allowed_host_roots=(".",),
)

# Configure isolation
adapter = ClaudeAgentSDKAdapter(
    model="claude-sonnet-4-5-20250929",
    client_config=ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        cwd=str(workspace.temp_dir),
        isolation=IsolationConfig(
            network_policy=NetworkPolicy(
                allowed_domains=("docs.python.org",),
            ),
            sandbox=SandboxConfig(
                enabled=True,
                readable_paths=(str(workspace.temp_dir),),
                bash_auto_allow=True,
            ),
        ),
    ),
)

response = adapter.evaluate(prompt, session=session)
workspace.cleanup()  # Clean up temp directory
```

Features:
- Native tools (Read, Write, Bash, Glob, Grep)
- Hermetic isolation with ephemeral home directory
- Network policy enforcement
- OS-level sandboxing (bubblewrap/seatbelt)
- MCP tool bridging for custom WINK tools

### Evaluation Response

All adapters return `PromptResponse`:

```python
response = adapter.evaluate(prompt, session=session)

# Typed output (if Prompt[OutputT] used)
output: OutputT = response.output

# Raw text response
text: str = response.text

# Token usage
print(f"Input: {response.usage.prompt_tokens}")
print(f"Output: {response.usage.completion_tokens}")
print(f"Total: {response.usage.total_tokens}")

# Tool calls made during evaluation
for call in response.tool_calls:
    print(f"Tool: {call.name}, Success: {call.success}")
```

### Deadlines and Budgets

Control execution time and cost:

```python
from weakincentives.deadlines import Deadline
from weakincentives import Budget, BudgetTracker

# Wall-clock deadline
deadline = Deadline(
    expires_at=datetime.now(UTC) + timedelta(minutes=5)
)

# Token budget
budget = Budget(
    max_total_tokens=100_000,
    max_prompt_tokens=50_000,
    max_completion_tokens=50_000,
)
tracker = BudgetTracker(budget=budget)

response = adapter.evaluate(
    prompt,
    session=session,
    deadline=deadline,
    budget=tracker,
)

# Tracker accumulates usage across calls
print(f"Total used: {tracker.total_tokens}")
```

### Throttle Policies

Handle rate limits gracefully:

```python
from weakincentives.adapters import new_throttle_policy

policy = new_throttle_policy(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
    backoff_multiplier=2.0,
)

adapter = OpenAIAdapter(
    model="gpt-4o",
    throttle_policy=policy,
)
```

---

## 8. Design-by-Contract

WINK uses design-by-contract decorators for API robustness.

### Contract Decorators

```python
from weakincentives.dbc import require, ensure, invariant, pure

@require(lambda x: x > 0, "x must be positive")
@ensure(lambda result: result >= 0, "result must be non-negative")
def compute(x: int) -> int:
    return x * 2

@pure()  # No side effects allowed
def calculate(a: int, b: int) -> int:
    return a + b

@invariant(
    lambda self: self.count >= 0,
    lambda self: self.name != "",
)
class Counter:
    def __init__(self, name: str):
        self.name = name
        self.count = 0

    def increment(self) -> None:
        self.count += 1
```

### Enabling Contracts

Contracts are no-ops by default for zero runtime cost. Enable for testing:

```python
from weakincentives.dbc import dbc_enabled, enable_dbc, disable_dbc

# Context manager
with dbc_enabled():
    compute(-1)  # Raises ContractViolation

# Global control
enable_dbc()   # Force enable everywhere
disable_dbc()  # Force disable

# Check state
from weakincentives.dbc import dbc_active
if dbc_active():
    print("Contracts are being checked")
```

### Testing with Contracts

```python
import pytest
from weakincentives.dbc import dbc_enabled

def test_contract_violation():
    with dbc_enabled():
        with pytest.raises(ContractViolation):
            compute(-1)  # Precondition fails
```

---

## 9. Advanced Patterns

### MainLoop for Production Agents

`MainLoop` standardizes agent orchestration:

```python
from weakincentives.runtime import MainLoop, MainLoopConfig

class MyLoop(MainLoop[RequestType, OutputType]):
    def __init__(self, adapter, bus):
        super().__init__(
            adapter=adapter,
            bus=bus,
            config=MainLoopConfig(
                max_retries=3,
            ),
        )
        self._session = Session(bus=bus)
        self._template = self._build_template()

    def create_prompt(self, request: RequestType) -> Prompt[OutputType]:
        return Prompt(self._template).bind(request)

    def create_session(self) -> Session:
        return self._session

    def _build_template(self) -> PromptTemplate[OutputType]:
        ...

# Use the loop
loop = MyLoop(adapter, bus)
response, session = loop.execute(request, deadline=deadline)
```

`MainLoop` handles:
- Visibility expansion (progressive disclosure)
- Deadline enforcement
- Budget tracking
- Error recovery

### Event-Driven Execution

Subscribe to loop events for reactive architectures:

```python
from weakincentives.runtime import (
    MainLoopRequest,
    MainLoopCompleted,
    MainLoopFailed,
)

bus.subscribe(MainLoopRequest, lambda e: print(f"Request: {e.request}"))
bus.subscribe(MainLoopCompleted, lambda e: print(f"Done: {e.response.output}"))
bus.subscribe(MainLoopFailed, lambda e: print(f"Error: {e.error}"))

# Publish request event (loop handles automatically)
bus.publish(MainLoopRequest(request=my_request, deadline=deadline))
```

### Transactional Tool Execution

Tool calls are atomic with automatic rollback:

```python
# Inside tool handler
def risky_operation(params: Params, *, context: ToolContext) -> ToolResult[Result]:
    # If handler raises exception, session state rolls back
    # If handler returns success=False, state is preserved (for error recording)

    try:
        result = do_work(params)
        return ToolResult(message="Success", value=result, success=True)
    except Exception as e:
        # State rolls back automatically
        raise
```

### Prompt Overrides

Iterate on prompts without code changes:

```python
from weakincentives.prompt.overrides import LocalPromptOverridesStore

# Create overrides store
store = LocalPromptOverridesStore()

# Seed initial override files
store.seed(template, tag="latest")

# Use overrides
prompt = Prompt(
    template,
    overrides_store=store,
    overrides_tag="experiment-v2",
).bind(params)
```

Override file structure:
```
~/.weakincentives/prompts/overrides/
  {namespace}/
    {prompt_key}/
      {tag}/
        sections/
          {section_key}.md
```

### Workspace Optimization

Auto-generate codebase summaries:

```python
from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer
from weakincentives.optimizers import OptimizationContext, PersistenceScope

# In MainLoop.execute():
if session[WorkspaceDigest].latest() is None:
    context = OptimizationContext(
        adapter=adapter,
        event_bus=bus,
    )
    optimizer = WorkspaceDigestOptimizer(
        context,
        store_scope=PersistenceScope.SESSION,
    )
    result = optimizer.optimize(prompt, session=session)
    # Digest stored in session for future requests
```

### Multi-Agent Systems

Create hierarchical agent structures:

```python
# Parent session for orchestration
orchestrator_session = Session(bus=shared_bus)

# Child sessions for sub-agents
worker1_session = Session(bus=shared_bus, parent=orchestrator_session)
worker2_session = Session(bus=shared_bus, parent=orchestrator_session)

# Child events propagate to parent bus
# Parent can observe all activity
```

### Custom Sections

Create specialized sections:

```python
from weakincentives.prompt import Section, RenderedSection, SectionContext

class CustomSection(Section[ParamsType]):
    """Custom section with special behavior."""

    def render(
        self,
        *,
        context: SectionContext,
        overrides: dict | None = None,
    ) -> RenderedSection:
        # Custom rendering logic
        content = self._generate_content(context)
        return RenderedSection(
            content=content,
            tools=self._tools,
            children=[],
        )

    @property
    def key(self) -> str:
        return "custom-section"

    @property
    def title(self) -> str:
        return "Custom Section"
```

---

## 10. Testing and Debugging

### Testing Patterns

```python
import pytest
from weakincentives.runtime import Session
from weakincentives.runtime.events import InProcessEventBus

@pytest.fixture
def session():
    """Fresh session for each test."""
    bus = InProcessEventBus()
    return Session(bus=bus)

def test_tool_handler(session):
    """Test a tool handler in isolation."""
    context = ToolContext(
        session=session,
        deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=1)),
        event_bus=session.event_bus,
        resources=ResourceRegistry(),
    )

    params = MyParams(value="test")
    result = my_handler(params, context=context)

    assert result.success
    assert result.value.expected_field == "expected"

def test_prompt_rendering():
    """Test prompt renders correctly."""
    template = build_my_template()
    prompt = Prompt(template).bind(MyParams(value="test"))
    rendered = prompt.render()

    assert "expected text" in rendered.text
    assert len(rendered.tools) == 2

def test_session_state(session):
    """Test session state management."""
    session[MyState].seed(MyState(value=0))
    session.broadcast(Increment(amount=5))

    state = session[MyState].latest()
    assert state.value == 5
```

### Contract Testing

```python
from weakincentives.dbc import dbc_enabled

def test_preconditions():
    """Verify contract enforcement."""
    with dbc_enabled():
        with pytest.raises(ContractViolation) as exc:
            compute(-1)
        assert "x must be positive" in str(exc.value)

def test_postconditions():
    """Verify results meet postconditions."""
    with dbc_enabled():
        result = compute(10)  # Should pass all contracts
        assert result == 20
```

### Snapshot Testing

```python
def test_session_rollback(session):
    """Test snapshot and restore."""
    session[Counter].seed(Counter(value=0))

    # Take snapshot
    snapshot = session.snapshot()

    # Make changes
    session.broadcast(Increment(amount=10))
    assert session[Counter].latest().value == 10

    # Restore
    session.restore(snapshot)
    assert session[Counter].latest().value == 0
```

### Debug UI

The `wink debug` command provides visual inspection:

```bash
# Serve debug UI for session snapshots
uv run --extra wink wink debug snapshots/session.jsonl --port 8000
```

Features:
- Slice browser with JSON viewer
- Session selector for multi-entry files
- Live reload on file changes
- Raw download

### Logging

WINK provides structured logging:

```python
from weakincentives.runtime.logging import configure_logging, get_logger

# Configure logging
configure_logging(json_mode=True)

# Get module-scoped logger
logger = get_logger("my-agent")

# Bind context
bound_logger = logger.bind(component="handler", request_id="abc123")

# Log with event and context
bound_logger.info(
    "Processing request",
    event="request.processing",
    context={"step": 1, "total": 5},
)
```

### Event Tracing

Subscribe to events for debugging:

```python
from weakincentives.runtime import PromptRendered, PromptExecuted, ToolInvoked

def trace_events(bus):
    """Attach tracing to event bus."""
    bus.subscribe(PromptRendered, lambda e:
        print(f"[RENDER] {e.prompt_name}: {len(e.text)} chars")
    )
    bus.subscribe(ToolInvoked, lambda e:
        print(f"[TOOL] {e.name}: {'OK' if e.success else 'FAIL'}")
    )
    bus.subscribe(PromptExecuted, lambda e:
        print(f"[EXEC] tokens={e.usage.total_tokens}")
    )
```

---

## 11. API Reference

### Core (`weakincentives`)

```python
# Budgeting & Deadlines
Budget                      # Token/time budget envelope
BudgetTracker              # Thread-safe cumulative tracker
BudgetExceededError        # Raised on limit breach
Deadline                   # Wall-clock expiration
DeadlineExceededError      # Raised on timeout

# Prompt System
MarkdownSection            # Render markdown with string.Template
Prompt                     # Bind params to template
Tool                       # Declare a capability
ToolContext                # Immutable context for handlers
ToolHandler                # Callable protocol
ToolResult                 # Return value from handler
parse_structured_output    # Parse model response

# Data Utilities
FrozenDataclass            # Immutable dataclass decorator
JSONValue                  # Type alias for JSON-safe types
SupportsDataclass          # Protocol for dataclass types

# Logging
StructuredLogger           # Logger with {event, context} schema
configure_logging          # Set up logging
get_logger                 # Module-scoped logger

# Errors
WinkError                  # Base exception
ToolValidationError        # Tool validation failed
```

### Prompt (`weakincentives.prompt`)

```python
# Templates and Prompts
PromptTemplate[OutputT]    # Immutable blueprint
Prompt                     # Wrapped template with bindings
RenderedPrompt            # Result of rendering
Section                   # Base section class
SectionVisibility         # Enum: FULL or SUMMARY
SectionNode               # Node in section tree

# Tools
Tool[ParamsT, ResultT]    # Typed tool descriptor
ToolContext               # Handler context
ResourceRegistry          # Runtime resources container
ToolExample               # Representative invocation
ToolResult[PayloadT]      # Handler return value

# Overrides
LocalPromptOverridesStore  # File-backed override store
PromptDescriptor          # Identifies prompt for versioning
SectionDescriptor         # Identifies section
ToolDescriptor            # Identifies tool
PromptOverride            # Override payload
hash_text, hash_json      # Hashing utilities

# Visibility
SetVisibilityOverride      # Override section visibility
ClearVisibilityOverride    # Clear override
VisibilityOverrides        # Session state for overrides

# Structured Output
StructuredOutputConfig     # Configuration
OutputParseError           # Parsing failed
parse_structured_output    # Parse model response

# Errors
PromptError                # Base
PromptValidationError      # Invalid prompt
PromptRenderError          # Render failed
VisibilityExpansionRequired # Model requested expansion
```

### Runtime (`weakincentives.runtime`)

```python
# Session
Session                    # Event ledger with Redux reducers
SessionProtocol            # Protocol for session interface
Snapshot                   # Immutable state capture
TypedReducer               # Reducer for typed state

# Events
EventBus                   # Abstract publisher
InProcessEventBus          # In-process implementation
PromptRendered             # After rendering
PromptExecuted             # After model response
ToolInvoked                # After tool call
TokenUsage                 # Usage from provider

# Main Loop
MainLoop[RequestT, OutputT] # Abstract orchestrator
MainLoopRequest            # Request event
MainLoopCompleted          # Success event
MainLoopFailed             # Failure event
MainLoopConfig             # Configuration

# Execution State
ExecutionState             # Unified mutable state
CompositeSnapshot          # Snapshot of execution state

# Reducers
append_all                 # Ledger reducer
replace_latest             # Keep only latest
replace_latest_by          # Replace by key
upsert_by                  # Insert or update
build_reducer_context      # Create context
```

### Adapters (`weakincentives.adapters`)

```python
# Base Protocol
ProviderAdapter[ConfigT]   # Synchronous evaluation
PromptResponse[OutputT]    # Structured result

# OpenAI
OpenAIAdapter              # Official OpenAI SDK
OpenAIClientConfig         # API configuration
OpenAIModelConfig          # Model parameters

# LiteLLM
LiteLLMAdapter             # 100+ providers
LiteLLMClientConfig        # Client config
LiteLLMModelConfig         # Model config

# Claude Agent SDK
ClaudeAgentSDKAdapter          # Claude Code via subprocess
ClaudeAgentSDKClientConfig     # Permission mode, isolation
ClaudeAgentWorkspaceSection    # Materialized workspace
HostMount                      # Host path mount config
IsolationConfig                # Hermetic isolation
NetworkPolicy                  # Network constraints
SandboxConfig                  # OS-level sandboxing
PermissionMode                 # "default", "acceptEdits", etc.

# Configuration
LLMConfig                  # Base for temperature, etc.
ThrottlePolicy             # Retry/backoff on rate limits
ThrottleError              # Raised on throttle
PromptEvaluationError      # Evaluation failure
```

### Contrib Tools (`weakincentives.contrib.tools`)

```python
# VFS
VfsToolsSection            # Section providing VFS tools
VirtualFileSystem          # In-memory filesystem
VfsFile                    # File within VFS
VfsPath                    # POSIX-style path
HostMount                  # Mount configuration
HostMountPreview           # Preview of mount contents

# Planning
PlanningToolsSection       # Section providing planning tools
Plan                       # Todo list in session
PlanningStrategy           # PLAN_ACT_REFLECT, etc.
Step                       # Individual todo item

# Workspace Digest
WorkspaceDigestSection     # Auto-generated summary
WorkspaceDigest            # The digest itself

# Advanced
AstevalSection             # Safe Python evaluation
PodmanSandboxSection       # Isolated container
PodmanSandboxConfig        # Podman configuration
```

### Serde (`weakincentives.serde`)

```python
parse(cls, data, *, extra, coerce, ...)  # JSON → dataclass
dump(obj, *, by_alias, exclude_none, ...)  # dataclass → JSON
clone(obj, **updates)                      # Deep clone with updates
schema(cls, *, alias_generator, extra)     # Generate JSON schema
```

### DBC (`weakincentives.dbc`)

```python
@require(predicate, message)   # Preconditions
@ensure(predicate, message)    # Postconditions
@invariant(predicates)         # Class invariants
@pure()                        # No side effects

dbc_active()                   # Check if enabled
dbc_enabled()                  # Context manager
enable_dbc()                   # Force enable
disable_dbc()                  # Force disable
skip_invariant                 # Mark method to skip
```

---

## Appendix: Quick Reference Card

### Creating an Agent

```python
# 1. Define output type
@dataclass(slots=True, frozen=True)
class Output:
    result: str

# 2. Create session
bus = InProcessEventBus()
session = Session(bus=bus)

# 3. Build template
template = PromptTemplate[Output](
    ns="myapp",
    key="agent",
    sections=(
        MarkdownSection(title="Instructions", template="...", key="inst"),
        VfsToolsSection(session=session, mounts=mounts, allowed_host_roots=(...,)),
    ),
)

# 4. Evaluate
adapter = OpenAIAdapter(model="gpt-4o")
prompt = Prompt(template).bind(params)
response = adapter.evaluate(prompt, session=session)
output: Output = response.output
```

### Common Tool Patterns

```python
# Read file from VFS
file_content = context.resources.get(Filesystem).read("/path/to/file")

# Query session state
plan = context.session[Plan].latest()

# Check deadline
if context.deadline.remaining() < timedelta(seconds=10):
    return ToolResult(message="Low time", success=False)

# Return success
return ToolResult(message="Done", value=Result(...), success=True)

# Return failure
return ToolResult(message="Error: invalid input", success=False)
```

### State Management

```python
# Query
session[Type].latest()              # Most recent
session[Type].all()                 # All values
session[Type].where(predicate)      # Filter

# Mutate
session[Type].seed(initial)         # Initialize
session[Type].register(Event, reducer)  # Add reducer
session.broadcast(Event(...))       # Dispatch event

# Snapshot
snapshot = session.snapshot()       # Capture
session.restore(snapshot)           # Restore
```

---

## Further Reading

- **`specs/`** — Design specifications for each subsystem
- **`guides/`** — Step-by-step tutorials and patterns
- **`AGENTS.md`** — Contributor workflow and conventions
- **`llms.md`** — PyPI README with full API reference
- **`code_reviewer_example.py`** — Complete working example

---

*WINK is alpha software. All APIs may change without backward compatibility.*
