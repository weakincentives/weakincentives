# Tools

*Canonical spec: [specs/TOOLS.md](../specs/TOOLS.md)*

The tool system is designed around one hard rule:

> **Tool handlers are the only place where side effects happen.**

Everything else—prompt rendering, state transitions, reducers—is meant to be
pure and deterministic. This constraint has a purpose: when something goes
wrong, you know exactly where to look.

## Tool Contracts

A tool is defined by:

- `name`: `^[a-z0-9_-]{1,64}$`
- `description`: short model-facing string
- `params_type`: a dataclass type (or `None`)
- `result_type`: a dataclass type (or `None`)
- `handler(params, *, context) -> ToolResult[result_type]`

**Skeleton:**

```python
from dataclasses import dataclass
from weakincentives.prompt import Tool, ToolContext, ToolResult

@dataclass(slots=True, frozen=True)
class MyParams:
    query: str

@dataclass(slots=True, frozen=True)
class MyResult:
    answer: str

    def render(self) -> str:
        return self.answer

def handler(params: MyParams, *, context: ToolContext) -> ToolResult[MyResult]:
    # Do work here (read files, call an API, etc.)
    return ToolResult.ok(MyResult(answer="42"), message="ok")

tool = Tool[MyParams, MyResult](
    name="my_tool",
    description="Do a thing.",
    handler=handler,
)
```

The type parameters matter. `Tool[MyParams, MyResult]` tells WINK how to
serialize parameters for the model and how to parse results. Type mismatches are
caught at construction time.

## ToolContext, Resources, and Filesystem

`ToolContext` provides access to execution-time state:

- `context.prompt`: the `Prompt` being executed
- `context.rendered_prompt`: the `RenderedPrompt` (when available)
- `context.adapter`: the adapter executing tools
- `context.session`: the current session
- `context.deadline`: optional wall-clock deadline
- `context.budget_tracker`: optional token budget tracker
- `context.resources`: a typed `ResourceRegistry`

**A key idea**: you can pass your own resources (HTTP clients, DB handles,
tracers) without adding new fields to the core dataclass.

```python
from weakincentives.resources import Binding, ResourceRegistry

registry = ResourceRegistry.of(
    Binding.instance(MyHttpClient, MyHttpClient(...)),
)
```

**Binding resources to prompts:**

Resources are bound to `Prompt` via the `bind()` method:

```python
from weakincentives.resources import Binding, Scope
from weakincentives.prompt import Prompt

# Simple case: pre-constructed instances
http_client = HTTPClient(base_url="https://api.example.com")
prompt = Prompt(template).bind(params, resources={HTTPClient: http_client})

# Advanced: lazy construction with dependencies and scopes
prompt = Prompt(template).bind(params, resources={
    Config: Binding(Config, lambda r: Config.from_env()),
    HTTPClient: Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
    Tracer: Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
})

# Use prompt.resources context manager for lifecycle management
with prompt.resources:
    response = adapter.evaluate(prompt, session=session)
```

**Scopes control instance lifetime:**

- `Scope.SINGLETON`: One instance per session (default)
- `Scope.TOOL_CALL`: Fresh instance per tool invocation
- `Scope.PROTOTYPE`: Fresh instance on every access

For workspace agents, the most common resource is a `Filesystem` implementation.
Many contributed tool suites install one automatically (VFS, Podman).

## ToolResult Semantics

Tool handlers return `ToolResult`. Use the convenience constructors:

```python
# Success with typed value (most common)
ToolResult.ok(MyResult(...), message="Done")

# Failure with no value
ToolResult.error("Something went wrong")

# Full form (when exclude_value_from_context is needed)
ToolResult(
    message="Human-readable status",
    value=...,                        # dataclass | mapping | sequence | str | None
    success=True,                     # if False, adapters treat as tool failure
    exclude_value_from_context=False, # hide large payloads from model context
)
```

**Key behaviors:**

- If `value` is a dataclass and implements `render() -> str`, adapters use that
  as the textual tool output. This lets you control exactly what the model sees.
- If `render()` is missing, WINK logs a warning and serializes the dataclass to
  JSON. This works but is less controlled.
- Exceptions raised by handlers are caught and converted into tool failures.

The `exclude_value_from_context=True` flag is useful for tools that return large
payloads (like file contents). The model sees a summary message, but the full
value is recorded in the session for debugging.

## Tool Examples

Tools can provide `ToolExample` entries for better model performance and better
debugging:

```python
from weakincentives.prompt import ToolExample

tool = Tool[NowParams, NowResult](
    name="now",
    description="Return UTC time.",
    handler=now_handler,
    examples=(
        ToolExample(
            description="Get UTC time",
            input=NowParams(tz="UTC"),
            output=NowResult(iso="2025-01-01T00:00:00+00:00"),
        ),
    ),
)
```

If you've ever seen a tool-capable model "almost" do the right call—wrong
parameter name, wrong format, etc.—examples tend to pay for themselves. One good
example often beats three paragraphs of instructions.

## Tool Suites as Sections

In WINK, "a tool suite" is usually a section:

- It adds instructions explaining when to use the tools
- It attaches tool contracts
- It often owns some session slice(s)

That co-location is intentional: **tools without guidance are unreliable, and
guidance without tools is toothless**.

Examples in `weakincentives.contrib.tools`:

- Planning tools (`PlanningToolsSection`)
- VFS tools (`VfsToolsSection`)
- Sandbox tools (`PodmanSandboxSection`, `AstevalSection`)

Each section bundles the instructions ("here's how to use these tools") with the
tools themselves. The model sees them together.

## Transactional Tool Execution

*Canonical spec: [specs/SESSIONS.md](../specs/SESSIONS.md)*

One of the hardest problems in building agents is handling partial failures.
When a tool call fails halfway through, you're left with corrupted session
state, inconsistent filesystem changes, or both.

WINK solves this with **transactional tool execution**. Every tool call is
wrapped in a transaction:

1. **Snapshot**: Before the tool runs, WINK captures the current state
2. **Execute**: The tool handler runs
3. **Commit or rollback**: If the tool succeeds, changes are kept. If it fails,
   WINK automatically restores the snapshot

This happens by default—you don't need to opt in or write rollback logic.

**What gets rolled back:**

- Session slices marked as `STATE` (working state like plans, visibility
  overrides)
- Filesystem changes (both in-memory VFS and disk-backed via git)

**What's preserved:**

- Session slices marked as `LOG` (historical records like `ToolInvoked`)
- External side effects (API calls, network requests)

**Why this matters:**

- Simpler error handling: no defensive rollback code in every handler
- Consistent state: failed operations never leave inconsistent state
- Easier debugging: when something fails, you know exactly what state you're in
- Adapter parity: all adapters use the same transaction semantics

## Tool Policies

*Canonical spec: [specs/TOOL_POLICIES.md](../specs/TOOL_POLICIES.md)*

Tool policies provide declarative constraints that govern when tools can be
invoked. Rather than embedding validation logic in each tool handler, policies
express cross-cutting concerns as composable constraints.

**Why policies exist:** Without constraints, models can call tools in
problematic orders—deploying code that was never tested, overwriting files they
haven't read, or skipping required validation steps.

**Built-in policies:**

```python
from weakincentives.prompt import (
    SequentialDependencyPolicy,
    ReadBeforeWritePolicy,
)

# Require 'test' and 'build' before 'deploy'
deploy_policy = SequentialDependencyPolicy(
    dependencies={
        "deploy": frozenset({"test", "build"}),
    }
)

# Require reading a file before overwriting it (new files allowed)
read_first = ReadBeforeWritePolicy()
```

**Attaching policies to tools:**

Policies are attached at the section level:

```python
from weakincentives.prompt import MarkdownSection, Tool

section = MarkdownSection(
    title="Deployment",
    key="deployment",
    template="Deploy the application after testing.",
    tools=(deploy_tool, test_tool, build_tool),
    policies=(deploy_policy,),
)
```

**Enforcement:** When a tool call violates a policy, WINK returns a
`ToolResult.error()` without executing the handler. The error message explains
which policy was violated.

**Default policies on contrib sections:** `VfsToolsSection` and
`PodmanSandboxSection` apply `ReadBeforeWritePolicy` by default.

## Next Steps

- [Sessions](sessions.md): Manage state with reducers
- [Adapters](adapters.md): Connect to different providers
- [Workspace Tools](workspace-tools.md): Use VFS, Podman, and planning tools
