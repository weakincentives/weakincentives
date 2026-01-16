# Tool Runtime Specification

## Purpose

Tool registration, context injection, failure semantics, policy enforcement,
and planning tools. Core at `src/weakincentives/prompt/tool.py`.

## Principles

- **Section-first**: Tools live within section hierarchy
- **Single source of truth**: Definitions alongside documenting sections
- **Type-safe**: Dataclass-based params and results
- **Predictable failures**: Never abort evaluation; return structured errors
- **Policy-governed**: Sequential dependencies enforced before execution

## Core Schemas

### Tool

At `src/weakincentives/prompt/tool.py` (`Tool` class):

| Field | Description |
| --- | --- |
| `name` | `^[a-z0-9_-]{1,64}$` |
| `description` | 1-200 chars |
| `handler` | `ToolHandler[ParamsT, ResultT] \| None` |
| `accepts_overrides` | Whether description overridable |
| `examples` | `tuple[ToolExample, ...]` |

Handler signature:

```python
def handle(params: ParamsT, *, context: ToolContext) -> ToolResult[ResultT]: ...
```

### ToolResult

At `src/weakincentives/prompt/tool_result.py` (`ToolResult` class):

| Field | Description |
| --- | --- |
| `message` | Text forwarded to model |
| `value` | Typed payload (may be None) |
| `success` | Normal vs. failure |
| `exclude_value_from_context` | Hide from provider |

**Factories:**

- `ToolResult.ok(value, message)` - Success
- `ToolResult.error(message)` - Failure

Rendering via `render()` uses `serde.dump`.

### ToolContext

At `src/weakincentives/prompt/tool.py` (`ToolContext` class):

| Field | Description |
| --- | --- |
| `prompt` | Active prompt |
| `rendered_prompt` | Rendered state |
| `adapter` | Provider adapter |
| `session` | Session for state |
| `deadline` | Optional deadline |
| `budget_tracker` | Optional budget tracker |

| Property | Description |
| --- | --- |
| `resources` | Access prompt's resource context |
| `filesystem` | Shortcut for Filesystem resource |
| `beat()` | Heartbeat for long operations |

Tool handlers publish events via `context.session.dispatcher`.

### Resource Access

Tools access resources through prompt:

```python
def my_handler(params: Params, *, context: ToolContext) -> ToolResult[Result]:
    fs = context.resources.get(Filesystem)
    # Or shorthand: fs = context.filesystem
```

### ToolExample

At `src/weakincentives/prompt/tool.py` (`ToolExample` class):

| Field | Description |
| --- | --- |
| `description` | ≤200 chars |
| `input` | Params dataclass instance |
| `output` | Result dataclass instance |

## Registration Lifecycle

### Section Integration

Tools declared on sections:

```python
section = MarkdownSection[Params](
    title="Guidance",
    key="guidance",
    template="Use tools when needed.",
    tools=[lookup_tool, search_tool],
    policies=[ReadBeforeWritePolicy()],
)
```

### Prompt Rendering

Validates:

1. Duplicate names → `PromptValidationError`
1. Examples against params/result dataclasses
1. Declaration order cached

`RenderedPrompt.tools` contains ordered tuple from enabled sections.

## Tool Policies

**Implementation:** `src/weakincentives/prompt/policy.py`

Policies enforce sequential dependencies between tool invocations. Declares
that tool B requires tool A first—unconditionally or keyed by parameter.

### Guiding Principles

- **Prompt-scoped declaration**: Bound to prompts alongside tools
- **Session-scoped state**: Invocation history in session slices
- **Composable**: Multiple policies can govern same tool; all must allow
- **Fail-closed**: Denied calls return error without executing

### ToolPolicy Protocol

| Method/Property | Description |
|-----------------|-------------|
| `name` | Unique identifier |
| `check(tool, params, *, context)` | Returns `PolicyDecision` |
| `on_result(tool, params, result, *, context)` | Update state after success |

### PolicyDecision

| Field | Type | Description |
|-------|------|-------------|
| `allowed` | `bool` | Whether to proceed |
| `reason` | `str \| None` | Denial explanation |

### PolicyState (Session Slice)

| Field | Type | Description |
|-------|------|-------------|
| `policy_name` | `str` | Policy identifier |
| `invoked_tools` | `frozenset[str]` | Successfully invoked tools |
| `invoked_keys` | `frozenset[tuple[str, str]]` | (tool, key) pairs |

### Built-in Policies

#### SequentialDependencyPolicy

Unconditional tool ordering: tool B requires tool A to have succeeded.

```python
policy = SequentialDependencyPolicy(
    dependencies={
        "deploy": frozenset({"test", "build"}),
        "build": frozenset({"lint"}),
    }
)
```

#### ReadBeforeWritePolicy

Parameter-keyed dependency for filesystem tools. Existing files must be read
before overwritten. New files can be created freely.

```python
policy = ReadBeforeWritePolicy()
# write_file("new.txt")      → OK (doesn't exist)
# write_file("config.yaml")  → DENIED (exists, not read)
# read_file("config.yaml")   → OK (records path)
# write_file("config.yaml")  → OK (was read)
```

### Policy Integration

```python
template = PromptTemplate(
    sections=[
        MarkdownSection(
            tools=[read_file, write_file],
            policies=[ReadBeforeWritePolicy()],
        ),
        MarkdownSection(
            tools=[lint, test, build, deploy],
            policies=[SequentialDependencyPolicy(dependencies={...})],
        ),
    ],
    policies=[...],  # Prompt-level policies
)
```

## Runtime Dispatch

Via `ToolExecutor` at `adapters/tool_executor.py`:

1. **Registry lookup** - Resolve tool name
1. **Argument parsing** - `serde.parse(..., extra="forbid")`
1. **Policy check** - All policies must allow (fail-closed)
1. **Deadline check** - Refuse if elapsed
1. **Context construction** - Build `ToolContext`
1. **Snapshot** - Capture session and resource state
1. **Handler execution** - Run with params/context
1. **Policy update** - Call `on_result` for successful invocations
1. **Restore on failure** - Rollback state
1. **Telemetry** - Publish `ToolInvoked` to `session.dispatcher`
1. **Response assembly** - Return result

### Execution Flow (Pseudocode)

```python
def execute_tool(call, *, context):
    tool, params = resolve_and_parse(call)
    policies = [*section.policies, *prompt.policies]

    for policy in policies:
        decision = policy.check(tool, params, context=context)
        if not decision.allowed:
            return ToolResult.error(decision.reason)

    result = tool.handler(params, context=context)

    if result.success:
        for policy in policies:
            policy.on_result(tool, params, result, context=context)

    return result
```

### Policy State Management

- **Snapshot/restore**: State captured with session snapshots
- **Reset**: `session.reset()` clears policy state
- **Isolation**: Each session has independent state

## Planning Tool Suite

Session-scoped todo list at `src/weakincentives/contrib/tools/planning.py`.

### Data Model

At `src/weakincentives/contrib/tools/planning.py`:

| Type | Description |
| --- | --- |
| `StepStatus` | `"pending"`, `"in_progress"`, `"done"` |
| `PlanStatus` | `"active"`, `"completed"` |
| `PlanStep` | step_id, title, status |
| `Plan` | objective, status, steps |

### Tools

| Tool | Purpose |
| --- | --- |
| `planning_setup_plan` | Create or replace plan |
| `planning_add_step` | Append steps |
| `planning_update_step` | Modify step title/status |
| `planning_read_plan` | Retrieve current state |

### Parameters

At `src/weakincentives/contrib/tools/planning.py`:

| Event | Fields |
| --- | --- |
| `SetupPlan` | objective, initial_steps |
| `AddStep` | steps |
| `UpdateStep` | step_id, title, status |
| `ReadPlan` | (none) |

### Behavior

- `setup_plan` creates/replaces; others require existing plan
- Step IDs: incrementing integers, never reused
- All steps `done` → plan `completed`
- Titles: non-empty, ≤500 chars

### Session Integration

`PlanningToolsSection` at `src/weakincentives/contrib/tools/planning.py` auto-registers
reducers for `Plan`, `SetupPlan`, `AddStep`, `UpdateStep`.

## Planning Strategies

At `src/weakincentives/contrib/tools/planning.py` (`PlanningStrategy` enum):

| Strategy | Description |
| --- | --- |
| `REACT` | Alternate reasoning, tool calls, observations |
| `PLAN_ACT_REFLECT` | Outline first, execute with reflections |
| `GOAL_DECOMPOSE_ROUTE_SYNTHESISE` | Restate goal, decompose, route, synthesize |

Same markdown structure; only mindset paragraphs vary.

## Failure Semantics

### ToolResult Contract

- `success=True`: Normal payload in `value`
- `success=False`: Error condition; `value=None` unless error payload

### Exception Handling

| Exception | Behavior |
| --- | --- |
| `ToolValidationError` | Wrap as `ToolResult(success=False)` |
| `VisibilityExpansionRequired` | Re-raise |
| `PromptEvaluationError` | Re-raise |
| `DeadlineExceededError` | Convert to `PromptEvaluationError` |
| `TypeError` | Wrap with descriptive message |
| Other | Wrap as `ToolResult(success=False)` |

All failure paths restore session and resource state before returning.

### Handler Validation

Fail-fast approach:

- **Development**: pyright strict mode catches mismatches
- **Runtime**: TypeErrors converted to failed results

Tool failures forward error messages to LLM via `role: "tool"` response.

## Limitations

- **Synchronous handlers**: Execute on provider loop thread
- **Dataclass-only schemas**: No TypedDict or arbitrary mappings
- **Payload visibility**: `exclude_value_from_context` not security boundary
- **Deadline enforcement**: Checked before entry, not per-invocation
- **Policy synchronous**: Policy checks run on tool execution thread
- **Policy session-scoped**: No cross-session persistence
- **No policy rollback notification**: Policies not notified on restore

## Related Specifications

- `specs/POLICIES_OVER_WORKFLOWS.md` - Design philosophy
- `specs/PROMPTS.md` - Prompt system and sections
- `specs/SESSIONS.md` - Session state
