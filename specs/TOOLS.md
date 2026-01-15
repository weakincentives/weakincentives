# Tool Runtime Specification

## Purpose

Tool registration, context injection, failure semantics, and planning tools.
Core at `prompt/tool.py`.

## Principles

- **Section-first**: Tools live within section hierarchy
- **Single source of truth**: Definitions alongside documenting sections
- **Type-safe**: Dataclass-based params and results
- **Predictable failures**: Never abort evaluation; return structured errors

## Core Schemas

### Tool

At `prompt/tool.py` (`Tool` class):

| Field | Description |
| --- | --- |
| `name` | `^[a-z0-9_-]{1,64}$` |
| `description` | 1-200 chars |
| `handler` | `ToolHandler[ParamsT, ResultT]` |
| `accepts_overrides` | Whether description overridable |
| `examples` | `tuple[ToolExample, ...]` |

Handler signature:
```python
def handle(params: ParamsT, *, context: ToolContext) -> ToolResult[ResultT]: ...
```

### ToolResult

At `prompt/tool_result.py` (`ToolResult` class):

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

At `prompt/tool.py` (`ToolContext` class):

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

At `prompt/tool.py` (`ToolExample` class):

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
)
```

### Prompt Rendering

Validates:
1. Duplicate names → `PromptValidationError`
2. Examples against params/result dataclasses
3. Declaration order cached

`RenderedPrompt.tools` contains ordered tuple from enabled sections.

## Runtime Dispatch

Via `ToolExecutor` at `adapters/tool_executor.py`:

1. **Registry lookup** - Resolve tool name
2. **Argument parsing** - `serde.parse(..., extra="forbid")`
3. **Deadline check** - Refuse if elapsed
4. **Context construction** - Build `ToolContext`
5. **Snapshot** - Capture session and resource state
6. **Handler execution** - Run with params/context
7. **Restore on failure** - Rollback state
8. **Telemetry** - Publish `ToolInvoked` to `session.dispatcher`
9. **Response assembly** - Return result

## Planning Tool Suite

Session-scoped todo list at `contrib/tools/planning.py`.

### Data Model

At `contrib/tools/planning.py`:

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

At `contrib/tools/planning.py`:

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

`PlanningToolsSection` at `contrib/tools/planning.py` auto-registers
reducers for `Plan`, `SetupPlan`, `AddStep`, `UpdateStep`.

## Planning Strategies

At `contrib/tools/planning.py` (`PlanningStrategy` enum):

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
