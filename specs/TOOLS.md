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
| `heartbeat` | Optional heartbeat for lease extension |
| `run_context` | Optional execution context with correlation IDs |

| Property/Method | Description |
| --- | --- |
| `resources` | Access prompt's resource context |
| `filesystem` | Shortcut for Filesystem resource |
| `budget_tracker` | Shortcut for BudgetTracker resource |
| `beat()` | Record heartbeat for long operations |

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

Policies enforce sequential dependencies between tool invocations. Declared on
sections via `policies=[...]` tuple; all must allow for execution to proceed.

**Full specification:** `GUARDRAILS.md` (Tool Policies section)

Built-in: `SequentialDependencyPolicy`, `ReadBeforeWritePolicy`

## Runtime Dispatch

Tool dispatch is handled by the adapter's execution hooks:

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

`PlanningToolsSection` at `src/weakincentives/contrib/tools/planning.py` installs
the `Plan` slice into the session. The `Plan` class defines `@reducer` methods
that handle `SetupPlan`, `AddStep`, and `UpdateStep` events.

### PlanningConfig

At `src/weakincentives/contrib/tools/planning.py` (`PlanningConfig` class):

| Field | Description |
| --- | --- |
| `strategy` | `PlanningStrategy` (default: `REACT`) |
| `accepts_overrides` | Whether section accepts parameter overrides |

Example:

```python
from weakincentives.contrib.tools import PlanningConfig, PlanningToolsSection

config = PlanningConfig(strategy=PlanningStrategy.PLAN_ACT_REFLECT)
section = PlanningToolsSection(session=session, config=config)
```

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

## Related Specifications

- `GUARDRAILS.md` - Tool policies, feedback providers, task completion
- `POLICIES_OVER_WORKFLOWS.md` - Design philosophy
- `PROMPTS.md` - Prompt system and sections
- `SESSIONS.md` - Session state
