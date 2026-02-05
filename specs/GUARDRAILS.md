# Guardrails Specification

## Purpose

Guardrails ensure agents operate within constraints while preserving reasoning
autonomy. Three complementary mechanisms—tool policies, feedback providers, and
task completion checking—enforce invariants, provide guidance, and verify goals.

**Philosophy:** See `POLICIES_OVER_WORKFLOWS.md` for design rationale.

## Overview

| Mechanism | Role | Enforcement |
|-----------|------|-------------|
| Tool Policies | Gate tool invocations | Hard block (fail-closed) |
| Feedback Providers | Soft guidance over time | Advisory (agent decides) |
| Task Completion | Verify goals before stopping | Block early termination |

______________________________________________________________________

## Tool Policies

**Implementation:** `src/weakincentives/prompt/policy.py`

Policies enforce sequential dependencies between tool invocations. Declares
that tool B requires tool A first—unconditionally or keyed by parameter.

### Principles

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

### Execution Flow

Policy enforcement happens in the adapter's tool execution hooks:

```python nocheck
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

### Limitations

- **Synchronous**: Policy checks run on tool execution thread
- **Session-scoped**: No cross-session persistence
- **No rollback notification**: Policies not notified on restore

______________________________________________________________________

## Feedback Providers

**Implementation:** `src/weakincentives/prompt/feedback.py`

Deliver ongoing progress feedback to agents during unattended execution. Analyze
patterns over time and inject guidance into context for soft course-correction.

Unlike tool policies that gate calls, feedback providers observe trajectory and
produce contextual feedback delivered immediately after tool execution.

### Characteristics

- **Non-blocking**: Guidance, not gates; agent decides response
- **Trigger-based**: Run when conditions met (every N calls/seconds)
- **Immediate delivery**: Inject via hook response, not next render

### FeedbackTrigger

| Field | Type | Description |
|-------|------|-------------|
| `every_n_calls` | `int \| None` | Run after N tool calls |
| `every_n_seconds` | `float \| None` | Run after N seconds elapsed |
| `on_file_created` | `FileCreatedTrigger \| None` | Run once when file created |

Conditions are OR'd together.

### FileCreatedTrigger

Triggers when a specified file is created on the filesystem. Fires exactly once
per session—after initial detection, subsequent tool calls will not re-trigger
even if the file is deleted and recreated.

| Field | Type | Description |
|-------|------|-------------|
| `filename` | `str` | Path to watch for creation |

#### Behavior

1. After tool execution completes, check if `filename` exists
1. If file exists and trigger has not fired → fire, mark as fired
1. If file does not exist or trigger already fired → skip
1. Trigger state persists in session; reset clears it

### StaticFeedbackProvider

A built-in provider that delivers a fixed feedback message. Useful with
`FileCreatedTrigger` for one-time guidance when specific files are detected.

| Field | Type | Description |
|-------|------|-------------|
| `feedback` | `str` | Feedback content to deliver |

```python
config = FeedbackProviderConfig(
    provider=StaticFeedbackProvider(
        feedback="AGENTS.md detected. Follow the conventions defined within.",
    ),
    trigger=FeedbackTrigger(
        on_file_created=FileCreatedTrigger(filename="AGENTS.md"),
    ),
)
```

### FeedbackProviderConfig

```python
template = PromptTemplate(
    ...,
    feedback_providers=(
        FeedbackProviderConfig(
            provider=DeadlineFeedback(),
            trigger=FeedbackTrigger(every_n_seconds=30),
        ),
    ),
)
```

### FeedbackProvider Protocol

| Method/Property | Description |
|-----------------|-------------|
| `name` | Unique identifier |
| `should_run(context)` | Additional filtering beyond trigger |
| `provide(context)` | Produce feedback |

### Feedback

| Field | Type | Description |
|-------|------|-------------|
| `provider_name` | `str` | Source provider |
| `summary` | `str` | Main message |
| `observations` | `tuple[Observation, ...]` | Detailed observations |
| `suggestions` | `tuple[str, ...]` | Recommendations |
| `severity` | `Literal["info", "caution", "warning"]` | Urgency level |
| `timestamp` | `datetime` | When produced |

### FeedbackContext

| Property/Method | Description |
|-----------------|-------------|
| `session` | Session protocol |
| `prompt` | Prompt protocol |
| `deadline` | Optional deadline |
| `last_feedback` | Most recent feedback for prompt |
| `tool_call_count` | Total calls for prompt |
| `tool_calls_since_last_feedback()` | Calls since last feedback |
| `recent_tool_calls(n)` | Last N tool calls |

### Execution Flow

1. Tool call completes
1. `ToolInvoked` dispatched
1. Check trigger conditions
1. Call `provider.should_run()`
1. Call `provider.provide()`
1. Store in session, return text
1. First match wins

### Adapter Integration

| Adapter | Delivery Method |
|---------|-----------------|
| Claude Agent SDK | `PostToolUse` hook `additionalContext` |
| OpenAI | Appended to tool result message |

### Built-in Provider: DeadlineFeedback

Reports remaining time until deadline.

```python
config = FeedbackProviderConfig(
    provider=DeadlineFeedback(warning_threshold_seconds=120),
    trigger=FeedbackTrigger(every_n_seconds=30),
)
```

Output varies by remaining time. Below threshold adds suggestions.

### Custom Provider Example

```python
@dataclass(frozen=True)
class ToolUsageMonitor:
    max_calls_without_progress: int = 20

    @property
    def name(self) -> str:
        return "ToolUsageMonitor"

    def should_run(self, *, context: FeedbackContext) -> bool:
        return True

    def provide(self, *, context: FeedbackContext) -> Feedback:
        count = context.tool_call_count
        if count > self.max_calls_without_progress:
            return Feedback(
                provider_name=self.name,
                summary=f"You have made {count} tool calls.",
                suggestions=("Review progress.",),
                severity="caution",
            )
        return Feedback(provider_name=self.name, summary="OK", severity="info")
```

### State Management

| Slice | Purpose |
|-------|---------|
| `ToolInvoked` | Tool invocation log |
| `Feedback` | Feedback history |

Feedback stored via `session.dispatch(feedback)` with `append_all` reducer.

When sessions reused across prompts, feedback/counts scoped to current prompt
via `prompt_name` field.

### Public API

```python
from weakincentives.prompt import (
    DeadlineFeedback,        # Built-in provider
    Feedback,                # Dataclass
    FeedbackContext,         # Context
    FeedbackProvider,        # Protocol
    FeedbackProviderConfig,  # Config
    FeedbackTrigger,         # Trigger
    FileCreatedTrigger,      # File creation trigger
    StaticFeedbackProvider,  # Built-in provider
    collect_feedback,        # Primary entry point
)
```

### Limitations

- **Single provider per check**: First match wins
- **Synchronous**: Providers block tool completion briefly
- **Text-based**: Agent interprets natural language

______________________________________________________________________

## Task Completion Checking

**Implementation:** `src/weakincentives/adapters/claude_agent_sdk/_task_completion.py`

Verify agents complete all assigned tasks before stopping. Critical for ensuring
agents don't prematurely terminate with work incomplete.

### TaskCompletionResult

| Field | Type | Description |
|-------|------|-------------|
| `complete` | `bool` | Whether tasks are complete |
| `feedback` | `str \| None` | Explanation for incomplete |

Factory methods: `TaskCompletionResult.ok()`, `TaskCompletionResult.incomplete(feedback)`

### TaskCompletionContext

| Field | Type | Description |
|-------|------|-------------|
| `session` | `Session` | Session containing state |
| `tentative_output` | `Any` | Output being produced |
| `filesystem` | `Filesystem \| None` | Optional filesystem |
| `adapter` | `ProviderAdapter \| None` | Optional adapter |
| `stop_reason` | `str \| None` | Why agent is stopping |

### TaskCompletionChecker Protocol

```python
class TaskCompletionChecker(Protocol):
    def check(self, context: TaskCompletionContext) -> TaskCompletionResult: ...
```

### Built-in Implementations

#### PlanBasedChecker

Checks session `Plan` state for incomplete steps.

```python
checker = PlanBasedChecker(plan_type=Plan)
```

Returns incomplete if plan steps exist with `status != "done"`.

#### CompositeChecker

Combines multiple checkers with configurable logic.

```python
# All must pass
checker = CompositeChecker(
    checkers=(PlanBasedChecker(plan_type=Plan), FileExistsChecker(("output.txt",))),
    all_must_pass=True,
)

# Any can pass
checker = CompositeChecker(checkers=(...), all_must_pass=False)
```

### Adapter Integration

Configure via `ClaudeAgentSDKClientConfig`:

```python
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        task_completion_checker=PlanBasedChecker(plan_type=Plan),
    ),
)
```

#### Hook Integration

- **PostToolUse Hook (StructuredOutput)**: If incomplete, adds feedback context
- **Stop Hook**: Returns `needsMoreTurns: True` if incomplete
- **Final Verification**: Raises `PromptEvaluationError` if incomplete

### Custom Checker Example

```python
class FileExistsChecker:
    def __init__(self, required_files: tuple[str, ...]) -> None:
        self._required = required_files

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        if context.filesystem is None:
            return TaskCompletionResult.ok()

        missing = [f for f in self._required if not context.filesystem.exists(f)]
        if missing:
            return TaskCompletionResult.incomplete(
                f"Missing required files: {', '.join(missing)}"
            )
        return TaskCompletionResult.ok()
```

### Operational Notes

- **Default disabled**: Must configure checker to enable
- **Budget/deadline bypass**: Skipped when exhausted
- **Feedback truncation**: Plan checker limits to 3 task titles

______________________________________________________________________

## Design Rationale

### Immediate Delivery (Feedback)

No outer workflow; hooks inject into current turn.

### Store If Delivered

Need history for trigger state and debugging.

### First-Match-Wins

Simplicity; order by priority.

### No Escalation

Budget provides backstop; feedback is soft guidance.

### Fail-Closed (Policies)

When uncertain, deny. Agent reasons about why and adjusts.

### Observable and Debuggable

Expose reasoning. Denial feedback enables self-correction.

______________________________________________________________________

## Related Specifications

- `POLICIES_OVER_WORKFLOWS.md` - Design philosophy
- `TOOLS.md` - Tool runtime, planning tools
- `SESSIONS.md` - Session state, snapshots
- `CLAUDE_AGENT_SDK.md` - SDK adapter integration
