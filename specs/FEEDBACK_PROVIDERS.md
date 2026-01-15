# Feedback Providers Specification

## Purpose

Deliver ongoing progress feedback to agents during unattended execution. Analyze
patterns over time and inject guidance into context for soft course-correction.

**Implementation:** `src/weakincentives/prompt/feedback.py`

## Overview

Unlike tool policies that gate calls, feedback providers observe trajectory and
produce contextual feedback delivered immediately after tool execution.

**Key characteristics:**
- **Non-blocking**: Guidance, not gates; agent decides response
- **Trigger-based**: Run when conditions met (every N calls/seconds)
- **Immediate delivery**: Inject via hook response, not next render

## Configuration

### FeedbackTrigger

| Field | Type | Description |
|-------|------|-------------|
| `every_n_calls` | `int \| None` | Run after N tool calls |
| `every_n_seconds` | `float \| None` | Run after N seconds elapsed |

Conditions are OR'd together.

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

## Core Types

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

## Execution Flow

1. Tool call completes
2. `ToolInvoked` dispatched
3. Check trigger conditions
4. Call `provider.should_run()`
5. Call `provider.provide()`
6. Store in session, return text
7. First match wins

## Adapter Integration

| Adapter | Delivery Method |
|---------|-----------------|
| Claude Agent SDK | `PostToolUse` hook `additionalContext` |
| OpenAI | Appended to tool result message |

## Built-in Provider: DeadlineFeedback

Reports remaining time until deadline.

```python
config = FeedbackProviderConfig(
    provider=DeadlineFeedback(warning_threshold_seconds=120),
    trigger=FeedbackTrigger(every_n_seconds=30),
)
```

Output varies by remaining time. Below threshold adds suggestions.

## Writing Custom Providers

Implement `FeedbackProvider` protocol:

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

## State Management

| Slice | Purpose |
|-------|---------|
| `ToolInvoked` | Tool invocation log |
| `Feedback` | Feedback history |

Feedback stored via `session.dispatch(feedback)` with `append_all` reducer.

### Prompt Scoping

When sessions reused across prompts, feedback/counts scoped to current prompt
via `prompt_name` field.

## Public API

```python
from weakincentives.prompt import (
    DeadlineFeedback,        # Built-in
    Feedback,                # Dataclass
    FeedbackContext,         # Context
    FeedbackProvider,        # Protocol
    FeedbackProviderConfig,  # Config
    FeedbackTrigger,         # Trigger
    collect_feedback,        # Primary entry point
)
```

## Design Rationale

- **Immediate delivery**: No outer workflow; hooks inject into current turn
- **Store if delivered**: Need history for trigger state, debugging
- **First-match-wins**: Simplicity; order by priority
- **No escalation**: Budget provides backstop; feedback is soft guidance

## Limitations

- **Single provider per check**: First match wins
- **Synchronous**: Providers block tool completion briefly
- **Text-based**: Agent interprets natural language
