# Feedback Providers

*Canonical spec: [specs/FEEDBACK_PROVIDERS.md](../specs/FEEDBACK_PROVIDERS.md)*

Feedback providers deliver periodic guidance to agents during long-running tasks.
Unlike hard constraints (budgets, deadlines), feedback is soft guidance—the agent
decides how to respond.

## When Feedback Helps

Agents working on multi-step tasks can lose track of time, repeat unsuccessful
patterns, or miss important context changes. Feedback providers address this by:

- Alerting agents to remaining time before deadlines
- Noting patterns in tool usage (too many calls, repeated failures)
- Surfacing context changes the agent might have missed

The key insight: **feedback is not control**. The agent receives information and
decides what to do with it. This respects agent autonomy while providing useful
signals.

## Core Types

### Feedback

The `Feedback` dataclass carries structured information from providers to agents:

```python nocheck
from weakincentives.prompt import Feedback, Observation

feedback = Feedback(
    provider_name="DeadlineFeedback",
    summary="You have 5 minutes remaining to complete the task.",
    observations=(
        Observation(label="elapsed", value="25 minutes"),
        Observation(label="remaining", value="5 minutes"),
    ),
    suggestions=("Consider wrapping up current work.",),
    severity="warning",  # "info", "caution", or "warning"
)
```

**Fields:**

- `provider_name`: Identifies the source
- `summary`: Main message (required)
- `observations`: Supporting evidence as label-value pairs
- `suggestions`: Actionable recommendations
- `severity`: Urgency level ("info", "caution", "warning")

### FeedbackTrigger

Triggers control when feedback providers run:

```python nocheck
from weakincentives.prompt import FeedbackTrigger

# Run every 5 tool calls OR every 30 seconds (whichever comes first)
trigger = FeedbackTrigger(every_n_calls=5, every_n_seconds=30)
```

**Trigger conditions are OR'd**—either condition can fire the trigger. This
ensures time-based feedback works even when tool calls are infrequent.

### FeedbackProviderConfig

Pairs a provider with its trigger:

```python nocheck
from weakincentives.prompt import FeedbackProviderConfig, DeadlineFeedback

config = FeedbackProviderConfig(
    provider=DeadlineFeedback(warning_threshold_seconds=120),
    trigger=FeedbackTrigger(every_n_seconds=60, every_n_calls=10),
)
```

## Built-in Provider: DeadlineFeedback

`DeadlineFeedback` reports remaining time with escalating urgency:

```python nocheck
from weakincentives.prompt import DeadlineFeedback

provider = DeadlineFeedback(warning_threshold_seconds=120)  # 2 minutes
```

**Behavior:**

- Returns "info" severity when time is plentiful
- Escalates to "warning" with suggestions when below threshold
- Provides special messaging when deadline has passed

**Example output:**

```
[Feedback - Deadline]

The work so far took 12 minutes. You have 8 minutes remaining
to complete the task.
```

## Configuring Feedback on Prompts

Attach feedback providers to `PromptTemplate`:

```python nocheck
from weakincentives.prompt import (
    PromptTemplate,
    FeedbackProviderConfig,
    FeedbackTrigger,
    DeadlineFeedback,
)

template = PromptTemplate[MyOutput](
    ns="my-namespace",
    key="my-prompt",
    sections=(task_section, tools_section),
    feedback_providers=(
        FeedbackProviderConfig(
            provider=DeadlineFeedback(warning_threshold_seconds=60),
            trigger=FeedbackTrigger(every_n_seconds=30, every_n_calls=5),
        ),
    ),
)
```

## Writing Custom Providers

Implement the `FeedbackProvider` protocol:

```python nocheck
from dataclasses import dataclass
from weakincentives.prompt import Feedback, FeedbackContext, FeedbackProvider


@dataclass(frozen=True)
class ToolUsageMonitor:
    """Alerts when tool call count exceeds threshold."""

    max_calls_without_progress: int = 20

    @property
    def name(self) -> str:
        return "ToolUsageMonitor"

    def should_run(self, *, context: FeedbackContext) -> bool:
        # Additional filtering beyond trigger conditions
        return True

    def provide(self, *, context: FeedbackContext) -> Feedback:
        count = context.tool_call_count
        if count > self.max_calls_without_progress:
            return Feedback(
                provider_name=self.name,
                summary=f"You have made {count} tool calls without completing.",
                suggestions=(
                    "Review your approach.",
                    "Consider whether you're making progress.",
                ),
                severity="caution",
            )
        return Feedback(
            provider_name=self.name,
            summary=f"Tool calls: {count}",
            severity="info",
        )
```

### FeedbackContext

Providers receive `FeedbackContext` with access to:

- `session`: Current session state
- `prompt`: The prompt being executed
- `deadline`: Optional deadline (if configured)
- `last_feedback`: Most recent feedback for this prompt
- `tool_call_count`: Total tool calls for this prompt
- `tool_calls_since_last_feedback()`: Calls since last feedback
- `recent_tool_calls(n)`: Last N tool invocations

## Execution Flow

1. Tool completes execution
1. `ToolInvoked` event dispatched to session
1. Trigger conditions evaluated
1. `provider.should_run()` called for additional filtering
1. `provider.provide()` called if both conditions pass
1. Feedback stored in session and returned to adapter
1. Adapter delivers feedback to agent

**First matching provider wins**—subsequent providers are skipped for that tool
call. This prevents feedback overload.

## Adapter Integration

### Claude Agent SDK

Feedback is delivered via the `PostToolUse` hook's `additionalContext` field.
The agent sees it immediately after the tool result.

### OpenAI/LiteLLM Adapters

Feedback is appended to the tool result message, becoming part of the response
the model sees.

## Prompt Scoping

Feedback is scoped to the prompt that triggered it:

- `prompt_name` filters feedback and tool calls to the current prompt
- Enables safe session reuse across multiple prompt evaluations
- Each prompt evaluation gets independent feedback tracking

This means trigger counters (calls since last feedback, time since last
feedback) reset when switching prompts.

## Design Philosophy

Feedback embodies the "weak incentives" approach:

1. **Soft guidance**: Feedback informs but doesn't force
1. **Agent autonomy**: The agent decides how to respond
1. **Observable**: All feedback is stored in the session for debugging
1. **Non-blocking**: Feedback never gates tool execution

The agent might ignore feedback, incorporate it immediately, or factor it into
future decisions. This flexibility is intentional—different tasks and agents
benefit from different response patterns.

## Next Steps

- [Sessions](sessions.md): Where feedback is stored and tracked
- [Lifecycle](lifecycle.md): Deadlines and budgets that feedback monitors
- [Claude Agent SDK](claude-agent-sdk.md): Production integration with feedback
