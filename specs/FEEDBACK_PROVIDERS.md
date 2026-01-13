# Feedback Providers

Feedback providers deliver ongoing progress feedback to agents during unattended
execution. They analyze patterns over time and inject guidance into the agent's
context, enabling soft course-correction without hard intervention.

## Overview

Unlike tool policies that gate individual calls, feedback providers observe the
agent's trajectory and produce contextual feedback. This feedback is delivered
immediately after tool execution via the adapter's hook mechanism.

Key characteristics:

- **Non-blocking**: Providers produce guidance, not gates. The agent decides how
  to respond.
- **Trigger-based**: Providers run only when configured conditions are met
  (e.g., every N calls or every N seconds).
- **Immediate delivery**: Feedback injects into the current turn via hook
  response, not the next prompt render.

## Configuration

Feedback providers are declared on the prompt template:

```python
from weakincentives.prompt import (
    DeadlineFeedback,
    FeedbackProviderConfig,
    FeedbackTrigger,
    PromptTemplate,
)

template = PromptTemplate[OutputType](
    ns="my-agent",
    key="main",
    sections=[...],
    feedback_providers=(
        FeedbackProviderConfig(
            provider=DeadlineFeedback(),
            trigger=FeedbackTrigger(every_n_seconds=30),
        ),
    ),
)
```

### FeedbackTrigger

Triggers control when a provider is evaluated. Conditions are OR'd together.

```python
@dataclass(frozen=True)
class FeedbackTrigger:
    every_n_calls: int | None = None      # Run after N tool calls
    every_n_seconds: float | None = None  # Run after N seconds elapsed
```

If `every_n_calls=10`, the provider runs after 10 tool calls since the last
feedback. If `every_n_seconds=30`, it runs 30 seconds after the last feedback
(or immediately if no prior feedback exists).

## Core Types

### FeedbackProvider Protocol

```python
class FeedbackProvider(Protocol):
    @property
    def name(self) -> str:
        """Unique identifier for this provider."""
        ...

    def should_run(self, *, context: FeedbackContext) -> bool:
        """Additional filtering beyond trigger conditions."""
        ...

    def provide(self, *, context: FeedbackContext) -> Feedback:
        """Produce feedback for context injection."""
        ...
```

Access session state via `context.session` for consistency with the `ToolContext`
pattern used elsewhere in the library.

The `should_run` method provides additional filtering after trigger conditions
are met. For example, `DeadlineFeedback.should_run` returns `False` when no
deadline is configured.

### Feedback

```python
@dataclass(frozen=True)
class Feedback:
    provider_name: str
    summary: str
    observations: tuple[Observation, ...] = ()
    suggestions: tuple[str, ...] = ()
    severity: Literal["info", "caution", "warning"] = "info"
    timestamp: datetime = field(default_factory=_utcnow)
    call_index: int = 0    # Set by runner after provide()
    prompt_name: str = ""  # Set by runner after provide()
```

The `render()` method produces text for injection:

```
[Feedback - Deadline]

You have 8 minutes remaining.

→ Prioritize completing critical remaining work.
```

### FeedbackContext

Provides access to session state and prompt resources:

```python
@dataclass(frozen=True)
class FeedbackContext:
    session: SessionProtocol
    prompt: PromptProtocol[Any]
    deadline: Deadline | None = None

    @property
    def prompt_name(self) -> str: ...
    @property
    def resources(self) -> PromptResources: ...
    @property
    def filesystem(self) -> Filesystem | None: ...
    @property
    def last_feedback(self) -> Feedback | None: ...
    @property
    def tool_call_count(self) -> int: ...

    def tool_calls_since_last_feedback(self) -> int: ...
    def recent_tool_calls(self, n: int) -> Sequence[ToolInvoked]: ...
```

The `last_feedback`, `tool_call_count`, `tool_calls_since_last_feedback()`, and
`recent_tool_calls()` methods are scoped to the current prompt (via `prompt_name`)
to ensure consistent behavior when sessions are reused across prompts.

## Execution Flow

```
Tool call completes
        │
        ▼
ToolInvoked dispatched to session
        │
        ▼
For each FeedbackProviderConfig:
    ├─ Check trigger conditions (every_n_calls, every_n_seconds)
    │   └─ If not met → skip
    ├─ Call provider.should_run()
    │   └─ If False → skip
    ├─ Call provider.provide()
    ├─ Store Feedback in session slice
    └─ Return rendered text (first match wins)
        │
        ▼
Inject feedback via adapter mechanism
```

First matching provider wins. Subsequent providers are not evaluated.

## Adapter Integration

### Claude Agent SDK

Feedback is returned in the `PostToolUse` hook via `additionalContext`:

```python
return {
    "hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "additionalContext": feedback_text,
    }
}
```

This mirrors how task completion feedback is delivered.

### OpenAI Adapter

Feedback is appended to the tool result message:

```python
tool_result = replace(
    tool_result,
    message=f"{tool_result.message}\n\n{feedback_text}",
)
```

## Built-in Provider: DeadlineFeedback

Reports remaining time until deadline.

```python
from weakincentives.prompt import (
    DeadlineFeedback,
    FeedbackProviderConfig,
    FeedbackTrigger,
)

config = FeedbackProviderConfig(
    provider=DeadlineFeedback(warning_threshold_seconds=120),
    trigger=FeedbackTrigger(every_n_seconds=30),
)
```

Output varies by remaining time:

```
[Feedback - Deadline]

You have 8 minutes remaining.
```

When remaining time drops below `warning_threshold_seconds`:

```
[Feedback - Deadline]

You have 90 seconds remaining.

→ Prioritize completing critical remaining work.
→ Consider summarizing progress and remaining tasks.
```

## Writing Custom Providers

Implement the `FeedbackProvider` protocol:

```python
from dataclasses import dataclass
from weakincentives.prompt import Feedback, FeedbackContext


@dataclass(frozen=True)
class ToolUsageMonitor:
    """Warn when tool usage is high without apparent progress."""

    max_calls_without_progress: int = 20

    @property
    def name(self) -> str:
        return "ToolUsageMonitor"

    def should_run(self, *, context: FeedbackContext) -> bool:
        return True  # Always run when triggered

    def provide(self, *, context: FeedbackContext) -> Feedback:
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
```

Register with a trigger:

```python
config = FeedbackProviderConfig(
    provider=ToolUsageMonitor(max_calls_without_progress=15),
    trigger=FeedbackTrigger(every_n_calls=10),
)
```

## State Management

| Slice | Purpose |
|-------|---------|
| `ToolInvoked` | Tool invocation log (existing) |
| `Feedback` | Feedback history for trigger calculations |

The `Feedback` slice stores all produced feedback:

```python
session[Feedback].all()      # All feedback in session
session[Feedback].latest()   # Most recent feedback
```

Feedback is stored via `session.dispatch(feedback)` using the default `append_all`
reducer. This follows the standard session dispatch pattern and allows external
handlers to observe feedback being produced if needed.

### Prompt Scoping

When sessions are reused across multiple prompt evaluations, feedback history
and tool call counts are scoped to the current prompt. Each `Feedback` object
includes a `prompt_name` field that the runner sets automatically. The
`FeedbackContext` filters by this field when calculating:

- `last_feedback` - Most recent feedback for the current prompt
- `tool_call_count` - Tool calls matching the current prompt
- `tool_calls_since_last_feedback()` - Calls since last feedback for this prompt

This ensures triggers behave consistently regardless of prior activity in the
session.

This history enables:

- Trigger calculations (`tool_calls_since_last_feedback`)
- Debugging and observability

## Design Rationale

### Why immediate delivery?

WINK uses a single continuous prompt. There is no outer workflow to inject
context between turns. Immediate delivery via hook response ensures feedback
reaches the agent without waiting for prompt re-rendering.

### Why store feedback if delivered immediately?

1. **Trigger state**: Need to know when last feedback occurred
1. **Debugging**: Feedback history aids troubleshooting

### Why first-match-wins?

Simplicity. Multiple feedback messages in one turn could overwhelm the agent.
Order providers by priority in the configuration.

### Why no escalation?

Budget exhaustion provides the backstop for unattended agents. Feedback provides
soft guidance; budget limits provide hard stops.

## Limitations

- **Single provider per check**: First matching provider wins
- **Synchronous**: Providers block tool completion briefly
- **Text-based**: Agent interprets natural language guidance

## Public API

Exported from `weakincentives.prompt`:

```python
from weakincentives.prompt import (
    DeadlineFeedback,        # Built-in provider
    Feedback,                # Feedback dataclass
    FeedbackContext,         # Context for providers
    FeedbackProvider,        # Protocol
    FeedbackProviderConfig,  # Configuration wrapper
    FeedbackTrigger,         # Trigger conditions
    Observation,             # Observation dataclass
    collect_feedback,        # Primary entry point for running providers
    run_feedback_providers,  # Lower-level runner (requires FeedbackContext)
)
```

### Entry Points

Use `collect_feedback` as the primary entry point:

```python
feedback_text = collect_feedback(
    prompt=prompt,
    session=session,
    deadline=deadline,  # Optional
)
if feedback_text:
    # Inject into agent context
    pass
```

The lower-level `run_feedback_providers` is available when you need to construct
the `FeedbackContext` yourself (e.g., for testing or custom context injection).
