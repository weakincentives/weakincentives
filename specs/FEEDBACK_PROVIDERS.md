# Feedback Providers Specification

Ongoing progress feedback for unattended agents.

**Source:** `src/weakincentives/prompt/feedback.py`

## Principles

- **Non-blocking**: Guidance, not gates
- **Trigger-based**: Run on conditions (N calls, N seconds)
- **Immediate delivery**: Via hook response, not next render

## Configuration

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

```python
class FeedbackProvider(Protocol):
    @property
    def name(self) -> str: ...
    def should_run(self, *, context: FeedbackContext) -> bool: ...
    def provide(self, *, context: FeedbackContext) -> Feedback: ...
```

### Feedback

```python
Feedback(
    provider_name: str,
    summary: str,
    observations: tuple[Observation, ...] = (),
    suggestions: tuple[str, ...] = (),
    severity: Literal["info", "caution", "warning"] = "info",
)
```

### FeedbackTrigger

```python
FeedbackTrigger(
    every_n_calls: int | None = None,
    every_n_seconds: float | None = None,
)
```

## Built-in: DeadlineFeedback

Reports remaining time:

```python
DeadlineFeedback(warning_threshold_seconds=120)
```

Output:
```
[Feedback - Deadline]
You have 8 minutes remaining.
→ Prioritize completing critical remaining work.
```

## Custom Provider

```python
@dataclass(frozen=True)
class ToolUsageMonitor:
    max_calls: int = 20

    @property
    def name(self) -> str: return "ToolUsageMonitor"

    def should_run(self, *, context: FeedbackContext) -> bool: return True

    def provide(self, *, context: FeedbackContext) -> Feedback:
        if context.tool_call_count > self.max_calls:
            return Feedback(
                provider_name=self.name,
                summary=f"You have made {context.tool_call_count} tool calls.",
                suggestions=("Review progress toward goal.",),
                severity="caution",
            )
        return Feedback(provider_name=self.name, summary="Progress check.")
```

## Execution

```
Tool completes → ToolInvoked dispatched → Check triggers →
provider.should_run() → provider.provide() → Store in slice → Inject via hook
```

First matching provider wins.

## Limitations

- Single provider per check
- Synchronous (blocks tool completion briefly)
- Text-based guidance
