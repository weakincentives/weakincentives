# Task Completion Specification

Verifies agent task completion before allowing stop or final output.

**Source:** `src/weakincentives/adapters/claude_agent_sdk/_task_completion.py`

## Principles

- **Protocol-based**: Custom implementations without inheritance
- **Context-rich**: Access to session, output, filesystem, adapter
- **Composable**: Multiple checkers combine with configurable logic
- **Feedback-oriented**: Results explain why tasks are incomplete

## Core Types

### TaskCompletionResult

```python
@FrozenDataclass()
class TaskCompletionResult:
    complete: bool
    feedback: str | None = None

    @classmethod
    def ok(cls, feedback: str | None = None) -> TaskCompletionResult: ...

    @classmethod
    def incomplete(cls, feedback: str) -> TaskCompletionResult: ...
```

### TaskCompletionContext

```python
@dataclass(slots=True)
class TaskCompletionContext:
    session: Session
    tentative_output: Any = None
    filesystem: Filesystem | None = None
    adapter: ProviderAdapter | None = None
    stop_reason: str | None = None
```

### TaskCompletionChecker Protocol

```python
class TaskCompletionChecker(Protocol):
    def check(self, context: TaskCompletionContext) -> TaskCompletionResult: ...
```

## Built-in Checkers

### PlanBasedChecker

Checks session `Plan` state for incomplete steps:

```python
checker = PlanBasedChecker(plan_type=Plan)
# Returns ok() if no plan, or all steps status == "done"
# Returns incomplete() with feedback listing incomplete tasks
```

### CompositeChecker

```python
CompositeChecker(
    checkers=(PlanBasedChecker(plan_type=Plan), TestPassingChecker()),
    all_must_pass=True,  # All must pass (default) or any can pass
)
```

## Claude Agent SDK Integration

Configure via `ClaudeAgentSDKClientConfig`:

```python
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        task_completion_checker=PlanBasedChecker(plan_type=Plan),
    ),
)
```

### Hook Integration

1. **PostToolUse (StructuredOutput)**: If incomplete, adds feedback to encourage continuation
2. **Stop Hook**: If incomplete, returns `needsMoreTurns: True`
3. **Final Verification**: Raises `PromptEvaluationError` if incomplete after SDK completes

### Bypass Conditions

Task completion checking is skipped when budget exhausted or deadline expired.

## Custom Checker Example

```python
class FileExistsChecker:
    def __init__(self, required_files: tuple[str, ...]) -> None:
        self._required = required_files

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        if context.filesystem is None:
            return TaskCompletionResult.ok()
        missing = [f for f in self._required if not context.filesystem.exists(f)]
        if missing:
            return TaskCompletionResult.incomplete(f"Missing: {', '.join(missing)}")
        return TaskCompletionResult.ok()
```

## Limitations

- Disabled by default; must configure explicitly
- Synchronous checking
- Feedback truncated to 3 task items
