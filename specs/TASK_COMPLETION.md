# Task Completion Checking Specification

## Purpose

Verify agents complete all assigned tasks before stopping. Critical for ensuring
agents don't prematurely terminate with work incomplete.

**Implementation:** `src/weakincentives/adapters/claude_agent_sdk/_task_completion.py`

## Core Types

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

## Built-in Implementations

### PlanBasedChecker

Checks session `Plan` state for incomplete steps.

```python
checker = PlanBasedChecker(plan_type=Plan)
```

Returns incomplete if plan steps exist with `status != "done"`.

### CompositeChecker

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

- **PostToolUse Hook (StructuredOutput)**: If incomplete, adds feedback context
- **Stop Hook**: Returns `needsMoreTurns: True` if incomplete
- **Final Verification**: Raises `PromptEvaluationError` if incomplete

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
            return TaskCompletionResult.incomplete(
                f"Missing required files: {', '.join(missing)}"
            )
        return TaskCompletionResult.ok()
```

## Operational Notes

- **Default disabled**: Must configure checker to enable
- **Budget/deadline bypass**: Skipped when exhausted
- **Feedback truncation**: Plan checker limits to 3 task titles

## Related Specifications

- `specs/CLAUDE_AGENT_SDK.md` - Adapter integration
- `specs/TOOLS.md` - Planning tools
