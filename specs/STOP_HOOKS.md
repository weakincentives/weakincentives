# Stop Hooks Specification

## Purpose

Stop hooks validate task completeness before agent termination. This spec
defines a `validate_completion` tool with a fixed contract that the Claude
Agent SDK adapter invokes automatically when the agent attempts to stop.

## Design

A single tool pattern replaces the previous validator abstraction:

```
Agent attempts to stop
        ↓
Adapter stop hook fires
        ↓
Invokes validate_completion tool (if attached)
        ↓
Tool returns ValidationResult
        ↓
complete=true  → allow stop
complete=false → block with reason, retry
```

## Contract

### Input: `ValidationParams`

```python
@dataclass(frozen=True, slots=True)
class ValidationParams:
    """Input to the validate_completion tool."""

    stop_reason: str
    """Reason provided by the agent (e.g., 'end_turn', 'max_turns')."""

    tool_count: int
    """Number of tools invoked during this execution."""

    retry_count: int
    """How many times stop has been blocked (0 on first attempt)."""
```

### Output: `ValidationResult`

```python
@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Output from the validate_completion tool."""

    complete: bool
    """True if the task is considered complete."""

    reason: str
    """Human-readable explanation."""

    def render(self) -> str:
        status = "complete" if self.complete else "incomplete"
        return f"[{status}] {self.reason}"
```

## Tool Definition

```python
from weakincentives.prompt import Tool, ToolContext, ToolResult


# Reserved tool name - adapter recognizes this
VALIDATE_COMPLETION_TOOL_NAME = "validate_completion"


def validate_completion(
    params: ValidationParams,
    *,
    context: ToolContext,
) -> ToolResult[ValidationResult]:
    """Validate whether the current task is complete."""
    # Default implementation - always complete
    return ToolResult(
        message="Validation passed",
        value=ValidationResult(complete=True, reason="No validation configured"),
    )


validate_completion_tool = Tool[ValidationParams, ValidationResult](
    name=VALIDATE_COMPLETION_TOOL_NAME,
    description="Validate task completeness before stopping",
    handler=validate_completion,
)
```

## Workspace Integration

The tool attaches to `ClaudeAgentWorkspaceSection` via the `completion_validator`
parameter:

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentWorkspaceSection,
    HostMount,
)


def my_validator(
    params: ValidationParams,
    *,
    context: ToolContext,
) -> ToolResult[ValidationResult]:
    """Check todos and tests before allowing stop."""
    from weakincentives.contrib.tools import Todo

    todos = context.session[Todo].all()
    incomplete = [t for t in todos if t.status != "completed"]

    if incomplete:
        names = ", ".join(t.content[:30] for t in incomplete[:3])
        return ToolResult(
            message="Incomplete todos",
            value=ValidationResult(
                complete=False,
                reason=f"{len(incomplete)} todos remaining: {names}",
            ),
        )

    return ToolResult(
        message="All complete",
        value=ValidationResult(complete=True, reason="All todos done"),
    )


workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="/path/to/repo", mount_path="repo"),),
    allowed_host_roots=("/path/to",),
    completion_validator=my_validator,  # Attached here
)
```

## Adapter Handling

The Claude Agent SDK adapter's stop hook checks for this tool:

```python
def create_stop_hook(
    hook_context: HookContext,
) -> AsyncHookCallback:
    """Create stop hook with completion validation support."""

    async def stop_hook(
        input_data: Any,
        tool_use_id: str | None,
        sdk_context: Any,
    ) -> dict[str, Any]:
        stop_reason = input_data.get("stopReason", "end_turn")
        hook_context.stop_reason = stop_reason

        # Check for completion validator tool
        validator_tool = hook_context.get_tool(VALIDATE_COMPLETION_TOOL_NAME)
        if validator_tool is None:
            return {}  # No validator, allow stop

        # Prevent infinite loops
        if hook_context.retry_count >= MAX_STOP_RETRIES:
            logger.warning("stop_hook.max_retries_exceeded")
            return {}

        # Invoke the validator
        params = ValidationParams(
            stop_reason=stop_reason,
            tool_count=hook_context._tool_count,
            retry_count=hook_context.retry_count,
        )

        result = validator_tool.handler(params, context=hook_context.tool_context)

        if not result.value.complete:
            hook_context.retry_count += 1
            return {
                "decision": "block",
                "reason": result.value.reason,
            }

        return {}

    return stop_hook
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `completion_validator` | `ToolHandler` | `None` | Custom validation handler |
| `max_stop_retries` | `int` | `3` | Max blocked stops before forcing |

## Common Validators

### Todo Validator

```python
def todo_validator(
    params: ValidationParams,
    *,
    context: ToolContext,
) -> ToolResult[ValidationResult]:
    """Block stop if todos remain incomplete."""
    from weakincentives.contrib.tools import Todo

    todos = context.session[Todo].all()
    incomplete = [t for t in todos if t.status != "completed"]

    if incomplete:
        return ToolResult(
            message="Incomplete",
            value=ValidationResult(
                complete=False,
                reason=f"{len(incomplete)} todos remaining",
            ),
        )

    return ToolResult(
        message="Complete",
        value=ValidationResult(complete=True, reason="All todos done"),
    )
```

### Composite Validator

```python
def composite_validator(
    *validators: ToolHandler[ValidationParams, ValidationResult],
) -> ToolHandler[ValidationParams, ValidationResult]:
    """Combine multiple validators with AND logic."""

    def combined(
        params: ValidationParams,
        *,
        context: ToolContext,
    ) -> ToolResult[ValidationResult]:
        for validator in validators:
            result = validator(params, context=context)
            if not result.value.complete:
                return result
        return ToolResult(
            message="All checks passed",
            value=ValidationResult(complete=True, reason="All validators passed"),
        )

    return combined
```

## Hook Taxonomy

For reference, Claude Code provides these hook types:

| Hook | When | Can Block | Use Case |
|------|------|-----------|----------|
| UserPromptSubmit | Before input processing | No | Context injection |
| **Stop** | Before termination | **Yes** | Completion validation |
| PreToolUse | Before tool execution | Yes | Constraint enforcement |
| PostToolUse | After tool execution | Yes | State rollback |
| SubagentStart | Subagent spawns | No | Tracking |
| SubagentStop | Subagent completes | No | Transcript capture |
| PreCompact | Before compaction | No | Logging |
| Notification | User notifications | No | Observability |

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Validator raises exception | Log warning, allow stop |
| Validator returns `complete=True` | Allow stop |
| Validator returns `complete=False` | Block with reason |
| `retry_count >= max_stop_retries` | Force stop, log warning |
| No validator attached | Allow stop (default) |

## Limitations

- Single validator per workspace (use `composite_validator` to combine)
- Validators are synchronous
- Cannot access raw LLM response content
- Read-only session access recommended
