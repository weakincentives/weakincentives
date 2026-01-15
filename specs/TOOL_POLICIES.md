# Tool Policy Specification

## Purpose

Tool policies enforce sequential dependencies between tool invocations. Declares
that tool B requires tool A first—unconditionally or keyed by parameter.

**Implementation:** `src/weakincentives/prompt/policies.py`

## Guiding Principles

- **Prompt-scoped declaration**: Bound to prompts alongside tools
- **Session-scoped state**: Invocation history in session slices
- **Composable**: Multiple policies can govern same tool; all must allow
- **Fail-closed**: Denied calls return error without executing

## Core Types

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

## Built-in Policies

### SequentialDependencyPolicy

Unconditional tool ordering: tool B requires tool A to have succeeded.

```python
policy = SequentialDependencyPolicy(
    dependencies={
        "deploy": frozenset({"test", "build"}),
        "build": frozenset({"lint"}),
    }
)
```

### ReadBeforeWritePolicy

Parameter-keyed dependency for filesystem tools. Existing files must be read
before overwritten. New files can be created freely.

```python
policy = ReadBeforeWritePolicy()
# write_file("new.txt")      → OK (doesn't exist)
# write_file("config.yaml")  → DENIED (exists, not read)
# read_file("config.yaml")   → OK (records path)
# write_file("config.yaml")  → OK (was read)
```

## Prompt Integration

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

## Execution Flow

```python
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

## State Management

- **Snapshot/restore**: State captured with session snapshots
- **Reset**: `session.reset()` clears policy state
- **Isolation**: Each session has independent state

## Limitations

- **Synchronous**: Policy checks run on tool execution thread
- **Session-scoped**: No cross-session persistence
- **No rollback notification**: Policies not notified on restore

## Related Specifications

- `specs/POLICIES_OVER_WORKFLOWS.md` - Design philosophy
- `specs/TOOLS.md` - Tool execution
- `specs/SESSIONS.md` - Session state
