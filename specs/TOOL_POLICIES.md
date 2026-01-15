# Tool Policies Specification

Enforce sequential dependencies between tool invocations.

**Source:** `src/weakincentives/prompt/policies.py`

## Principles

- **Prompt-scoped declaration**: Policies bound to prompts alongside tools
- **Session-scoped state**: Invocation history in session slices (snapshot/restore)
- **Composable**: Multiple policies govern same tool; all must allow
- **Fail-closed**: Denied calls return error without executing

## Core Types

### ToolPolicy Protocol

```python
class ToolPolicy(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, tool: Tool, params: Any, *, context: ToolContext) -> PolicyDecision: ...
    def on_result(self, tool: Tool, params: Any, result: ToolResult, *, context: ToolContext) -> None: ...
```

### PolicyDecision

```python
@dataclass(slots=True, frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str | None = None

    @classmethod
    def allow(cls) -> PolicyDecision: ...
    @classmethod
    def deny(cls, reason: str) -> PolicyDecision: ...
```

### PolicyState (Session Slice)

```python
@FrozenDataclass()
class PolicyState:
    policy_name: str
    invoked_tools: frozenset[str] = frozenset()
    invoked_keys: frozenset[tuple[str, str]] = frozenset()  # (tool, key) pairs
```

## Built-in Policies

### SequentialDependencyPolicy

Tool B requires tool A to have succeeded:

```python
policy = SequentialDependencyPolicy(
    dependencies={
        "deploy": frozenset({"test", "build"}),
        "build": frozenset({"lint"}),
    }
)
# Required order: lint → build, then test, then deploy
```

### ReadBeforeWritePolicy

File must be read before overwriting (new files can be created freely):

```python
policy = ReadBeforeWritePolicy(
    read_tools=frozenset({"read_file"}),
    write_tools=frozenset({"write_file", "edit_file"}),
)

# write_file(path="new.txt")      → OK (doesn't exist)
# write_file(path="config.yaml")  → DENIED (exists, not read)
# read_file(path="config.yaml")   → OK (records path)
# write_file(path="config.yaml")  → OK (was read)
```

## Prompt Integration

```python
template = PromptTemplate(
    sections=[
        MarkdownSection(
            tools=[read_file, write_file],
            policies=[ReadBeforeWritePolicy()],  # Section-level
        ),
    ],
    policies=[SequentialDependencyPolicy(...)],  # Prompt-level
)
```

## Execution Flow

```python
def execute_tool(call, *, context):
    policies = [*section.policies, *context.prompt.policies]
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

## Built-in Section Defaults

| Section | Default Policy |
|---------|----------------|
| `VFSToolsSection` | `ReadBeforeWritePolicy` |
| `PodmanToolsSection` | `ReadBeforeWritePolicy` |

Override with `policies=()` or custom policies.

## Limitations

- Synchronous policy checks
- Session-scoped only (no cross-session persistence)
- No rollback notification on session restore
