# Claude Agent SDK Adapter Specification

> **SDK Version**: `claude-agent-sdk>=0.1.15`

## Purpose

The Claude Agent SDK adapter enables weakincentives prompts to leverage Claude's
full agentic capabilities through the official `claude-agent-sdk` Python package.
The adapter uses SDK hooks to synchronize state between SDK execution and the
weakincentives Session, publishing events while delegating tool execution to
Claude Code's native tools.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ClaudeAgentSDKAdapter                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │   Prompt    │    │              sdk.query() streaming               │   │
│  │  Rendering  │───▶│                                                  │   │
│  └─────────────┘    │  ┌────────────────────────────────────────────┐  │   │
│                     │  │           SDK Agentic Loop                 │  │   │
│  ┌─────────────┐    │  │                                            │  │   │
│  │   Session   │◀───┼──┼──  PreToolUse ──▶ Tool Exec                │  │   │
│  │   (Events)  │    │  │       │               │                    │  │   │
│  │             │◀───┼──┼── PostToolUse ◀───────┤                    │  │   │
│  │             │◀───┼──┼──    Stop    ◀─── End │                    │  │   │
│  └─────────────┘    │  └───────────────────────┼────────────────────┘  │   │
│                     └──────────────────────────┼───────────────────────┘   │
│                                                │                           │
│                     ┌──────────────────────────┴───────────────────────┐   │
│                     │                   Tools                          │   │
│                     ├──────────────────────────────────────────────────┤   │
│                     │  Native (Read, Write, Bash, ...)                 │   │
│                     │       └── Executed by Claude Code CLI            │   │
│                     │                                                  │   │
│                     │  Custom (via MCP Server "wink")                  │   │
│                     │       └── Planning tools, VFS, etc.              │   │
│                     │       └── Bridged via in-process MCP server      │   │
│                     └──────────────────────────────────────────────────┘   │
│                                                                             │
│  Output: PromptResponse[OutputT] with structured output + events published  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## SDK API Selection

The adapter uses `sdk.query()` in **streaming mode** for hook support. The SDK's
`query()` function only initializes hooks when `is_streaming_mode=True`, which
requires passing an `AsyncIterable` prompt instead of a string.

| Feature | `query()` (string) | `query()` (streaming) |
| -------------------- | ------------------ | --------------------- |
| Hooks | ❌ | ✅ |
| Custom Tools via MCP | ✅ | ✅ |
| One-shot queries | ✅ | ✅ |
| Lifecycle management | Automatic | Automatic |

The adapter converts prompts to streaming format:

```python
async def stream_prompt() -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "user",
        "message": {"role": "user", "content": prompt_text},
        "parent_tool_use_id": None,
        "session_id": prompt_name,
    }

messages = [msg async for msg in sdk.query(prompt=stream_prompt(), options=options)]
```

## Hook Integration

### Hook Event Flow

```
Prompt submitted
       │
       ▼
┌──────────────────┐
│ UserPromptSubmit │  (placeholder - no-op currently)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   PreToolUse     │──▶ Check deadline/budget, deny if violated
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Tool Execution   │    (SDK handles natively)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   PostToolUse    │──▶ Publish ToolInvoked event to session bus
└──────────────────┘
       │
       ▼
┌──────────────────┐
│      Stop        │──▶ Record stop reason
└──────────────────┘
```

### HookContext

Shared context passed to all hooks:

```python
class HookContext:
    session: SessionProtocol      # Session for event publishing
    adapter_name: str             # "claude_agent_sdk"
    prompt_name: str              # For event attribution
    deadline: Deadline | None     # For deadline enforcement
    budget_tracker: BudgetTracker | None  # For budget enforcement
    stop_reason: str | None       # Set by Stop hook
```

### PreToolUse Hook

Enforces deadline and budget constraints before tool execution:

```python
async def pre_tool_use_hook(input_data, tool_use_id, sdk_context):
    # Check deadline
    if deadline and deadline.remaining().total_seconds() <= 0:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Deadline exceeded",
            }
        }

    # Check budget
    if budget_tracker and budget_tracker.consumed >= budget.max_total_tokens:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Token budget exhausted",
            }
        }

    return {}  # Allow tool execution
```

### PostToolUse Hook

Records tool execution by publishing `ToolInvoked` events:

```python
async def post_tool_use_hook(input_data, tool_use_id, sdk_context):
    # SDK provides tool_response, not tool_output
    # tool_response has keys: stdout, stderr, interrupted, isImage
    event = ToolInvoked(
        prompt_name=hook_context.prompt_name,
        adapter=hook_context.adapter_name,
        name=input_data.get("tool_name", ""),
        params=input_data.get("tool_input", {}),
        result=input_data.get("tool_response", {}),
        call_id=tool_use_id,
        # ... other fields
    )
    session.event_bus.publish(event)
    return {}
```

### Stop Hook

Records the stop reason for result construction:

```python
async def stop_hook(input_data, tool_use_id, sdk_context):
    hook_context.stop_reason = input_data.get("stopReason", "end_turn")
    return {}
```

## Configuration

### ClaudeAgentSDKClientConfig

```python
@FrozenDataclass()
class ClaudeAgentSDKClientConfig:
    permission_mode: PermissionMode = "bypassPermissions"
    cwd: str | None = None
    max_turns: int | None = None
    suppress_stderr: bool = True
    stop_on_structured_output: bool = True
```

| Field | Default | Description |
| -------------------------- | --------------------- | ----------------------------------------- |
| `permission_mode` | `"bypassPermissions"` | Tool permission handling |
| `cwd` | `None` | Working directory for SDK ops |
| `max_turns` | `None` | Maximum conversation turns |
| `suppress_stderr` | `True` | Suppress CLI stderr (hides bun errors) |
| `stop_on_structured_output` | `True` | End turn after StructuredOutput tool call |

### PermissionMode

```python
PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
```

- `"bypassPermissions"`: Allow all tool use without prompts (recommended for automation)
- `"acceptEdits"`: Auto-accept file edits
- `"plan"`: Planning mode only
- `"default"`: Interactive permission prompts

### Sandboxing

The Claude Agent SDK supports OS-level sandboxing (Linux bubblewrap, macOS seatbelt)
that isolates filesystem and network access. Sandboxing is configured externally
via Claude Code settings, not via the Python SDK options.

**Configuration** (in `~/.claude/settings.json` or project `.claude/settings.json`):

```json
{
  "sandbox": {
    "enabled": true,
    "network": {
      "allowedDomains": []
    }
  }
}
```

**Key settings**:

| Setting | Description |
| ---------------------------------- | ------------------------------------------- |
| `sandbox.enabled` | Enable OS-level sandboxing |
| `sandbox.network.allowedDomains` | Domains accessible (empty = no internet) |
| `sandbox.allowUnixSockets` | Allow Unix socket access (security risk) |
| `sandbox.allowUnsandboxedCommands` | Allow escape hatch for specific commands |
| `sandbox.excludedCommands` | Commands that bypass sandbox (e.g., docker) |

**Autonomous sandbox mode**: For fully autonomous operation within a
network-restricted sandbox:

1. Configure sandbox with `enabled: true` and `allowedDomains: []`
1. Use `permission_mode="bypassPermissions"` in the adapter
1. Claude Code can execute any operation without permission prompts, but cannot
   access the network or files outside the working directory

This reduces permission prompts by ~84% while maintaining security boundaries.

### ClaudeAgentSDKModelConfig

```python
@FrozenDataclass()
class ClaudeAgentSDKModelConfig(LLMConfig):
    model: str = "claude-sonnet-4-5-20250929"
```

Unsupported LLMConfig fields (`seed`, `stop`, `presence_penalty`, `frequency_penalty`)
raise `ValueError` if provided.

## ClaudeAgentWorkspaceSection

A prompt section that manages a temporary workspace directory for SDK operations.
Copies host files into a temp directory and renders workspace information for the
prompt. The SDK's native tools (Read, Write, Edit, Glob, Grep, Bash) operate
directly on this temp directory.

### HostMount

Configuration for mounting host files:

```python
@FrozenDataclass()
class HostMount:
    host_path: str                    # Path on host
    mount_path: str | None = None     # Path in temp dir (default: basename)
    include_glob: tuple[str, ...] = () # Patterns to include
    exclude_glob: tuple[str, ...] = () # Patterns to exclude
    max_bytes: int | None = None       # Byte budget
    follow_symlinks: bool = False
```

### Usage

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    HostMount,
)

# Create workspace section with host mounts
workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=[
        HostMount(
            host_path="src",
            mount_path="project/src",
            exclude_glob=("*.pyc", "__pycache__/*"),
            max_bytes=1_000_000,
        ),
    ],
    allowed_host_roots=["/home/user/myproject"],
)

# Configure adapter to use workspace temp_dir
adapter = ClaudeAgentSDKAdapter(
    model="claude-sonnet-4-5-20250929",
    client_config=ClaudeAgentSDKClientConfig(
        cwd=str(workspace.temp_dir),
    ),
)

# Include workspace in prompt sections
prompt = Prompt[ReviewResult](
    ns="review",
    key="code",
    sections=[
        MarkdownSection(title="Task", key="task", template="Review the code"),
        workspace,
    ],
)

response = adapter.evaluate(prompt, session=session)

# Cleanup when done
workspace.cleanup()
```

### Security

- `allowed_host_roots` restricts which host paths can be mounted
- Paths outside allowed roots raise `WorkspaceSecurityError`
- `max_bytes` prevents excessive copying

## Custom Tool Bridging

Weakincentives tools with handlers are bridged to the SDK via MCP servers:

```python
def create_mcp_server(bridged_tools: tuple[BridgedTool, ...]) -> McpSdkServerConfig:
    """Create MCP server config exposing tools to the SDK."""
```

Tools are registered with the SDK via `mcp_servers` option:

```python
options_kwargs["mcp_servers"] = {
    "wink": create_mcp_server(bridged_tools),
}
```

Each tool handler is wrapped to:

1. Parse arguments via `serde.parse()`
1. Build `ToolContext` with session/deadline/budget
1. Execute handler and call `result.render()` for output
1. Return MCP-format result with rendered text

The bridge uses `ToolResult.render()` to produce output text, mirroring the OpenAI
adapter's behavior. This calls `render_tool_payload()` on the result value, which:

- Invokes the value's `render()` method if defined (for custom formatting)
- Falls back to JSON serialization for dataclasses without `render()`
- Falls back to `result.message` if render returns empty

## Structured Output

The SDK supports JSON schema validation via `output_format`:

```python
def _build_output_format(rendered: RenderedPrompt[OutputT]) -> dict | None:
    if output_type is None or output_type is type(None):
        return None
    return {
        "type": "json_schema",
        "schema": schema(output_type),
    }
```

Output is extracted from `ResultMessage.structured_output` and parsed via
`serde.parse()`.

## Error Handling

SDK exceptions are normalized to weakincentives error types:

| SDK Exception | weakincentives Error |
| ----------------------- | ----------------------- |
| `CLINotFoundError` | `PromptEvaluationError` |
| `CLIConnectionError` | `ThrottleError` |
| `ProcessError` | `PromptEvaluationError` |
| `CLIJSONDecodeError` | `PromptEvaluationError` |
| `MaxTurnsExceededError` | `PromptEvaluationError` |

## Events Published

| Event | When | Data |
| ---------------- | ----------------------- | ------------------------- |
| `PromptRendered` | After prompt render | Rendered text, tools |
| `ToolInvoked` | Each SDK tool execution | Tool name, params, result |
| `PromptExecuted` | After SDK completion | Output, usage, duration |

## File Structure

```
src/weakincentives/adapters/claude_agent_sdk/
├── __init__.py           # Public exports
├── adapter.py            # ClaudeAgentSDKAdapter
├── config.py             # Configuration dataclasses
├── workspace.py          # ClaudeAgentWorkspaceSection, HostMount
├── _hooks.py             # Hook implementations
├── _bridge.py            # MCP tool bridge
├── _async_utils.py       # Async/sync bridging (run_async)
└── _errors.py            # Error normalization
```

## Dependencies

```toml
[project.optional-dependencies]
claude-agent-sdk = ["claude-agent-sdk>=0.1.15"]
```

Requires Claude Code CLI: `npm install -g @anthropic-ai/claude-code`

## Limitations

- **CLI dependency**: Requires Claude Code CLI installation
- **Async overhead**: `asyncio.run()` creates new event loop per call
- **Hook latency**: Each tool call incurs hook overhead
- **No streaming in evaluate()**: Results collected after completion

## Testing

Unit tests mock SDK types to test:

- Hook wiring and responses
- Error normalization
- Tool bridging
- Config validation

Integration tests require Claude Code CLI and test:

- Full prompt evaluation flow
- Structured output parsing
- Tool invocation events
