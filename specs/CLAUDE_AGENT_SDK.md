# Claude Agent SDK Adapter Specification

> **SDK**: `claude-agent-sdk>=0.1.15`

## Purpose

`ClaudeAgentSDKAdapter` evaluates prompts via the Claude Agent SDK (Claude Code
CLI) while maintaining orchestration state in a Session. Provides Claude Code
native tools, MCP tool bridging, structured output, and optional isolation.

**Implementation:** `src/weakincentives/adapters/claude_agent_sdk/`

## Requirements

- Python: `pip install 'weakincentives[claude-agent-sdk]'`
- Claude Code CLI: `npm install -g @anthropic-ai/claude-code`
- Linux sandboxing: bubblewrap (`bwrap`) on PATH

## Configuration

### ClaudeAgentSDKClientConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `permission_mode` | `PermissionMode` | `"bypassPermissions"` | Permission handling |
| `cwd` | `str \| None` | `None` | Working directory |
| `max_turns` | `int \| None` | `None` | Maximum conversation turns |
| `max_budget_usd` | `float \| None` | `None` | Maximum budget in USD |
| `suppress_stderr` | `bool` | `True` | Suppress stderr output |
| `stop_on_structured_output` | `bool` | `True` | Stop after structured output |
| `isolation` | `IsolationConfig \| None` | `None` | Isolation configuration |
| `betas` | `tuple[str, ...] \| None` | `None` | Beta features to enable |
| `transcript_collection` | `TranscriptCollectorConfig \| None` | `TranscriptCollectorConfig()` | Transcript collection config (enabled by default) |

At `src/weakincentives/adapters/claude_agent_sdk/config.py`:
`ClaudeAgentSDKClientConfig`.

### ClaudeAgentSDKModelConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"claude-opus-4-6"` | Claude model identifier |
| `reasoning` | `ReasoningEffort \| None` | `"high"` | Adaptive reasoning effort level |

`ReasoningEffort = Literal["low", "medium", "high", "max"]`. Default `"high"` enables
deep reasoning. `"max"` removes all thinking constraints (Opus 4.6 only). `None`
disables reasoning entirely.

At `src/weakincentives/adapters/claude_agent_sdk/config.py`:
`ClaudeAgentSDKModelConfig`, `ReasoningEffort`.

**Note:** `seed`, `stop`, `presence_penalty`, `frequency_penalty` not supported—raises `ValueError`.

### IsolationConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `network_policy` | `NetworkPolicy \| None` | `None` | Tool network restrictions |
| `sandbox` | `SandboxConfig \| None` | `None` | Sandbox configuration |
| `env` | `Mapping[str, str] \| None` | `None` | Environment variables |
| `api_key` | `str \| None` | `None` | Explicit API key (disables Bedrock) |
| `aws_config_path` | `Path \| str \| None` | `None` | AWS config path for Docker |
| `include_host_env` | `bool` | `False` | Copy non-sensitive vars |

**Note:** Skills are attached to prompt sections, not IsolationConfig.
See `specs/SKILLS.md` for skill attachment.

**Authentication modes:**

1. **Inherit host auth** (default): When `api_key` is `None`, inherits authentication
   from the host environment. Works with both Anthropic API (via `ANTHROPIC_API_KEY`)
   and AWS Bedrock (via `CLAUDE_CODE_USE_BEDROCK=1` + AWS credentials).

1. **Explicit API key**: When `api_key` is set, uses that key with the Anthropic API
   and disables Bedrock.

**Factory methods** (recommended for explicit intent and fail-fast validation):

| Factory | Description |
|---------|-------------|
| `IsolationConfig.inherit_host_auth()` | Inherit auth, fail if none configured |
| `IsolationConfig.with_api_key(key)` | Use explicit API key |
| `IsolationConfig.for_anthropic_api()` | Require Anthropic API key from env |
| `IsolationConfig.for_bedrock()` | Require Bedrock, fail if not configured |

**Docker support:** When running in a container, use `aws_config_path` to specify
where AWS credentials are mounted (e.g., `/mnt/aws` instead of `~/.aws`).

### NetworkPolicy

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allowed_domains` | `tuple[str, ...]` | `()` | Domains tools can access |
| `allow_unix_sockets` | `tuple[str, ...]` | `()` | Unix socket paths (macOS) |
| `allow_all_unix_sockets` | `bool` | `False` | Allow all Unix sockets |
| `allow_local_binding` | `bool` | `False` | Allow localhost binding (macOS) |
| `http_proxy_port` | `int \| None` | `None` | Custom HTTP proxy port |
| `socks_proxy_port` | `int \| None` | `None` | Custom SOCKS5 proxy port |

| Factory | Description |
|---------|-------------|
| `NetworkPolicy.no_network()` | No tool network access |
| `NetworkPolicy.with_domains(*domains)` | Allow specific domains |

### SandboxConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable sandboxing |
| `writable_paths` | `tuple[str, ...]` | `()` | Writable paths |
| `readable_paths` | `tuple[str, ...]` | `()` | Readable paths |
| `excluded_commands` | `tuple[str, ...]` | `()` | Excluded commands |
| `allow_unsandboxed_commands` | `bool` | `False` | Allow unsandboxed |
| `bash_auto_allow` | `bool` | `True` | Auto-allow bash |
| `enable_weaker_nested_sandbox` | `bool` | `False` | Weaker sandbox for Docker (Linux) |
| `ignore_file_violations` | `tuple[str, ...]` | `()` | File paths to ignore violations |
| `ignore_network_violations` | `tuple[str, ...]` | `()` | Network hosts to ignore violations |

## Isolation Behavior

Isolation is **always active**. When `isolation` is `None` on the client config,
the adapter creates a default `IsolationConfig()` — it does not run against the
host environment. The default config enables the sandbox and blocks all tool
network access.

For any `IsolationConfig` (explicit or default):

- Creates ephemeral `HOME` with `$HOME/.claude/settings.json`
- Applies `SandboxConfig()` defaults (enabled) when `sandbox` is `None`
- Applies `NetworkPolicy.no_network()` when `network_policy` is `None`
- Passes `env` and `setting_sources=["user"]` to SDK

**Notes:**

- `NetworkPolicy` restricts tools only, not API connection
- Without `api_key`, `ANTHROPIC_API_KEY` from process env is forwarded

## Tool Bridging (MCP)

Tools attached to sections are exposed to Claude Code as MCP tools under server
key `"wink"`. Tool calls publish `ToolInvoked` events with `call_id` correlation.

### call_id Correlation

The MCP protocol does not pass `tool_use_id` to tool handlers. To correlate
`ToolInvoked` events with SDK tool calls, `MCPToolExecutionState` maintains a
thread-safe mapping of `(tool_name, params_hash)` to `tool_use_id` queues:

- **PreToolUse hook** enqueues `tool_use_id` keyed by tool name + MD5 hash of
  parameters (via `_hash_params`)
- **BridgedTool handler** dequeues the matching `tool_use_id` when executing
- Different tools or same tool with different params use different keys
- Same tool with identical params uses a bounded FIFO `deque` per key

At `src/weakincentives/adapters/_shared/_bridge.py`: `MCPToolExecutionState`.

## Skill Mounting

Skills attached to sections are collected during prompt rendering and
automatically mounted to `$HOME/.claude/skills/` in the ephemeral home.
Skills follow the same visibility rules as tools—sections with `SUMMARY`
visibility do not contribute their skills until expanded.

See `specs/SKILLS.md` for skill attachment and format details.

## Events

| Event | When |
|-------|------|
| `PromptRendered` | After render (carries `event_id` as UUID4) |
| `RenderedTools` | Alongside `PromptRendered`, correlated via `render_event_id` |
| `ToolInvoked` | Each tool call (native + bridged), with `call_id` |
| `PromptExecuted` | Completion (includes `TokenUsage`) |

`RenderedTools` captures tool schemas (name, description, JSON Schema parameters)
at render time. The `render_event_id` field links back to the `PromptRendered`
event's `event_id` (UUID4) for correlation.

At `src/weakincentives/runtime/session/rendered_tools.py`: `RenderedTools`, `ToolSchema`.

## Hooks

The adapter registers six hook types with the SDK. All hooks use SDK-native
input/output types for type-safe integration.

At `src/weakincentives/adapters/claude_agent_sdk/_hooks.py`.

### Hook Types

| Hook | SDK Input Type | Purpose |
|------|---------------|---------|
| `PreToolUse` | `PreToolUseHookInput` | Constraint enforcement, state snapshots |
| `PostToolUse` | `PostToolUseHookInput` | Tool result recording, state rollback |
| `Stop` | `StopHookInput` | Execution finalization, task completion |
| `UserPromptSubmit` | `UserPromptSubmitHookInput` | Turn boundary tracking |
| `SubagentStop` | `SubagentStopHookInput` | Subagent completion tracking |
| `PreCompact` | `PreCompactHookInput` | Context compaction tracking |

Return type for all hooks: `SyncHookJSONOutput`. PreToolUse uses
`PreToolUseHookSpecificOutput` for deny decisions. PostToolUse uses
`PostToolUseHookSpecificOutput` for feedback injection.

**Removed hooks:** `create_subagent_start_hook` and `create_notification_hook`
are not available in the Python SDK.

### HookConstraints

Groups optional parameters passed to hooks via `HookContext`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deadline` | `Deadline \| None` | `None` | Deadline for constraint checking |
| `budget_tracker` | `BudgetTracker \| None` | `None` | Token budget limits |
| `heartbeat` | `Heartbeat \| None` | `None` | Liveness monitoring |
| `run_context` | `RunContext \| None` | `None` | Tracing context |
| `mcp_tool_state` | `MCPToolExecutionState \| None` | `None` | MCP tool_use_id correlation |

## Error Handling

SDK exceptions are normalized to `PromptEvaluationError` via `normalize_sdk_error`.
Uses `isinstance()` checks against SDK-native exception types.

At `src/weakincentives/adapters/claude_agent_sdk/_errors.py`.

| SDK Exception | Normalized To | Description |
|---------------|---------------|-------------|
| `CLINotFoundError` | `PromptEvaluationError` | Claude Code CLI not installed |
| `CLIConnectionError` | `ThrottleError` | Connection/timeout issues (retryable) |
| `ProcessError` | `PromptEvaluationError` | CLI process failure (includes exit code) |
| `CLIJSONDecodeError` | `PromptEvaluationError` | Malformed SDK response |
| `ExceptionGroup` | `PromptEvaluationError` | TaskGroup cleanup errors |

## Usage Patterns

### Structured Output

```python
template = PromptTemplate[Hello](
    ns="demo", key="hello",
    sections=[MarkdownSection(title="Task", key="task", template="Say hello.")],
)
response = ClaudeAgentSDKAdapter().evaluate(Prompt(template), session=session)
```

### Secure Code Review (No Tool Network)

```python
workspace = WorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="/abs/path/to/repo", mount_path="repo"),),
    allowed_host_roots=("/abs/path/to",),
)

adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        cwd=str(workspace.temp_dir),
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            sandbox=SandboxConfig(readable_paths=(str(workspace.temp_dir),)),
        ),
    ),
)
```

### Domain Allowlist

```python
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.with_domains("docs.python.org", "pypi.org"),
            sandbox=SandboxConfig(enabled=True),
        ),
    ),
)
```

### AWS Bedrock Authentication

When Bedrock is configured, isolation automatically inherits authentication.
Auth detection checks both shell environment and host `~/.claude/settings.json`:

**Priority order for auth vars:**

1. Shell environment variables (highest priority)
1. Host `~/.claude/settings.json` env section (fallback)

This ensures that if `claude` works on the host, WINK agents will too.

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
)

# Inherits host auth (works with Bedrock or Anthropic API)
# Uses AWS credential chain: env vars, ~/.aws/credentials, instance profile, etc.
adapter = ClaudeAgentSDKAdapter(
    model="us.anthropic.claude-opus-4-6-v1",  # Bedrock model ID
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(),  # No api_key = inherit host auth
    ),
)

# Docker: specify where AWS config is mounted
adapter = ClaudeAgentSDKAdapter(
    model="us.anthropic.claude-opus-4-6-v1",
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(aws_config_path="/mnt/aws"),
    ),
)
```

**Bedrock model ID format:** `us.anthropic.claude-opus-4-6-v1` (no `:0` suffix
for Opus 4.6). Older models retain the `:0` suffix (e.g.,
`us.anthropic.claude-sonnet-4-5-20250929-v1:0`).

Environment variables passed through for Bedrock: `AWS_PROFILE`, `AWS_REGION`,
`AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
`AWS_ROLE_ARN`, `AWS_WEB_IDENTITY_TOKEN_FILE`, `CLAUDE_CODE_USE_BEDROCK`.

At `src/weakincentives/adapters/claude_agent_sdk/isolation.py`:
`IsolationConfig`, `EphemeralHome`, `get_default_model`,
`DEFAULT_MODEL`, `DEFAULT_BEDROCK_MODEL`.

### Model ID Helpers

| Function | Description |
|----------|-------------|
| `get_default_model()` | Returns Opus 4.6 in appropriate format for auth mode |
| `get_supported_bedrock_models()` | Returns mapping of Anthropic names to Bedrock IDs |
| `to_bedrock_model_id(name)` | Convert Anthropic model name to Bedrock ID |
| `to_anthropic_model_name(id)` | Convert Bedrock ID to Anthropic model name |
| `DEFAULT_MODEL` | `"claude-opus-4-6"` |
| `DEFAULT_BEDROCK_MODEL` | `"us.anthropic.claude-opus-4-6-v1"` |

### MCP Tool Exposure

Attach `Tool` to sections; they're automatically bridged as MCP tools under
`"wink"` server. See `prompt/tool.py` for `Tool` definition.

## Operational Notes

- Use `budget_tracker` in `evaluate()` for token usage tracking
- `stop_on_structured_output=True` ends turn after `StructuredOutput` tool
- Transcript collection enabled by default via `TranscriptCollectorConfig()`
- Windows: Sandbox settings may not be enforced; HOME redirection applies
