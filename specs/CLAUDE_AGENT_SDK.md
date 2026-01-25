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
| `task_completion_checker` | `TaskCompletionChecker \| None` | `None` | Task completion verification |
| `isolation` | `IsolationConfig \| None` | `None` | Isolation configuration |
| `betas` | `tuple[str, ...] \| None` | `None` | Beta features to enable |

### ClaudeAgentSDKModelConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"claude-sonnet-4-5-20250929"` | Claude model identifier |
| `max_thinking_tokens` | `int \| None` | `None` | Extended thinking mode tokens |

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
| `skills` | `SkillConfig \| None` | `None` | Skills to mount |

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

## Isolation Behavior

When `isolation` is set:

- Creates ephemeral `HOME` with `$HOME/.claude/settings.json`
- Passes `env` and `setting_sources=["user"]` to SDK

**Notes:**

- `NetworkPolicy` restricts tools only, not API connection
- Without `api_key`, `ANTHROPIC_API_KEY` from process env is forwarded

## Tool Bridging (MCP)

Tools attached to sections are exposed to Claude Code as MCP tools under server
key `"wink"`. Tool calls publish `ToolInvoked` events.

## Events

| Event | When |
|-------|------|
| `PromptRendered` | After render |
| `ToolInvoked` | Each tool call (native + bridged) |
| `PromptExecuted` | Completion (includes `TokenUsage`) |

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
workspace = ClaudeAgentWorkspaceSection(
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
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",  # Bedrock model ID
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(),  # No api_key = inherit host auth
    ),
)

# Docker: specify where AWS config is mounted
adapter = ClaudeAgentSDKAdapter(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(aws_config_path="/mnt/aws"),
    ),
)
```

Environment variables passed through for Bedrock: `AWS_PROFILE`, `AWS_REGION`,
`AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
`AWS_ROLE_ARN`, `AWS_WEB_IDENTITY_TOKEN_FILE`, `CLAUDE_CODE_USE_BEDROCK`.

### Model ID Helpers

| Function | Description |
|----------|-------------|
| `get_default_model()` | Returns Sonnet 4.5 in appropriate format for auth mode |
| `get_supported_bedrock_models()` | Returns mapping of Anthropic names to Bedrock IDs |
| `to_bedrock_model_id(name)` | Convert Anthropic model name to Bedrock ID |
| `to_anthropic_model_name(id)` | Convert Bedrock ID to Anthropic model name |
| `DEFAULT_MODEL` | Default Anthropic model (Sonnet 4.5) |
| `DEFAULT_BEDROCK_MODEL` | Default Bedrock model ID (Sonnet 4.5) |

### MCP Tool Exposure

Attach `Tool` to sections; they're automatically bridged as MCP tools under
`"wink"` server. See `prompt/tool.py` for `Tool` definition.

## Operational Notes

- Use `budget_tracker` in `evaluate()` for token usage tracking
- `stop_on_structured_output=True` ends turn after `StructuredOutput` tool
- Windows: Sandbox settings may not be enforced; HOME redirection applies
