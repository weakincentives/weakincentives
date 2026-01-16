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
| `api_key` | `str \| None` | `None` | API key override |
| `include_host_env` | `bool` | `False` | Copy non-sensitive vars |

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

### MCP Tool Exposure

Attach `Tool` to sections; they're automatically bridged as MCP tools under
`"wink"` server. See `prompt/tool.py` for `Tool` definition.

## Operational Notes

- Use `budget_tracker` in `evaluate()` for token usage tracking
- `stop_on_structured_output=True` ends turn after `StructuredOutput` tool
- Windows: Sandbox settings may not be enforced; HOME redirection applies
