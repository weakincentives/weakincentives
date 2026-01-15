# Claude Agent SDK Adapter Specification

Adapter for Claude Code CLI with native tools and MCP bridging.

**Source:** `src/weakincentives/adapters/claude_agent_sdk.py`

**SDK:** `claude-agent-sdk>=0.1.15`

## Features

- Native tools: Read/Write/Edit/Glob/Grep/Bash
- MCP bridging for weakincentives `Tool` handlers
- Structured output via JSON Schema
- Optional isolation with ephemeral HOME

## Configuration

### ClaudeAgentSDKClientConfig

```python
ClaudeAgentSDKClientConfig(
    permission_mode="bypassPermissions",
    cwd=None,
    max_turns=None,
    suppress_stderr=True,
    stop_on_structured_output=True,
    isolation=IsolationConfig(...),
)
```

### IsolationConfig

```python
IsolationConfig(
    network_policy=NetworkPolicy.no_network(),
    sandbox=SandboxConfig(enabled=True),
    env={...},
    api_key=None,
    include_host_env=False,
)
```

### NetworkPolicy

```python
NetworkPolicy.no_network()                    # No tool egress
NetworkPolicy.with_domains("docs.python.org") # Allowlist
```

**Note:** Restricts tools only, not model API connection.

### SandboxConfig

```python
SandboxConfig(
    enabled=True,
    writable_paths=(),
    readable_paths=(),
    bash_auto_allow=True,
)
```

## Tool Bridging

Tools on sections are exposed as MCP tools under server key `"wink"`:

```python
section = MarkdownSection(..., tools=(my_tool,))
```

## Events

| Event | When |
|-------|------|
| `PromptRendered` | After render |
| `ToolInvoked` | Each tool call (native + bridged) |
| `PromptExecuted` | Completion with token usage |

## Usage

```python
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter

adapter = ClaudeAgentSDKAdapter()
response = adapter.evaluate(Prompt(template), session=session)
```

With isolation:

```python
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
        ),
    ),
)
```

## Requirements

- Python: `pip install 'weakincentives[claude-agent-sdk]'`
- CLI: `npm install -g @anthropic-ai/claude-code`
- Linux sandbox: bubblewrap (`bwrap`)
