# Adapters

*Canonical spec: [specs/ADAPTERS.md](../specs/ADAPTERS.md)*

Adapters bridge a prompt to a provider and enforce consistent semantics:

- Render prompt markdown
- Expose tools to the model
- Execute tool calls synchronously
- Parse structured output when declared

## ProviderAdapter.evaluate

All adapters implement the same interface:

```python
response = adapter.evaluate(
    prompt,
    session=session,
    deadline=...,        # optional
    budget=...,          # optional
    budget_tracker=...,  # optional
    resources=...,       # optional ResourceRegistry
)
```

It returns `PromptResponse[OutputT]`:

- `prompt_name`: string
- `text`: raw assistant text
- `output`: parsed structured output (or `None`)

The adapter handles all the provider-specific details: API formatting, tool
schema translation, response parsing. Your code just calls `evaluate()` and gets
back typed results.

## OpenAIAdapter

**Install:** `pip install "weakincentives[openai]"`

```python
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.adapters import OpenAIClientConfig, OpenAIModelConfig

adapter = OpenAIAdapter(
    model="gpt-4.1-mini",
    client_config=OpenAIClientConfig(),
    model_config=OpenAIModelConfig(max_tokens=800),
)
response = adapter.evaluate(prompt, session=session)
```

**Key configs:**

- `OpenAIClientConfig(api_key=..., base_url=..., timeout=..., max_retries=...)`
- `OpenAIModelConfig(temperature=..., max_tokens=..., top_p=..., ...)`

The adapter uses OpenAI's native JSON schema response format for structured
output. It handles tool calls synchronously, executing each tool and feeding
results back to the model.

## LiteLLMAdapter

**Install:** `pip install "weakincentives[litellm]"`

LiteLLM provides a unified interface to many providers. Use this when you want
to switch between providers without changing code.

```python
from weakincentives.adapters.litellm import LiteLLMAdapter
from weakincentives.adapters import LiteLLMClientConfig, LiteLLMModelConfig

adapter = LiteLLMAdapter(
    model="openai/gpt-4.1-mini",
    completion_config=LiteLLMClientConfig(),
    model_config=LiteLLMModelConfig(max_tokens=800),
)
```

## Claude Agent SDK Adapter

**Install:** `pip install "weakincentives[claude-agent-sdk]"`

The Claude Agent SDK adapter is WINK's recommended integration for production
applications. Instead of WINK executing tools itself, it delegates to Claude
Code's native tool execution—giving you Claude's battle-tested tooling (Read,
Write, Bash, Glob, Grep) with WINK's prompt composition and session management.

```python
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter

adapter = ClaudeAgentSDKAdapter()
response = adapter.evaluate(prompt, session=session)
```

The adapter provides:

- **Native tooling quality**: Claude Code's tools handle edge cases that custom
  implementations often miss
- **Built-in sandboxing**: Hermetic isolation prevents access to host
  configuration and credentials
- **MCP bridging**: Your WINK tools are automatically exposed as MCP tools
- **Workspace management**: `ClaudeAgentWorkspaceSection` provides structured
  access to host files with security boundaries

**See [Claude Agent SDK Guide](claude-agent-sdk.md) for complete documentation**
covering workspace sections, isolation configuration, tool bridging, skill
mounting, and production patterns.

## Adapter Events

All adapters publish events to the session's dispatcher:

| Event | When | Fields |
| --- | --- | --- |
| `PromptRendered` | After prompt render, before API call | `rendered_prompt`, `adapter` |
| `ToolInvoked` | Each tool call (native + bridged) | `name`, `params`, `result`, `usage` |
| `PromptExecuted` | After evaluation completes | `result`, `usage` (TokenUsage) |

For Claude Agent SDK, native tools (Read, Write, Bash) are tracked via SDK hooks
and also publish `ToolInvoked` events.

## Throttling

Adapters support throttle policies for rate limiting:

```python
from weakincentives.adapters import new_throttle_policy

policy = new_throttle_policy(requests_per_minute=60)
adapter = OpenAIAdapter(model="gpt-4o", throttle_policy=policy)
```

**Full throttle configuration:**

```python
from weakincentives.adapters import ThrottlePolicy

policy = ThrottlePolicy(
    requests_per_minute=60,
    max_attempts=5,           # Total attempts before giving up
    base_delay=1.0,           # Initial backoff delay (seconds)
    max_delay=60.0,           # Maximum backoff delay
    max_total_delay=300.0,    # Total time budget for retries
    jitter=0.1,               # Randomization factor (0-1)
)
```

**How throttling works:**

1. When a rate limit is hit, the adapter backs off exponentially
1. Jitter prevents thundering herd when multiple workers retry
1. If `Retry-After` header is present, it's respected
1. `ThrottleError` is raised if all attempts fail

## Next Steps

- [Claude Agent SDK](claude-agent-sdk.md): Production integration guide
- [Orchestration](orchestration.md): Use MainLoop for request handling
- [Evaluation](evaluation.md): Test agents with datasets
