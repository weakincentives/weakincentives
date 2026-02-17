# Adapters

*Canonical spec: [specs/ADAPTERS.md](../specs/ADAPTERS.md)*

Adapters bridge a prompt to a provider and enforce consistent semantics:

- Render prompt markdown
- Expose tools to the model
- Execute tool calls synchronously
- Parse structured output when declared

## ProviderAdapter.evaluate

All adapters implement the same interface:

```python nocheck
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

## Design Philosophy

WINK only integrates with agentic harnesses and their SDKs. Native SDK
integrations (like direct OpenAI or Anthropic API calls) are too low-level to
qualify as an execution harness.

An **execution harness** provides:

- Planning loops and tool orchestration
- Sandboxing and isolation
- Retry handling and crash recovery
- Deadline and budget enforcement

WINK's agent definition (prompts, tools, policies, feedback) is portable across
harnesses. The harness owns execution; you own the definition.

## Claude Agent SDK Adapter

**Install:** `pip install "weakincentives[claude-agent-sdk]"`

The Claude Agent SDK adapter is WINK's recommended integration for production
applications. Instead of WINK executing tools itself, it delegates to Claude
Code's native tool execution—giving you Claude's battle-tested tooling (Read,
Write, Bash, Glob, Grep) with WINK's prompt composition and session management.

```python nocheck
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
- **Workspace management**: `WorkspaceSection` provides structured
  access to host files with security boundaries

**See [Claude Agent SDK Guide](claude-agent-sdk.md) for complete documentation**
covering workspace sections, isolation configuration, tool bridging, skill
mounting, and production patterns.

## Guardrails

All three adapters support the full guardrails stack:

- **Tool policies**: Gate tool invocations via session-scoped state
- **Feedback providers**: Inject advisory guidance after tool calls
- **Task completion**: Verify goals before the agent stops (with continuation
  loop, max 10 rounds)

Guardrails are declared on the prompt, not configured per-adapter. See the
[Guardrails guide](guardrails-and-feedback.md) for details.

## Adapter Events

All adapters publish events to the session's dispatcher:

| Event | When | Fields |
| --- | --- | --- |
| `PromptRendered` | After prompt render, before API call | `rendered_prompt`, `adapter` |
| `RenderedTools` | After prompt render, correlated with `PromptRendered` | `tools`, `render_event_id` |
| `ToolInvoked` | Each tool call (native + bridged) | `name`, `params`, `result`, `usage` |
| `PromptExecuted` | After evaluation completes | `result`, `usage` (TokenUsage) |

For Claude Agent SDK, native tools (Read, Write, Bash) are tracked via SDK hooks
and also publish `ToolInvoked` events.

## Throttling

Adapters support throttle policies for retry handling on rate limits:

```python nocheck
from weakincentives.adapters import new_throttle_policy

policy = new_throttle_policy(max_attempts=5)
adapter = ClaudeAgentSDKAdapter(throttle_policy=policy)
```

**Full throttle configuration:**

```python nocheck
from datetime import timedelta
from weakincentives.adapters import ThrottlePolicy

policy = ThrottlePolicy(
    max_attempts=5,                        # Total attempts before giving up
    base_delay=timedelta(milliseconds=500),  # Initial backoff delay
    max_delay=timedelta(seconds=8),        # Maximum backoff delay
    max_total_delay=timedelta(seconds=30), # Total time budget for retries
)
```

**How throttling works:**

1. When a rate limit is hit, the adapter backs off exponentially
1. If `Retry-After` header is present, it's respected
1. `ThrottleError` is raised if all attempts fail

## Next Steps

- [Claude Agent SDK](claude-agent-sdk.md): Production integration guide
- [Orchestration](orchestration.md): Use AgentLoop for request handling
- [Evaluation](evaluation.md): Test agents with datasets
