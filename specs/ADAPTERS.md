# Provider Adapters Specification

## Purpose

Adapters bridge prompts and external LLM services, handling request formatting,
response parsing, rate limiting, and error recovery. Core at `src/weakincentives/adapters/core.py`.

## Design Philosophy

WINK only integrates with **agentic harnesses** and their SDKs. Native SDK
integrations (like direct OpenAI or Anthropic API calls) are too low-level to
qualify as an execution harness.

An **execution harness** provides:

- Planning loops and tool orchestration
- Sandboxing and isolation
- Retry handling and crash recovery
- Deadline and budget enforcement

WINK's agent definition (prompts, tools, policies, feedback) is portable across
harnesses. The harness owns execution; you own the definition.

## Principles

- **Provider-agnostic orchestration**: Uniform protocol; provider differences encapsulated
- **Prompt-owned resources**: Adapters access resources via `prompt.resources`
- **Predictable failures**: Typed exceptions with retry/abort context
- **Observable by default**: Structured events and logs at each decision point
- **Protect upstream health**: Reactive rate limiting respecting provider signals

## Adapter Protocol

All adapters implement `ProviderAdapter` at `src/weakincentives/adapters/core.py`:

| Parameter | Description |
| --- | --- |
| `prompt` | Prompt to evaluate (must be in context manager) |
| `session` | Session for state management |
| `deadline` | Optional wall-clock deadline |
| `budget` | Optional token/time budget |
| `budget_tracker` | Optional shared budget tracker |
| `heartbeat` | Optional heartbeat for liveness monitoring |
| `run_context` | Optional execution context with correlation identifiers |

| Property | Description |
| --- | --- |
| `adapter_name` | Canonical name (e.g., `"claude_agent_sdk"`, `"codex_app_server"`, `"acp"`). Default: `type(self).__name__`. |

Returns `PromptResponse[OutputT]` at `src/weakincentives/adapters/core.py`.

### Configuration

Base config `LLMConfig` at `src/weakincentives/adapters/config.py` with fields: `temperature`,
`max_tokens`, `top_p`, `presence_penalty`, `frequency_penalty`, `stop`, `seed`.
Provider-specific configs extend this.

### Lifecycle

1. **Validate context** - Verify prompt within context manager
1. **Render** - `prompt.render()` → `RenderedPrompt`
1. **Format** - Convert to provider wire format
1. **Call** - Issue request with throttle protection and deadline checks
1. **Parse** - Extract content, dispatch tool calls
1. **Emit** - Publish `PromptRendered`, `RenderedTools`, `PromptExecuted` to `session.dispatcher`

## Provider Implementations

### Claude Agent SDK Adapter

At `src/weakincentives/adapters/claude_agent_sdk/adapter.py`:

- Async execution with MCP tool bridging
- Skill mounting support
- Hermetic isolation and sandboxing
- Native Claude Code tooling (Read, Write, Bash, Glob, Grep)
- Adaptive reasoning effort levels (`ReasoningEffort`)
- Transcript collection (enabled by default)
- `call_id` correlation via `MCPToolExecutionState`
- SDK-native exception types for error normalization

See [CLAUDE_AGENT_SDK.md](CLAUDE_AGENT_SDK.md) for complete documentation.

### Codex App Server Adapter

At `src/weakincentives/adapters/codex_app_server/adapter.py`:

- Delegates execution to Codex via the app-server protocol (stdio NDJSON)
- WINK tools bridged as Codex dynamic tools
- No additional Python dependencies beyond WINK and `codex` CLI on PATH
- Native Codex tools (commands, file changes, web search)
- Structured output via native `outputSchema`

See [CODEX_APP_SERVER.md](CODEX_APP_SERVER.md) for complete documentation.

### ACP Adapter

At `src/weakincentives/adapters/acp/`:

- Generic Agent Communication Protocol (ACP) adapter
- Bridges WINK prompts and tools to ACP-compatible harnesses
- Protocol-level tool bridging and session management

See [ACP_ADAPTER.md](ACP_ADAPTER.md) for complete documentation.

### OpenCode ACP Adapter

At `src/weakincentives/adapters/opencode_acp/`:

- ACP adapter specialized for OpenCode harness
- Quirk handling for OpenCode-specific behaviors

See [OPENCODE_ACP_ADAPTER.md](OPENCODE_ACP_ADAPTER.md) for complete documentation.

### Gemini CLI ACP Adapter

At `src/weakincentives/adapters/gemini_acp/` (planned):

- ACP adapter specialized for Gemini CLI harness
- Model selection via CLI `--model` flag (not ACP protocol)
- No token usage reporting
- HTTP/SSE MCP only (no stdio)

See [GEMINI_ACP_ADAPTER.md](GEMINI_ACP_ADAPTER.md) for complete documentation.

## Shared Adapter Module

At `src/weakincentives/adapters/_shared/`:

Code shared across multiple adapter implementations lives in the `_shared`
package. Individual adapters import from `_shared` rather than from each
other's private modules.

| Component | Location | Purpose |
|-----------|----------|---------|
| `BridgedTool` | `_shared/_bridge.py` | Transactional tool wrapper for MCP/SDK consumption |
| `MCPToolExecutionState` | `_shared/_bridge.py` | Thread-safe `call_id` correlation between hooks and bridge |
| `create_bridged_tools()` | `_shared/_bridge.py` | Factory for `BridgedTool` instances |
| `create_mcp_server()` | `_shared/_bridge.py` | In-process MCP server for Claude Agent SDK |
| `VisibilityExpansionSignal` | `_shared/_visibility_signal.py` | Thread-safe signal for progressive disclosure |
| `run_async()` | `_shared/_async_utils.py` | Async/sync bridging via `asyncio.run()` |

Both the Claude Agent SDK adapter and Codex App Server adapter re-export from
`_shared` via thin compatibility modules (`claude_agent_sdk/_bridge.py`,
`codex_app_server/_async.py`, etc.).

## Guardrails

All adapters support the full guardrails stack declared on the prompt. See
[GUARDRAILS.md](GUARDRAILS.md) for the complete specification.

| Mechanism | Integration Point |
|-----------|------------------|
| Tool policies | Checked in `BridgedTool` before handler execution |
| Feedback providers | Collected after successful tool calls and appended to content |
| Task completion | Continuation loop (max 10 rounds) when checker reports incomplete |

Guardrails are declared on `PromptTemplate` (not adapter config):

```python
template = PromptTemplate(
    ns="my-agent", key="main",
    sections=[...],
    task_completion_checker=FileOutputChecker(files=("report.md",)),
)
```

Each adapter implements guardrails through adapter-specific `_guardrails.py`
modules that handle protocol-appropriate feedback injection and continuation
semantics.

## Rate Limiting and Throttling

### ThrottlePolicy

At `src/weakincentives/adapters/throttle.py` via `new_throttle_policy()`:

| Field | Default | Description |
| --- | --- | --- |
| `max_attempts` | 5 | Maximum retry attempts |
| `base_delay` | 500ms | Initial backoff |
| `max_delay` | 8s | Cap on individual delays |
| `max_total_delay` | 30s | Total time budget |

### Signal Classification

| Signal | Examples | Behavior |
| --- | --- | --- |
| Rate limit | HTTP 429 | Retry with backoff |
| Quota exhaustion | `insufficient_quota` | Longer backoff, alerting |
| Timeout | Connection/read timeout | Retry if deadline permits |
| Server error | HTTP 500-503 | Retry with backoff |

### ThrottleError

At `src/weakincentives/adapters/throttle.py`:

- `kind`: rate_limit, quota_exhausted, timeout, unknown
- `retry_after`: Provider-suggested delay
- `attempts`: Retry count
- `retry_safe`: Whether retry is safe

## Tool Bridging

Both adapters use the shared `BridgedTool` abstraction at
`src/weakincentives/adapters/_shared/_bridge.py` for transactional tool
execution. Each `BridgedTool` invocation:

1. **Snapshot** — Capture session and resource state
1. **Execute** — Call handler with parsed parameters via `serde.parse`
1. **Dispatch** — Emit `ToolInvoked` event with `call_id` for correlation
1. **Rollback** — Restore snapshot on failure

Tool arguments are decoded via `serde.parse()` with `extra="forbid"`. Resources
are accessed through `prompt.resources`. Exceptions are caught and converted to
error results (never abort).

### Transactional Tool Execution

Tool execution is transactional via `src/weakincentives/runtime/transactions.py`:

- `create_snapshot(session, resource_context, tag)` - Capture state
- `restore_snapshot(session, resource_context, snapshot)` - Rollback
- `tool_transaction` context manager for simpler cases

Failed or aborted tools leave no trace in mutable state.

## Error Handling

### Exception Hierarchy

| Exception | Location | Description |
| --- | --- | --- |
| `PromptEvaluationError` | `src/weakincentives/errors.py` | Base for evaluation failures |
| `ThrottleError` | `src/weakincentives/adapters/throttle.py` | Retryable provider errors |
| `PromptRenderError` | `prompt/errors.py` | Template/section failures |
| `OutputParseError` | `prompt/structured_output.py` | Structured output validation |
| `DeadlineExceededError` | `errors.py` | Time budget exhausted |

### Error Propagation

- **Tool failures**: Wrapped as `ToolResult(success=False)`, returned to model
- **Parse failures**: Raise `OutputParseError` with raw response
- **Throttle exhaustion**: Raise `ThrottleError` with `retry_safe=False`
- **Deadline exceeded**: Raise `DeadlineExceededError` immediately

## Budget Tracking

Via `Budget` and `BudgetTracker` at `src/weakincentives/budget.py`:

- Records token usage after each response
- Checks limits at defined checkpoints
- Thread-safe for concurrent execution

## Telemetry

Events via `session.dispatcher`:

| Event | When | Payload |
| --- | --- | --- |
| `PromptRendered` | After render | Text, tools, metadata |
| `RenderedTools` | After render | Tool schemas, correlated with `PromptRendered` via `render_event_id` |
| `PromptExecuted` | After parse | Response, tokens, timing |
| `ToolInvoked` | After dispatch | Name, params, result, `call_id` |

Logs: `prompt.render.start`, `prompt.render.complete`, `prompt.call.start`,
`prompt.call.complete`, `prompt.throttled`, `prompt.error`.

## Implementing New Adapters

1. Define `ClientConfig` and `ModelConfig` extending `LLMConfig`
1. Accept concrete client or config for test injection
1. Call `prompt.render` once
1. Access resources via `prompt.resources`
1. Use `create_bridged_tools()` from `_shared/_bridge.py` for tool bridging
1. Wrap SDK failures as `PromptEvaluationError`
1. Dispatch `PromptRendered`, `RenderedTools`, `ToolInvoked`, and `PromptExecuted` events

## Testing

- **Unit**: Mock provider responses; verify backoff; test structured output parsing
- **Integration**: Provider test endpoints; tool dispatch round-trips; throttle recovery
- **Fixtures**: `tests/helpers/adapters.py` provides adapter name constants; mock adapters are created ad-hoc in test files
