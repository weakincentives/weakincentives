# Provider Adapters Specification

## Purpose

Adapters bridge prompts and external LLM services, handling request formatting,
response parsing, rate limiting, and error recovery. Core at `src/weakincentives/adapters/core.py`.

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
1. **Emit** - Publish `PromptRendered`, `PromptExecuted` to `session.dispatcher`

## Provider Implementations

### OpenAI Adapter

At `src/weakincentives/adapters/openai.py`:

| Config | Description |
| --- | --- |
| `OpenAIClientConfig` | api_key, base_url, organization, timeout, max_retries |
| `OpenAIModelConfig` | logprobs, top_logprobs, parallel_tool_calls, store, user |

**Note:** `max_tokens` renamed to `max_output_tokens` for Responses API. `seed`,
`stop`, `presence_penalty`, `frequency_penalty` not accepted—raises `ValueError`.

**Structured output:** Uses native JSON schema response format with `.parsed` payload.

### LiteLLM Adapter

At `src/weakincentives/adapters/litellm.py`:

| Config | Description |
| --- | --- |
| `LiteLLMClientConfig` | api_key, api_base, timeout, num_retries |
| `LiteLLMModelConfig` | Model parameters |

**Caveats:** Tool calling/structured output varies by provider. LiteLLM exceptions
normalized to `ThrottleError` or `PromptEvaluationError`. Always sets
`require_structured_output_text=True`.

### Claude Agent SDK Adapter

At `src/weakincentives/adapters/claude_agent_sdk/adapter.py`:

- Async execution with MCP tool bridging
- Skill mounting support

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

## Inner Loop Architecture

Shared `InnerLoop` at `adapters/inner_loop.py` drives request/response:

1. **Render** → RenderedPrompt
1. **Publish PromptRendered**
1. **Prepare tools** via `tool_to_spec`
1. **Call provider** (failures → `PromptEvaluationError` with `phase="request"`)
1. **Handle tool calls** via `ToolExecutor` at `adapters/tool_executor.py`
1. **Loop** until message without tool calls
1. **Parse output** from `.parsed` or text
1. **Publish PromptExecuted**

### Tool Execution

`ToolExecutor` at `adapters/tool_executor.py`:

- Arguments decoded via `parse_tool_arguments`
- Dataclass parsing via `serde.parse`
- Tool handlers access resources via `context.prompt.resources`
- Exceptions logged and converted to failed results

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
| `PromptEvaluationError` | `src/weakincentives/adapters/core.py` | Base for evaluation failures |
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
| `PromptExecuted` | After parse | Response, tokens, timing |
| `ToolInvoked` | After dispatch | Name, params, result |

Logs: `prompt.render.start`, `prompt.render.complete`, `prompt.call.start`,
`prompt.call.complete`, `prompt.throttled`, `prompt.error`.

## Implementing New Adapters

1. Define `ClientConfig` and `ModelConfig` extending `LLMConfig`
1. Accept concrete client or config for test injection
1. Call `prompt.render` once
1. Access resources via `prompt.resources`
1. Delegate to `run_inner_loop`
1. Wrap SDK failures as `PromptEvaluationError`

## Testing

- **Unit**: Mock provider responses; verify backoff; test structured output parsing
- **Integration**: Provider test endpoints; tool dispatch round-trips; throttle recovery
- **Fixtures**: `tests/helpers/adapters.py` provides adapter name constants; mock adapters are created ad-hoc in test files
