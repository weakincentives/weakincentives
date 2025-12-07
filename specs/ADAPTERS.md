# Provider Adapters Specification

## Purpose

Provider adapters bridge the prompt abstraction and external LLM services. They
handle request formatting, response parsing, rate limiting, and error recovery
so orchestration code remains provider-agnostic. This specification covers the
shared adapter protocol, provider-specific implementations, structured output
handling, and throttling behavior.

## Guiding Principles

- **Provider-agnostic orchestration**: Callers interact with a uniform protocol;
  provider differences stay encapsulated in adapter implementations.
- **Fail predictable**: Errors surface as typed exceptions with enough context
  for callers to retry, degrade, or abort gracefully.
- **Observable by default**: Adapters emit structured events and logs at each
  decision point so operators can trace requests, retries, and failures.
- **Protect upstream health**: Rate limiting is reactive rather than
  pre-emptive; adapters respect provider signals and avoid amplifying overload.

## Adapter Protocol

All adapters implement `ProviderAdapter[ConfigT]`:

```python
class ProviderAdapter(ABC):
    @abstractmethod
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *,
        parse_output: bool = True,
        bus: EventBus,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponse[OutputT]: ...
```

**Parameters:**

- `prompt` - The prompt template to evaluate
- `parse_output` - Whether to parse structured output from response
- `bus` - Event bus for telemetry
- `session` - Session for state management
- `deadline` - Optional wall-clock deadline
- `visibility_overrides` - Section visibility controls for progressive disclosure
- `budget` - Optional token/time budget limits
- `budget_tracker` - Optional shared tracker for budget consumption

### Configuration

Adapters use frozen dataclass configurations for type-safe instantiation:

```python
@FrozenDataclass()
class LLMConfig:
    """Base configuration for LLM model parameters."""
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: tuple[str, ...] | None = None
    seed: int | None = None
```

Provider-specific configs extend this base with additional fields. Only non-None
fields are included in request payloads.

### Lifecycle

1. **Render** - Call `Prompt(template).bind(params).render()` to produce a `RenderedPrompt`
   with markdown text, tools, and structured output metadata.
1. **Format** - Convert the rendered prompt into the provider wire format.
1. **Call** - Issue the provider request with throttle protection and deadline
   checks.
1. **Parse** - Extract assistant content and dispatch tool calls.
1. **Emit** - Publish `PromptRendered` and `PromptExecuted` events.

## Provider Implementations

### OpenAI Adapter

`OpenAIAdapter` targets the OpenAI Responses API via the official SDK.

```python
from weakincentives.adapters.openai import OpenAIAdapter, OpenAIClientConfig, OpenAIModelConfig

client_config = OpenAIClientConfig(api_key="sk-...", timeout=30.0)
model_config = OpenAIModelConfig(temperature=0.7, max_tokens=1024)

adapter = OpenAIAdapter(
    model="gpt-4o",
    client_config=client_config,
    model_config=model_config,
)
```

**Configuration:**

| Field | Type | Description |
|-------|------|-------------|
| `api_key` | `str \| None` | API key (falls back to env) |
| `base_url` | `str \| None` | Custom API endpoint |
| `organization` | `str \| None` | Organization ID |
| `timeout` | `float \| None` | Request timeout seconds |
| `max_retries` | `int \| None` | SDK-level retries |

**Model Parameters (OpenAIModelConfig):**

| Field | Type | Description |
|-------|------|-------------|
| `logprobs` | `bool \| None` | Return log probabilities |
| `top_logprobs` | `int \| None` | Number of top logprobs |
| `parallel_tool_calls` | `bool \| None` | Allow parallel tool calls |
| `store` | `bool \| None` | Store conversation |
| `user` | `str \| None` | End-user identifier |

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model identifier |
| `client_config` | `OpenAIClientConfig \| None` | `None` | Client settings |
| `model_config` | `OpenAIModelConfig \| None` | `None` | Model parameters |

Note: `max_tokens` is renamed to `max_output_tokens` for Responses API. The
Responses API does not accept `seed`, `stop`, `presence_penalty`, or
`frequency_penalty`; supplying these raises `ValueError` at construction.

**Structured Output:**

When a prompt declares structured output, the adapter uses OpenAI's native
JSON schema response format. The adapter sets `response_format.type = "json_schema"`
with the dataclass schema. The provider enforces structure at generation time
and returns a `.parsed` payload.

### LiteLLM Adapter

`LiteLLMAdapter` provides access to 100+ providers through LiteLLM.

```python
from weakincentives.adapters.litellm import LiteLLMAdapter, LiteLLMClientConfig, LiteLLMModelConfig

completion_config = LiteLLMClientConfig(api_key="...", timeout=60.0)
model_config = LiteLLMModelConfig(temperature=0.5, max_tokens=2048)

adapter = LiteLLMAdapter(
    model="anthropic/claude-3-opus",
    completion_config=completion_config,
    model_config=model_config,
)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model identifier |
| `completion_config` | `LiteLLMClientConfig \| None` | `None` | Client settings |
| `model_config` | `LiteLLMModelConfig \| None` | `None` | Model parameters |
| `completion` | `LiteLLMCompletion \| None` | `None` | Pre-configured completion |
| `completion_factory` | `Callable \| None` | `None` | Factory for completions |
| `completion_kwargs` | `Mapping \| None` | `None` | Extra completion kwargs |

**Configuration:**

| Field | Type | Description |
|-------|------|-------------|
| `api_key` | `str \| None` | Provider API key |
| `api_base` | `str \| None` | Custom API endpoint |
| `timeout` | `float \| None` | Request timeout seconds |
| `num_retries` | `int \| None` | LiteLLM-level retries |

**Constraints and Caveats:**

- Tool calling, structured output, and streaming support varies by underlying
  provider.
- LiteLLM exceptions are normalized to `ThrottleError` or
  `PromptEvaluationError`.
- Token counting uses LiteLLM's estimation which may differ from actuals.
- Structured outputs always set `require_structured_output_text=True` because
  LiteLLM does not return structured `.parsed` payloads.

### Anthropic Adapter

`AnthropicAdapter` targets the Anthropic Messages API via the official SDK,
with native support for structured outputs (beta) and tool use.

```python
from weakincentives.adapters.anthropic import AnthropicAdapter
from weakincentives.adapters import AnthropicClientConfig, AnthropicModelConfig

client_config = AnthropicClientConfig(api_key="sk-ant-...", timeout=30.0)
model_config = AnthropicModelConfig(temperature=0.7, max_tokens=4096, top_k=40)

adapter = AnthropicAdapter(
    model="claude-sonnet-4-20250514",
    client_config=client_config,
    model_config=model_config,
)
```

**Configuration (AnthropicClientConfig):**

| Field | Type | Description |
|-------|------|-------------|
| `api_key` | `str \| None` | API key (falls back to env) |
| `base_url` | `str \| None` | Custom API endpoint |
| `timeout` | `float \| None` | Request timeout seconds |
| `max_retries` | `int \| None` | SDK-level retries |

**Model Parameters (AnthropicModelConfig):**

| Field | Type | Description |
|-------|------|-------------|
| `top_k` | `int \| None` | Sample from top K tokens |
| `metadata` | `Mapping[str, str] \| None` | Request metadata |

Note: Anthropic does not support `seed`, `presence_penalty`, or
`frequency_penalty`; supplying these raises `ValueError` at construction.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `claude-opus-4-5-20250929` | Model identifier |
| `client_config` | `AnthropicClientConfig \| None` | `None` | Client settings |
| `model_config` | `AnthropicModelConfig \| None` | `None` | Model parameters |
| `tool_choice` | `ToolChoice` | `"auto"` | Tool selection directive |
| `use_native_structured_output` | `bool` | `True` | Use beta structured outputs |
| `client` | `AnthropicProtocol \| None` | `None` | Pre-configured client |

**Structured Output:**

When `use_native_structured_output=True` and a prompt declares structured
output, the adapter uses Anthropic's beta structured outputs feature
(`structured-outputs-2025-11-13`). This guarantees the response matches the
JSON schema by constraining token generation at inference time.

The adapter passes `output_format={"type": "json_schema", "schema": ...}` and
disables prompt-level output instructions when native structured output is
enabled.

**Message Format Normalization:**

The adapter normalizes WINK's message format to Anthropic's API:

- System prompts are extracted and passed as the separate `system` parameter
- Tool result messages use `tool_result` content blocks under `user` role
- Assistant messages with tool calls use `tool_use` content blocks

**Tool Choice Translation:**

| WINK ToolChoice | Anthropic Equivalent |
|-----------------|----------------------|
| `"auto"` | `{"type": "auto"}` |
| `{"type": "function", "function": {"name": "x"}}` | `{"type": "tool", "name": "x"}` |

**Throttle Detection:**

The adapter detects and normalizes these error conditions:

| Signal | Detection | ThrottleKind |
|--------|-----------|--------------|
| Rate limit | HTTP 429 or "rate" in message | `rate_limit` |
| API overload | HTTP 529 or "overloaded" in message | `rate_limit` |
| Timeout | Class name contains "timeout" | `timeout` |

**Constraints and Caveats:**

- Native structured outputs require `anthropic>=0.75.0` and are in public beta.
- Claude Opus 4.5 supports up to 200K input tokens and 8K output tokens.
- The adapter requires the optional `anthropic` dependency: `uv sync --extra anthropic`.

## Rate Limiting and Throttling

Adapters implement reactive throttling to protect upstream services.

### Throttle Policy

```python
from weakincentives.adapters.shared import ThrottlePolicy, new_throttle_policy

policy = new_throttle_policy(
    max_attempts=5,
    base_delay_ms=500,
    max_delay_ms=8000,
    max_total_delay_ms=30000,
)
```

| Field | Default | Description |
|-------|---------|-------------|
| `max_attempts` | `5` | Maximum retry attempts |
| `base_delay_ms` | `500` | Initial backoff delay |
| `max_delay_ms` | `8000` | Cap on individual delays |
| `max_total_delay_ms` | `30000` | Total time budget |

### Signal Classification

| Signal | Examples | Behavior |
|--------|----------|----------|
| **Rate limit** | HTTP 429, `RateLimitError` | Retry with backoff |
| **Quota exhaustion** | `insufficient_quota` | Longer backoff, alerting |
| **Timeout** | Connection/read timeout | Retry if deadline permits |
| **Server error** | HTTP 500-503 | Retry with backoff |

### Backoff Strategy

- **Exponential with jitter**: Delays double from base, capped at max, with full
  jitter to avoid thundering herd.
- **Retry-After respect**: Provider-supplied values set the minimum delay.
- **Deadline awareness**: Retries abort early if remaining time is insufficient.
- **No pre-send shaping**: Requests are never altered before hitting the
  provider; all mitigation is reactive.

### ThrottleError

```python
@dataclass(slots=True)
class ThrottleDetails:
    kind: ThrottleKind  # RATE_LIMIT, QUOTA_EXHAUSTED, TIMEOUT
    retry_after: timedelta | None
    attempts: int
    retry_safe: bool
    provider_payload: dict[str, Any] | None

@dataclass(slots=True)
class ThrottleError(PromptEvaluationError):
    details: ThrottleDetails

    # Properties provide access to details fields
    @property
    def kind(self) -> ThrottleKind: ...
    @property
    def retry_after(self) -> timedelta | None: ...
    @property
    def attempts(self) -> int: ...
    @property
    def retry_safe(self) -> bool: ...
```

## Inner Loop Architecture

The shared `InnerLoop` drives the request/response cycle:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Render    │───▶│    Call     │───▶│    Parse    │
│   Prompt    │    │   Provider  │    │   Response  │
└─────────────┘    └─────────────┘    └─────────────┘
                         │                   │
                         ▼                   ▼
                   ┌───────────┐       ┌───────────┐
                   │  Throttle │       │   Tools   │
                   │  Handler  │       │  Dispatch │
                   └───────────┘       └───────────┘
```

1. **Render** - Produce `RenderedPrompt` from prompt + params.
1. **Publish `PromptRendered`** - Emit event with namespace, key, adapter label,
   and rendered markdown.
1. **Prepare tools** - Convert tools to provider-agnostic JSON schemas via
   `tool_to_spec`.
1. **Call provider** - Build chat payload and invoke provider callable. Failures
   wrap in `PromptEvaluationError` with `phase="request"`.
1. **Handle tool calls** - Execute handlers via `ToolExecutor`, emit
   `ToolInvoked` events, collect results.
1. **Loop** - Repeat until a message arrives without tool calls.
1. **Parse output** - Extract structured output from `.parsed` or text.
1. **Publish `PromptExecuted`** - Emit final response event.

### Tool Execution

`ToolExecutor` owns tool scheduling, argument parsing, and event publication:

- Arguments are decoded via `parse_tool_arguments` (JSON objects with string
  keys).
- Dataclass parsing uses `serde.parse`. Validation errors produce
  `ToolResult(success=False)` rather than raising.
- `ToolContext` exposes prompt, adapter, session, event bus, and deadline.
- Handlers run synchronously. Exceptions are logged and converted to failed
  results.
- Before event emission, the adapter captures a session snapshot. On publish
  failure, it rolls back and replaces the tool message with error details.

## Error Handling

### Exception Hierarchy

```
PromptEvaluationError
├── ThrottleError          # Retryable provider errors
├── PromptRenderError      # Template/section failures
├── OutputParseError       # Structured output validation
└── DeadlineExceededError  # Time budget exhausted
```

### Error Propagation

- **Tool failures** - Wrapped in `ToolResult(success=False)` and returned to the
  model; never abort evaluation.
- **Parse failures** - Raise `OutputParseError` with raw response attached.
- **Throttle exhaustion** - Raise `ThrottleError` with `retry_safe=False`.
- **Deadline exceeded** - Raise `DeadlineExceededError` immediately.

## Budget Tracking

Adapters integrate with the budget system for token and time limits:

```python
from weakincentives.budget import Budget, BudgetTracker

budget = Budget(
    deadline=Deadline(expires_at=...),
    max_total_tokens=10000,
    max_input_tokens=8000,
    max_output_tokens=2000,
)

tracker = BudgetTracker(budget)

response = adapter.evaluate(
    prompt,
    bus=bus,
    session=session,
    budget=budget,
    budget_tracker=tracker,
)
```

The adapter records token usage after each provider response and checks limits
at defined checkpoints. Budget tracking is thread-safe for concurrent execution.

## Telemetry

Adapters emit events through the provided `EventBus`:

| Event | When | Payload |
|-------|------|---------|
| `PromptRendered` | After render | Text, tools, metadata |
| `PromptExecuted` | After parse | Response, tokens, timing |
| `ToolInvoked` | After dispatch | Name, params, result |

Structured logs include:

- `prompt.render.start` / `prompt.render.complete`
- `prompt.call.start` / `prompt.call.complete`
- `prompt.throttled` (on retry)
- `prompt.error` (on failure)

## Implementing New Adapters

1. **Define configuration** - Create `ClientConfig` and `ModelConfig` frozen
   dataclasses extending `LLMConfig`.
1. **Instantiate client** - Accept either concrete client or config so tests can
   inject fakes.
1. **Render prompt** - Call `prompt.render` once, respecting overrides and
   instruction toggles.
1. **Delegate to `run_inner_loop`** - Supply provider-specific `call_provider`
   and `select_choice` functions.
1. **Wrap SDK failures** - Catch exceptions and re-raise
   `PromptEvaluationError`.
1. **Expose extras** - Raise helpful `RuntimeError` when optional dependencies
   are missing.

## Testing

### Unit Tests

- Mock provider responses for success, throttle, and error paths.
- Verify backoff calculations respect policy and jitter bounds.
- Test structured output parsing for valid and malformed payloads.
- Confirm deadline checks abort before provider calls when expired.

### Integration Tests

- Use provider test endpoints or sandboxed accounts.
- Verify tool dispatch round-trips through the full stack.
- Test throttle recovery with artificial 429 responses.

### Fixtures

- `tests/helpers/adapters.py` provides `MockAdapter` for prompt tests.
- `tests/fixtures/responses/` contains sample provider payloads.
