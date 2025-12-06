# Anthropic Adapter Specification

## Overview

**Scope:** This document specifies the design for an Anthropic adapter that integrates the native
`anthropic` Python SDK with WINK's adapter architecture. The adapter will leverage Anthropic's native
structured output capabilities (currently in public beta) and default to Claude Opus 4.5 as the
primary model.

**Design goals**

- Present a uniform adapter surface consistent with existing OpenAI and LiteLLM adapters so prompts
  behave identically across providers.
- Leverage Anthropic's native structured output feature for schema-guaranteed JSON responses,
  eliminating reliance on prompt-based instruction injection when possible.
- Support tool use (function calling) with optional strict mode for guaranteed tool definition
  compliance.
- Preserve deterministic, synchronous execution aligned with session reducers and the DbC
  enforcement model.
- Keep all provider interactions observable through the `EventBus` while isolating prompt rendering
  from transport details.

**Constraints**

- The adapter must run synchronously and honor the caller-supplied `Deadline` at every blocking
  boundary (request, tools, response parsing).
- Native structured outputs are a beta feature requiring the `anthropic-beta:
  structured-outputs-2025-11-13` header; the adapter must gracefully fall back to prompt-based
  parsing when the feature is unavailable or disabled.
- Tool execution must reuse the active `Session` and its `EventBus` so reducers observe the same
  state mutations that the adapter returns.
- The Anthropic SDK is an optional dependency; missing packages must fail fast with actionable
  errors.

**Default model**

The adapter defaults to `claude-opus-4-5-20250929` (Claude Opus 4.5), the most capable model
available for complex reasoning and agentic tasks. Callers may override this via the `model`
constructor parameter.

## Anthropic API Integration

### SDK Requirements

- **Package:** `anthropic` (version >= 0.75.0 for structured outputs and Opus 4.5 support)
- **Optional dependency:** `uv sync --extra anthropic` or `pip install weakincentives[anthropic]`
- **Import guard:** Raise a descriptive `RuntimeError` if the package is missing, mirroring
  OpenAI/LiteLLM patterns.

### Native Structured Outputs

Anthropic's structured outputs feature (public beta) guarantees that API responses match specified
JSON schemas by constraining token generation at inference time. This differs from prompt-based
approaches that rely on the model following instructions.

**Beta header:** `anthropic-beta: structured-outputs-2025-11-13`

**Supported models:** Claude Sonnet 4.5, Claude Opus 4.5, Claude Haiku 4.5

**Two implementation approaches:**

1. **`client.beta.messages.parse()`** - Higher-level helper that handles schema transformation
   automatically:
   ```python
   response = client.beta.messages.parse(
       model="claude-opus-4-5-20250929",
       max_tokens=8192,
       betas=["structured-outputs-2025-11-13"],
       messages=[{"role": "user", "content": prompt_text}],
       output_format=OutputDataclass,  # Pydantic model or dataclass
   )
   output = response.parsed_output
   ```

2. **`client.beta.messages.create()`** - Lower-level control with explicit schema:
   ```python
   response = client.beta.messages.create(
       model="claude-opus-4-5-20250929",
       max_tokens=8192,
       betas=["structured-outputs-2025-11-13"],
       messages=[{"role": "user", "content": prompt_text}],
       output_format={
           "type": "json_schema",
           "schema": json_schema_dict,
       },
   )
   # Parse response.content[0].text as JSON
   ```

**Response format:** When structured outputs are enabled, the response text is guaranteed to be
valid JSON matching the provided schema. Access via `response.content[0].text` for the raw JSON
string, or `response.parsed_output` when using the `parse()` helper.

### Tool Use (Function Calling)

Anthropic supports tool definitions with JSON Schema-based input specifications:

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    }
]
```

**Strict tool use:** Add `"strict": True` to tool definitions for guaranteed schema compliance
(requires the structured outputs beta header).

**Tool call response format:** When the model invokes a tool, the response includes:
```python
response.content[0].type == "tool_use"
response.content[0].id      # Tool call ID
response.content[0].name    # Tool name
response.content[0].input   # Parsed arguments dict
```

**Tool result message format:**
```python
{
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": result_string,
        }
    ]
}
```

### Message Format

Anthropic uses a different message structure than OpenAI:

- **System prompt:** Passed as a separate `system` parameter, not a message
- **User messages:** `{"role": "user", "content": "..."}`
- **Assistant messages:** `{"role": "assistant", "content": "..."}`
- **Multi-part content:** Content can be a list of content blocks for mixed media

## Adapter Architecture

### Module Structure

```
src/weakincentives/adapters/
├── _names.py           # Add ANTHROPIC_ADAPTER_NAME
├── config.py           # Add AnthropicClientConfig, AnthropicModelConfig
├── anthropic.py        # New adapter implementation
└── __init__.py         # Export AnthropicAdapter and config classes
```

### Typed Configuration

Following the pattern established by OpenAI and LiteLLM adapters, the Anthropic adapter uses frozen
dataclass configurations for type-safe instantiation and model parameter control.

**Client Configuration**

```python
@FrozenDataclass()
class AnthropicClientConfig:
    """Configuration for Anthropic client instantiation.

    Attributes:
        api_key: Anthropic API key. None uses the ANTHROPIC_API_KEY environment variable.
        base_url: Base URL for API requests. None uses the default Anthropic endpoint.
        timeout: Request timeout in seconds. None uses the client default.
        max_retries: Maximum number of retries. None uses the client default.
    """

    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    max_retries: int | None = None

    def to_client_kwargs(self) -> dict[str, Any]:
        """Convert non-None fields to client constructor kwargs."""
        kwargs: dict[str, Any] = {}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.max_retries is not None:
            kwargs["max_retries"] = self.max_retries
        return kwargs
```

**Model Configuration**

```python
@FrozenDataclass()
class AnthropicModelConfig(LLMConfig):
    """Anthropic-specific model configuration.

    Extends LLMConfig with parameters specific to Anthropic's API.

    Attributes:
        top_k: Sample from the top K most likely tokens. None uses the provider default.
        metadata: Optional metadata to include with the request.

    Notes:
        Anthropic does not support ``presence_penalty``, ``frequency_penalty``, or ``seed``.
        If any of these fields are provided, ``AnthropicModelConfig`` raises ``ValueError``
        so callers fail fast instead of issuing an invalid request.
    """

    top_k: int | None = None
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        unsupported: dict[str, object | None] = {
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        set_unsupported = [
            key for key, value in unsupported.items() if value is not None
        ]
        if set_unsupported:
            raise ValueError(
                "Unsupported Anthropic parameters: "
                + ", ".join(sorted(set_unsupported))
                + ". Remove them from AnthropicModelConfig."
            )

    @override
    def to_request_params(self) -> dict[str, Any]:
        """Convert non-None fields to request parameters."""
        params: dict[str, Any] = {}

        # Supported core fields
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.stop is not None:
            params["stop_sequences"] = list(self.stop)

        # Anthropic-specific fields
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.metadata is not None:
            params["metadata"] = dict(self.metadata)

        return params
```

### Adapter Constructor

```python
class AnthropicAdapter(ProviderAdapter[Any]):
    """Adapter that evaluates prompts against Anthropic's Messages API.

    Args:
        model: Model identifier (e.g., "claude-opus-4-5-20250929").
        model_config: Typed configuration for model parameters like temperature,
            max_tokens, etc. When provided, these values are merged into each
            request payload.
        tool_choice: Tool selection directive. Defaults to "auto".
        use_native_structured_output: When True, uses Anthropic's beta structured
            outputs feature for JSON schema enforcement. Defaults to True.
        client: Pre-configured Anthropic client instance. Mutually exclusive with
            client_config.
        client_config: Typed configuration for client instantiation. Used when
            client is not provided.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-5-20250929",
        model_config: AnthropicModelConfig | None = None,
        tool_choice: ToolChoice = "auto",
        use_native_structured_output: bool = True,
        client: _AnthropicProtocol | None = None,
        client_config: AnthropicClientConfig | None = None,
    ) -> None:
        ...
```

**Parameters:**

- `model`: Model identifier (default: `claude-opus-4-5-20250929`)
- `model_config`: Typed configuration for model parameters (temperature, max_tokens, top_k, etc.)
- `tool_choice`: Tool selection strategy (`"auto"`, `"any"`, `"none"`, or specific tool)
- `use_native_structured_output`: Enable beta structured outputs feature (default: `True`)
- `client`: Pre-configured Anthropic client instance
- `client_config`: Typed configuration for client instantiation

**Mutual exclusivity:** Reject providing `client` alongside `client_config` to avoid ambiguous
execution paths.

**Usage Examples**

```python
from weakincentives.adapters import AnthropicAdapter
from weakincentives.adapters.config import AnthropicClientConfig, AnthropicModelConfig

# Configure client and model parameters
client_config = AnthropicClientConfig(
    api_key="sk-ant-...",
    timeout=30.0,
)
model_config = AnthropicModelConfig(
    temperature=0.7,
    max_tokens=4096,
    top_k=40,
)

adapter = AnthropicAdapter(
    model="claude-opus-4-5-20250929",
    client_config=client_config,
    model_config=model_config,
)

# Or with pre-configured client
from anthropic import Anthropic
client = Anthropic(api_key="sk-ant-...")
adapter = AnthropicAdapter(
    model="claude-opus-4-5-20250929",
    client=client,
    model_config=model_config,  # Model params still apply
)
```

### Adapter Name

Add to `_names.py`:
```python
ANTHROPIC_ADAPTER_NAME: Final[AdapterName] = "anthropic"
"""Canonical label for the Anthropic adapter."""
```

### Protocol Definitions

```python
class _AnthropicProtocol(Protocol):
    """Structural type for the Anthropic client."""

    class beta:
        class messages:
            @staticmethod
            def create(*args: object, **kwargs: object) -> Any: ...

            @staticmethod
            def parse(*args: object, **kwargs: object) -> Any: ...


class _AnthropicClientFactory(Protocol):
    def __call__(self, **kwargs: object) -> _AnthropicProtocol: ...
```

## Execution Lifecycle

### Evaluate Signature

The adapter implements the standard `ProviderAdapter.evaluate()` signature with full support for
budgets and visibility overrides:

```python
@override
def evaluate(
    self,
    prompt: Prompt[OutputT],
    *,
    bus: EventBus,
    session: SessionProtocol,
    parse_output: bool = True,
    deadline: Deadline | None = None,
    visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
    budget: Budget | None = None,
    budget_tracker: BudgetTracker | None = None,
) -> PromptResponse[OutputT]:
    """Evaluate the prompt and return a structured response.

    When ``budget`` is provided and ``budget_tracker`` is not, a new tracker
    is created. When ``budget_tracker`` is supplied (typically by a parent
    during subagent dispatch), it is used directly for shared limit enforcement.
    """
    ...
```

**Budget Integration:**

- When `budget` is provided without `budget_tracker`, create a new `BudgetTracker` for this
  evaluation.
- When `budget_tracker` is supplied (e.g., from a parent agent), use it directly for shared limit
  enforcement across subagent calls.
- Token usage is recorded after each provider response via `budget_tracker.record_cumulative()`.
- Budget limits are checked after provider responses and tool executions; violations raise
  `PromptEvaluationError` with `phase="budget"`.

**Visibility Overrides:**

- Pass `visibility_overrides` to `prompt.render()` for progressive disclosure control.
- Sections can be hidden, collapsed, or expanded based on the override mapping.

### Evaluate Flow

1. **Validate deadline and budget** - Reject already-expired deadlines before rendering. Initialize
   budget tracker if budget is provided.

2. **Render prompt** - Call `prompt.render(visibility_overrides=...)`, optionally disabling output
   instructions when native structured outputs are enabled.

3. **Build request payload:**
   - Extract system prompt from rendered text
   - Assemble user messages
   - Convert tools via `tool_to_anthropic_spec()`
   - Build `output_format` when structured output is requested
   - Merge model config parameters via `model_config.to_request_params()`

4. **Call provider:**
   - Use `client.beta.messages.create()` with the structured outputs beta header
   - Pass `betas=["structured-outputs-2025-11-13"]` when native structured output is enabled

5. **Record and check budget** - After each provider response:
   - Extract token usage from response
   - Record cumulative usage via `budget_tracker.record_cumulative()`
   - Check budget limits via `budget_tracker.check()`

6. **Handle tool calls** - When response contains `tool_use` blocks:
   - Execute tools via `ToolExecutor` (which receives `budget_tracker` in `ToolContext`)
   - Format tool results as Anthropic-style `tool_result` messages
   - Check budget after tool execution
   - Continue conversation loop

7. **Parse response:**
   - Extract text from content blocks
   - When structured output is enabled, parse JSON from response text
   - Fall back to prompt-based parsing when native parsing fails
   - Final budget check before returning

8. **Publish events** - Emit `PromptRendered`, `ToolInvoked`, and `PromptExecuted` events.

### Tool Specification Translation

```python
def tool_to_anthropic_spec(
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Convert WINK tool to Anthropic tool specification."""

    if tool.params_type is type(None):
        input_schema = {"type": "object", "properties": {}}
    else:
        input_schema = schema(tool.params_type, extra="forbid")
        _ = input_schema.pop("title", None)

    spec: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "input_schema": input_schema,
    }
    if strict:
        spec["strict"] = True
    return spec
```

### Tool Choice Translation

Anthropic tool choice differs from OpenAI:

| WINK ToolChoice | Anthropic Equivalent |
|-----------------|---------------------|
| `"auto"` | `{"type": "auto"}` |
| `"none"` | Not passed (omit tools) |
| `{"type": "function", "function": {"name": "x"}}` | `{"type": "tool", "name": "x"}` |

### Message Normalization

Convert WINK conversation messages to Anthropic format:

```python
def _normalize_messages_for_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Extract system prompt and normalize messages for Anthropic API."""

    system_prompt = ""
    normalized: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            system_prompt = content or ""
            continue

        if role == "tool":
            # Convert tool result message
            normalized.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id"),
                    "content": content or "",
                }],
            })
            continue

        if role == "assistant" and message.get("tool_calls"):
            # Convert assistant message with tool calls
            content_blocks: list[dict[str, Any]] = []
            if content:
                content_blocks.append({"type": "text", "text": content})
            for tool_call in message["tool_calls"]:
                func = tool_call.get("function", {})
                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_call.get("id"),
                    "name": func.get("name"),
                    "input": json.loads(func.get("arguments", "{}")),
                })
            normalized.append({"role": "assistant", "content": content_blocks})
            continue

        normalized.append({"role": role, "content": content})

    return system_prompt, normalized
```

## Structured Output Handling

### Schema Construction

Reuse `build_json_schema_response_format()` from shared utilities to construct the JSON schema,
then extract the schema portion for Anthropic's `output_format` parameter:

```python
def _build_anthropic_output_format(
    rendered: RenderedPrompt[Any],
    prompt_name: str,
) -> dict[str, Any] | None:
    """Build Anthropic output_format from rendered prompt metadata."""

    response_format = build_json_schema_response_format(rendered, prompt_name)
    if response_format is None:
        return None

    json_schema = response_format["json_schema"]
    return {
        "type": "json_schema",
        "schema": json_schema["schema"],
    }
```

### Response Parsing Strategy

1. **Native structured output path:** When `use_native_structured_output=True` and the response
   contains valid JSON in `content[0].text`, parse directly into the target dataclass.

2. **Fallback path:** When native parsing fails or is disabled, use `parse_structured_output()` with
   the rendered prompt's parsing helpers.

3. **Text-only path:** When `parse_output=False`, return raw text without structured parsing.

### Content Extraction

```python
def _extract_anthropic_content(response: Any) -> tuple[str, list[Any]]:
    """Extract text content and tool use blocks from Anthropic response."""

    text_parts: list[str] = []
    tool_uses: list[Any] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_uses.append(block)

    return "".join(text_parts), tool_uses
```

## Throttle Handling

Map Anthropic-specific errors to `ThrottleKind`:

```python
def _normalize_anthropic_throttle(
    error: Exception,
    *,
    prompt_name: str,
) -> ThrottleError | None:
    """Detect and normalize Anthropic throttling errors."""

    message = str(error) or "Anthropic request failed."
    lower_message = message.lower()
    status_code = getattr(error, "status_code", None)
    class_name = error.__class__.__name__.lower()
    kind: ThrottleKind | None = None

    if status_code == 429 or "rate" in lower_message:
        kind = "rate_limit"
    elif "overloaded" in lower_message or status_code == 529:
        kind = "rate_limit"  # API overloaded
    elif "timeout" in class_name:
        kind = "timeout"

    if kind is None:
        return None

    retry_after = _retry_after_from_error(error)
    return ThrottleError(
        message,
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_REQUEST,
        details=throttle_details(
            kind=kind,
            retry_after=retry_after,
            provider_payload=_error_payload(error),
        ),
    )
```

## Caveats and Limitations

### Beta Feature Status

Native structured outputs are currently in public beta. The feature:

- Requires the `anthropic-beta: structured-outputs-2025-11-13` header
- May change without notice before GA
- Guarantees format compliance, not content accuracy (hallucinations remain possible)

### Model Availability

- Claude Opus 4.5 (`claude-opus-4-5-20250929`) is the default
- Structured outputs require Claude Sonnet 4.5, Opus 4.5, or Haiku 4.5
- Older models require prompt-based parsing fallback

### API Differences from OpenAI

- System prompts are a separate parameter, not messages
- Tool results use `tool_result` content blocks, not `tool` role messages
- Content is always a list of typed blocks, not a simple string
- No direct `.parsed` attribute on responses (must parse `content[0].text`)

### Token Limits

Claude Opus 4.5 supports up to 200K input tokens and 8K output tokens by default. Configure
output token limits via `AnthropicModelConfig.max_tokens`.

### Budget Enforcement

The adapter fully integrates with WINK's `Budget` abstraction:

- **Token limits:** `max_total_tokens`, `max_input_tokens`, `max_output_tokens` are enforced
  cumulatively across all provider calls within an evaluation.
- **Deadline:** `Budget.deadline` is checked alongside the `deadline` parameter.
- **Subagent sharing:** When `budget_tracker` is passed (e.g., from a parent prompt), token usage
  is aggregated across the entire agent tree.
- **Error phase:** Budget violations raise `PromptEvaluationError` with `phase="budget"`.

Anthropic responses include `usage.input_tokens` and `usage.output_tokens` which the adapter
extracts and records via `token_usage_from_payload()`.

## Testing Expectations

- **Unit tests:** Mock the Anthropic client and verify request payload construction, response
  parsing, and error handling.
- **Integration tests:** Require `ANTHROPIC_API_KEY` environment variable; opt-in via
  `make integration-tests`.
- **Structured output tests:** Verify both native and fallback parsing paths.
- **Tool execution tests:** Cover single and multi-turn tool conversations.
- **Budget tests:** Verify token tracking, cumulative usage across tool calls, and
  `BudgetExceededError` propagation as `PromptEvaluationError` with `phase="budget"`.
- **Visibility tests:** Verify `visibility_overrides` are passed through to prompt rendering.
- **Config tests:** Verify `AnthropicClientConfig` and `AnthropicModelConfig` produce correct
  request parameters, and that unsupported fields raise `ValueError`.
- **Error handling tests:** Verify throttle detection, deadline enforcement, and graceful
  degradation when the beta feature is unavailable.

## Public API

```python
from weakincentives.adapters import AnthropicAdapter
from weakincentives.adapters.config import AnthropicClientConfig, AnthropicModelConfig
from weakincentives.budget import Budget
from weakincentives.deadlines import Deadline
from datetime import timedelta

# Basic usage with defaults (Claude Opus 4.5)
adapter = AnthropicAdapter()

response = adapter.evaluate(
    prompt,
    bus=bus,
    session=session,
)

# Configured adapter with model parameters
model_config = AnthropicModelConfig(
    temperature=0.7,
    max_tokens=4096,
    top_k=40,
)

adapter = AnthropicAdapter(
    model="claude-opus-4-5-20250929",
    model_config=model_config,
    use_native_structured_output=True,
)

# Evaluate with budget and deadline constraints
budget = Budget(
    deadline=Deadline.after(timedelta(minutes=5)),
    max_total_tokens=100000,
    max_output_tokens=8000,
)

response = adapter.evaluate(
    prompt,
    bus=bus,
    session=session,
    budget=budget,
)

# Evaluate with visibility overrides for progressive disclosure
from weakincentives.prompt import SectionVisibility

response = adapter.evaluate(
    prompt,
    bus=bus,
    session=session,
    visibility_overrides={
        ("details",): SectionVisibility.COLLAPSED,
        ("advanced", "options"): SectionVisibility.HIDDEN,
    },
)
```

## Migration Notes

- Callers currently using LiteLLM with `anthropic/` model prefixes can migrate to the native
  adapter for improved structured output support.
- The adapter follows identical `evaluate()` signature and `PromptResponse` return type as
  OpenAI/LiteLLM adapters.
- Tool handlers require no changes; the shared `ToolExecutor` handles provider-agnostic execution.

## References

- [Anthropic Structured Outputs Documentation](https://docs.claude.com/en/docs/build-with-claude/structured-outputs)
- [Anthropic Python SDK Releases](https://github.com/anthropics/anthropic-sdk-python/releases)
- [Claude API Tool Use](https://docs.claude.com/en/docs/build-with-claude/tool-use)
