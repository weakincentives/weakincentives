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
├── anthropic.py        # New adapter implementation
└── __init__.py         # Export AnthropicAdapter
```

### Configuration Surfaces

```python
class AnthropicAdapter(ProviderAdapter[Any]):
    def __init__(
        self,
        *,
        model: str = "claude-opus-4-5-20250929",
        tool_choice: ToolChoice = "auto",
        use_native_structured_output: bool = True,
        max_tokens: int = 8192,
        client: _AnthropicProtocol | None = None,
        client_factory: _AnthropicClientFactory | None = None,
        client_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        ...
```

**Parameters:**

- `model`: Model identifier (default: `claude-opus-4-5-20250929`)
- `tool_choice`: Tool selection strategy (`"auto"`, `"any"`, `"none"`, or specific tool)
- `use_native_structured_output`: Enable beta structured outputs feature (default: `True`)
- `max_tokens`: Maximum tokens in response (default: 8192)
- `client`: Pre-configured Anthropic client instance
- `client_factory`: Factory function for creating clients
- `client_kwargs`: Additional kwargs passed to client construction

**Mutual exclusivity:** Reject mixing `client` with `client_factory`/`client_kwargs` to avoid
ambiguous execution paths.

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

### Evaluate Flow

1. **Validate deadline** - Reject already-expired deadlines before rendering.

2. **Render prompt** - Call `prompt.render()`, optionally disabling output instructions when native
   structured outputs are enabled.

3. **Build request payload:**
   - Extract system prompt from rendered text
   - Assemble user messages
   - Convert tools via `tool_to_anthropic_spec()`
   - Build `output_format` when structured output is requested

4. **Call provider:**
   - Use `client.beta.messages.create()` with the structured outputs beta header
   - Pass `betas=["structured-outputs-2025-11-13"]` when native structured output is enabled

5. **Handle tool calls** - When response contains `tool_use` blocks:
   - Execute tools via `ToolExecutor`
   - Format tool results as Anthropic-style `tool_result` messages
   - Continue conversation loop

6. **Parse response:**
   - Extract text from content blocks
   - When structured output is enabled, parse JSON from response text
   - Fall back to prompt-based parsing when native parsing fails

7. **Publish events** - Emit `PromptRendered`, `ToolInvoked`, and `PromptExecuted` events.

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

Claude Opus 4.5 supports up to 200K input tokens and 8K output tokens by default. The adapter
exposes `max_tokens` as a constructor parameter for customization.

## Testing Expectations

- **Unit tests:** Mock the Anthropic client and verify request payload construction, response
  parsing, and error handling.
- **Integration tests:** Require `ANTHROPIC_API_KEY` environment variable; opt-in via
  `make integration-tests`.
- **Structured output tests:** Verify both native and fallback parsing paths.
- **Tool execution tests:** Cover single and multi-turn tool conversations.
- **Error handling tests:** Verify throttle detection, deadline enforcement, and graceful
  degradation when the beta feature is unavailable.

## Public API

```python
from weakincentives.adapters import AnthropicAdapter

adapter = AnthropicAdapter(
    model="claude-opus-4-5-20250929",  # Default
    use_native_structured_output=True,  # Enable beta feature
    max_tokens=8192,
)

response = adapter.evaluate(
    prompt,
    bus=bus,
    session=session,
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
