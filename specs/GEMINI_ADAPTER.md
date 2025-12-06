# Gemini Adapter Implementation Specification

## Overview

**Scope:** This document specifies the `GeminiAdapter` implementation for integrating Google's
Gemini models via the official `google-genai` SDK. The adapter enables native structured output
support and defaults to the Gemini 3.0 Pro model.

**Design goals**

- Provide first-class Gemini support using the official `google-genai` SDK (not `google-generativeai`).
- Enable native JSON schema constrained output via `response_schema` configuration.
- Default to `gemini-3.0-pro` while supporting all Gemini models available through the API.
- Maintain full compatibility with the existing adapter architecture and `ConversationRunner`.
- Support both Gemini Developer API and Vertex AI deployment modes.

**Constraints**

- Must follow the synchronous execution model required by `ProviderAdapter`.
- Must honor `Deadline` at every blocking boundary.
- Structured output parsing must work with and without native schema enforcement.
- Tool execution must reuse the active `Session` and `EventBus`.
- All errors must be wrapped as `PromptEvaluationError` with appropriate phase metadata.

## SDK Selection

The adapter uses the `google-genai` package, which is Google's current official SDK:

```bash
uv sync --extra gemini
# or
pip install weakincentives[gemini]
```

**Rationale:** The older `google-generativeai` package is deprecated with support ending August 31,
2025. The `google-genai` SDK provides:

- Unified API for both Gemini Developer API and Vertex AI
- Native structured output via `response_schema`
- Improved async support
- Active development and feature parity with Google AI Studio

## Default Model

The adapter defaults to `gemini-3.0-pro`:

```python
DEFAULT_GEMINI_MODEL = "gemini-3.0-pro"
```

**Gemini 3.0 Pro capabilities:**

- 1M token context window
- Native tool/function calling
- JSON schema constrained output
- Multimodal input (text, images, audio, video, PDFs)
- Thinking level control (`low`, `high`) for reasoning depth
- Top benchmark performance (1501 Elo on LMArena)

Model identifier mapping:

| Display Name | API Identifier |
|-------------|----------------|
| Gemini 3.0 Pro | `gemini-3.0-pro` |
| Gemini 2.5 Flash | `gemini-2.5-flash` |
| Gemini 2.5 Pro | `gemini-2.5-pro` |
| Gemini 2.0 Flash | `gemini-2.0-flash-001` |

## Architecture

### File Structure

```
src/weakincentives/adapters/
├── __init__.py           # Export GeminiAdapter
├── _names.py             # Add GEMINI_ADAPTER_NAME
├── gemini.py             # Main adapter implementation
└── _gemini_protocols.py  # Structural typing for SDK types (optional)
```

### Adapter Class

```python
class GeminiAdapter(ProviderAdapter[Any]):
    """Adapter that evaluates prompts against Google's Gemini API.

    Args:
        model: Model identifier (e.g., "gemini-3.0-pro").
        model_config: Typed configuration for model parameters like temperature,
            max_tokens, thinking_level, etc. When provided, these values are
            merged into each request payload.
        tool_choice: Tool selection directive. Defaults to "auto".
        use_native_response_format: When True, uses Gemini's JSON schema response
            format for structured outputs. Defaults to True.
        client: Pre-configured Gemini client instance. Mutually exclusive with
            client_config.
        client_config: Typed configuration for client instantiation. Used when
            client is not provided.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3.0-pro",
        model_config: GeminiModelConfig | None = None,
        tool_choice: ToolChoice = "auto",
        use_native_response_format: bool = True,
        client: GeminiClientProtocol | None = None,
        client_config: GeminiClientConfig | None = None,
    ) -> None: ...

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
    ) -> PromptResponse[OutputT]: ...
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gemini-3.0-pro"` | Gemini model identifier |
| `model_config` | `GeminiModelConfig \| None` | `None` | Typed model parameters |
| `tool_choice` | `ToolChoice` | `"auto"` | Tool invocation mode |
| `use_native_response_format` | `bool` | `True` | Enable JSON schema output |
| `client` | `GeminiClientProtocol \| None` | `None` | Pre-configured client |
| `client_config` | `GeminiClientConfig \| None` | `None` | Typed client configuration |

### Typed Configuration

Gemini adapter uses frozen dataclass configurations for type-safe instantiation and model parameter
control, following the pattern established by OpenAI and LiteLLM adapters.

**Client Configuration**

```python
@FrozenDataclass()
class GeminiClientConfig:
    """Configuration for Gemini client instantiation.

    Attributes:
        api_key: Google API key. None uses the GOOGLE_API_KEY environment variable.
        vertexai: Whether to use Vertex AI instead of Gemini Developer API.
        project: GCP project ID for Vertex AI. Required when vertexai=True.
        location: GCP region for Vertex AI. Defaults to "us-central1".
        api_version: API version string. Defaults to "v1".
    """

    api_key: str | None = None
    vertexai: bool = False
    project: str | None = None
    location: str | None = None
    api_version: str = "v1"

    def __post_init__(self) -> None:
        if self.vertexai and self.project is None:
            raise ValueError("project is required when vertexai=True")

    def to_client_kwargs(self) -> dict[str, Any]:
        """Convert non-None fields to client constructor kwargs."""
        kwargs: dict[str, Any] = {}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.vertexai:
            kwargs["vertexai"] = True
            kwargs["project"] = self.project
            if self.location is not None:
                kwargs["location"] = self.location
        kwargs["http_options"] = HttpOptions(api_version=self.api_version)
        return kwargs
```

**Model Configuration**

```python
@FrozenDataclass()
class GeminiModelConfig(LLMConfig):
    """Gemini-specific model configuration.

    Extends LLMConfig with parameters specific to Gemini's API.

    Attributes:
        thinking_level: Reasoning depth control ("low" or "high"). Gemini 3.0+
            feature that balances response quality vs latency/cost.
        candidate_count: Number of response variations to generate. None uses
            the provider default (1).
        safety_settings: Safety threshold configuration. None uses provider defaults.

    Notes:
        Gemini uses ``max_output_tokens`` instead of ``max_tokens`` in the API,
        but this config accepts ``max_tokens`` for consistency with LLMConfig
        and performs the rename in ``to_request_params()``.
    """

    thinking_level: Literal["low", "high"] | None = None
    candidate_count: int | None = None
    safety_settings: Mapping[str, str] | None = None

    @override
    def to_request_params(self) -> dict[str, Any]:
        """Convert non-None fields to GenerateContentConfig parameters."""
        params: dict[str, Any] = {}

        # Core LLM parameters
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_output_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.stop is not None:
            params["stop_sequences"] = list(self.stop)
        if self.seed is not None:
            params["seed"] = self.seed
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty

        # Gemini-specific parameters
        if self.thinking_level is not None:
            params["thinking_level"] = self.thinking_level
        if self.candidate_count is not None:
            params["candidate_count"] = self.candidate_count
        if self.safety_settings is not None:
            params["safety_settings"] = dict(self.safety_settings)

        return params
```

**Configuration in config.py**

Add these classes to `src/weakincentives/adapters/config.py` and export from the package:

```python
# In config.py
__all__ = [
    # ... existing exports
    "GeminiClientConfig",
    "GeminiModelConfig",
]
```

## SDK Integration

### Client Initialization

```python
from google import genai
from google.genai.types import HttpOptions

def create_gemini_client(**kwargs: object) -> genai.Client:
    """Create a Gemini client, raising a helpful error if the extra is missing."""
    try:
        from google import genai
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Gemini support requires the optional 'google-genai' dependency. "
            "Install it with `uv sync --extra gemini` or "
            "`pip install weakincentives[gemini]`."
        ) from exc

    return genai.Client(**kwargs)
```

The adapter instantiates clients via `GeminiClientConfig.to_client_kwargs()`:

```python
# In adapter __init__
if client is not None:
    if client_config is not None:
        raise ValueError(
            "client_config cannot be provided when an explicit client is supplied."
        )
else:
    client_kwargs = client_config.to_client_kwargs() if client_config else {}
    client = create_gemini_client(**client_kwargs)
```

### Authentication

The SDK supports multiple authentication modes, all configurable via `GeminiClientConfig`:

1. **API Key (Gemini Developer API):**
   ```python
   config = GeminiClientConfig(api_key="YOUR_API_KEY")
   adapter = GeminiAdapter(model="gemini-3.0-pro", client_config=config)
   ```

2. **Application Default Credentials (Vertex AI):**
   ```python
   config = GeminiClientConfig(
       vertexai=True,
       project="your-project",
       location="us-central1",
   )
   adapter = GeminiAdapter(model="gemini-3.0-pro", client_config=config)
   ```

3. **Environment Variables:**
   - `GOOGLE_API_KEY` for API key authentication (used when `api_key=None`)
   - `GOOGLE_APPLICATION_CREDENTIALS` for service account

## Structured Output

### Native JSON Schema Support

When `use_native_response_format=True`, the adapter configures native schema enforcement:

```python
from google.genai import types

config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=schema,  # JSON Schema or Pydantic model
)
```

### Schema Translation

The adapter translates weakincentives JSON schemas to Gemini format:

```python
def _build_gemini_response_schema(
    rendered: RenderedPrompt[Any],
    *,
    prompt_name: str,
) -> dict[str, Any] | None:
    """Build Gemini-compatible JSON schema from rendered prompt."""
    if rendered.output_type is None or rendered.container is None:
        return None

    # Use build_json_schema_response_format to get the base schema
    base_format = build_json_schema_response_format(rendered, prompt_name)
    json_schema = base_format["json_schema"]["schema"]

    # Gemini uses OpenAPI-style type names (uppercase)
    return _convert_to_gemini_schema(json_schema)


def _convert_to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON Schema to Gemini schema format.

    Gemini uses uppercase type names: STRING, NUMBER, INTEGER, BOOLEAN,
    ARRAY, OBJECT instead of lowercase JSON Schema types.
    """
    converted = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            converted[key] = value.upper()
        elif isinstance(value, dict):
            converted[key] = _convert_to_gemini_schema(value)
        elif isinstance(value, list):
            converted[key] = [
                _convert_to_gemini_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            converted[key] = value
    return converted
```

### Response Parsing

The SDK provides parsed output directly when schemas are used:

```python
# Access parsed output
parsed_data = response.parsed  # Returns dict matching schema

# Or access raw text
raw_text = response.text
```

The adapter should prefer `response.parsed` when available, falling back to text parsing.

## Tool/Function Calling

### Tool Declaration

Gemini supports function declarations via the `tools` config parameter:

```python
from google.genai import types

def _tool_to_gemini_declaration(
    spec: Mapping[str, Any],
) -> types.FunctionDeclaration:
    """Convert weakincentives tool spec to Gemini FunctionDeclaration."""
    function_payload = spec.get("function", {})
    return types.FunctionDeclaration(
        name=function_payload.get("name"),
        description=function_payload.get("description"),
        parameters=_convert_to_gemini_schema(
            function_payload.get("parameters", {})
        ),
    )

# In provider invoker:
config = types.GenerateContentConfig(
    tools=[
        types.Tool(function_declarations=[
            _tool_to_gemini_declaration(spec)
            for spec in tool_specs
        ])
    ],
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode=tool_mode),
    ),
)
```

### Tool Choice Configuration

Map weakincentives `ToolChoice` to Gemini's `FunctionCallingConfig`:

| weakincentives | Gemini Mode |
|----------------|-------------|
| `"auto"` | `"AUTO"` |
| `"none"` | `"NONE"` |
| `"required"` | `"ANY"` |
| `{"type": "function", "name": "..."}` | `"ANY"` + `allowed_function_names` |

```python
def _normalize_tool_choice(
    tool_choice: ToolChoice | None,
    tool_specs: Sequence[Mapping[str, Any]],
) -> types.ToolConfig | None:
    """Convert tool choice to Gemini ToolConfig.

    GenerateContentConfig.tool_config expects a ToolConfig wrapper around
    FunctionCallingConfig, not the inner FunctionCallingConfig directly.
    """
    function_calling_config: types.FunctionCallingConfig | None = None

    if tool_choice is None or tool_choice == "auto":
        function_calling_config = types.FunctionCallingConfig(mode="AUTO")
    elif tool_choice == "none":
        function_calling_config = types.FunctionCallingConfig(mode="NONE")
    elif tool_choice == "required":
        function_calling_config = types.FunctionCallingConfig(mode="ANY")
    elif isinstance(tool_choice, Mapping):
        # Force specific function
        function_name = tool_choice.get("function", {}).get("name")
        if function_name:
            function_calling_config = types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=[function_name],
            )

    if function_calling_config is None:
        return None

    return types.ToolConfig(function_calling_config=function_calling_config)
```

### Function Call Response Handling

```python
def _extract_tool_calls(
    response: types.GenerateContentResponse,
) -> list[ProviderToolCallData]:
    """Extract tool calls from Gemini response."""
    tool_calls = []
    for part in response.candidates[0].content.parts:
        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            tool_calls.append(
                ProviderToolCallData(
                    id=None,  # Gemini doesn't use call IDs
                    function=ProviderFunctionCallData(
                        name=fc.name,
                        arguments=json.dumps(dict(fc.args)) if fc.args else None,
                    ),
                )
            )
    return tool_calls
```

### Function Response Format

Gemini expects function responses in a specific format:

```python
def _serialize_tool_result(
    tool_name: str,
    result: str,
) -> types.Content:
    """Create Gemini-compatible function response."""
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name=tool_name,
                    response={"result": result},
                )
            )
        ],
    )
```

## Message Format

### Content Structure

Gemini uses a `Content` structure with `parts`:

```python
def _normalize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[types.Content]:
    """Convert weakincentives messages to Gemini Content format."""
    contents = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "system":
            # System messages go in config.system_instruction
            continue

        gemini_role = "model" if role == "assistant" else "user"
        contents.append(
            types.Content(
                role=gemini_role,
                parts=[types.Part(text=content)],
            )
        )
    return contents
```

### System Instructions

Gemini handles system prompts via `system_instruction` in config:

```python
config = types.GenerateContentConfig(
    system_instruction=system_prompt,
    # ... other config
)
```

## Error Handling

### Exception Mapping

```python
from google.genai import errors as genai_errors

def _normalize_gemini_error(
    error: Exception,
    *,
    prompt_name: str,
) -> PromptEvaluationError | ThrottleError:
    """Convert Gemini SDK errors to weakincentives error types."""
    if isinstance(error, genai_errors.APIError):
        code = error.code
        message = error.message or str(error)

        # Rate limiting
        if code == 429:
            return ThrottleError(
                message,
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
                details=throttle_details(
                    kind="rate_limit",
                    retry_after=_extract_retry_after(error),
                    provider_payload={"code": code, "message": message},
                ),
            )

        # Quota exhausted
        if code == 403 and "quota" in message.lower():
            return ThrottleError(
                message,
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
                details=throttle_details(
                    kind="quota_exhausted",
                    provider_payload={"code": code, "message": message},
                ),
            )

    return PromptEvaluationError(
        str(error),
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_REQUEST,
        provider_payload=_error_payload(error),
    )
```

### HTTP Status Mapping

| HTTP Status | Error Type | ThrottleKind |
|-------------|------------|--------------|
| 429 | `ThrottleError` | `"rate_limit"` |
| 403 (quota) | `ThrottleError` | `"quota_exhausted"` |
| 408, 504 | `ThrottleError` | `"timeout"` |
| 400 | `PromptEvaluationError` | - |
| 401, 403 | `PromptEvaluationError` | - |
| 404 | `PromptEvaluationError` | - |
| 500+ | `PromptEvaluationError` | - |

## Provider Invoker Implementation

```python
def _build_provider_invoker(
    self,
    prompt_name: str,
    system_instruction: str,
) -> ProviderInvoker:
    """Build the callable for ConversationRunner."""

    def _call_provider(
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice_directive: ToolChoice | None,
        response_format_payload: Mapping[str, Any] | None,
    ) -> types.GenerateContentResponse:
        # Build configuration
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
        }

        # Merge model config parameters (temperature, max_output_tokens, thinking_level, etc.)
        if self._model_config is not None:
            config_kwargs.update(self._model_config.to_request_params())

        # Add tools if provided
        if tool_specs:
            declarations = [
                _tool_to_gemini_declaration(spec) for spec in tool_specs
            ]
            config_kwargs["tools"] = [
                types.Tool(function_declarations=declarations)
            ]
            tool_config = _normalize_tool_choice(tool_choice_directive, tool_specs)
            if tool_config:
                config_kwargs["tool_config"] = tool_config

        # Add response schema if structured output requested
        if response_format_payload:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_format_payload.get("schema")

        config = types.GenerateContentConfig(**config_kwargs)

        # Convert messages to Gemini format
        contents = _normalize_messages(messages)

        try:
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as error:
            raise _normalize_gemini_error(error, prompt_name=prompt_name) from error

    return _call_provider
```

## Choice Selector Implementation

```python
def _build_choice_selector(
    prompt_name: str,
) -> Callable[[object], ProviderChoice]:
    """Build selector for extracting response data."""

    def _select_choice(response: object) -> ProviderChoice:
        if not hasattr(response, "candidates") or not response.candidates:
            raise PromptEvaluationError(
                "Provider response did not include any candidates.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_RESPONSE,
            )

        candidate = response.candidates[0]
        content = candidate.content

        # Check for tool calls
        tool_calls = _extract_tool_calls(response)
        if tool_calls:
            return ProviderChoiceData(
                message=ProviderMessageData(
                    content=(),
                    tool_calls=tuple(tool_calls),
                    parsed=None,
                )
            )

        # Extract text content
        text_parts = []
        for part in content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        # Check for parsed structured output
        parsed = getattr(response, "parsed", None)

        return ProviderChoiceData(
            message=ProviderMessageData(
                content=tuple({"type": "text", "text": t} for t in text_parts),
                tool_calls=None,
                parsed=parsed,
            )
        )

    return _select_choice
```

## Dependency Configuration

### pyproject.toml

```toml
[project.optional-dependencies]
gemini = ["google-genai>=1.0.0"]
```

### __init__.py Export

```python
# src/weakincentives/adapters/__init__.py
from .config import GeminiClientConfig, GeminiModelConfig
from .gemini import GeminiAdapter, create_gemini_client

__all__ = [
    # ... existing exports
    "GeminiAdapter",
    "GeminiClientConfig",
    "GeminiModelConfig",
    "create_gemini_client",
]
```

### Adapter Name

```python
# src/weakincentives/adapters/_names.py
GEMINI_ADAPTER_NAME: Final[str] = "gemini"
```

## Testing Requirements

### Unit Tests

```python
# tests/adapters/test_gemini_adapter.py

class TestGeminiAdapter:
    def test_evaluate_basic_prompt(self, mock_gemini_client): ...
    def test_evaluate_with_structured_output(self, mock_gemini_client): ...
    def test_evaluate_with_tools(self, mock_gemini_client): ...
    def test_tool_choice_auto(self, mock_gemini_client): ...
    def test_tool_choice_none(self, mock_gemini_client): ...
    def test_tool_choice_required(self, mock_gemini_client): ...
    def test_tool_choice_specific_function(self, mock_gemini_client): ...
    def test_deadline_enforcement(self, mock_gemini_client): ...
    def test_rate_limit_error_handling(self, mock_gemini_client): ...
    def test_quota_exhausted_error_handling(self, mock_gemini_client): ...
    def test_schema_conversion_to_gemini_format(self): ...
    def test_message_normalization(self): ...
    def test_model_config_merged_into_request(self, mock_gemini_client): ...
    def test_budget_tracking(self, mock_gemini_client): ...
    def test_budget_exceeded_error(self, mock_gemini_client): ...
    def test_visibility_overrides(self, mock_gemini_client): ...


class TestGeminiClientConfig:
    def test_api_key_config(self): ...
    def test_vertexai_requires_project(self): ...
    def test_to_client_kwargs_api_key(self): ...
    def test_to_client_kwargs_vertexai(self): ...


class TestGeminiModelConfig:
    def test_to_request_params_basic(self): ...
    def test_to_request_params_thinking_level(self): ...
    def test_max_tokens_renamed_to_max_output_tokens(self): ...
    def test_stop_renamed_to_stop_sequences(self): ...
```

### Test Stubs

```python
# tests/adapters/_gemini_stubs.py

@dataclass
class MockFunctionCall:
    name: str
    args: dict[str, Any]

@dataclass
class MockPart:
    text: str | None = None
    function_call: MockFunctionCall | None = None

@dataclass
class MockContent:
    role: str
    parts: list[MockPart]

@dataclass
class MockCandidate:
    content: MockContent
    finish_reason: str = "STOP"

@dataclass
class MockGenerateContentResponse:
    candidates: list[MockCandidate]
    parsed: Any | None = None
    text: str | None = None
```

### Integration Tests

```python
# tests/integration/test_gemini_integration.py

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Requires GOOGLE_API_KEY")
class TestGeminiIntegration:
    def test_basic_generation(self): ...
    def test_structured_output(self): ...
    def test_tool_execution(self): ...
    def test_multi_turn_conversation(self): ...
```

## Usage Examples

### Basic Usage

```python
from weakincentives import Prompt, MarkdownSection
from weakincentives.adapters import GeminiAdapter

adapter = GeminiAdapter()  # Uses gemini-3.0-pro by default

prompt = Prompt[str](
    ns="example",
    key="greeting",
    name="greeting_prompt",
    sections=[
        MarkdownSection(
            title="Task",
            template="Say hello to $name",
            key="task",
        ),
    ],
)

response = adapter.evaluate(
    prompt.with_params(name="World"),
    bus=bus,
    session=session,
)
print(response.text)  # "Hello, World!"
```

### With Typed Configuration

```python
from weakincentives.adapters import (
    GeminiAdapter,
    GeminiClientConfig,
    GeminiModelConfig,
)

# Configure client and model parameters
client_config = GeminiClientConfig(
    api_key="YOUR_API_KEY",
)
model_config = GeminiModelConfig(
    temperature=0.7,
    max_tokens=1024,
    thinking_level="high",
)

adapter = GeminiAdapter(
    model="gemini-3.0-pro",
    client_config=client_config,
    model_config=model_config,
)

# Or with pre-configured client
from google import genai
client = genai.Client(api_key="YOUR_API_KEY")
adapter = GeminiAdapter(
    model="gemini-3.0-pro",
    client=client,
    model_config=model_config,  # Model params still apply
)
```

### Structured Output

```python
from dataclasses import dataclass
from weakincentives import Prompt, MarkdownSection, StructuredOutput

@dataclass(frozen=True, slots=True)
class Analysis:
    sentiment: str
    confidence: float
    keywords: list[str]

prompt = Prompt[Analysis](
    ns="example",
    key="analyze",
    name="analysis_prompt",
    sections=[
        MarkdownSection(
            title="Task",
            template="Analyze the sentiment of: $text",
            key="task",
        ),
    ],
    structured_output=StructuredOutput(output_type=Analysis),
)

adapter = GeminiAdapter(use_native_response_format=True)
response = adapter.evaluate(prompt.with_params(text="I love this!"), bus=bus, session=session)
analysis = response.output  # Analysis(sentiment="positive", confidence=0.95, ...)
```

### With Tools

```python
from weakincentives import Tool, ToolResult

@dataclass(frozen=True, slots=True)
class WeatherParams:
    location: str

def get_weather(params: WeatherParams, *, context: ToolContext) -> ToolResult[str]:
    return ToolResult(message=f"Weather in {params.location}: Sunny, 72F", success=True)

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a location",
    params_type=WeatherParams,
    handler=get_weather,
)

prompt = Prompt[str](
    ns="example",
    key="weather",
    sections=[
        MarkdownSection(
            title="Task",
            template="What's the weather in $city?",
            key="task",
            tools=(weather_tool,),
        ),
    ],
)
```

### Vertex AI Deployment

```python
from weakincentives.adapters import GeminiAdapter, GeminiClientConfig

client_config = GeminiClientConfig(
    vertexai=True,
    project="my-gcp-project",
    location="us-central1",
)

adapter = GeminiAdapter(
    model="gemini-3.0-pro",
    client_config=client_config,
)
```

### With Budget Tracking

```python
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from datetime import timedelta

# Create a budget with token and time limits
budget = Budget(
    deadline=Deadline.from_timeout(timedelta(minutes=5)),
    max_total_tokens=10000,
    max_output_tokens=2000,
)

# Optionally share a tracker across multiple evaluations
tracker = BudgetTracker(budget=budget)

response = adapter.evaluate(
    prompt.with_params(name="World"),
    bus=bus,
    session=session,
    budget=budget,
    budget_tracker=tracker,  # Optional: share across calls
)

# Check consumed tokens
print(f"Tokens used: {tracker.consumed.total_tokens}")
```

### With Thinking Level (via Model Config)

```python
from weakincentives.adapters import GeminiAdapter, GeminiModelConfig

# For complex reasoning tasks
model_config = GeminiModelConfig(
    thinking_level="high",  # Use deeper reasoning
    temperature=0.3,        # Lower temperature for more focused output
)

adapter = GeminiAdapter(
    model="gemini-3.0-pro",
    model_config=model_config,
)
```

## Migration Notes

### From LiteLLM with Gemini

If currently using `LiteLLMAdapter` with Gemini models, migrate to `GeminiAdapter` for:

- Native structured output (no text parsing fallback needed)
- Direct SDK error handling
- Access to Gemini-specific features (thinking level, etc.)
- Reduced latency (no LiteLLM proxy layer)
- Type-safe configuration via `GeminiClientConfig` and `GeminiModelConfig`

### API Key Environment Variable

The SDK reads from `GOOGLE_API_KEY` by default. For explicit configuration:

```python
from weakincentives.adapters import GeminiAdapter, GeminiClientConfig

client_config = GeminiClientConfig(
    api_key=os.getenv("MY_GEMINI_KEY"),
)
adapter = GeminiAdapter(
    model="gemini-3.0-pro",
    client_config=client_config,
)
```

## References

- [Google GenAI Python SDK](https://github.com/googleapis/python-genai)
- [SDK Documentation](https://googleapis.github.io/python-genai/)
- [Gemini 3 for Developers](https://blog.google/technology/developers/gemini-3-developers/)
- [Structured Output Guide](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output)
- [Gemini 3 Pro on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-pro)

## Changelog

- **v2**: Updated to use typed configuration pattern (`GeminiClientConfig`, `GeminiModelConfig`)
  matching OpenAI/LiteLLM adapters. Added budget tracking support (`Budget`, `BudgetTracker`),
  visibility overrides, and moved `thinking_level` to `GeminiModelConfig`.
- **v1**: Initial specification with native structured output support and Gemini 3.0 Pro
  default model.
