# Hosted Tools Specification

## Purpose

Hosted tools are provider-executed capabilities that run server-side rather than
locally. Unlike user-defined tools with handlers, hosted tools delegate execution
to the LLM provider and receive structured results. This specification covers how
hosted tools extend the core Tool interface, adapter-specific implementations,
and integration patterns.

## Guiding Principles

- **Unified tool interface**: Hosted tools extend the core `Tool` type rather than
  introducing a separate abstraction; they flow through the existing `tools` tuple.
- **No lowest common denominator**: Different providers have different capabilities;
  users import adapter-specific types explicitly.
- **Codec-based extensibility**: Adapters register codecs that own serialization,
  output parsing, and type definitions.
- **Section-first integration**: Hosted tools use the existing `tools` parameter
  on sections, respecting visibility and enablement rules.

## Core Abstraction

### Extended Tool Interface

Hosted tools are regular `Tool` instances with a `hosted` configuration field.
When `hosted` is set, the adapter delegates execution to the provider instead of
invoking a local handler:

```python
@dataclass(slots=True, frozen=True)
class HostedToolConfig(Generic[ConfigT]):
    """Configuration for a provider-executed tool.

    ConfigT is a provider-specific configuration type defined in the adapter
    layer. Users import the appropriate config type from the adapter they
    are using.
    """

    kind: str           # Codec lookup key (e.g., "web_search")
    config: ConfigT     # Provider-specific configuration
```

The `Tool` class gains an optional `hosted` field:

```python
@dataclass(slots=True)
class Tool[ParamsT, ResultT]:
    name: str
    description: str
    handler: ToolHandler[ParamsT, ResultT] | None
    examples: tuple[ToolExample[ParamsT, ResultT], ...] = ()
    accepts_overrides: bool = True
    hosted: HostedToolConfig[Any] | None = None  # Provider-executed tool config
```

**Construction Rules:**

- When `hosted` is `None`, the tool is a standard user-defined tool
- When `hosted` is set, `kind` identifies which codec handles this tool
- `name` follows the same constraints: `^[a-z0-9_-]{1,64}$`
- `description` must be 1-200 ASCII characters
- `hosted.config` is opaque to the core—adapters define and validate config types

**Hosted vs. User-Defined Tools:**

| Aspect | User-Defined Tool | Hosted Tool |
|--------|-------------------|-------------|
| Execution | Local handler | Provider server-side |
| `handler` | Required callable | `None` (ignored if set) |
| `hosted` | `None` | `HostedToolConfig` instance |
| Serialization | `{"type": "function", ...}` | Adapter-specific via codec |
| Result | `ToolResult[ResultT]` | Adapter-defined output type |

### Section Integration

Hosted tools use the existing `tools` parameter on sections:

```python
section = MarkdownSection[Params](
    title="Research",
    template="Find information about $topic",
    key="research",
    tools=[user_tool, hosted_tool],  # Both tool types in same tuple
)
```

No separate `hosted_tools` method or attribute is needed.

### RenderedPrompt

`RenderedPrompt.tools` contains all tools (user-defined and hosted):

```python
@dataclass(slots=True, frozen=True)
class RenderedPrompt(Generic[OutputT]):
    # ... existing fields ...
    tools: tuple[Tool[Any, Any], ...] = ()  # Includes hosted tools
```

Adapters inspect `tool.hosted` to determine serialization strategy.

## Adapter Layer

All configuration types, output types, codecs, and convenience helpers live in
the adapter layer. Users import directly from the adapter they are targeting.

### HostedToolCodec Protocol

Adapters implement codecs for each supported hosted tool:

```python
class HostedToolCodec(Protocol[ConfigT, OutputT]):
    """Adapter-specific serialization and parsing for a hosted tool."""

    @property
    def kind(self) -> str:
        """The tool kind this codec handles."""
        ...

    def serialize(
        self,
        tool: Tool[Any, Any],
    ) -> dict[str, Any]:
        """Convert tool configuration to provider wire format.

        The tool is guaranteed to have `tool.hosted` set with `kind` matching
        this codec's kind.
        """
        ...

    def parse_output(
        self,
        response_items: Sequence[Any],
        tool: Tool[Any, Any],
    ) -> OutputT | None:
        """Extract typed output from provider response, if present."""
        ...
```

### Codec Registry

Adapters maintain a registry mapping kinds to codecs:

```python
@dataclass(slots=True)
class OpenAIAdapter:
    hosted_tool_codecs: Mapping[str, HostedToolCodec[Any, Any]] = field(
        default_factory=lambda: {
            "web_search": OpenAIWebSearchCodec(),
        }
    )
```

## OpenAI Implementation

### File Structure

```
adapters/openai/
├── __init__.py
├── adapter.py
└── hosted/
    ├── __init__.py          # Re-exports for convenience
    ├── web_search.py        # Config, output, codec, section
    └── code_interpreter.py  # Future: another hosted tool
```

### Web Search Types

All web search types are OpenAI-specific:

```python
# adapters/openai/hosted/web_search.py

@dataclass(slots=True, frozen=True)
class OpenAIWebSearchFilters:
    """Domain filtering for OpenAI web search."""
    allowed_domains: tuple[str, ...] = ()  # Max 100
    blocked_domains: tuple[str, ...] = ()

@dataclass(slots=True, frozen=True)
class OpenAIUserLocation:
    """Geographic context for OpenAI web search."""
    country: str | None = None   # ISO 3166-1 alpha-2
    city: str | None = None
    region: str | None = None
    timezone: str | None = None  # IANA timezone

@dataclass(slots=True, frozen=True)
class OpenAIWebSearchConfig:
    """Configuration for OpenAI web_search tool."""
    filters: OpenAIWebSearchFilters | None = None
    user_location: OpenAIUserLocation | None = None
    external_web_access: bool = True
```

### Web Search Output

```python
@dataclass(slots=True, frozen=True)
class OpenAIUrlCitation:
    """A citation from OpenAI web search results."""
    url: str
    title: str
    start_index: int
    end_index: int

@dataclass(slots=True, frozen=True)
class OpenAIWebSearchResult:
    """Parsed output from OpenAI web_search invocation."""
    text: str
    citations: tuple[OpenAIUrlCitation, ...] = ()
    source_urls: tuple[str, ...] = ()
```

### Web Search Codec

```python
class OpenAIWebSearchCodec:
    """Codec for OpenAI web_search tool."""

    @property
    def kind(self) -> str:
        return "web_search"

    def serialize(
        self,
        tool: Tool[Any, Any],
    ) -> dict[str, Any]:
        spec: dict[str, Any] = {"type": "web_search"}
        # tool.hosted is guaranteed to be set with matching kind
        hosted = cast(HostedToolConfig[OpenAIWebSearchConfig], tool.hosted)
        config = hosted.config

        if config.filters:
            filters: dict[str, Any] = {}
            if config.filters.allowed_domains:
                filters["allowed_domains"] = list(config.filters.allowed_domains)
            if config.filters.blocked_domains:
                filters["blocked_domains"] = list(config.filters.blocked_domains)
            if filters:
                spec["filters"] = filters

        if config.user_location:
            loc = config.user_location
            location: dict[str, Any] = {"type": "approximate"}
            if loc.country:
                location["country"] = loc.country
            if loc.city:
                location["city"] = loc.city
            if loc.region:
                location["region"] = loc.region
            if loc.timezone:
                location["timezone"] = loc.timezone
            spec["user_location"] = location

        if not config.external_web_access:
            spec["external_web_access"] = False

        return spec

    def parse_output(
        self,
        response_items: Sequence[Any],
        tool: Tool[Any, Any],
    ) -> OpenAIWebSearchResult | None:
        web_search_call = None
        message_content = None

        for item in response_items:
            item_type = getattr(item, "type", None)
            if item_type == "web_search_call":
                web_search_call = item
            elif item_type == "message":
                message_content = item

        if web_search_call is None:
            return None

        text = ""
        citations: list[OpenAIUrlCitation] = []

        if message_content:
            content = getattr(message_content, "content", [])
            for part in content:
                if getattr(part, "type", None) == "output_text":
                    text = getattr(part, "text", "")
                    for ann in getattr(part, "annotations", []):
                        if getattr(ann, "type", None) == "url_citation":
                            citations.append(OpenAIUrlCitation(
                                url=getattr(ann, "url", ""),
                                title=getattr(ann, "title", ""),
                                start_index=getattr(ann, "start_index", 0),
                                end_index=getattr(ann, "end_index", 0),
                            ))

        return OpenAIWebSearchResult(
            text=text,
            citations=tuple(citations),
            source_urls=(),
        )
```

### Convenience Factory

```python
def openai_web_search(
    config: OpenAIWebSearchConfig = OpenAIWebSearchConfig(),
    *,
    name: str = "web_search",
) -> Tool[None, None]:
    """Create an OpenAI web search hosted tool."""
    return Tool[None, None](
        name=name,
        description="Search the web for current information and cite sources.",
        handler=None,
        hosted=HostedToolConfig(
            kind="web_search",
            config=config,
        ),
    )
```

### Convenience Section

```python
class OpenAIWebSearchSection(MarkdownSection[EmptyParams]):
    """Section that enables OpenAI web search."""

    def __init__(
        self,
        config: OpenAIWebSearchConfig = OpenAIWebSearchConfig(),
        *,
        key: str = "web_search",
    ) -> None:
        super().__init__(
            title="Web Search",
            key=key,
            template=_OPENAI_WEB_SEARCH_TEMPLATE,
            tools=(openai_web_search(config),),  # Uses existing tools parameter
        )
```

### Response Structure

The adapter populates `PromptResponse.hosted_outputs`:

```python
@dataclass(slots=True, frozen=True)
class PromptResponse(Generic[OutputT]):
    output: OutputT
    # ... existing fields ...
    hosted_outputs: Mapping[str, Any] = field(default_factory=dict)
```

Output types are adapter-specific; callers cast based on the tool they used:

```python
response = adapter.evaluate(prompt, bus=bus, session=session)
web_result = cast(
    OpenAIWebSearchResult | None,
    response.hosted_outputs.get("web_search"),
)
if web_result:
    for citation in web_result.citations:
        print(f"- [{citation.title}]({citation.url})")
```

## Usage Examples

### OpenAI Web Search

```python
from weakincentives import Prompt, MarkdownSection
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.adapters.openai.hosted import (
    OpenAIWebSearchSection,
    OpenAIWebSearchConfig,
    OpenAIWebSearchFilters,
    OpenAIWebSearchResult,
)

# Domain-restricted medical research
config = OpenAIWebSearchConfig(
    filters=OpenAIWebSearchFilters(
        allowed_domains=("pubmed.ncbi.nlm.nih.gov", "www.cdc.gov"),
    ),
)

prompt = Prompt[Answer](
    ns="research",
    key="medical",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Research: $topic",
        ),
        OpenAIWebSearchSection(config=config),
    ],
)

adapter = OpenAIAdapter(model="gpt-4o")
response = adapter.evaluate(prompt, bus=bus, session=session)

# Access typed output
web_result = cast(OpenAIWebSearchResult | None, response.hosted_outputs.get("web_search"))
```

### Inline Tool Construction

```python
from weakincentives.adapters.openai.hosted import (
    openai_web_search,
    OpenAIWebSearchConfig,
    OpenAIUserLocation,
)

tool = openai_web_search(
    OpenAIWebSearchConfig(
        user_location=OpenAIUserLocation(
            country="GB",
            city="London",
        ),
        external_web_access=False,
    ),
)

section = MarkdownSection(
    title="Research",
    key="research",
    template="Find information about $topic",
    tools=(tool,),  # Hosted tools use the same parameter as user-defined tools
)
```

## Adding New Hosted Tools

All new hosted tools follow the same pattern within the adapter layer.

### Step 1: Define Types (Adapter)

```python
# adapters/openai/hosted/code_interpreter.py

@dataclass(slots=True, frozen=True)
class OpenAICodeInterpreterConfig:
    """Configuration for OpenAI code interpreter."""
    container: str | None = None  # Container type hint

@dataclass(slots=True, frozen=True)
class OpenAICodeInterpreterResult:
    """Output from OpenAI code interpreter."""
    code: str
    output: str
    error: str | None = None
```

### Step 2: Implement Codec (Adapter)

```python
class OpenAICodeInterpreterCodec:
    @property
    def kind(self) -> str:
        return "code_interpreter"

    def serialize(
        self,
        tool: Tool[Any, Any],
    ) -> dict[str, Any]:
        spec: dict[str, Any] = {"type": "code_interpreter"}
        hosted = cast(HostedToolConfig[OpenAICodeInterpreterConfig], tool.hosted)
        if hosted.config.container:
            spec["container"] = hosted.config.container
        return spec

    def parse_output(
        self,
        response_items: Sequence[Any],
        tool: Tool[Any, Any],
    ) -> OpenAICodeInterpreterResult | None:
        # Parse code_interpreter_call items
        ...
```

### Step 3: Register Codec (Adapter)

```python
# adapters/openai/adapter.py

hosted_tool_codecs: Mapping[str, HostedToolCodec[Any, Any]] = {
    "web_search": OpenAIWebSearchCodec(),
    "code_interpreter": OpenAICodeInterpreterCodec(),
}
```

### Step 4: Create Convenience Factory (Adapter)

```python
def openai_code_interpreter(
    config: OpenAICodeInterpreterConfig = OpenAICodeInterpreterConfig(),
    *,
    name: str = "code_interpreter",
) -> Tool[None, None]:
    """Create an OpenAI code interpreter hosted tool."""
    return Tool[None, None](
        name=name,
        description="Execute Python code in a sandboxed environment.",
        handler=None,
        hosted=HostedToolConfig(
            kind="code_interpreter",
            config=config,
        ),
    )
```

### Step 5: Export Conveniences (Adapter)

```python
# adapters/openai/hosted/__init__.py

from .web_search import (
    OpenAIWebSearchConfig,
    OpenAIWebSearchFilters,
    OpenAIUserLocation,
    OpenAIWebSearchResult,
    OpenAIUrlCitation,
    OpenAIWebSearchSection,
    openai_web_search,
)
from .code_interpreter import (
    OpenAICodeInterpreterConfig,
    OpenAICodeInterpreterResult,
    openai_code_interpreter,
)
```

## Error Handling

### Unsupported Kind

When an adapter encounters an unknown hosted tool kind:

```python
def _serialize_tool(
    self,
    tool: Tool[Any, Any],
    prompt_name: str,
) -> dict[str, Any]:
    if tool.hosted is None:
        # Standard user-defined tool serialization
        return self._serialize_function_tool(tool)

    # Hosted tool - lookup codec by kind
    codec = self.hosted_tool_codecs.get(tool.hosted.kind)
    if codec is None:
        raise PromptEvaluationError(
            f"Unsupported hosted tool kind '{tool.hosted.kind}' for {self.name} adapter.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_RENDER,
        )
    return codec.serialize(tool)
```

### Wrong Adapter

Using an OpenAI-specific hosted tool with a different adapter raises at
serialization time. This is intentional—hosted tools are not portable.

### Parse Failures

Codec `parse_output` returns `None` when the tool was not invoked or parsing
fails gracefully. For hard failures, codecs raise `PromptEvaluationError`.

## Limitations

- **No portability**: Hosted tools are adapter-specific by design
- **No local fallback**: Cannot execute hosted tools locally
- **Provider constraints**: Availability varies by model and account tier
- **Rate limits**: Hosted tool calls may have separate rate limits
- **Streaming**: Output metadata may only be available after streaming completes

## Testing

### Unit Tests

- Test codec serialization produces correct wire format
- Test codec parsing handles various response shapes
- Verify unknown kinds raise appropriate errors
- Validate config dataclass constraints
- Test `Tool` with `hosted` field set behaves as hosted tool
- Test `Tool` with `hosted=None` behaves as user-defined tool
- Verify hosted tools can be mixed with user-defined tools in sections

### Integration Tests

- Round-trip through actual provider with mocked responses
- Verify citations are correctly extracted
- Test domain filtering behavior
- Test section rendering includes both tool types in `tools` tuple

### Fixtures

- `tests/fixtures/openai/hosted/` contains sample response payloads
- `tests/helpers/openai_hosted.py` provides mock codecs
