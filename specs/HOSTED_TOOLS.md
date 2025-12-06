# Hosted Tools Specification

## Purpose

Hosted tools are provider-executed capabilities that run server-side rather than
locally. Unlike user-defined tools with handlers, hosted tools delegate execution
to the LLM provider and receive structured results. This specification covers the
core abstraction, configuration patterns, adapter integration, and output parsing.

## Guiding Principles

- **Clear layer separation**: Core abstractions remain provider-agnostic; adapter
  implementations handle wire formats and provider-specific behavior.
- **Type-safe configuration**: Tool options are frozen dataclasses validated at
  construction time.
- **Codec-based extensibility**: Adapters register codecs that serialize configs
  and parse outputs without modifying core abstractions.
- **Section-first integration**: Hosted tools attach to sections like user-defined
  tools, respecting visibility and enablement rules.

## Core Abstractions

### HostedToolConfig Protocol

Marker protocol for hosted tool configuration types:

```python
@runtime_checkable
class HostedToolConfig(Protocol):
    """Marker protocol for hosted tool configurations."""
    pass
```

All configuration dataclasses implement this protocol implicitly when they are
frozen dataclasses with slots.

### HostedToolOutput Protocol

Marker protocol for parsed hosted tool outputs:

```python
@runtime_checkable
class HostedToolOutput(Protocol):
    """Marker protocol for parsed hosted tool outputs."""
    pass
```

Adapters parse provider responses into typed outputs implementing this protocol.

### HostedTool

`HostedTool[ConfigT]` describes a provider-executed capability:

```python
@dataclass(slots=True, frozen=True)
class HostedTool(Generic[ConfigT: HostedToolConfig]):
    kind: str           # Abstract identifier (e.g., "web_search")
    name: str           # Registry key, unique within prompt
    description: str    # Documentation (1-200 ASCII chars)
    config: ConfigT     # Tool-specific configuration
```

**Construction Rules:**

- `kind` identifies the capability type; adapters map kinds to wire formats
- `name` follows the same constraints as `Tool.name`: `^[a-z0-9_-]{1,64}$`
- `description` must be 1-200 ASCII characters
- `config` must be a frozen dataclass implementing `HostedToolConfig`

**Key Difference from Tool:**

| Aspect | Tool | HostedTool |
|--------|------|------------|
| Execution | Local handler | Provider server-side |
| Parameters | `ParamsT` dataclass | `ConfigT` configuration |
| Result | `ToolResult[ResultT]` | Provider-specific, parsed by codec |
| Serialization | `{"type": "function", ...}` | Adapter-specific via codec |

## Tool Definitions

Tool-specific configurations and outputs live in `weakincentives.tools.*` and
remain provider-agnostic.

### Web Search

#### Configuration

```python
@dataclass(slots=True, frozen=True)
class DomainFilter:
    """Restrict search to specific domains."""
    allowed: tuple[str, ...] = ()   # Include only these domains
    blocked: tuple[str, ...] = ()   # Exclude these domains

@dataclass(slots=True, frozen=True)
class GeoHint:
    """Geographic context for search personalization."""
    country_code: str | None = None   # ISO 3166-1 alpha-2
    city: str | None = None
    region: str | None = None
    timezone: str | None = None       # IANA timezone

@dataclass(slots=True, frozen=True)
class WebSearchConfig(HostedToolConfig):
    """Configuration for web search capability."""
    domain_filter: DomainFilter | None = None
    geo_hint: GeoHint | None = None
    allow_live_access: bool = True
```

**Validation:**

- `DomainFilter.allowed` and `blocked` are mutually exclusive in some providers
- Domain strings omit protocol prefix (e.g., `"cdc.gov"` not `"https://cdc.gov"`)
- `GeoHint.country_code` must be valid ISO 3166-1 alpha-2 when provided
- `GeoHint.timezone` must be valid IANA timezone when provided

#### Output

```python
@dataclass(slots=True, frozen=True)
class Citation:
    """A reference to a web source."""
    url: str
    title: str
    span: tuple[int, int] | None = None  # Character range in response text

@dataclass(slots=True, frozen=True)
class WebSearchResult(HostedToolOutput):
    """Parsed result from a web search invocation."""
    text: str                              # Response text with inline citations
    citations: tuple[Citation, ...] = ()   # Extracted citation metadata
    source_urls: tuple[str, ...] = ()      # All consulted URLs
```

#### Factory Function

```python
def web_search_tool(
    config: WebSearchConfig = WebSearchConfig(),
    *,
    name: str = "web_search",
) -> HostedTool[WebSearchConfig]:
    """Create a web search hosted tool with the given configuration."""
    return HostedTool(
        kind="web_search",
        name=name,
        description="Search the web for current information and cite sources.",
        config=config,
    )
```

## Section Integration

### Section.hosted_tools

Sections expose hosted tools via an override point:

```python
class Section(ABC, Generic[ParamsT]):
    def hosted_tools(self) -> tuple[HostedTool[Any], ...]:
        """Hosted tools exposed by this section. Override in subclasses."""
        return ()
```

### WebSearchSection

Convenience section for web search:

```python
class WebSearchSection(MarkdownSection[EmptyParams]):
    """Section that enables web search capability."""

    def __init__(
        self,
        config: WebSearchConfig = WebSearchConfig(),
        *,
        key: str = "web_search",
    ) -> None:
        self._tool = web_search_tool(config)
        super().__init__(
            title="Web Search",
            key=key,
            template=_WEB_SEARCH_TEMPLATE,
        )

    def hosted_tools(self) -> tuple[HostedTool[WebSearchConfig], ...]:
        return (self._tool,)
```

### RenderedPrompt

Rendered prompts include both tool types:

```python
@dataclass(slots=True, frozen=True)
class RenderedPrompt(Generic[OutputT]):
    # ... existing fields ...
    tools: tuple[Tool[Any, Any], ...] = ()
    hosted_tools: tuple[HostedTool[Any], ...] = ()
```

Rendering collects hosted tools depth-first from enabled sections, same as
user-defined tools.

## Adapter Integration

### HostedToolCodec Protocol

Adapters implement codecs for each supported hosted tool kind:

```python
class HostedToolCodec(Protocol[ConfigT, OutputT]):
    """Adapter-specific serialization and parsing for a hosted tool kind."""

    @property
    def kind(self) -> str:
        """The abstract tool kind this codec handles."""
        ...

    def serialize(
        self,
        tool: HostedTool[ConfigT],
    ) -> dict[str, Any]:
        """Convert tool configuration to provider wire format."""
        ...

    def parse_output(
        self,
        response_items: Sequence[Any],
        tool: HostedTool[ConfigT],
    ) -> OutputT | None:
        """Extract typed output from provider response, if present."""
        ...
```

**Codec Responsibilities:**

- Map abstract `kind` to provider-specific `type` field
- Serialize `ConfigT` fields to provider wire format
- Parse provider response items into `OutputT`
- Return `None` from `parse_output` if the tool was not invoked

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

### Serialization Flow

```
HostedTool[ConfigT]
       │
       │ kind="web_search"
       ▼
┌──────────────────┐
│ Adapter lookup   │
│ codecs[kind]     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ codec.serialize  │
│ (tool)           │
└────────┬─────────┘
         │
         ▼
{"type": "web_search", "filters": {...}, ...}
```

### Output Parsing Flow

```
Provider Response Items
         │
         ▼
┌──────────────────────┐
│ For each hosted tool │
│ in rendered_prompt   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ codec.parse_output   │
│ (items, tool)        │
└──────────┬───────────┘
           │
           ▼
WebSearchResult(text=..., citations=(...), ...)
```

## OpenAI Implementation

### Wire Format Mapping

| Core Concept | OpenAI Wire Format |
|--------------|-------------------|
| `kind="web_search"` | `{"type": "web_search"}` |
| `DomainFilter.allowed` | `filters.allowed_domains` |
| `DomainFilter.blocked` | `filters.blocked_domains` |
| `GeoHint` | `user_location` with `type: "approximate"` |
| `allow_live_access=False` | `external_web_access: false` |

### OpenAIWebSearchCodec

```python
class OpenAIWebSearchCodec(HostedToolCodec[WebSearchConfig, WebSearchResult]):
    """OpenAI Responses API codec for web_search."""

    @property
    def kind(self) -> str:
        return "web_search"

    def serialize(
        self,
        tool: HostedTool[WebSearchConfig],
    ) -> dict[str, Any]:
        spec: dict[str, Any] = {"type": "web_search"}
        config = tool.config

        if config.domain_filter:
            filters: dict[str, Any] = {}
            if config.domain_filter.allowed:
                filters["allowed_domains"] = list(config.domain_filter.allowed)
            if config.domain_filter.blocked:
                filters["blocked_domains"] = list(config.domain_filter.blocked)
            if filters:
                spec["filters"] = filters

        if config.geo_hint:
            location: dict[str, Any] = {"type": "approximate"}
            if config.geo_hint.country_code:
                location["country"] = config.geo_hint.country_code
            if config.geo_hint.city:
                location["city"] = config.geo_hint.city
            if config.geo_hint.region:
                location["region"] = config.geo_hint.region
            if config.geo_hint.timezone:
                location["timezone"] = config.geo_hint.timezone
            spec["user_location"] = location

        if not config.allow_live_access:
            spec["external_web_access"] = False

        return spec

    def parse_output(
        self,
        response_items: Sequence[Any],
        tool: HostedTool[WebSearchConfig],
    ) -> WebSearchResult | None:
        # Find web_search_call items and message with annotations
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

        # Extract text and citations from message
        text = ""
        citations: list[Citation] = []

        if message_content:
            content = getattr(message_content, "content", [])
            for part in content:
                if getattr(part, "type", None) == "output_text":
                    text = getattr(part, "text", "")
                    for ann in getattr(part, "annotations", []):
                        if getattr(ann, "type", None) == "url_citation":
                            citations.append(Citation(
                                url=getattr(ann, "url", ""),
                                title=getattr(ann, "title", ""),
                                span=(
                                    getattr(ann, "start_index", 0),
                                    getattr(ann, "end_index", 0),
                                ),
                            ))

        return WebSearchResult(
            text=text,
            citations=tuple(citations),
            source_urls=(),  # Requires include=["web_search_call.action.sources"]
        )
```

### Response Structure

The adapter populates `PromptResponse.hosted_outputs`:

```python
@dataclass(slots=True, frozen=True)
class PromptResponse(Generic[OutputT]):
    output: OutputT
    # ... existing fields ...
    hosted_outputs: Mapping[str, HostedToolOutput] = field(default_factory=dict)
```

Callers access parsed outputs by tool name:

```python
response = adapter.evaluate(prompt, bus=bus, session=session)
web_result = response.hosted_outputs.get("web_search")
if web_result:
    for citation in web_result.citations:
        print(f"- [{citation.title}]({citation.url})")
```

## Usage Examples

### Basic Web Search

```python
from weakincentives import Prompt, MarkdownSection
from weakincentives.tools.web_search import WebSearchSection

prompt = Prompt[Answer](
    ns="research",
    key="lookup",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Answer this question: $question",
        ),
        WebSearchSection(),
    ],
)

response = adapter.evaluate(prompt, bus=bus, session=session)
```

### Domain-Restricted Search

```python
from weakincentives.tools.web_search import (
    WebSearchSection,
    WebSearchConfig,
    DomainFilter,
)

config = WebSearchConfig(
    domain_filter=DomainFilter(
        allowed=("pubmed.ncbi.nlm.nih.gov", "www.cdc.gov", "www.who.int"),
    ),
)

section = WebSearchSection(config=config)
```

### Geo-Localized Search

```python
from weakincentives.tools.web_search import (
    WebSearchSection,
    WebSearchConfig,
    GeoHint,
)

config = WebSearchConfig(
    geo_hint=GeoHint(
        country_code="GB",
        city="London",
        timezone="Europe/London",
    ),
)

section = WebSearchSection(config=config)
```

### Inline Tool Construction

```python
from weakincentives.tools.web_search import web_search_tool, WebSearchConfig

tool = web_search_tool(
    WebSearchConfig(allow_live_access=False),
    name="cached_search",
)

section = MarkdownSection(
    title="Research",
    key="research",
    template="Find information about $topic",
    hosted_tools=(tool,),
)
```

## Adding New Hosted Tools

### Step 1: Define Configuration (Core)

Create a new module in `weakincentives/tools/`:

```python
# tools/code_interpreter/config.py

@dataclass(slots=True, frozen=True)
class CodeInterpreterConfig(HostedToolConfig):
    """Configuration for code interpreter capability."""
    allowed_languages: tuple[str, ...] = ("python",)
    timeout_seconds: int = 30
```

### Step 2: Define Output (Core)

```python
# tools/code_interpreter/output.py

@dataclass(slots=True, frozen=True)
class CodeInterpreterResult(HostedToolOutput):
    """Parsed result from code interpreter execution."""
    code: str
    stdout: str
    stderr: str
    exit_code: int
```

### Step 3: Create Factory (Core)

```python
# tools/code_interpreter/__init__.py

def code_interpreter_tool(
    config: CodeInterpreterConfig = CodeInterpreterConfig(),
    *,
    name: str = "code_interpreter",
) -> HostedTool[CodeInterpreterConfig]:
    return HostedTool(
        kind="code_interpreter",
        name=name,
        description="Execute code in a sandboxed environment.",
        config=config,
    )
```

### Step 4: Implement Codec (Adapter)

```python
# adapters/openai/codecs/code_interpreter.py

class OpenAICodeInterpreterCodec(
    HostedToolCodec[CodeInterpreterConfig, CodeInterpreterResult]
):
    @property
    def kind(self) -> str:
        return "code_interpreter"

    def serialize(
        self,
        tool: HostedTool[CodeInterpreterConfig],
    ) -> dict[str, Any]:
        # Map to OpenAI wire format
        ...

    def parse_output(
        self,
        response_items: Sequence[Any],
        tool: HostedTool[CodeInterpreterConfig],
    ) -> CodeInterpreterResult | None:
        # Parse provider response
        ...
```

### Step 5: Register Codec (Adapter)

```python
# adapters/openai.py

hosted_tool_codecs: Mapping[str, HostedToolCodec[Any, Any]] = {
    "web_search": OpenAIWebSearchCodec(),
    "code_interpreter": OpenAICodeInterpreterCodec(),  # NEW
}
```

## Error Handling

### Unsupported Kind

When an adapter encounters an unknown hosted tool kind:

```python
def _serialize_hosted_tool(
    self,
    tool: HostedTool[Any],
    prompt_name: str,
) -> dict[str, Any]:
    codec = self.hosted_tool_codecs.get(tool.kind)
    if codec is None:
        raise PromptEvaluationError(
            f"Unsupported hosted tool kind '{tool.kind}' for {self.name} adapter.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_RENDER,
        )
    return codec.serialize(tool)
```

### Parse Failures

Codec `parse_output` returns `None` when the tool was not invoked or parsing
fails gracefully. For hard failures, codecs raise `PromptEvaluationError` with
phase `"parse"`.

### Provider Errors

Provider-side tool failures (e.g., search timeout) surface through the normal
response content. The codec extracts error information into the output type
or returns partial results.

## Limitations

- **No local execution**: Hosted tools cannot fall back to local handlers
- **Provider dependency**: Availability varies by provider and model
- **Rate limits**: Hosted tool calls may have separate rate limits
- **Output fidelity**: Parsed outputs are best-effort; some provider details may
  be lost in translation
- **Streaming**: Citation metadata may only be available after streaming completes

## Testing

### Unit Tests

- Validate configuration dataclass constraints
- Test codec serialization produces correct wire format
- Test codec parsing handles various response shapes
- Verify unknown kinds raise appropriate errors

### Integration Tests

- Round-trip through actual provider with mocked responses
- Verify citations are correctly extracted
- Test domain filtering behavior
- Validate geo-hint localization effects

### Fixtures

- `tests/fixtures/hosted_tools/` contains sample provider responses
- `tests/helpers/hosted_tools.py` provides mock codecs for prompt tests
