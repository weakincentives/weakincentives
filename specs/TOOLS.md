# Tool Registration Specification

## Introduction

Large language model runtimes expect prompts to advertise structured "tools" (a.k.a. function calls) that can be invoked
mid-interaction. The prompt abstraction now allows every `Section` to contribute tools directly through a shared
interface, eliminating the need for a dedicated `ToolsSection`. This keeps instructions and callable affordances
co-located while reusing the existing section hierarchy for ordering and enablement.

## Goals

- **Section-First Integration**: Allow tools to live inside the existing section hierarchy so ordering and enablement stay
  consistent with the rest of the prompt.
- **Single Source of Truth**: Co-locate tool contracts with the prompt that introduces them, removing ad-hoc external
  registries.
- **Type-Safe Tooling**: Retain the dataclass-first approach to argument validation so schema issues surface before a
  request reaches an LLM.
- **Deterministic Exposure**: Provide stable, machine-readable tool definitions that downstream adapters can map onto
  OpenAI or Anthropic clients without guessing.

## Guiding Principles

- **Explicit Contracts**: Tool names, parameter shapes, and summaries are declared statically in code.
- **Composable Structure**: Any section can expose tools; the prompt collects their contents in render order while
  honoring enablement predicates.
- **Strict Validation**: Duplicate tool names, missing required fields, or parameter mismatches raise
  `PromptValidationError` during construction.
- **Runtime Agnostic**: The prompt layer stops at describing tools; adapter layers handle provider-specific payloads and
  execution semantics.

## Core Concepts

### `ToolResult` Dataclass

Tools return structured data that pairs a language-model-facing message with typed value metadata. `ToolResult[PayloadT]`
instances capture that response tuple, and every tool handler returns one directly:

- `message: str` – short textual reply returned to the LLM.
- `value: PayloadT` – strongly typed result produced by the tool's business logic. The payload type is declared on the
  `Tool` so downstream adapters can reason about the schema without guessing.

### `Tool` Dataclass

`Tool[ParamsT, ResultT]` instances describe callable affordances. They carry:

- `name: str` – unique identifier scoped to a single prompt instance. Validation enforces ASCII-only lowercase letters,
  digits, or underscores with length ≤ 64 characters for SDK compatibility.
- `description: str` – concise model-facing summary. We constrain summaries to ASCII, strip surrounding whitespace, and
  require 1–200 characters so payloads stay portable.
- `handler: Callable[[ParamsT], ToolResult[ResultT]] | None` – optional runtime hook surfaced to orchestration layers.
  Handlers must accept exactly one argument of type `ParamsT`, and when provided they must return a `ToolResult[ResultT]`.

Parameter and result dataclasses inherit the same validation rules as section params: every placeholder referenced in
markdown must exist on the dataclass, and required fields without defaults must be supplied when rendering. Tools bind the
dataclass *types* through their generic parameters—no redundant instance plumbing lives in the prompt configuration.

### Section Tool Registration

`Section.__init__` accepts an optional `tools` sequence. Sections normalize the sequence into a tuple, validate each entry
is a `Tool`, and expose the collection via `Section.tools()`. Because every section shares this capability, authors can:

- Attach tools to existing `MarkdownSection`s without creating new subclasses.
- Associate tools with otherwise minimal sections that only emit headings or act as grouping nodes.
- Allow child sections to contribute additional tooling while parent enablement gates the entire branch.

## Prompt Integration

`Prompt` continues to accept an ordered tree of sections. During initialization it walks the tree depth-first, validating
all tools contributed by each section:

1. Validation enforces unique tool names across the entire prompt; composite prompts must coordinate naming themselves
   until hierarchical namespaces are introduced.
1. Parameter and result dataclasses may repeat across tools to encourage reuse.
1. Declaration order is cached so callers can retrieve tools without re-traversing the tree.

`Prompt.render(...)` still returns the rendered markdown string, accepting dataclass overrides exactly as before. The
resulting `RenderedPrompt` now exposes a `.tools` property that surfaces an ordered tuple of `Tool` objects contributed by
enabled sections, honoring section-level defaults and enablement rules. Callers no longer re-traverse the prompt tree to
resolve tooling; they rely on the rendered instance for both markdown and tool metadata.

## Runtime Execution

Sections only document callable capabilities. Higher-level orchestration code is responsible for invoking the appropriate
handler when an LLM emits a tool call, passing the params dataclass instance as the sole argument. Handlers return a
`ToolResult[result_type]` directly. The prompt layer remains side effect free—it surfaces handlers and `result_type`
metadata without executing them. The package provides no sync/async bridging helpers; orchestrators decide when to `await`
or call handlers directly.

## Example

```python
from dataclasses import dataclass, field
from weakincentives.prompt import Prompt, MarkdownSection, Tool, ToolResult

@dataclass
class LookupParams:
    entity_id: str = field(metadata={"description": "Global identifier to fetch"})
    include_related: bool = field(default=False)

@dataclass
class LookupResult:
    entity_id: str
    document_url: str

@dataclass
class GuidanceParams:
    primary_tool: str

@dataclass
class ToolDescriptionParams:
    primary_tool: str = "lookup_entity"

def lookup_handler(params: LookupParams) -> ToolResult[LookupResult]:
    result = LookupResult(entity_id=params.entity_id, document_url="https://example.com")
    message = f"Fetched entity {result.entity_id}."
    return ToolResult(message=message, value=result)

lookup_tool = Tool[LookupParams, LookupResult](
    name="lookup_entity",
    description="Fetch structured information for a given entity id.",
    handler=lookup_handler,
)

tooling_overview = Prompt(
    name="tools_overview",
    sections=[
        MarkdownSection[GuidanceParams](
            title="Guidance",
            template="""
            Use tools when you need up-to-date context. Prefer ${primary_tool} for critical lookups.
            """,
            children=[
                MarkdownSection[ToolDescriptionParams](
                    title="Available Tools",
                    template="""
                    Invoke ${primary_tool} whenever you need fresh entity context.
                    """,
                    tools=[lookup_tool],
                    default_params=ToolDescriptionParams(),
                )
            ],
        )
    ],
)

rendered = tooling_overview.render(
    GuidanceParams(primary_tool="lookup_entity"),
    ToolDescriptionParams(),
)
markdown = rendered.text
tools = rendered.tools
assert tools[0].name == "lookup_entity"
```

In the example the nested `MarkdownSection` documents the tooling guidance while registering the `lookup_entity` tool. Because
sections own their tool collections directly, no additional subclasses are needed to describe a toolbox.

## Validation and Error Handling

- Construction failures raise `PromptValidationError` with contextual data (`section path`, `tool.name`, parameter
  dataclass).
- Rendering without required tool parameters raises `PromptRenderError`, mirroring existing section behavior.
- Registering two tools with the same name triggers `PromptValidationError` to preserve lookup determinism.
- Handler references that do not accept exactly one argument matching the `ParamsT` dataclass raise
  `PromptValidationError` during prompt validation.
- Disabled sections contribute neither markdown nor entries in `RenderedPrompt.tools`.
