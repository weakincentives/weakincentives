# Tool Registration Specification

## Introduction
Large language model runtimes expect prompts to advertise structured "tools" (a.k.a. function calls) that can be invoked
mid-interaction. The existing `Prompt` abstraction centralizes markdown instructions but lacks a native way to describe
those tools. This document defines `ToolsSection`, a `Section` subclass that collects tool definitions alongside the rest
of the prompt tree so authors can manage instructions and callable affordances in one place.

## Goals
- **Section-First Integration**: Allow tools to live inside the existing section hierarchy so ordering and enablement stay
  consistent with the rest of the prompt.
- **Single Source of Truth**: Co-locate tool contracts with the prompt that introduces them, eliminating ad-hoc external
  registries.
- **Type-Safe Tooling**: Retain the dataclass-first approach to argument validation so schema issues surface before a
  request reaches an LLM.
- **Deterministic Exposure**: Provide stable, machine-readable tool definitions that downstream adapters can map onto
  OpenAI or Anthropic clients without guessing.

## Guiding Principles
- **Explicit Contracts**: Tool names, parameter shapes, and summaries are declared statically in code.
- **Composable Structure**: Multiple `ToolsSection` instances can appear anywhere in the tree; the prompt collects their
  contents in render order.
- **Strict Validation**: Duplicate tool names, missing required fields, or parameter mismatches raise
  `PromptValidationError` during construction.
- **Runtime Agnostic**: The prompt layer stops at describing tools; adapter layers handle provider-specific payloads and
  execution semantics.

## Core Concepts
### `ToolResult` Dataclass
Tools return structured data that pairs a language-model-facing message with typed payload metadata. `ToolResult[PayloadT]`
instances capture that response tuple, and every tool handler returns one directly:
- `message: str` – short textual reply returned to the LLM.
- `payload: PayloadT` – strongly typed result produced by the tool's business logic. The payload type is declared on the
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

### `ToolsSection`
`ToolsSection` subclasses `Section`, unlocking existing enablement logic, heading management, and child composition:
- Accepts `tools: Sequence[Tool]` and an optional markdown `description` template that can reference the section's
  parameters for additional guidance.
- Shares the surrounding section's `params` dataclass so authors can keep instructional text and tool definitions aligned.
- Minimizes rendered output by omitting per-tool listings from markdown; authors can add a concise description if human
  readers need inline guidance. Deeper schema output is intentionally deferred to adapter layers, and additional
  formatting hooks will be introduced only if real prompts require customization beyond this default.
- Exposes the contained tools back to the prompt runtime so they can be returned via `Prompt.tools()`.

Multiple `ToolsSection` instances can appear in one prompt. During validation the prompt aggregates every tool, ensures
names remain unique, and records declaration order. Tools are free to share parameter or result dataclasses when that
improves reuse. Disabled sections (via `is_enabled`) omit
both markdown and tool entries. Per-tool enablement remains out of scope for this iteration.

## Schema Generation
Schema serialization is out of scope for this iteration. The prompt layer guarantees access to strongly typed `Tool`
objects; provider adapters (OpenAI, Anthropic, etc.) can translate those into JSON Schema or API-specific payloads as
needed without touching the prompt core.

## Prompt Integration
`Prompt` continues to accept an ordered tree of sections. `ToolsSection` integrates without new constructor arguments:
1. During initialization the prompt walks the section tree depth-first, collecting tools from each `ToolsSection`.
2. Validation enforces unique tool names across the entire prompt; composite prompts must coordinate naming themselves
   until we revisit hierarchical namespaces. Parameter and result dataclasses may repeat across tools.
3. Declaration order is cached so callers can retrieve tools without re-traversing the tree.

`Prompt.render(...)` still returns the rendered markdown string, accepting dataclass overrides exactly as before. A new
`Prompt.tools()` accessor exposes an ordered list of `Tool` objects contributed by enabled `ToolsSection`s, honoring
section-level defaults and enablement rules.

## Runtime Execution
`ToolsSection` only documents callable capabilities. Higher-level orchestration code is responsible for invoking the
appropriate handler when an LLM emits a tool call, passing the params dataclass instance as the sole argument. Handlers
return a `ToolResult[result_type]` directly. The prompt layer remains side effect free—it surfaces handlers and
`result_type` metadata without executing them. The package provides no sync/async bridging helpers; orchestrators decide
when to `await` or call handlers directly.

## Example
```python
from dataclasses import dataclass, field
from weakincentives.prompts import Prompt, TextSection, Tool, ToolResult, ToolsSection

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

def lookup_handler(params: LookupParams) -> ToolResult[LookupResult]:
    result = LookupResult(entity_id=params.entity_id, document_url="https://example.com")
    message = f"Fetched entity {result.entity_id}."
    return ToolResult(message=message, payload=result)

lookup_tool = Tool[LookupParams, LookupResult](
    name="lookup_entity",
    description="Fetch structured information for a given entity id.",
    handler=lookup_handler,
)

tooling_overview = Prompt(
    name="tools_overview",
    sections=[
        TextSection(
            title="Guidance",
            body="""
            Use tools when you need up-to-date context. Prefer ${primary_tool} for critical lookups.
            """,
            params=GuidanceParams,
            children=[
                ToolsSection(
                    title="Available Tools",
                    tools=[lookup_tool],
                    params=GuidanceParams,
                    description="""
                    Invoke ${primary_tool} whenever you need fresh entity context.
                    """,
                )
            ],
        )
    ],
)

markdown = tooling_overview.render(GuidanceParams(primary_tool="lookup_entity"))
tools = tooling_overview.tools()
assert tools[0].name == "lookup_entity"
```

In the example above the `ToolsSection` contributes its title and optional description to the markdown output but omits
the tool listing, preserving context budget while keeping a single source of truth for adapters.

## Validation and Error Handling
- Construction failures raise `PromptValidationError` with contextual data (`section path`, `tool.name`, parameter
  dataclass).
- Rendering without required tool parameters raises `PromptRenderError`, mirroring existing section behavior.
- Registering two tools with the same name triggers `PromptValidationError` to preserve lookup determinism.
- Handler references that do not accept exactly one argument matching the `ParamsT` dataclass raise
  `PromptValidationError` during prompt validation.
- Disabled `ToolsSection` instances contribute neither markdown nor entries in `Prompt.tools()`.
