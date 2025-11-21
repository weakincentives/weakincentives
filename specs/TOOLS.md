# Tool Runtime Specification

## Introduction

Large language model runtimes expect prompts to advertise structured "tools" (a.k.a.
function calls) that can be invoked mid-interaction. The prompt abstraction now allows
every `Section` to contribute tools directly through a shared interface, eliminating the
need for a dedicated `ToolsSection`. This keeps instructions and callable affordances
co-located while reusing the existing section hierarchy for ordering and enablement.

This combined runtime specification documents the full contract—from how tools register
against a prompt, to the context objects injected into handlers, through to the failure
semantics adapters must expose to large language models (LLMs). Treat it as the single
source of truth for tool affordances.

The runtime focuses on three pillars:

1. **Registration Lifecycle** – how sections declare tools and how prompts validate and
   expose them.
1. **Context Injection** – the immutable metadata bundle threaded into every handler
   invocation.
1. **Failure Semantics** – the uniform success/failure contract that handlers and
   adapters honor.

## Goals

- **Section-first integration** – keep tooling within the section hierarchy so enablement
  and ordering align with rendered instructions.
- **Single source of truth** – co-locate tool contracts with the prompts that introduce
  them instead of maintaining ad-hoc registries.
- **Type-safe tooling** – lean on dataclass-based params and result payloads so schema
  issues surface before a request reaches an LLM.
- **Deterministic exposure** – present a stable, machine-readable tool list so adapters
  negotiate provider payloads without guesswork.
- **Unified context contract** – describe the immutable metadata handlers expect through
  `ToolContext` to keep orchestration predictable.
- **Predictable failure semantics** – document shared success/failure handling so tool
  adapters never abort evaluation unexpectedly.

## Registration Lifecycle

### Registration Goals

- **Section-first integration** – any `Section` can register tools so enablement logic and
  ordering remain consistent with the rendered prompt.
- **Single source of truth** – tool definitions live alongside the sections that document
  them; no out-of-band registry exists.
- **Typed contracts** – dataclass-based params and result payloads keep schemas explicit
  and validated before requests reach an LLM.
- **Deterministic exposure** – rendered prompts surface a stable, machine-readable tuple
  of tools in declaration order.

### Key Structures

#### `ToolResult`

`ToolResult[PayloadT]` models the data returned to both orchestrators and the LLM:

- `message: str` – short textual reply forwarded to the model.
- `value: PayloadT | None` – typed payload produced by the handler. Successful executions
  return the documented dataclass; failures must set this to `None` unless they emit a
  structured error payload.
- `success: bool` – indicates whether the handler completed normally. Downstream reducers
  rely on this flag instead of inspecting `value`.
- `exclude_value_from_context: bool` – when `True`, adapters omit the payload from the
  provider-facing tool message while still returning it out-of-band.

##### Result Rendering Protocol

`ResultT` payloads must satisfy `ToolRenderableResult`, a simple protocol that guarantees
every result can produce a provider-safe string representation:

```python
from typing import Protocol

class ToolRenderableResult(Protocol):
    def render(self) -> str: ...
```

- `render()` **must** return a string and defaults to serialising the dataclass via
  `weakincentives.serde.dump(self, exclude_none=True)` and `json.dumps(...)`. Using the
  shared serde helpers keeps aliases, extras policies, and computed properties aligned with
  the structured response orchestrators forward to reducers.
- Tool authors may override `render()` when a friendlier summary helps the LLM reason
  about the response. For example, a search tool can emit a table-style plaintext view
  instead of raw JSON.
- Adapters treat `render()` as the canonical textual body when emitting `role: "tool"`
  messages, so providing deterministic output avoids confusing deltas between the
  message and `ToolResult.value`.
- Instrumentation also records `render()` output; `ToolInvoked` events MUST capture the
  rendered string alongside the structured payload so reducers and debuggers see identical
  content to what the provider observed.

#### `Tool`

`Tool[ParamsT, ResultT]` instances describe callable affordances:

- `name: str` – lowercase ASCII letters, digits, underscores, or hyphens (≤64 characters).
  Names must be unique per rendered prompt; collisions raise `PromptValidationError`.
- `description: str` – 1–200 character ASCII summary presented to the LLM.
- `handler: ToolHandler[ParamsT, ResultT] | None` – optional runtime hook that
  must accept a positional `params` argument **and** a keyword-only
  `context: ToolContext` parameter. When provided the handler returns a `ToolResult`.
- `accepts_overrides: bool` – flag that determines whether tooling participates in
  automatic override pipelines. Tools opt in by default (`True`), and sections can disable
  overrides on a case-by-case basis when contracts are still stabilizing.

Handlers follow the canonical signature:

```python
from weakincentives.prompt import ToolContext, ToolHandler, ToolResult

def handle_tool(params: ParamsT, *, context: ToolContext) -> ToolResult[ResultT]:
    ...
```

The `ToolHandler` protocol enforces this calling convention at type-check time, ensuring
implementations always accept the keyword-only `context` parameter and annotate their
return value with `ToolResult[ResultT]`.

### Section Integration

`Section.__init__` accepts an optional `tools` sequence. Sections normalize, validate, and
expose that collection via `Section.tools()`. Because every section supports the same
interface, authors can:

- Attach tools to existing `MarkdownSection`s without inventing bespoke subclasses.
- Register tooling on otherwise minimal sections that only emit headings or act as
  grouping nodes.
- Allow child sections to contribute additional tooling while parent enablement gates the
  branch.

### Prompt Rendering

`Prompt` continues to accept an ordered tree of sections. During initialization it walks
that tree depth-first to validate all contributed tools:

1. Duplicate names trigger `PromptValidationError`.
1. Parameter and result dataclasses reuse existing placeholder validation rules—required
   fields must be supplied when rendering.
1. Declaration order is cached so callers can retrieve tools without re-traversing the
   tree.

`Prompt.render(...)` returns a `RenderedPrompt` containing both the rendered markdown and
an ordered tuple of `Tool` instances contributed by enabled sections. The rendered object
also exposes `.tool_param_descriptions`, a mapping of tool name to per-field metadata
overrides sourced from the override system (description, type, and default hints for
every parameter attribute).

### Runtime Responsibilities

Sections merely document tools; they never execute handlers. Orchestrators are
responsible for:

- Creating the appropriate params dataclass instance when an LLM invokes a tool.
- Injecting the [`ToolContext`](#context-injection) keyword argument.
- Propagating the resulting `ToolResult` back to both the session reducers and the LLM.

### Example

```python
from dataclasses import dataclass, field

from weakincentives.prompt import MarkdownSection, Prompt, Tool, ToolContext, ToolResult


@dataclass
class LookupParams:
    entity_id: str = field(metadata={"description": "Global identifier to fetch"})
    include_related: bool = field(default=False)


@dataclass
class LookupResult:
    entity_id: str
    document_url: str


def lookup_handler(
    params: LookupParams, *, context: ToolContext
) -> ToolResult[LookupResult]:
    result = LookupResult(entity_id=params.entity_id, document_url="https://example.com")
    message = f"Fetched entity {result.entity_id}."
    return ToolResult(message=message, value=result)


lookup_tool = Tool[LookupParams, LookupResult](
    name="lookup_entity",
    description="Fetch structured information for a given entity id.",
    handler=lookup_handler,
)

prompt = Prompt(
    ns="examples/tooling",
    key="tools_overview",
    name="tools_overview",
    sections=[
        MarkdownSection(
            title="Guidance",
            template="Use tools when you need up-to-date context.",
            tools=[lookup_tool],
        )
    ],
)

rendered = prompt.render()
markdown = rendered.text
tools = rendered.tools
assert tools[0].name == "lookup_entity"
```

## Context Injection

Tool handlers receive an immutable snapshot of the surrounding runtime via `ToolContext`.
The dataclass surfaces only the objects the orchestrator already maintains while
executing a tool call.

### Data Model

```python
from dataclasses import dataclass
from typing import Any

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.prompt.prompt import Prompt, RenderedPrompt
from weakincentives.runtime.events import EventBus
from weakincentives.runtime.session.protocols import SessionProtocol


@dataclass(slots=True, frozen=True)
class ToolContext:
    prompt: Prompt[Any]
    rendered_prompt: RenderedPrompt[Any] | None
    adapter: ProviderAdapter[Any]
    session: SessionProtocol
    event_bus: EventBus
```

- `prompt` – the prompt instance currently executing. Handlers may inspect the section
  tree or helper methods when they need to mirror instructions.
- `rendered_prompt` – the `RenderedPrompt` generated for the outer call. Some adapters
  render lazily; in that case the value is `None` until rendering completes.
- `adapter` – the provider adapter orchestrating the outer request. Handlers can reuse it
  to execute nested prompts while sharing retry, tracing, and serialization behaviour.
- `session` – the active session instance, ensuring delegated work records against the
  same reducers.
- `event_bus` – the telemetry bus used for in-process events. Tools reuse it when
  publishing events for nested prompts or custom instrumentation.

### Construction Flow

1. The orchestrator renders (or prepares to render) the prompt and determines the active
   tool call.
1. Immediately before invoking the handler it builds a `ToolContext`, supplying the
   prompt, rendered prompt (if available), adapter, session, and event bus.
1. The handler executes with the `context` keyword argument.
1. After invocation the context instance is discarded; no references are reused across
   tool calls.

### Safety and Determinism

- The dataclass is frozen; handlers must not mutate the stored objects.
- Adapters should continue redacting sensitive provider payloads before attaching them to
  the prompt or session.
- Nested prompt calls must publish their events through the shared bus so the session
  state remains coherent.

## Failure Semantics

Tool execution must never abort the surrounding prompt evaluation. Instead, adapters and
handlers cooperate through a consistent contract so the LLM can recover gracefully.

### ToolResult Contract

- `success=True` indicates a normal payload; `success=False` signals any failure.
- Successful executions return the documented result dataclass via `value`.
- Failures set `value=None` unless the tool emits an explicit error payload. The
  `message` string still contains human-readable guidance.
- Session reducers and telemetry respect `success` when recording outcomes.

### Adapter Behaviour

- Wrap all handler exceptions—validation errors and unexpected exceptions alike—and
  convert them into `ToolResult(success=False, value=None, message="…")` responses.
- Forward the failure message back to the LLM as a `role: "tool"` message so the model
  can decide how to proceed.
- Log or attach the original exception for observability but avoid raising
  `PromptEvaluationError` for tool-level issues. Short-circuit only when provider
  communication fails or prompt parsing is impossible.

### Session and Telemetry

- Session reducers must tolerate `ToolResult.value is None` without dropping events.
- `ToolInvoked` events continue to fire even when tools fail; reducers MUST include the
  corresponding `result.render()` output (or `""` when `value is None`) so downstream
  observers can reconcile telemetry with provider-visible tool messages. Reducers may track separate
  slices for failures if desired.
- Consider dedicated failure events in the future, but the `success` flag suffices for the
  current design.

### Acceptance Checklist

- Adapters never abort evaluation solely because a tool handler failed.
- Unit tests assert the `success` semantics and nullable values.
- Documentation (including this file and adapter specs) references the contract.
