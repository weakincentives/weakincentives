# Tool Context Specification

## Introduction

Complex tools sometimes need access to the surrounding prompt execution so they can
compose follow-up prompts, delegate work to a sub-agent, and feed the result back to
the outer run. Today that metadata is already collected by the orchestrator (prompt,
adapter, session, and event bus) but tool handlers do not receive it directly. This
spec introduces a lightweight `ToolContext` container that forwards those existing
references to handlers without inventing new telemetry or identifiers.

## Goals

- **Enable nested orchestration**: Give handlers the prompt, adapter, session, and
  event bus they need to spawn a child prompt and surface its result.
- **Keep the contract minimal**: Only surface data the runtime already tracks rather
  than expanding the metadata footprint.
- **Stay deterministic**: Make the context immutable and scoped to a single
  invocation so tools cannot accidentally persist shared state.

## Non-goals

- Recording new execution metadata (IDs, timestamps, retry counters, etc.).
- Providing mutation hooks for prompts or sessions—tools still communicate outcomes
  through their `ToolResult` payloads.
- Replacing the existing parameter dataclasses passed to handlers.

## Scope

`ToolContext` is constructed immediately before invoking a tool handler and is passed
as a keyword-only argument. Built-in adapters and orchestration helpers adopt the new
signature; external callers that execute tools directly are responsible for creating
and threading the context themselves.

## Data Model

`ToolContext` is a frozen dataclass exposing only the objects the runtime already
holds while executing a tool:

```python
from dataclasses import dataclass
from typing import Any

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.runtime.events import EventBus
from weakincentives.prompt.prompt import Prompt, RenderedPrompt
from weakincentives.runtime.session.session import Session


@dataclass(slots=True, frozen=True)
class ToolContext:
    prompt: Prompt[Any]
    rendered_prompt: RenderedPrompt[Any] | None
    adapter: ProviderAdapter[Any]
    session: Session
    event_bus: EventBus
```

- `prompt` is the prompt instance currently executing. Handlers may inspect the
  section tree or helper methods to understand the calling contract.
- `rendered_prompt` mirrors the `RenderedPrompt` generated for the outer call. Some
  adapters render lazily; in that case the value is `None` until rendering completes.
- `adapter` is the provider adapter already orchestrating the outer request. Handlers
  can reuse it to execute nested prompts while sharing retry, tracing, and
  serialization behaviour.
- `session` is the active session instance. Forwarding it ensures child prompts
  update the same reducers and selectors.
- `event_bus` is the in-process event bus the runtime uses for telemetry. Tools reuse
  it to publish tool or prompt events when delegating work.

## Handler Signature

Handlers remain synchronous and accept the context via a keyword-only parameter:

```python
def handle_tool(params: ParamsT, *, context: ToolContext) -> ToolResult[ResultT]:
    ...
```

The orchestrator injects `context=` via keyword arguments when calling the handler.
Handlers **must** declare this parameter. If a handler omits the `context` keyword,
the orchestrator raises an error instead of attempting to call the tool with a
partial signature.

## Construction Flow

1. The orchestrator renders (or prepares to render) the prompt and determines the
   active tool.
1. Right before calling the handler it builds a `ToolContext`, supplying the prompt,
   rendered prompt (if already available), adapter, session, and event bus.
1. The handler executes with the `context` keyword argument.
1. After the invocation returns the context instance is discarded; no references are
   reused across tool calls.

## Usage Sketch

A delegation tool can now evaluate a follow-up prompt with the same adapter and
session state:

```python
rendered_child = context.adapter.evaluate(
    sub_prompt,
    child_params,
    bus=context.event_bus,
    session=context.session,
)

return ToolResult(
    message="Delegated work to sub-agent",
    value=rendered_child.output,
)
```

The handler inspects `context.prompt` or `context.rendered_prompt` when it needs to
mirror instructions, while session and bus forwarding keep telemetry consistent.

## Safety Considerations

- The dataclass is frozen; handlers must not mutate the stored objects. If mutable
  data needs to cross the boundary, adapters should provide read-only views.
- Adapters should continue redacting sensitive provider payloads before attaching
  them to the prompt or session; `ToolContext` does not add new exposure paths.
- Nested prompt calls must still publish their events through the shared bus so the
  session stays coherent.

## Migration Plan

1. Implement the `ToolContext` dataclass and helper constructors in the prompt
   runtime.
1. Update built-in tool handlers and adapters to accept and thread the `context`
   keyword argument.
1. Enforce the required `context` keyword at call time so handlers that fail to opt
   in raise immediately, ensuring no silent shims mask missing context.
