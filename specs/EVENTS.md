# Prompt Event Emission

## Introduction

Adapters need a unified way to surface evaluation telemetry without forcing every consumer to wrap provider SDKs
individually. This document defines the minimal event vocabulary and delivery mechanism that adapters must adopt to make
prompt execution observable inside a single process.

## Goals

- **Observable evaluations**: Surface prompt execution milestones that downstream systems can inspect and record.
- **In-process pub/sub**: Standardise on a lightweight message bus abstraction so adapters emit structured events without
  taking external dependencies.
- **Schema reuse**: Reuse existing prompt and tooling dataclasses wherever possible so events stay consistent with the
  synchronous API.

## Event Bus Abstraction

- Adapters MUST emit prompt lifecycle events through an in-process message bus exposed by `weakincentives.events`.
- The bus presents a minimal synchronous API: `subscribe(event_type, handler)` for registration and `publish(event)` for
  dispatch. Implementations MAY batch or queue under the hood but MUST deliver callbacks on the publishing thread by
  default.
- Subscriptions are type-scoped. Handlers receive concrete event dataclasses, not envelopes or loosely typed dictionaries.
- The module ships with two ready-made implementations:
  - `NullEventBus` satisfies the interface and drops every event. It is the module-level default returned by
    `get_default_bus()` so adapters can publish safely even when no subscribers opt in.
  - `InProcessEventBus` stores handlers in a per-type registry and iterates over a snapshot on publish. It guarantees
    in-order delivery and isolates subscriber exceptions by logging and continuing.
- `set_default_bus(bus)` lets embedders override the process-wide bus. Adapters MUST call `get_default_bus()` when a bus
  is not supplied explicitly so global configuration is respected.

## Event Dataclasses

Events live under `weakincentives.events` and follow the same conventions as other core dataclasses (frozen, typed,
ASCII-friendly field names). The minimal spec keeps payloads strictly typed; provider-specific metadata can be added in
follow-on patches once concrete types are defined.

### `PromptExecuted`

Emitted exactly once per successful adapter evaluation, after all tool invocations and response parsing finish. Fields:

- `prompt_name: str` – logical name taken from the `Prompt` instance.
- `adapter: str` – identifier for the adapter emitting the event (e.g. `openai`, `anthropic`).
- `response: PromptResponse[Any]` – the structured result returned to the caller.

Adapters SHOULD reuse the `PromptResponse` object they already produce to avoid redundant allocations. Provider-specific
metadata is intentionally excluded from this minimal event until we have a concrete, typed schema to expose.

### `ToolInvoked`

A thin rename of the existing `ToolCallRecord`, emitted every time an adapter executes a tool handler. Fields mirror the
record so aggregated tooling data stays consistent across APIs:

- `prompt_name: str` – name of the prompt that requested the tool invocation.
- `adapter: str` – adapter identifier.
- `name: str` – tool identifier provided by the prompt.
- `params: SupportsDataclass` – dataclass instance passed to the tool handler.
- `result: ToolResult[Any]` – structured return value from the handler.
- `call_id: str | None` – provider-specific correlation identifier when available.

Adapters SHOULD publish `ToolInvoked` immediately after the handler returns. If a handler raises, adapters MAY emit a
separate error event in the future; error semantics are out of scope for this minimal spec.

## Delivery Semantics

- Event ordering MUST follow publish order per adapter instance so observers can reconstruct execution timelines.
- Publishing MUST NOT raise; adapters are responsible for isolating subscriber exceptions so evaluation can proceed.
- Events are currently synchronous and in-memory only. Any cross-process forwarding or persistence happens in subscriber
  code, not inside adapters.
