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
  - `NullEventBus` satisfies the interface and drops every event. Callers can pass it to adapter evaluations when
    telemetry is not required.
  - `InProcessEventBus` stores handlers in a per-type registry and iterates over a snapshot on publish. It guarantees
    in-order delivery and isolates subscriber exceptions by logging and continuing.
- Callers MUST provide an `EventBus` instance for each adapter evaluation. There is no module-level default; pass
  `NullEventBus()` to opt out of telemetry or wire up a custom bus scoped to the current request.

## Event Dataclasses

Events live under `weakincentives.events` and follow the same conventions as other core dataclasses (frozen, typed,
ASCII-friendly field names). The minimal spec keeps payloads strictly typed; provider-specific metadata can be added in
follow-on patches once concrete types are defined.

### `PromptExecuted`

Emitted exactly once per successful adapter evaluation, after all tool invocations and response parsing finish. Fields:

- `prompt_name: str` – logical name taken from the `Prompt` instance.
- `adapter: str` – identifier for the adapter emitting the event (e.g. `openai`, `anthropic`).
- `result: PromptResponse[Any]` – the structured result returned to the caller.

Adapters SHOULD reuse the `PromptResponse` object they already produce to avoid redundant allocations. Provider-specific
metadata is intentionally excluded from this minimal event until we have a concrete, typed schema to expose.

### `ToolInvoked`

Emitted every time an adapter executes a tool handler. The dataclass mirrors the information returned through
`PromptResponse.tool_results` so aggregated tooling data stays consistent across APIs:

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
