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

- Adapters MUST emit prompt lifecycle events through an in-process message bus exposed by `weakincentives.runtime.events`.
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

Events live under `weakincentives.runtime.events` and follow the same conventions as other core dataclasses (frozen, typed,
ASCII-friendly field names). The minimal spec keeps payloads strictly typed; provider-specific metadata can be added in
follow-on patches once concrete types are defined. Token accounting introduces an additional telemetry event so sessions and
observers can mirror the live ledger state without bespoke hooks.

### `PromptRendered`

Published immediately after rendering completes and before the adapter dispatches a request to the provider. Fields:

- `event_id: UUID` – immutable identifier generated with `uuid4()` when the event is constructed.
- `prompt_ns: str` – namespace of the prompt instance.
- `prompt_key: str` – key identifying the prompt within the namespace.
- `prompt_name: str | None` – human readable label when provided.
- `adapter: str` – adapter identifier preparing to execute the call.
- `session_id: UUID | None` – session identifier threading through the orchestration layer.
- `render_inputs: tuple[SupportsDataclass, ...]` – dataclass params used for rendering.
- `rendered_prompt: str` – fully rendered prompt text as dispatched to the provider.
- `created_at: datetime` – timezone-aware timestamp captured before handing off to the provider SDK.

### `PromptExecuted`

Emitted exactly once per successful adapter evaluation, after all tool invocations and response parsing finish. Fields:

- `event_id: UUID` – immutable identifier generated with `uuid4()` when the event is constructed.
- `prompt_name: str` – logical name taken from the `Prompt` instance.
- `adapter: str` – identifier for the adapter emitting the event (e.g. `openai`, `anthropic`).
- `result: PromptResponse[Any]` – the structured result returned to the caller.
- `session_id: UUID | None` – session identifier threading through the orchestration layer.
- `created_at: datetime` – timezone-aware timestamp captured immediately after evaluation completes.

Adapters SHOULD reuse the `PromptResponse` object they already produce to avoid redundant allocations. Provider-specific
metadata is intentionally excluded from this minimal event until we have a concrete, typed schema to expose.

### `ToolInvoked`

Emitted every time an adapter executes a tool handler. The dataclass mirrors the information returned through
`PromptResponse.tool_results` so aggregated tooling data stays consistent across APIs:

- `event_id: UUID` – immutable identifier generated with `uuid4()` when the event is constructed.
- `prompt_name: str` – name of the prompt that requested the tool invocation.
- `adapter: str` – adapter identifier.
- `name: str` – tool identifier provided by the prompt.
- `params: SupportsDataclass` – dataclass instance passed to the tool handler.
- `result: ToolResult[Any]` – structured return value from the handler.
- `call_id: str | None` – provider-specific correlation identifier when available.
- `session_id: UUID | None` – session identifier threading through the orchestration layer.
- `created_at: datetime` – timezone-aware timestamp captured immediately after the handler returns.

Adapters SHOULD publish `ToolInvoked` immediately after the handler returns. A failure is signalled via
`result.success=False`, and in that case `result.value` MAY be `None`.

### `TokenLedgerUpdated`

Published any time the token ledger reserves or consumes usage on behalf of an adapter evaluation. The event carries a
complete snapshot of the ledger after applying the mutation so observers, including the active `Session`, can record token
usage without additional queries. Fields:

- `event_id: UUID` – immutable identifier generated with `uuid4()` when the event is constructed.
- `adapter: str` – adapter identifier whose ledger mutated.
- `session_id: UUID | None` – session identifier threading through the orchestration layer.
- `created_at: datetime` – timezone-aware timestamp captured immediately after applying the mutation.
- `mutation: Literal["reserve", "consume", "release"]` – mutation type applied to the ledger. `reserve` covers optimistic
  deductions, `consume` reconciles actual usage, and `release` rolls back a reservation after a failed call.
- `aggregate: TokenTotals` – aggregate ledger totals after applying the mutation. `TokenTotals` is a frozen, slot-based
  dataclass with `total`, `input`, and `output` integer fields (all zero or positive).
- `providers: Mapping[str, TokenTotals]` – per-provider totals keyed by adapter slug. Missing providers imply zero usage.

`TokenTotals.total` MUST equal `TokenTotals.input + TokenTotals.output`. Implementations MAY store additional metadata on the
ledger but MUST NOT extend the event surface without updating this specification. Sessions capture `TokenLedgerUpdated`
events via a dedicated slice so downstream tooling can inspect historical usage.

## Delivery Semantics

- Event ordering MUST follow publish order per adapter instance so observers can reconstruct execution timelines.
- Publishing MUST NOT raise; adapters are responsible for isolating subscriber exceptions so evaluation can proceed.
- Events are currently synchronous and in-memory only. Any cross-process forwarding or persistence happens in subscriber
  code, not inside adapters.

## Publish Results

The existing `publish` contract drops handler exceptions after logging, which makes it difficult for callers to
understand whether reducers completed successfully. We will extend the API to surface delivery outcomes without changing
the core synchronous semantics.

### Goals

- Preserve fire-and-forget publishing while giving adapters and higher-level orchestration visibility into handler
  failures.
- Keep exception isolation: no handler failure should interrupt delivery to the remaining subscribers.
- Provide a structured summary that downstream code can inspect or log without parsing log output.

### Non-Goals

- Changing subscription mechanics or how handlers are stored.
- Introducing asynchronous dispatching or retries.
- Altering how adapters emit events; publish still executes synchronously on the caller's thread.

### API Additions

- Introduce `PublishResult` in `weakincentives.runtime.events` as a frozen, slot-based dataclass.
  - Fields:
    - `event: object` – the event instance that was dispatched.
    - `handlers_invoked: tuple[EventHandler, ...]` – ordered snapshot of handlers targeted for this publish call.
    - `errors: tuple[HandlerFailure, ...]` – ordered tuple of failures captured during dispatch. Empty when all handlers
      succeed.
    - `handled_count: int` – convenience count of handlers invoked, mirroring `len(handlers_invoked)` to avoid repeated
      tuple length checks in hot paths.
  - Methods/properties:
    - `ok: bool` property returning `True` when `errors` is empty.
    - `raise_if_errors()` helper that raises an `ExceptionGroup` composed of the captured exceptions while preserving the
      handler identity in the error message. This lets callers opt in to exception propagation without reimplementing the
      aggregation logic.
- Add `HandlerFailure` dataclass in the same module to represent a single handler error.
  - Fields: `handler: EventHandler` and `error: BaseException`.
  - Implement `__str__` to emit a concise diagnostic (`"{handler!r} -> {error!r}"`) for logging and debugging.
- Export `PublishResult` and `HandlerFailure` via `__all__` so downstream code can type against them.

### Publisher Responsibilities

- Publishers MUST treat a non-empty `PublishResult.errors` as a signal that delivery degraded.
  - Emit a single log line that aggregates the `HandlerFailure` diagnostics in order. Reuse the
    existing `logger.exception` calls for the individual failures but add a summary entry so
    operators have a stable breadcrumb tying the publish call to its failures.
  - After logging, publishers MAY surface additional diagnostics (for example by attaching the
    aggregated string to a tracing span or structured log event) but MUST keep evaluation flows
    uninterrupted by default.
- Publishers SHOULD expose a configuration or contextual knob that lets callers opt into
  propagation by invoking `PublishResult.raise_if_errors()`.
- When publishers choose to ignore the errors, they MUST still log the aggregated diagnostic and
  continue without raising.

#### Adapter and Session Guidance

- Adapter implementations emitting events during prompt execution SHOULD inspect the returned
  `PublishResult` immediately.
  - Call `raise_if_errors()` in environments where instrumentation failures invalidate the run
    (for example, adapter conformance suites or integration tests that assert telemetry). This
    keeps observability guarantees strict when required.
  - Otherwise, adapters SHOULD log the aggregated diagnostic and proceed without raising so prompt
    responses remain uninterrupted for end users.
- Session managers that orchestrate multiple adapter invocations SHOULD mirror this behaviour:
  raise on errors when running in verification or debugging modes, but treat telemetry failures as
  non-fatal in production flows after logging the summary diagnostic.

### EventBus Contract Changes

- Update the `EventBus` protocol so `publish` returns `PublishResult` instead of `None`.
- `NullEventBus.publish` returns a `PublishResult` with the provided event, an empty handler snapshot, zero `handled_count`,
  and no errors.
- `InProcessEventBus.publish` collects a snapshot of handlers for the event type before iterating, preserving the current
  isolation guarantee.
  - On each handler invocation:
    - Append the handler to a local list of invoked handlers.
    - Wrap calls in `try/except` to log exceptions (same logger call as today) and store a `HandlerFailure` with the
      original exception instance.
  - After iterating, construct a `PublishResult` with the event, the tuple of invoked handlers, the tuple of accumulated
    failures, and the handled count.
  - Logging remains unchanged; we continue to emit the `logger.exception` entry for observability even though the caller
    can now inspect the returned errors.
- Existing callers that ignore the return value continue to operate; the return is additive and does not introduce side
  effects. New code can opt in by inspecting `PublishResult.ok` or calling `raise_if_errors()`.

### Testing Strategy

- Extend `tests/test_events.py`:
  - Add a focused test verifying that publishing without subscribers returns a `PublishResult` with no handlers and
    `ok` set to `True`.
  - Add a test where multiple handlers are registered; ensure the result reflects the ordered handler list and aggregates
    all raised exceptions without interrupting remaining handlers.
  - Add a test covering `raise_if_errors()` to confirm it produces an `ExceptionGroup` with the captured exceptions and
    leaves successes untouched.
- Update existing tests that interact with the event bus (adapters, sessions, tools) to either assert `.ok` where
  appropriate or to ignore the return value explicitly so static type checkers remain satisfied. Helper utilities may
  need `assert bus.publish(...).ok` or assignment to `_` to avoid unused-value linting depending on context.
- Consider adding a smoke test ensuring that `NullEventBus.publish` returns a result object with `handled_count == 0` and
  no errors.
- Add targeted coverage in adapter/session tests to assert that a publish result with errors triggers the expected
  aggregated logging or, when configured, a propagated `ExceptionGroup` via `raise_if_errors()`.
