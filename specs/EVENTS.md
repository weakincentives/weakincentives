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

## Guiding principles

- **Typed, minimal payloads**: Events are frozen dataclasses in
  `weakincentives.runtime.events`; prefer explicit fields and reuse prompt/tool
  dataclasses over opaque dictionaries so reducers can rely on static typing.
- **Publisher isolation**: Adapter and session code MUST treat event dispatch as
  fire-and-forget. Publishers log handler failures and continue execution unless
  a caller explicitly opts into propagation via `PublishResult.raise_if_errors()`.
- **Subscriber determinism**: Subscriptions are scoped to exact event types and
  triggered synchronously on publish. Keep handlers idempotent and side-effect
  aware because they run on the adapter thread.
- **Session-first reducers**: The session module (`runtime/session/session.py`)
  is the canonical consumer; emitters should prefer enriching events rather than
  manipulating session state directly so reducers stay the single source of
  truth.
- **No implicit globals**: There is no module-level default bus. Callers MUST
  provide an `EventBus` instance per evaluation to avoid cross-request bleed.

## Scope

Event emission currently covers the synchronous prompt lifecycle inside a single
process:

- Prompt rendering, prompt execution, and tool invocations are the only
  lifecycle milestones with first-class events.
- Events are dispatched on the caller's thread with in-order delivery per bus
  instance. Cross-process forwarding, async fan-out, or persistence layers are
  outside the current scope and should be built in subscribers.
- The bus abstraction lives in `runtime/events/_types.py` and the reference
  implementation `InProcessEventBus` lives in `runtime/events/__init__.py`.
  Tests often depend on `tests.helpers.events.NullEventBus` to disable
  telemetry while satisfying the interface.
- Sessions subscribe to all defined event types to build reducers and snapshots
  without mutating adapter code.

## Event Bus Abstraction

- Adapters MUST emit prompt lifecycle events through an in-process message bus exposed by `weakincentives.runtime.events`.
- The bus presents a minimal synchronous API: `subscribe(event_type, handler)` for registration,
  `unsubscribe(event_type, handler)` for removal, and `publish(event)` for dispatch. Implementations MAY batch or queue
  under the hood but MUST deliver callbacks on the publishing thread by default.
- Subscriptions are type-scoped. Handlers receive concrete event dataclasses, not envelopes or loosely typed dictionaries.
- The module ships with an in-process implementation: `InProcessEventBus` stores handlers in a per-type registry and
  iterates over a snapshot on publish. It guarantees in-order delivery and isolates subscriber exceptions by logging
  and continuing.
- Unsubscription removes a handler from the per-type registry. `unsubscribe(event_type, handler)` returns `True` if the
  handler was found and removed, `False` otherwise. Unsubscription is thread-safe and can race with publishes without
  errors; handlers already captured in a publish snapshot will still be invoked for that publish call.
- Tests use `tests.helpers.events.NullEventBus` to satisfy the interface while discarding telemetry.
- Callers MUST provide an `EventBus` instance for each adapter evaluation. There is no module-level default; provide
  `InProcessEventBus()` or a custom implementation scoped to the current request.

## Event Dataclasses

Events live under `weakincentives.runtime.events` and follow the same conventions as other core dataclasses (frozen, typed,
ASCII-friendly field names). The minimal spec keeps payloads strictly typed; provider-specific metadata can be added in
follow-on patches once concrete types are defined.

### `PromptRendered`

Published immediately after rendering completes and before the adapter dispatches a request to the provider. Fields:

- `event_id: UUID` – immutable identifier generated with `uuid4()` when the event is constructed.
- `prompt_ns: str` – namespace of the prompt instance.
- `prompt_key: str` – key identifying the prompt within the namespace.
- `prompt_name: str | None` – human readable label when provided.
- `descriptor: PromptDescriptor | None` – descriptor describing the rendered prompt when available.
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
`ToolResult` so aggregated tooling data stays consistent across APIs:

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

### Emitters and listeners

- **Adapters**: `adapters/shared.py` publishes `PromptRendered` when the provider
  payload is ready, `ToolInvoked` as tools complete, and `PromptExecuted` once
  parsing finishes. Publish failures are logged and optionally bubbled via
  `PublishResult.raise_if_errors()` depending on adapter configuration.
- **Tool executor**: `ToolExecutor` in `adapters/shared.py` drives the
  per-tool `ToolInvoked` emission and surfaces aggregated handler errors in tool
  outputs when reducers fail.
- **Sessions**: `runtime/session/session.py` subscribes to all three events to
  populate slices and snapshots. Default reducers append the full events and, if
  present, the underlying dataclass payloads (`value`) so downstream reducers can
  react to typed prompt outputs or tool results.
- **Test helpers**: `tests.helpers.events.NullEventBus` satisfies the
  `EventBus` protocol while dropping events, useful for benchmarks or tests that
  mock telemetry.

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
- The test helper `tests.helpers.events.NullEventBus.publish` returns a `PublishResult` with the provided event, an empty
  handler snapshot, zero `handled_count`, and no errors.
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

## Caveats and known gaps

- **No persistence or buffering**: Events are process-local and discarded after
  handler execution. Adding queues, retries, or cross-process forwarding must
  happen in subscribers or custom `EventBus` implementations.
- **Schema gaps**: Provider-specific metadata (token counts, latency per tool,
  call IDs beyond `ToolInvoked.call_id`) is intentionally absent. Extend
  dataclasses with typed fields rather than stuffing dictionaries when new
  provider data becomes available.
- **Partial value propagation**: `PromptExecuted` and `ToolInvoked` expose a
  `value` field when a dataclass payload is present. Non-dataclass outputs will
  not enrich session slices; reducers that depend on raw strings or lists should
  inspect `result.output` and handle normalization themselves.
- **Listener discipline**: Subscribers run synchronously and can mutate shared
  state. Keep handlers cheap, idempotent, and side-effect aware to avoid
  stalling adapter threads.

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
- Consider adding a smoke test ensuring that `tests.helpers.events.NullEventBus.publish` returns a result object with
  `handled_count == 0` and no errors.
- Add targeted coverage in adapter/session tests to assert that a publish result with errors triggers the expected
  aggregated logging or, when configured, a propagated `ExceptionGroup` via `raise_if_errors()`.

## Logging Schema and Conventions

This section describes the runtime logging mini-framework that accompanies
event emission. It is an **internal** facility shared by runtime modules;
callers do not consume it directly.

### Design Intent

The logging framework is intentionally minimal: it provides shared semantics for
event names, severity, and structured payloads without wrapping the standard
library API. Key design choices:

- **Module isolation**: Each module owns its logger instance via
  `logging.getLogger(__name__)`.
- **Structured-first**: Prefer stable key/value pairs (`extra`) over message
  formatting.
- **Event taxonomy**: Every non-error record SHOULD carry an `event` key so
  collectors can bucket logs predictably.

### Current Implementation

Runtime modules attach to Python's standard library logging without custom
handlers:

| Module | Logger | Level | Message / Event | Context Fields |
| --- | --- | --- | --- | --- |
| `runtime/events/__init__.py` | `logger` | ERROR | "Error delivering event %s to handler %r" | `event_type`, `handler` |
| `runtime/session/session.py` | `logger` | ERROR | "Reducer %r failed for data type %s" | `reducer`, `data_type` |
| `adapters/shared.py` | `logger` | ERROR | "Tool '%s' raised an unexpected exception." | `tool_name` |
| `prompt/overrides/local_store.py` | `_LOGGER` | DEBUG | Override resolution events | `ns`, `prompt_key`, `tag`, `section_path` |
| `tools/asteval.py` | `_logger` | DEBUG | event="asteval.run" | `event`, `mode`, `stdout_len`, `stderr_len` |

### Required Context Keys

Logging calls SHOULD include the following fields when available:

- `event`: Stable event name categorizing the log entry.
- `prompt_name`: Name of the prompt being evaluated.
- `adapter`: Adapter identifier.
- `tool`: Tool identifier.
- `mode`: Execution mode for tools that support multiple behaviors.

### Severity Conventions

- `DEBUG`: Diagnostic and lifecycle messages for local development.
- `INFO`: High-level lifecycle events for default logs.
- `WARNING`: Recoverable conditions that may require operator attention.
- `ERROR`: Unexpected exceptions caught and converted into fallback paths.
- `CRITICAL`: Process about to exit or enter unrecoverable degraded state.

### Structured Context Delivery

Pass structured fields via the logger's `extra` mapping:

```python
logger.info(
    "Tool execution completed",
    extra={
        "event": "tool.run",
        "prompt_name": prompt.name,
        "adapter": adapter_id,
        "tool": tool.name,
    },
)
```

### Error Handling Expectations

- Publishing events MUST NOT raise from subscriber failures; the bus logs and
  exposes failures through the `PublishResult`.
- Reducers that raise are logged and skipped, leaving previous state in place.
- Tool handlers that raise are logged and converted into `ToolResult` instances
  with `success=False`.
