# Tool Context Specification

## Introduction

Tools often need additional metadata about the prompt execution pipeline to make well-informed decisions. Existing
interfaces only deliver the declared parameters to each tool handler, leaving adapters, session state, prompt metadata,
and orchestration utilities hidden. This specification introduces a `ToolContext` object that exposes those execution
components in a structured, read-only container.

## Goals

- **Provide execution visibility**: Supply tool handlers with a snapshot of prompt, adapter, and session state during each
  invocation.
- **Improve cross-cutting capabilities**: Unlock tooling that needs to publish events, inspect rendered prompts, or query
  session-scoped resources without bespoke plumbing for every handler.
- **Preserve safety and determinism**: Ensure the context is explicit, immutable where possible, and clearly scoped to a
  single tool invocation to avoid leaking mutable shared state.
- **Remain adapter-agnostic**: Offer a consistent API across adapters while permitting providers to contribute optional
  extensions.

## Non-goals

- Persisting arbitrary mutable state back into the prompt or session. Tool handlers must continue to communicate
  outcomes through `ToolResult` and orchestrator-defined channels.
- Replacing existing argument dataclasses. Business inputs remain separate from the contextual metadata provided by this
  object.

## Scope

`ToolContext` is constructed immediately before a tool handler executes and is passed to the handler alongside its
parameter payload. All built-in tooling helpers and adapter integrations must support the new signature. Orchestrators
that bypass the standard execution pipeline must construct and thread `ToolContext` instances themselves.

## Data Model

`ToolContext` is a frozen dataclass that surfaces the following fields:

- `prompt: Prompt` – Reference to the prompt instance being executed.
  - `original_sections: Sequence[Section]` – Optional view into the unrendered section tree.
  - `rendered_prompt: RenderedPrompt | None` – The prompt payload emitted after rendering (if available). Rendering may
    occur lazily; adapters that render on demand populate this field post-render.
- `adapter: Adapter` – The adapter orchestrating the current interaction. Exposes provider-specific utilities such as
  request factories or telemetry helpers.
- `session: Session` – The session driving the prompt lifecycle. Provides access to runtime configuration, identity, and
  session storage APIs.
- `event_bus: EventBus` – Publisher that handlers can use to emit structured events. Events must remain namespaced under
  the tool key and respect existing validation rules.
- `tool: Tool[Any, Any]` – The metadata entry describing the tool being invoked. Useful for introspection (e.g., name,
  description, override flags).
- `invocation: ToolInvocationContext` – Structured runtime metadata for the specific call. Includes unique invocation ID,
  timestamp, retry count, and parent message references when applicable.
- `request: ProviderRequest | None` – The provider-bound request object if the adapter renders prompts prior to tool
  execution. Allows instrumentation or mutation-free inspection.
- `response_accumulator: ResponseAccumulator | None` – Handle for streaming or partial response pipelines when the
  adapter supports them.
- `extras: Mapping[str, Any]` – Optional extension map reserved for adapter- or orchestrator-specific metadata. Keys must
  use snake_case ASCII to remain portable.

All fields are read-only. Mutating exposed objects is discouraged; adapters should pass immutable views or copies for
sensitive structures.

## API Surface

Tool handlers adopt the following signature:

```python
async def handle_tool(params: ParamsT, *, context: ToolContext) -> ToolResult[ResultT]:
    ...
```

- The `context` parameter is keyword-only to avoid ambiguity with existing positional arguments.
- Synchronous handlers continue to be supported; the orchestration layer injects `context` via keyword arguments.
- Legacy handlers that do not accept the `context` parameter receive a deprecation warning and eventually raise a
  validation error once the migration window closes.

### Construction

- Prompts expose a `build_tool_context(...)` helper that assembles the base fields (`prompt`, `tool`, `session`, and
  `invocation`).
- Adapters extend the base context with provider-specific data (`adapter`, `request`, `response_accumulator`, `extras`).
- Eventing utilities register themselves through the session to populate the `event_bus` reference.
- Rendering hooks capture the pre-render `prompt` state and the post-render `RenderedPrompt` payload when available.

## Lifecycle

1. Prompt execution identifies the next tool call.
2. Orchestrator constructs `ToolInvocationContext` capturing IDs, timing, and ancestry.
3. `build_tool_context` is invoked, threading in the prompt, session, event bus, and tool metadata.
4. Adapter enriches the context with provider details.
5. Tool handler executes with `(params, context=context)`.
6. Context is discarded after invocation; no references are reused across tool calls.

## Usage Patterns

- **Event emission**: Handlers publish telemetry via `context.event_bus.publish(...)` to surface structured events to
  observers or logging sinks.
- **Prompt inspection**: Tools validate their instructions or extract structured data by inspecting
  `context.rendered_prompt`.
- **Session storage**: Tools persist or retrieve state with `context.session.storage` utilities while preserving the
  single source of truth for user-specific data.
- **Adapter utilities**: Provider-specific adapters may expose rate limiting, retry helpers, or tracing interfaces through
  `context.adapter`.

## Safety Considerations

- Context objects must avoid holding on to large mutable payloads unnecessarily. Use lightweight references or
  explicitly documented streaming interfaces.
- Any object exposed via `extras` must be serializable or provide clear documentation about lifecycle and thread-safety
  expectations.
- Avoid leaking credentials or secrets through the context; adapters should redact or mask sensitive information.

## Migration Plan

1. Introduce the `ToolContext` dataclass alongside helper constructors.
2. Update built-in tool handlers and tests to accept the new `context` parameter.
3. Provide adapter shims that detect legacy handlers and supply `context` only when supported.
4. Emit deprecation warnings for handlers without a `context` parameter.
5. Remove compatibility shims and enforce the new signature in a future minor release.

## Open Questions

- Should we provide utility mixins for common `extras` entries (e.g., tracing spans, retry policies)?
- Do we need a stable serialization format for `ToolContext` to support remote execution environments?
- How should we represent adapters that render prompts lazily so handlers can differentiate between pre- and post-render
  contexts without checking for `None`?

