# Error Handling and Retry Specification

## Overview

Agents interacting with tools must recover gracefully from validation errors and
invalid structured outputs. This specification introduces a consistent retry
configuration surface, structured feedback channels, and reducer-driven error
reporting so the inner inference loop can react immediately when a tool call or
state update fails.

## Design Principles

1. **Retry only on recoverable faults** – Retries should trigger exclusively for
   validation errors (e.g., argument schema mismatches, enum violations) or
   structured output parsing failures. Transport errors, timeouts, or runtime
   exceptions fall outside this scope.
2. **Deterministic retries** – Retry decisions are deterministic functions of the
   current attempt count, the validation failure metadata, and explicit
   configuration. Hidden random backoff is prohibited unless seeded and
   reproducible.
3. **Prompt-level observability** – Every retry reason and corrective feedback
   must surface through the session state so prompts can reflect on their last
   attempt before producing a follow-up tool invocation.
4. **Reducer integrity** – Session reducers are the single source of truth for
   error-aware state transitions. They must not mutate external state and should
   communicate issues synchronously.

## Configuration Surface

### Retry Policies

Expose a `RetryPolicy` dataclass that captures the contract for recoverable
validation errors:

```python
@dataclass(slots=True, frozen=True)
class RetryPolicy:
    max_attempts: int
    backoff: BackoffStrategy
    retryable_errors: tuple[type[Exception], ...]
    emit_feedback: bool = True
```

- `max_attempts` counts the initial attempt plus retries. The default is `3`.
- `backoff` encodes a deterministic delay strategy (e.g., constant, linear). Use
  typed strategy objects to avoid ad-hoc sleeps.
- `retryable_errors` enumerates validation exception types. Include
  `ToolValidationError` and `StructuredOutputError` by default.
- `emit_feedback` toggles reducer-driven commentary for the inference loop.

Policies live alongside tool definitions or structured output specs and support
three merge layers: global defaults, per-tool overrides, and per-call overrides
supplied by the planner. Merging follows "most specific wins" semantics.

### Backoff Strategies

Provide a family of deterministic strategies:

```python
Protocol BackoffStrategy:
    def delay_seconds(self, attempt: int, *, seed: str | None = None) -> float: ...
```

- `ConstantBackoff(seconds: float)` – Always return the same delay.
- `LinearBackoff(base: float, step: float)` – `base + step * (attempt - 1)`.
- `ExponentialBackoff(base: float, factor: float)` – `base * factor ** (attempt - 1)`.

The inference loop is responsible for sleeping according to the returned delay
before the next attempt.

## Execution Flow

1. **Tool invocation** – When the LLM proposes arguments, the tool adapter
   validates them. On success, continue normally.
2. **Validation failure** – If validation fails with a retryable exception, the
   policy increments the attempt count. The inference loop consults the backoff
   strategy, emits structured feedback, and resubmits the tool call if attempts
   remain.
3. **Structured output parsing** – After a prompt returns, parse the structured
   output. Failures follow the same retry flow as tool validations.
4. **Retry exhaustion** – When `max_attempts` is reached, surface a terminal
   `RetryExhaustedError` that contains the history of attempts and reasons. The
   planner decides whether to abort or fall back to a different action.

## Structured Feedback

Introduce a `RetryFeedback` dataclass stored in the session state:

```python
@dataclass(slots=True, frozen=True)
class RetryFeedback:
    source: Literal["tool", "structured_output"]
    tool_name: str | None
    attempt: int
    max_attempts: int
    error_type: str
    message: str
    retry_scheduled_at: datetime
    next_delay_seconds: float
```

Reducers append `RetryFeedback` events for every retryable validation failure.
Prompts can query the latest feedback to craft corrected arguments or outputs.

## Reducer Error Propagation

Reducers run synchronously after each tool invocation or structured output event.
If a reducer raises a validation or state consistency error while applying the
update:

1. Capture the exception and wrap it in a `ReducerError` dataclass containing the
   reducer identifier, the offending payload, and the error message.
2. Publish the `ReducerError` back into the session state via a dedicated slice
   so downstream prompts can detect it.
3. Notify the inference loop immediately through a callback interface
   (`Session.on_reducer_error`). The loop halts further tool retries for the
   current action and prompts the LLM with the reducer feedback instead.

Reducers must not swallow errors silently. Every caught exception becomes either
retryable feedback (if the policy allows) or a terminal failure that surfaces to
operators.

## Observability & Instrumentation

- Emit structured logs for each retry attempt, including the error type, attempt
  number, and delay.
- Maintain metrics counters (`validation_retry_total`, `reducer_error_total`) and
  histograms for delays and attempt counts.
- Attach correlation IDs from the session to every log entry to simplify tracing
  across retries.

## Open Questions

- Should retry policies be dynamically adjustable mid-run based on reducer
  feedback or external signals?
- Do planners need a declarative way to opt-out of retries when operating under
  strict time budgets?
- How should long-running tools communicate partial progress when a retry is
  required?
