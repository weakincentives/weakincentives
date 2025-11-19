# Throttling and Rate Limit Recovery

This spec describes a principled approach for detecting and recovering from
`openai.RateLimitError` and analogous throttling responses from other model
providers. The goal is to protect upstream services, preserve session-level
invariants, and provide predictable caller experience while maximizing useful
throughput.

## Goals and guardrails

- **Protect providers**: Never amplify overload with aggressive retries or burst
  amplification. Favor jittered, reactive backoff over any pre-emptive shaping
  that would alter caller intent before a throttle signal arrives.
- **Preserve correctness**: Keep requests idempotent under retry and expose
  enough context for callers to decide whether to reissue work.
- **Bound latency**: Cap retry windows so flows terminate in a predictable time.
- **Visibility first**: Emit structured telemetry for every throttle event to
  guide tuning and capacity planning.

## Taxonomy of throttling signals

- **HTTP 429 / explicit rate limit error**: OpenAI `RateLimitError` or provider
  equivalents. Prefer the provider-supplied `Retry-After` header or error field
  when present.
- **Token/budget exhaustion**: Errors indicating quota exhaustion (e.g., OpenAI
  `insufficient_quota`). Treat as throttling with a longer cool-down and
  elevated alerting because retries will not succeed until limits reset.
- **Soft timeouts**: Errors from local circuit breakers or queue timeouts. Handle
  as throttling when caused by backpressure; avoid retry when upstream latency is
  unknown or unbounded.

## Reactive handling only

- **No pre-send shaping**: Do not alter or down-scope requests before they hit
  the provider. All mitigation is triggered by explicit throttle signals (e.g.,
  HTTP 429, `Retry-After`, or provider error codes) to avoid silently degrading
  caller intent.
- **Backoff kicks in after signals**: The first provider throttle immediately
  enters the retry/backoff path; there is no pre-emptive smoothing. Subsequent
  throttles continue to use the same reactive policy until the budget is
  exhausted.
- **Fairness during recovery**: When multiple sessions are retrying, arbitrate
  retry scheduling so one session does not monopolize the backoff window, but
  never reshape or split the original request payload.

## Adapter integration map

- **Shared conversation runner**: Insert retry and backoff handling inside
  `adapters.shared.run_conversation` by wrapping the `call_provider` callable.
  The runner owns the request/response loop for both the OpenAI and LiteLLM
  adapters, so a single throttle policy there keeps behavior consistent while
  preserving the existing `PromptEvaluationError` surface area.
- **Provider-specific detection**:
  - OpenAI: inspect `openai.RateLimitError`, `openai.APITimeoutError`, and
    payloads that contain `"insufficient_quota"` or HTTP 429 responses inside
    `adapters.openai.OpenAIAdapter.evaluate`'s `_call_provider` closure.
    Normalize those into a structured `ThrottleError` before re-raising through
    `PromptEvaluationError` so telemetry and backoff logic can branch on
    `ThrottleError.kind`.
  - LiteLLM: handle `litellm.RateLimitError` and 429 payloads surfaced through
    the completion callable built in `adapters.litellm.create_litellm_completion`
    and `_call_provider`. Normalize LiteLLM's `retry_after` hints (when
    present) into the shared `ThrottleError.retry_after` field.
- **Session/runtime hooks**: propagate throttle metadata into `PromptExecuted`
  events published by `ConversationRunner` so downstream handlers (e.g., UI or
  orchestrators) can present retry guidance without re-inspecting provider
  payloads. Add a structured context field such as `"throttle": {"kind": ...}`
  on the structured logger events in `adapters.shared` so log-based monitors can
  alert on throttle rates.
- **Configuration plumbing**: centralize backoff and concurrency settings in the
  adapter constructors (e.g., optional `throttle_policy: ThrottlePolicy` kwarg
  on `OpenAIAdapter` and `LiteLLMAdapter`). The default policy should be shared
  via a helper in `adapters.shared` so both adapters stay aligned and new
  providers can opt in without duplicating constants.

## Retry policy

- **Exponential backoff with jitter**: Start with a small base delay (e.g.,
  250–500 ms) and double until capped (e.g., 8–16 s). Apply full jitter to avoid
  synchronization. Respect provider-specified `Retry-After` values; treat them
  as the minimum backoff.
- **Retry budget**: Cap total retry duration (e.g., 30–60 s) or attempt count to
  bound latency. Surface the exhausted budget as a structured failure with the
  last provider payload attached for context.
- **Idempotency**: Only retry operations that are safe to repeat. Prefer
  idempotency keys (where supported) or deterministic request construction so the
  provider can deduplicate.
- **Cancel on caller abort**: If the session or task is cancelled while waiting
  to retry, abort immediately and return a cancellation reason instead of
  continuing retries in the background.

## Adaptive degradation

- **Quality shaping**: When retry budgets are nearly exhausted, consider falling
  back to cheaper or smaller models, shorter contexts, or truncated tool traces
  to reduce pressure.
- **Queue shedding**: Reject new work with a clear "try later" error when local
  queues exceed safe depth; do not enqueue unbounded retries.
- **Circuit breakers**: Open a short circuit (e.g., 30–90 s) after repeated
  throttles from a provider/model pair to prevent hammering a constrained
  dependency.

## Telemetry and observability

- **Structured events**: Log throttle events with provider, model, status code,
  request type, retry count, backoff duration, and correlation/session ids.
- **Metrics**: Emit counters for throttles and quota failures plus timers for
  backoff delay. Track per-model rates to surface localized contention.
- **Tracing**: Attach retry spans with attributes for decision inputs (retry
  budget remaining, retry-after value, jittered delay). Mark the terminal span
  when retries are exhausted or cancelled.

## Surface area to callers

- **Typed errors**: Wrap provider exceptions in a local `ThrottleError` that
  records provider payloads, retry-after hints, whether retry is safe, and the
  number of attempts performed.
- **User messaging**: Provide actionable error messages that distinguish quota
  exhaustion from transient throttling. Include suggested wait durations when
  available.
- **Retry hints**: Expose retry-after timestamps on responses so interactive
  clients or schedulers can defer work without guessing.

## Testing strategy

- **Unit**: Simulate provider responses for 429s with and without `Retry-After`,
  quota exhaustion, and malformed payloads. Assert backoff calculations, jitter,
  retry ceilings, and idempotency key propagation.
- **Integration**: Use the OpenAI test endpoint or a stub server to validate
  envelope shape, telemetry, and circuit breaker behavior under sustained
  throttling.
- **Load**: Stress concurrency budgets with synthetic traffic to confirm fairness
  and that shedding happens before upstream saturation.

## Operational playbook

- **Alarm thresholds**: Alert when throttle rate exceeds a baseline (e.g., >1%
  over five minutes) or when quota failures occur. Distinguish transient spikes
  from sustained limit breaches.
- **Config levers**: Centralize backoff constants, retry budgets, and concurrency
  caps in configuration so they can be tuned without code changes. Validate at
  startup and disallow zero/negative values.
- **Runbooks**: Document per-provider quirks (e.g., headers to respect, error
  fields to parse) and escalation paths for quota raises. Keep examples of
  throttle payloads to aid debugging.
