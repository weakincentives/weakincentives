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

- **Reactive detection**: Requests are issued unchanged until the provider
  returns an explicit throttling signal. OpenAI and LiteLLM adapters normalize
  429 responses, rate limit errors, and `insufficient_quota` payloads into a
  shared `ThrottleError` that triggers retry.
- **Backoff kicks in after signals**: The first throttle raises `ThrottleError`,
  which `run_conversation` handles by backing off with jitter and retrying until
  the budget is exhausted. There is no pre-emptive smoothing.
- **Independence across sessions**: Backoff decisions and budgets are scoped to
  the `ConversationRunner` instance so one throttled session will not alter the
  pacing of others.

## Adapter integration map

- **Shared conversation runner**: Retry and backoff handling live in
  `adapters.shared.run_conversation`, which wraps `call_provider` for OpenAI and
  LiteLLM adapters. Throttle handling preserves the existing
  `PromptEvaluationError` surface area while adding a `throttle` payload when
  retries are exhausted.
- **Provider-specific detection**:
  - OpenAI: `OpenAIAdapter` maps `RateLimitError`, `APITimeoutError`, HTTP 429
    statuses, and messages containing `"insufficient_quota"` into
    `ThrottleError` with retry-after hints pulled from the SDK error payloads.
  - LiteLLM: `LiteLLMAdapter` maps `RateLimitError`, 429 statuses, and
    `insufficient_quota` codes exposed by LiteLLM into the same `ThrottleError`
    shape, carrying through any `retry_after` hint when present.
- **Session/runtime hooks**: `ConversationRunner` logs a structured
  `prompt_throttled` event with throttle metadata for each backoff interval. If
  the retry budget is exhausted, the resulting `PromptEvaluationError` includes
  a `{"throttle": ...}` provider payload for downstream consumers.
- **Configuration plumbing**: Both adapters accept an optional
  `throttle_policy: ThrottlePolicy` keyword argument. The default policy is
  shared via `DEFAULT_THROTTLE_POLICY` in `adapters.shared` so providers stay in
  sync.

## Retry policy

- **Exponential backoff with jitter**: The default `ThrottlePolicy` starts at
  0.5 s, doubles per attempt, and caps at 8 s with full jitter applied to the
  ceiling. Provider `retry_after` hints raise the jitter ceiling when present.
- **Retry budget**: The default budget allows up to five throttle retries within
  a 30 s window. When the budget is exhausted the runner raises
  `PromptEvaluationError` with throttle metadata in `provider_payload`.
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
