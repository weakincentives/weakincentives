# Limits Specification

## Goal

Provide a single configuration surface that caps runtime resource usage across
prompts, tool invocations, and adapter requests. The feature prevents runaway
executions while keeping the integration surface minimal.

## Configuration Object

- Define `RunLimits` in `src/weakincentives/runtime/limits.py` and export it from
  `weakincentives.runtime`.
- Implement it as an immutable `@dataclass(slots=True, frozen=True)` with the
  following optional fields (all default to `None`, meaning "no limit"):
  - `max_duration: datetime.timedelta | None`
  - `max_tool_calls: int | None`
  - `max_delegation_depth: int | None`
  - `max_parallel_subagents: int | None`
  - `adapter_rate_limit: AdapterRateLimit | None`
- Add `AdapterRateLimit` beside `RunLimits` with slots/frozen, containing:
  - `max_requests: int`
  - `per: datetime.timedelta`
- Both dataclasses MUST validate positive integers and non-zero durations during
  `__post_init__`, raising `ValueError` when misconfigured.

## Deadline Enforcement

- The orchestrator records a monotonic start time for every prompt run.
- When `max_duration` is set, adapters and sections consult
  `limits.deadline` (start + duration) through a helper on `RunLimits`.
- The helper returns `None` when no deadline is configured.
- Before executing any tool or subagent dispatch, compare the current monotonic
  time to the deadline and abort with a structured timeout error when exceeded.
- Timeout errors propagate as failed `ToolResult` or prompt evaluation results
  without raising unhandled exceptions.

## Tool Call Ceiling

- Maintain a counter on the execution context that increments before each tool
  invocation (including `dispatch_subagents`).
- When `max_tool_calls` is set and the counter would exceed the limit, short
  circuit with a failed `ToolResult` (`success=False`, `message="tool call
  limit reached"`).
- Do not attempt the tool call once the limit trips; return immediately so
  downstream reducers see a clean failure state.

## Delegation Depth Guardrails

- Extend the subagent orchestration context with a `delegation_depth` integer
  that starts at `0` for the root run.
- Every child spawned by `dispatch_subagents` increments the depth by `1`.
- When `max_delegation_depth` is provided and the next depth would exceed it,
  the handler MUST reject the delegation batch before any children start.
- The rejection returns a single failed `ToolResult` with a descriptive error
  and no partial children.

## Parallel Subagent Limit

- Before submitting a delegation batch, compute the total number of children
  that would be active concurrently (`len(delegations)` plus any currently
  running siblings).
- If `max_parallel_subagents` is set and the total would cross the limit, raise a
  failed `ToolResult` immediately with `success=False` and an explanatory
  message.
- When allowed, pass `min(len(delegations), max_parallel_subagents)` as the
  executor's `max_workers` so the handler never spawns more threads than
  permitted.

## Adapter Rate Limits

- Adapter implementations call `limits.record_adapter_call(adapter_id)` before
  contacting the provider.
- `RunLimits` tracks per-adapter windows using a monotonic timestamp queue.
- If `adapter_rate_limit` is `None`, the helper becomes a no-op.
- When the rate limit is configured, the helper clears timestamps older than the
  configured window and blocks once `max_requests` would be exceeded.
- Blocking manifests as a retry-friendly error: return a failed adapter result
  with `success=False`, `error="rate limit exceeded"`, and the time until the
  next slot becomes available.

## Surfaces to Update

- Extend the session or executor builder to accept an optional `RunLimits`
  instance; default to `RunLimits()` when nothing is supplied.
- Thread the limits through adapters, tool dispatch, and subagent orchestration
  without mutating them (the objects are frozen and shared).
- Document the new keyword argument in any public factory or CLI entry point
  that constructs prompt runs.
