# Deadline Specification

## Goal

Define a single deadline concept that governs how long a prompt evaluation and
any delegated subagents may run. The deadline must propagate everywhere a run
is orchestrated so runaway executions terminate deterministically.

## Deadline Configuration

- Represent deadlines with a frozen `@dataclass(slots=True)` named
  `RunDeadline` in `src/weakincentives/runtime/deadline.py` and export it from
  `weakincentives.runtime`.
- Require a single field: `deadline: datetime.datetime`.
  - The timestamp MUST be timezone-aware and in UTC.
  - Reject naive datetimes and values that are already in the past during
    `__post_init__` by raising `ValueError`.
- Provide a helper method `time_remaining(monotonic_started: float) -> datetime.timedelta | None`.
  - The helper computes the remaining time by comparing the stored deadline to
    `datetime.datetime.now(datetime.timezone.utc)`.
  - Use the provided `monotonic_started` to fall back to monotonic computations
    when the wall clock appears to move backwards. When the fallback is used,
    return `None` if the elapsed monotonic time already exceeds the original
    deadline horizon.
- Add `expired(now: datetime.datetime | None = None) -> bool` to centralize the
  comparison logic. Default `now` to the current UTC time.

## Orchestrator Responsibilities

- When a run starts, record two timestamps:
  1. `started_at`: the UTC wall-clock time obtained via
     `datetime.datetime.now(datetime.timezone.utc)`.
  2. `started_monotonic`: `time.monotonic()` captured at the same moment.
- Store the `RunDeadline` instance alongside these timestamps in the execution
  context so every subsystem can reference the same deadline and monotonic
  baseline without mutation.

## Prompt Evaluation Enforcement

- Before evaluating a prompt section, call `deadline.expired()` and short-circuit
  with a structured timeout result when it returns `True`.
- During long-running sections, periodically recompute
  `deadline.time_remaining(started_monotonic)` and abort once it returns
  `datetime.timedelta(0)` or `None`.
- Timeout results should set `success=False`, include a machine-parsable error
  code such as `"deadline_exceeded"`, and attach metadata that surfaces the
  captured `deadline`, `started_at`, and remaining time (if any) to aid logging.

## Tool and Adapter Calls

- Tool adapters MUST consult `deadline.expired()` before issuing network or file
  system calls. If the deadline has passed, return the same structured timeout
  used for prompt sections.
- For operations that support partial progress reporting, supply the remaining
  wall-clock time to downstream clients so they can honor the same cutoff.

## Subagent Delegation

- When spawning subagents, thread the parent `RunDeadline` unchanged into every
  child context.
- Subagents may maintain their own `started_at` / `started_monotonic` pair for
  accurate remaining-time calculations, but they must never extend or ignore the
  inherited deadline.
- Any delegation helper that fans out to multiple subagents should check the
  deadline before dispatching each child and after aggregating responses. If the
  deadline expires mid-flight, cancel in-flight subagents where possible and
  surface a single consolidated timeout result to the parent run.

## Observability

- Emit structured logs whenever a deadline-related abort occurs. Include the
  deadline timestamp, elapsed time, and call site (prompt section, tool adapter,
  or subagent) to simplify debugging.
- Expose a lightweight metrics hook (counter or timer) so operators can monitor
  how often deadlines trigger and adjust default horizons accordingly.

