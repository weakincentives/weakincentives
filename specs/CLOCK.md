# CLOCK.md

Specification for controllable time dependencies in the WINK runtime.

## Overview

All time-dependent code in WINK uses explicit dependency injection via the
`Clock` protocol and its implementations. This enables deterministic testing
without real delays or fragile monkeypatching.

**Implementation:** `src/weakincentives/clock/`

**Core types:**

- `MonotonicClock` — Protocol for elapsed time measurement (timeouts, rate limiting)
- `WallClock` — Protocol for UTC timestamps (deadlines, event recording)
- `Sleeper` — Protocol for synchronous delay operations
- `AsyncSleeper` — Protocol for asynchronous delay operations
- `Clock` — Unified protocol combining all four capabilities
- `SystemClock` / `SYSTEM_CLOCK` — Production implementation using real system calls
- `FakeClock` — Test implementation that advances time instantly

## Time Domains

WINK distinguishes two distinct time domains:

**Monotonic time** (float seconds): For measuring elapsed time, timeouts, and
rate limiting. Guaranteed to never go backwards. The zero point is arbitrary
and unrelated to wall-clock time.

**Wall-clock time** (UTC datetime): For timestamps, deadlines, and recording
when events occurred. Can jump due to NTP adjustments but always
timezone-aware (UTC).

## Clock Protocols

Narrow protocols at `src/weakincentives/clock.py`:

| Protocol | Method | Use for |
|----------|--------|---------|
| `MonotonicClock` | `monotonic() -> float` | Elapsed time, timeouts |
| `WallClock` | `utcnow() -> datetime` | Timestamps, deadlines |
| `Sleeper` | `sleep(seconds: float) -> None` | Synchronous delays |
| `AsyncSleeper` | `async_sleep(seconds: float) -> None` | Async delays |
| `Clock` | All of the above | Full time control |

Components depend on the narrowest protocol that meets their needs:

| Component | Protocol | Reason |
| --- | --- | --- |
| `Deadline` | `WallClock` | Compares against wall-clock time |
| `Heartbeat` | `MonotonicClock` | Measures elapsed intervals |
| `LeaseExtender` | `MonotonicClock` | Tracks extension intervals |
| `wait_until()` | `Clock` | Needs sleep + monotonic |
| `sleep_for()` | `Sleeper` | Only needs sleep |
| `InMemoryMailbox` | `MonotonicClock` | Message visibility timing |
| `TranscriptCollector` | `AsyncSleeper` | Async polling delays |
| `CodexAppServerAdapter` | `AsyncSleeper` | Async deadline sleep |
| `ACPAdapter` | `AsyncSleeper` | Async drain wait |

## FakeClock

`FakeClock` is the key testing tool. It advances time instantly without
blocking—`sleep(10)` returns immediately and advances the internal clock by 10
seconds:

```python
clock = FakeClock()
start = clock.monotonic()
clock.sleep(10)  # Returns instantly
assert clock.monotonic() - start == 10
```

### FakeClock API

| Method | Description |
| --- | --- |
| `monotonic()` | Return current monotonic time |
| `utcnow()` | Return current wall-clock time |
| `sleep(seconds)` | Advance time instantly (calls `advance()`) |
| `async_sleep(seconds)` | Async version — advance time instantly |
| `advance(seconds)` | Advance both clocks by duration (non-negative only) |
| `set_monotonic(value)` | Set monotonic time to absolute value |
| `set_wall(value)` | Set wall-clock time (must be timezone-aware) |

Both `advance()`, `sleep()`, and `async_sleep()` advance monotonic and
wall-clock time together. All operations are thread-safe. `advance()` raises
`ValueError` if given negative seconds.

Test helpers and pytest fixtures at `tests/helpers/time.py`.

## Prohibited: Direct `datetime.now(UTC)` Calls

**`datetime.now(UTC)` must never appear in production code outside of
`SystemClock.utcnow()`.**

`datetime.now(UTC)` is syntactically correct (timezone-aware), but it bypasses
the `WallClock` protocol and makes the calling code untestable without
monkeypatching. All wall-clock access must flow through the protocol so that
tests can inject `FakeClock` for deterministic time control.

Equally prohibited: `time.time()`, `time.sleep()`, `threading.Event.wait()`,
and `asyncio.sleep()` — always use the injectable protocol instead.

### How to Get the Current Time

Choose the approach that matches your context, from most testable to least:

| Approach | When to use | Testability |
| --- | --- | --- |
| `self.clock.utcnow()` | Component has an injected `WallClock` field | Full control via `FakeClock` |
| `SYSTEM_CLOCK.utcnow()` | No injected clock available (utility functions, event DTOs) | Patchable singleton |
| `datetime.now(UTC)` | **Never** (only inside `SystemClock.utcnow()`) | None |

**Preferred pattern** — Components accept a `clock` parameter typed to the
narrowest protocol, defaulting to `SYSTEM_CLOCK`:

```python
@dataclass(slots=True)
class MyComponent:
    clock: WallClock = field(default=SYSTEM_CLOCK, repr=False, compare=False)
```

**For event DTOs and utilities** without a clock parameter, use
`SYSTEM_CLOCK.utcnow()` as the `default_factory` rather than
`datetime.now(UTC)`. This routes through the protocol and is patchable.

## Non-Goals

- **Timezones**: WINK uses UTC exclusively
- **Calendar operations**: No date arithmetic needed
- **High-resolution timing**: Millisecond precision is sufficient
- **Distributed clock sync**: Single-process focus
