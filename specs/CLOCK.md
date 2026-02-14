# CLOCK.md

Specification for controllable time dependencies in the WINK runtime.

## Overview

All time-dependent code in WINK uses explicit dependency injection via the
`Clock` protocol and its implementations. This enables deterministic testing
without real delays or fragile monkeypatching.

**Core types:**

- `MonotonicClock` - Protocol for elapsed time measurement (timeouts, rate
  limiting)
- `WallClock` - Protocol for UTC timestamps (deadlines, event recording)
- `Sleeper` - Protocol for synchronous delay operations
- `AsyncSleeper` - Protocol for asynchronous delay operations
- `Clock` - Unified protocol combining all capabilities
- `SystemClock` - Production implementation using real system calls
- `FakeClock` - Test implementation that advances time instantly
- `SYSTEM_CLOCK` - Module-level singleton for production use

## Time Domains

WINK distinguishes two distinct time domains:

**Monotonic time** (float seconds): For measuring elapsed time, timeouts, and
rate limiting. Guaranteed to never go backwards. The zero point is arbitrary and
unrelated to wall-clock time.

**Wall-clock time** (UTC datetime): For timestamps, deadlines, and recording
when events occurred. Can jump due to NTP adjustments but always timezone-aware
(UTC).

## Clock Protocols

```python
from weakincentives.clock import (
    AsyncSleeper,
    Clock,
    MonotonicClock,
    WallClock,
    Sleeper,
)

# Narrow protocols for specific needs
class MonotonicClock(Protocol):
    def monotonic(self) -> float: ...

class WallClock(Protocol):
    def utcnow(self) -> datetime: ...

class Sleeper(Protocol):
    def sleep(self, seconds: float) -> None: ...

class AsyncSleeper(Protocol):
    async def async_sleep(self, seconds: float) -> None: ...

# Combined protocol for components needing multiple capabilities
class Clock(MonotonicClock, WallClock, Sleeper, AsyncSleeper, Protocol):
    pass
```

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
| `deadline_watchdog()` | `AsyncSleeper` | Async deadline sleep |
| `ACPAdapter._drain_quiet_period()` | `AsyncSleeper` | Async drain wait |

## SystemClock

The production implementation delegates to Python's standard library:

```python
from weakincentives.clock import SystemClock, SYSTEM_CLOCK

# Use the singleton
start = SYSTEM_CLOCK.monotonic()
SYSTEM_CLOCK.sleep(1.0)
elapsed = SYSTEM_CLOCK.monotonic() - start  # ~1.0

# Or create an instance
clock = SystemClock()
now = clock.utcnow()  # datetime.now(UTC)
```

`SystemClock` is a frozen, slotted dataclass:

- `monotonic()` → `time.monotonic()`
- `utcnow()` → `datetime.now(UTC)`
- `sleep()` → `time.sleep()`
- `async_sleep()` → `asyncio.sleep()`

## FakeClock

The test implementation advances time instantly without blocking:

```python
from weakincentives.clock import FakeClock

clock = FakeClock()

start = clock.monotonic()
clock.sleep(10)  # Advances immediately, no real delay
assert clock.monotonic() - start == 10

# Manual advancement
clock.advance(60)
assert clock.monotonic() - start == 70
```

### FakeClock API

| Method | Description |
| --- | --- |
| `monotonic()` | Return current monotonic time |
| `utcnow()` | Return current wall-clock time |
| `sleep(seconds)` | Advance time instantly (calls `advance()`) |
| `async_sleep(seconds)` | Async version — advance time instantly (calls `advance()`) |
| `advance(seconds)` | Advance both clocks by duration (non-negative only) |
| `set_monotonic(value)` | Set monotonic time to absolute value |
| `set_wall(value)` | Set wall-clock time (must be timezone-aware) |

Both `advance()`, `sleep()`, and `async_sleep()` advance monotonic and
wall-clock time together.
All operations are thread-safe. `advance()` raises `ValueError` if given
negative seconds; use `set_monotonic()` and `set_wall()` for explicit clock
positioning when needed.

## Usage Patterns

### Deadline Testing

```python
from datetime import UTC, datetime, timedelta
from weakincentives import Deadline
from weakincentives.clock import FakeClock

clock = FakeClock()
clock.set_wall(datetime(2024, 6, 1, 12, 0, tzinfo=UTC))

deadline = Deadline(
    expires_at=datetime(2024, 6, 1, 13, 0, tzinfo=UTC),
    clock=clock,
)

assert deadline.remaining() == timedelta(hours=1)
clock.advance(1800)  # 30 minutes
assert deadline.remaining() == timedelta(minutes=30)
```

### Heartbeat Testing

```python
from weakincentives.runtime.watchdog import Heartbeat
from weakincentives.clock import FakeClock

clock = FakeClock()
hb = Heartbeat(clock=clock)

hb.beat()
assert hb.elapsed() == 0.0

clock.advance(10)
assert hb.elapsed() == 10.0
```

### wait_until Testing

```python
from weakincentives.runtime.lifecycle import wait_until
from weakincentives.clock import FakeClock

clock = FakeClock()
calls = []

def eventually_true() -> bool:
    calls.append(clock.monotonic())
    clock.advance(0.5)
    return len(calls) >= 3

result = wait_until(
    eventually_true,
    timeout=2.0,
    poll_interval=0.5,
    clock=clock,
)

assert result is True
assert len(calls) == 3
```

### Throttle Testing

```python
from datetime import timedelta
from weakincentives.adapters.throttle import sleep_for
from weakincentives.clock import FakeClock

clock = FakeClock()
initial = clock.monotonic()

sleep_for(timedelta(seconds=5), sleeper=clock)

assert clock.monotonic() == initial + 5
```

### Async Sleep Testing

```python
import asyncio
from weakincentives.clock import FakeClock

clock = FakeClock()
start = clock.monotonic()

await clock.async_sleep(5.0)  # Advances instantly, no real delay
assert clock.monotonic() - start == 5.0
```

Async components accept an `async_sleeper` parameter defaulting to
`SYSTEM_CLOCK`:

```python
from weakincentives.clock import FakeClock

clock = FakeClock()
collector = TranscriptCollector(
    prompt_name="test",
    async_sleeper=clock,
)
```

## Component Integration

All clock-dependent components accept a clock parameter with a sensible default:

```python
# Deadline uses SYSTEM_CLOCK by default
deadline = Deadline(expires_at=datetime(2024, 12, 31, tzinfo=UTC))

# Inject FakeClock for testing
clock = FakeClock()
deadline = Deadline(expires_at=..., clock=clock)
```

The pattern is consistent across all components:

| Component | Parameter | Type | Default |
| --- | --- | --- | --- |
| `Deadline` | `clock` | `WallClock` | `SYSTEM_CLOCK` |
| `Heartbeat` | `clock` | `MonotonicClock` | `SYSTEM_CLOCK` |
| `LeaseExtender` | `clock` | `MonotonicClock` | `SYSTEM_CLOCK` |
| `wait_until()` | `clock` | `Clock` | `SYSTEM_CLOCK` |
| `sleep_for()` | `sleeper` | `Sleeper` | `SYSTEM_CLOCK` |
| `InMemoryMailbox` | `clock` | `MonotonicClock` | `SYSTEM_CLOCK` |
| `TranscriptCollector` | `async_sleeper` | `AsyncSleeper` | `SYSTEM_CLOCK` |
| `deadline_watchdog()` | `async_sleeper` | `AsyncSleeper` | `SYSTEM_CLOCK` |
| `ACPAdapter` | `async_sleeper` | `AsyncSleeper` | `SYSTEM_CLOCK` |

## Prohibited: Direct `datetime.now(UTC)` Calls

**`datetime.now(UTC)` must never appear in production code outside of
`SystemClock.utcnow()`.**

`datetime.now(UTC)` is syntactically correct (timezone-aware), but it bypasses
the `WallClock` protocol and makes the calling code untestable without
monkeypatching. All wall-clock access must flow through the protocol so that
tests can inject `FakeClock` for deterministic time control.

### How to Get the Current Time

Choose the approach that matches your context, from most testable to least:

| Approach | When to use | Testability |
| --- | --- | --- |
| `self.clock.utcnow()` | Component has an injected `WallClock` field | Full control via `FakeClock` |
| `SYSTEM_CLOCK.utcnow()` | No injected clock available (utility functions, event DTOs) | Patchable singleton |
| `datetime.now(UTC)` | **Never** (only inside `SystemClock.utcnow()`) | None |

### Preferred: Inject a `WallClock` Parameter

Components that need wall-clock time should accept a `clock` parameter typed to
the narrowest protocol, defaulting to `SYSTEM_CLOCK`:

```python
from weakincentives.clock import SYSTEM_CLOCK, WallClock

@dataclass(slots=True)
class MyComponent:
    clock: WallClock = field(default=SYSTEM_CLOCK, repr=False, compare=False)

    def record_event(self) -> Event:
        return Event(created_at=self.clock.utcnow())
```

### Acceptable: `SYSTEM_CLOCK.utcnow()` for DTOs and Utilities

Event dataclasses and utility functions that don't accept a clock parameter
should use `SYSTEM_CLOCK.utcnow()` rather than `datetime.now(UTC)`:

```python
from weakincentives.clock import SYSTEM_CLOCK

@dataclass(frozen=True, slots=True)
class MyEvent:
    created_at: datetime = field(default_factory=SYSTEM_CLOCK.utcnow)
```

This routes through the protocol and is patchable at `SYSTEM_CLOCK` level when
needed.

### Module-Level `_utcnow()` Helpers

Several adapter modules define a private `_utcnow()` helper as a testability
seam. These must delegate to `SYSTEM_CLOCK`:

```python
from weakincentives.clock import SYSTEM_CLOCK

def _utcnow() -> datetime:
    return SYSTEM_CLOCK.utcnow()
```

## Test Helpers

The `tests/helpers/time.py` module re-exports `FakeClock` and provides a pytest
fixture:

```python
# In tests
from tests.helpers.time import fake_clock  # pytest fixture

def test_something(fake_clock: FakeClock) -> None:
    fake_clock.set_wall(datetime(2024, 1, 1, tzinfo=UTC))
    # ...
```

`ControllableClock` is preserved for backward compatibility with tests that use
the callable interface (`clock()` instead of `clock.monotonic()`).

## Public Exports

From the main package:

```python
from weakincentives import (
    SYSTEM_CLOCK,
    AsyncSleeper,
    Clock,
    FakeClock,
    MonotonicClock,
    Sleeper,
    SystemClock,
    WallClock,
)
```

From the clock module:

```python
from weakincentives.clock import FakeClock, SYSTEM_CLOCK
```

## Non-Goals

This specification does not address:

- **Timezones**: WINK uses UTC exclusively
- **Calendar operations**: No date arithmetic needed
- **High-resolution timing**: Millisecond precision is sufficient
- **Distributed clock sync**: Single-process focus

## Summary

All core time-dependent code in WINK accepts injectable clocks with sensible
defaults. Production code uses `SYSTEM_CLOCK` unchanged, while tests inject
`FakeClock` for instant, deterministic time advancement without real delays or
monkeypatching.
