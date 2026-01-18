# CLOCK.md

Specification for controllable time dependencies in the WINK runtime.

## Overview

All time-dependent code in WINK uses explicit dependency injection via the
`Clock` protocol and its implementations. This enables deterministic testing
without real delays or fragile monkeypatching.

**Key types:**

- `Clock` - Unified protocol combining monotonic time, wall-clock time, and sleep
- `MonotonicClock` - Protocol for elapsed time measurement (timeouts, rate limiting)
- `WallClock` - Protocol for UTC timestamps (deadlines, event recording)
- `Sleeper` - Protocol for delay operations
- `SystemClock` - Production implementation using real system calls
- `FakeClock` - Test implementation that advances time instantly
- `SYSTEM_CLOCK` - Module-level singleton for production use

## Motivation

Time-dependent code is notoriously difficult to test. Direct calls to
`time.monotonic()`, `time.sleep()`, and `datetime.now(UTC)` create hidden
dependencies that force tests to either use real delays (slow, flaky) or
resort to fragile monkeypatching (brittle, error-prone).

WINK standardizes time handling using explicit dependency injection.

## Implementation Status

All core time-dependent components now support clock injection:

| Component | Clock Type | Injection Point | Status |
|-----------|-----------|-----------------|--------|
| `Deadline` | WallClock | `clock` field | ✅ Implemented |
| `Heartbeat` | MonotonicClock | `clock` field | ✅ Implemented |
| `LeaseExtender` | MonotonicClock | `clock` field | ✅ Implemented |
| `wait_until()` | Clock | `clock` parameter | ✅ Implemented |
| `sleep_for()` | Sleeper | `sleeper` parameter | ✅ Implemented |
| `InMemoryMailbox` | MonotonicClock | `clock` field | ✅ Implemented |

Components that still use system time directly (acceptable for now):

| Component | Usage | Notes |
|-----------|-------|-------|
| `MainLoopResult.completed_at` | Completion timestamp | Field default |
| `MainLoopRequest.created_at` | Request timestamp | Field default |
| `DeadLetter.dead_lettered_at` | DLQ timestamp | Explicit construction |
| `Watchdog._run()` | Check interval | Uses Event.wait() |
| `InMemoryMailbox._reaper_loop()` | Reap interval | Uses Event.wait() |
| `HookContext.elapsed_ms` | Hook timing | Non-critical |
| `EvalLoop._evaluate_sample()` | Latency measurement | Non-critical |

## Design Goals

1. **All time dependencies injectable** - No hidden calls to system clocks
2. **Default to real clocks** - Production code unchanged
3. **Test doubles advance without sleeping** - Fast, deterministic tests
4. **Two clock types** - Monotonic for intervals, wall for timestamps
5. **Minimal API surface** - Simple protocols, easy to implement
6. **Thread-safe** - All clock operations must be safe for concurrent access

## Proposed Architecture

### Clock Protocols

Define two separate protocols for the two distinct time domains:

```python
# src/weakincentives/clock.py

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable


@runtime_checkable
class MonotonicClock(Protocol):
    """Protocol for monotonic time measurement.

    Monotonic clocks measure elapsed time and are guaranteed to never
    go backwards. They are suitable for timeouts, rate limiting, and
    measuring durations.

    The zero point is arbitrary and not related to wall-clock time.
    """

    def monotonic(self) -> float:
        """Return monotonic time in seconds."""
        ...


@runtime_checkable
class WallClock(Protocol):
    """Protocol for wall-clock time measurement.

    Wall clocks provide the current UTC datetime. They are suitable for
    timestamps, deadlines, and recording when events occurred.

    Unlike monotonic clocks, wall clocks can jump (NTP adjustments,
    daylight saving, etc.) and should not be used for measuring durations.
    """

    def utcnow(self) -> datetime:
        """Return current UTC datetime (timezone-aware)."""
        ...


@runtime_checkable
class Sleeper(Protocol):
    """Protocol for sleep/delay operations.

    Separating sleep from clock allows tests to advance time without
    actually sleeping, while production code uses real delays.
    """

    def sleep(self, seconds: float) -> None:
        """Sleep for the specified duration in seconds."""
        ...


class Clock(MonotonicClock, WallClock, Sleeper, Protocol):
    """Unified clock combining monotonic, wall-clock, and sleep.

    Components that need multiple time capabilities should depend on
    this combined protocol. Components that only need monotonic time
    or wall-clock time should depend on the narrower protocol.
    """

    pass
```

### System Clock Implementation

The default implementation uses real system calls:

```python
# src/weakincentives/clock.py (continued)

import time as _time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemClock:
    """Production clock using system time functions.

    This is the default clock used throughout WINK. It delegates to:
    - time.monotonic() for monotonic time
    - datetime.now(UTC) for wall-clock time
    - time.sleep() for delays

    Example::

        clock = SystemClock()
        start = clock.monotonic()
        clock.sleep(1.0)
        elapsed = clock.monotonic() - start  # ~1.0
    """

    def monotonic(self) -> float:
        """Return monotonic time from time.monotonic()."""
        return _time.monotonic()

    def utcnow(self) -> datetime:
        """Return current UTC datetime."""
        return datetime.now(UTC)

    def sleep(self, seconds: float) -> None:
        """Sleep using time.sleep()."""
        _time.sleep(seconds)


# Module-level singleton for convenience
SYSTEM_CLOCK: Clock = SystemClock()
```

### Test Clock Implementation

A controllable clock for tests:

```python
# tests/helpers/time.py (enhanced)

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import threading


@dataclass
class FakeClock:
    """Controllable clock for deterministic testing.

    Both monotonic and wall-clock time advance together when advance()
    is called. Sleep operations advance time immediately without blocking.

    Example::

        clock = FakeClock()

        start = clock.monotonic()
        clock.sleep(10)  # Advances immediately, no real delay
        assert clock.monotonic() - start == 10

        # Or advance manually
        clock.advance(60)
        assert clock.monotonic() - start == 70

    Thread-safety:
        All operations are thread-safe. Multiple threads can read and
        advance the clock concurrently.
    """

    _monotonic: float = 0.0
    _wall: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=UTC))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def monotonic(self) -> float:
        """Return current monotonic time."""
        with self._lock:
            return self._monotonic

    def utcnow(self) -> datetime:
        """Return current wall-clock time."""
        with self._lock:
            return self._wall

    def sleep(self, seconds: float) -> None:
        """Advance time immediately without blocking."""
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        """Advance both clocks by the given duration.

        Args:
            seconds: Duration to advance in seconds.
        """
        with self._lock:
            self._monotonic += seconds
            self._wall += timedelta(seconds=seconds)

    def set_monotonic(self, value: float) -> None:
        """Set monotonic time to an absolute value."""
        with self._lock:
            self._monotonic = value

    def set_wall(self, value: datetime) -> None:
        """Set wall-clock time to an absolute value."""
        if value.tzinfo is None:
            msg = "Wall clock time must be timezone-aware"
            raise ValueError(msg)
        with self._lock:
            self._wall = value
```

## Usage Examples

### Deadline with Clock Injection

```python
from datetime import UTC, datetime, timedelta
from weakincentives import Deadline
from weakincentives.clock import FakeClock

# Production: uses system clock by default
deadline = Deadline(datetime.now(UTC) + timedelta(hours=1))

# Testing: inject FakeClock for deterministic behavior
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

### Heartbeat with Clock Injection

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

### wait_until with Clock Injection

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

### sleep_for with Sleeper Injection

```python
from datetime import timedelta
from weakincentives.adapters.throttle import sleep_for
from weakincentives.clock import FakeClock

clock = FakeClock()
initial = clock.monotonic()

sleep_for(timedelta(seconds=5), sleeper=clock)

# FakeClock advances time instantly
assert clock.monotonic() == initial + 5
```

## Testing Guidelines

1. **Unit tests**: Always inject `FakeClock` for deterministic behavior
2. **Integration tests**: May use `SystemClock` for real timing verification
3. **Property tests**: Use `FakeClock` with hypothesis strategies

## Public API

All clock types are exported from the main package:

```python
from weakincentives import (
    Clock,
    FakeClock,
    MonotonicClock,
    SYSTEM_CLOCK,
    Sleeper,
    SystemClock,
    WallClock,
)
```

Or import directly from the clock module:

```python
from weakincentives.clock import FakeClock, SYSTEM_CLOCK
```

## Non-Goals

This spec does not address:

1. **Timezones** - WINK uses UTC exclusively
2. **Calendar operations** - No date arithmetic needed
3. **High-resolution timing** - Millisecond precision sufficient
4. **Distributed clock sync** - Single-process focus

## Summary

All core time-dependent code is now deterministically testable without real
delays or fragile monkeypatching. Components accept a clock parameter that
defaults to `SYSTEM_CLOCK` for production use, and tests can inject `FakeClock`
for instant, controllable time advancement.
