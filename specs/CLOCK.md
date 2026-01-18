# CLOCK.md

Specification for controllable time dependencies in the WINK runtime.

## Motivation

Time-dependent code is notoriously difficult to test. Direct calls to
`time.monotonic()`, `time.sleep()`, and `datetime.now(UTC)` create hidden
dependencies that force tests to either use real delays (slow, flaky) or
resort to fragile monkeypatching (brittle, error-prone).

WINK currently uses a mix of approaches:

| Pattern | Example | Testability |
|---------|---------|-------------|
| Module-level function | `deadlines._utcnow()` | Monkeypatchable |
| Constructor parameter | `InMemoryMailbox(clock=...)` | Excellent |
| Direct global call | `time.monotonic()` in Heartbeat | Poor |
| Direct sleep | `time.sleep()` in throttle | Poor |

This spec standardizes time handling across the codebase using explicit
dependency injection.

## Current State Analysis

### Wall-Clock Time (UTC datetime)

Used for timestamps, deadlines, and event recording:

| Location | Usage | Current Mockability |
|----------|-------|---------------------|
| `deadlines._utcnow()` | Deadline validation | ✅ Monkeypatch |
| `MainLoopResult.completed_at` | Completion timestamp | ❌ Global default |
| `MainLoopRequest.created_at` | Request timestamp | ❌ Global default |
| `DeadLetter.dead_lettered_at` | DLQ timestamp | ❌ Direct call |
| `InMemoryMailbox._enqueued_at` | Message timestamp | ❌ Direct call |

### Monotonic Clock (float seconds)

Used for elapsed time, timeouts, and rate limiting:

| Location | Usage | Current Mockability |
|----------|-------|---------------------|
| `InMemoryMailbox.clock` | Visibility timeout | ✅ Injectable |
| `Heartbeat._last_beat` | Heartbeat tracking | ❌ Global call |
| `Heartbeat.elapsed()` | Staleness check | ❌ Global call |
| `LeaseExtender._on_beat()` | Rate limiting | ❌ Global call |
| `lifecycle.wait_until()` | Polling timeout | ❌ Global call |
| `HookContext.elapsed_ms` | Hook timing | ❌ Global call |
| `EvalLoop._evaluate_sample()` | Latency measurement | ❌ Global call |

### Sleep/Delay

Used for throttling and polling:

| Location | Usage | Current Mockability |
|----------|-------|---------------------|
| `throttle.sleep_for()` | Backoff delay | ❌ Real sleep |
| `lifecycle.wait_until()` | Poll interval | ❌ Real sleep |
| `Watchdog._run()` | Check interval | ❌ Event.wait() |
| `InMemoryMailbox._reaper_loop()` | Reap interval | ❌ Event.wait() |

### Existing Test Helpers

Two helpers exist in `tests/helpers/time.py`:

**FrozenUtcNow** - Controls wall-clock time for deadline tests:

```python
class FrozenUtcNow:
    """Controller for deadlines._utcnow() during tests."""

    def now(self) -> datetime: ...
    def set(self, current: datetime) -> datetime: ...
    def advance(self, delta: timedelta) -> datetime: ...
```

**ControllableClock** - Controls monotonic time:

```python
class ControllableClock:
    """Controllable clock for testing without sleeping."""

    def __call__(self) -> float: ...
    def advance(self, seconds: float) -> float: ...
    def set(self, value: float) -> float: ...
```

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
class TestClock:
    """Controllable clock for deterministic testing.

    Both monotonic and wall-clock time advance together when advance()
    is called. Sleep operations advance time immediately without blocking.

    Example::

        clock = TestClock()

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

## Refactoring Plan

### Phase 1: Core Clock Module

Create the clock protocols and implementations:

1. Create `src/weakincentives/clock.py` with protocols and `SystemClock`
2. Enhance `tests/helpers/time.py` with unified `TestClock`
3. Export from `src/weakincentives/__init__.py`

### Phase 2: Deadline Module

Refactor `deadlines.py` to accept a clock:

**Before:**
```python
def _utcnow() -> datetime:
    return datetime.now(UTC)

@FrozenDataclass()
class Deadline:
    expires_at: datetime

    def __post_init__(self) -> None:
        now = _utcnow()
        if self.expires_at <= now:
            raise ValueError("Deadline expires_at must be in the future.")
```

**After:**
```python
from weakincentives.clock import WallClock, SYSTEM_CLOCK

@FrozenDataclass()
class Deadline:
    expires_at: datetime
    _clock: WallClock = field(default=SYSTEM_CLOCK, repr=False, compare=False)

    def __post_init__(self) -> None:
        now = self._clock.utcnow()
        if self.expires_at <= now:
            raise ValueError("Deadline expires_at must be in the future.")

    def remaining(self, *, now: datetime | None = None) -> timedelta:
        current = now if now is not None else self._clock.utcnow()
        return self.expires_at - current
```

**Test usage:**
```python
def test_deadline_remaining(test_clock: TestClock) -> None:
    test_clock.set_wall(datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC))

    deadline = Deadline(
        expires_at=datetime(2024, 6, 1, 13, 0, 0, tzinfo=UTC),
        _clock=test_clock,
    )

    assert deadline.remaining() == timedelta(hours=1)

    test_clock.advance(1800)  # 30 minutes
    assert deadline.remaining() == timedelta(minutes=30)
```

### Phase 3: Heartbeat and Watchdog

Inject clock into `Heartbeat` for testable elapsed time:

**Before:**
```python
@dataclass(slots=True)
class Heartbeat:
    _last_beat: float = field(default_factory=time.monotonic)

    def beat(self) -> None:
        with self._lock:
            self._last_beat = time.monotonic()

    def elapsed(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_beat
```

**After:**
```python
from weakincentives.clock import MonotonicClock, SYSTEM_CLOCK

@dataclass(slots=True)
class Heartbeat:
    clock: MonotonicClock = field(default=SYSTEM_CLOCK, repr=False)
    _last_beat: float = field(init=False)

    def __post_init__(self) -> None:
        self._last_beat = self.clock.monotonic()

    def beat(self) -> None:
        with self._lock:
            self._last_beat = self.clock.monotonic()

    def elapsed(self) -> float:
        with self._lock:
            return self.clock.monotonic() - self._last_beat
```

### Phase 4: LeaseExtender

Inject clock for rate-limit testing:

**Before:**
```python
def _on_beat(self) -> None:
    with self._lock:
        now = time.monotonic()
        elapsed = now - self._last_extension
        if elapsed < self.config.interval:
            return
```

**After:**
```python
from weakincentives.clock import MonotonicClock, SYSTEM_CLOCK

@dataclass(slots=True)
class LeaseExtender:
    config: LeaseExtenderConfig = field(default_factory=LeaseExtenderConfig)
    clock: MonotonicClock = field(default=SYSTEM_CLOCK, repr=False)

    def _on_beat(self) -> None:
        with self._lock:
            now = self.clock.monotonic()
            elapsed = now - self._last_extension
            if elapsed < self.config.interval:
                return
```

### Phase 5: Lifecycle Utilities

Refactor `wait_until()` to accept clock and sleeper:

**Before:**
```python
def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float,
    poll_interval: float = 0.1,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return predicate()
```

**After:**
```python
from weakincentives.clock import Clock, SYSTEM_CLOCK

def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float,
    poll_interval: float = 0.1,
    clock: Clock = SYSTEM_CLOCK,
) -> bool:
    deadline = clock.monotonic() + timeout
    while clock.monotonic() < deadline:
        if predicate():
            return True
        clock.sleep(poll_interval)
    return predicate()
```

**Test usage:**
```python
def test_wait_until_timeout(test_clock: TestClock) -> None:
    """Test wait_until returns False on timeout."""
    calls = 0

    def never_true() -> bool:
        nonlocal calls
        calls += 1
        test_clock.advance(0.5)  # Each check advances time
        return False

    result = wait_until(
        never_true,
        timeout=2.0,
        poll_interval=0.5,
        clock=test_clock,
    )

    assert result is False
    assert calls == 5  # 0.5s intervals over 2s timeout
```

### Phase 6: Throttle Module

Inject sleeper for backoff testing:

**Before:**
```python
def sleep_for(delay: timedelta) -> None:
    time.sleep(delay.total_seconds())
```

**After:**
```python
from weakincentives.clock import Sleeper, SYSTEM_CLOCK

def sleep_for(delay: timedelta, *, sleeper: Sleeper = SYSTEM_CLOCK) -> None:
    sleeper.sleep(delay.total_seconds())
```

### Phase 7: Timestamp Fields

For dataclass fields that capture timestamps, use a factory pattern:

**Before:**
```python
@dataclass(frozen=True, slots=True)
class MainLoopResult[OutputT]:
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

**After:**
```python
from weakincentives.clock import WallClock, SYSTEM_CLOCK

def _default_timestamp(clock: WallClock = SYSTEM_CLOCK) -> datetime:
    return clock.utcnow()

@dataclass(frozen=True, slots=True)
class MainLoopResult[OutputT]:
    completed_at: datetime = field(default_factory=_default_timestamp)
```

For testing, use `replace()` or explicit construction:

```python
def test_result_timestamp(test_clock: TestClock) -> None:
    test_clock.set_wall(datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC))

    result = MainLoopResult(
        request_id=uuid4(),
        output="test",
        completed_at=test_clock.utcnow(),
    )

    assert result.completed_at.year == 2024
```

### Phase 8: InMemoryMailbox Enhancement

The mailbox already supports clock injection but uses different naming:

**Current:**
```python
clock: Callable[[], float] = field(default=time.monotonic, repr=False)
```

**Recommended:**
```python
clock: MonotonicClock = field(default=SYSTEM_CLOCK, repr=False)

# Update usage from clock() to clock.monotonic()
expires_at = self.clock.monotonic() + visibility_timeout
```

### Phase 9: Background Thread Timing

For threads that use `Event.wait(timeout=...)`:

**Watchdog pattern:**
```python
class Watchdog:
    def __init__(
        self,
        heartbeats: Sequence[Heartbeat],
        *,
        stall_threshold: float = 720.0,
        check_interval: float = 60.0,
        clock: MonotonicClock = SYSTEM_CLOCK,
    ) -> None:
        self._clock = clock
        # ...

    def _run(self) -> None:
        # Event.wait() still uses real time for blocking
        # But heartbeat.elapsed() uses the injected clock
        while not self._stop_event.wait(timeout=self._check_interval):
            stalled = self._check_heartbeats()
            # ...
```

Note: Background thread loop timing (`Event.wait()`) cannot be easily mocked
without restructuring. This is acceptable because:

1. Watchdog tests focus on heartbeat staleness detection
2. The clock injection in `Heartbeat.elapsed()` enables that testing
3. Integration tests verify actual timeout behavior

## Test Fixtures

Add pytest fixtures for common test scenarios:

```python
# tests/conftest.py

import pytest
from tests.helpers.time import TestClock


@pytest.fixture
def test_clock() -> TestClock:
    """Provide a fresh TestClock for each test."""
    return TestClock()


@pytest.fixture
def frozen_time(test_clock: TestClock) -> TestClock:
    """Provide a TestClock frozen at a known point."""
    from datetime import datetime, UTC
    test_clock.set_wall(datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC))
    test_clock.set_monotonic(0.0)
    return test_clock
```

## Migration Strategy

### Backward Compatibility

During migration, maintain backward compatibility:

1. Clock parameters default to `SYSTEM_CLOCK`
2. Existing tests continue to work
3. New tests can inject `TestClock`

### Deprecation of Monkeypatch Approach

The existing `FrozenUtcNow` helper uses monkeypatching:

```python
monkeypatch.setattr(deadlines, "_utcnow", self.now)
```

After migration, this pattern should be replaced with clock injection.
Keep `FrozenUtcNow` for backward compatibility but mark as deprecated.

### Testing Guidelines

1. **Unit tests**: Always inject `TestClock`
2. **Integration tests**: May use `SystemClock` for real timing
3. **Property tests**: Use `TestClock` with hypothesis strategies

## Public API

Export from main package:

```python
# src/weakincentives/__init__.py

from .clock import (
    Clock,
    MonotonicClock,
    WallClock,
    Sleeper,
    SystemClock,
    SYSTEM_CLOCK,
)
```

## Non-Goals

This spec does not address:

1. **Timezones** - WINK uses UTC exclusively
2. **Calendar operations** - No date arithmetic needed
3. **High-resolution timing** - Millisecond precision sufficient
4. **Distributed clock sync** - Single-process focus

## Summary

| Component | Clock Type | Injection Point |
|-----------|-----------|-----------------|
| Deadline | WallClock | Constructor field |
| Heartbeat | MonotonicClock | Constructor field |
| LeaseExtender | MonotonicClock | Constructor field |
| wait_until | Clock | Function parameter |
| sleep_for | Sleeper | Function parameter |
| InMemoryMailbox | MonotonicClock | Constructor field |
| Watchdog | MonotonicClock | Constructor (for Heartbeat) |
| MainLoopResult | WallClock | Explicit construction |
| DeadLetter | WallClock | Explicit construction |

After refactoring, all time-dependent code will be deterministically testable
without real delays or fragile monkeypatching.
