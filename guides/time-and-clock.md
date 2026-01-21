# Time and Clock Patterns

*Canonical spec: [specs/CLOCK.md](../specs/CLOCK.md)*

WINK provides explicit time abstractions that enable deterministic testing
without real delays or fragile monkeypatching. All time-dependent code accepts
injectable clocks with sensible defaults.

## Time Domains

WINK distinguishes two distinct time domains:

**Monotonic time** (float seconds): For measuring elapsed time, timeouts, and
rate limiting. Guaranteed to never go backwards. The zero point is arbitrary and
unrelated to wall-clock time. Use for durations and intervals.

**Wall-clock time** (UTC datetime): For timestamps, deadlines, and recording
when events occurred. Can jump due to NTP adjustments. Always timezone-aware
UTC. Use for absolute points in time.

## Clock Protocols

Components depend on the narrowest protocol that meets their needs:

```python nocheck
from weakincentives import (
    Clock,
    MonotonicClock,
    WallClock,
    Sleeper,
    SYSTEM_CLOCK,
)

# Narrow protocols for specific needs
class MonotonicClock(Protocol):
    def monotonic(self) -> float: ...

class WallClock(Protocol):
    def utcnow(self) -> datetime: ...

class Sleeper(Protocol):
    def sleep(self, seconds: float) -> None: ...

# Combined protocol for components needing multiple capabilities
class Clock(MonotonicClock, WallClock, Sleeper, Protocol):
    pass
```

| Component | Protocol | Reason |
| --- | --- | --- |
| `Deadline` | `WallClock` | Compares against wall-clock time |
| `Heartbeat` | `MonotonicClock` | Measures elapsed intervals |
| `LeaseExtender` | `MonotonicClock` | Tracks extension intervals |
| `wait_until()` | `Clock` | Needs sleep + monotonic |
| `sleep_for()` | `Sleeper` | Only needs sleep |
| `InMemoryMailbox` | `MonotonicClock` | Message visibility timing |

## SystemClock and SYSTEM_CLOCK

Production code uses the `SYSTEM_CLOCK` singleton, which delegates to Python's
standard library:

```python nocheck
from weakincentives import SYSTEM_CLOCK

# Measure elapsed time
start = SYSTEM_CLOCK.monotonic()
SYSTEM_CLOCK.sleep(1.0)
elapsed = SYSTEM_CLOCK.monotonic() - start  # ~1.0

# Get current UTC time
now = SYSTEM_CLOCK.utcnow()  # datetime.now(UTC)
```

All time-dependent components default to `SYSTEM_CLOCK`, so you typically don't
need to pass it explicitly in production code.

## FakeClock for Testing

`FakeClock` advances time instantly without blocking, enabling fast and
deterministic tests:

```python nocheck
from weakincentives import FakeClock

clock = FakeClock()

start = clock.monotonic()
clock.sleep(10)  # Advances immediately, no real delay
assert clock.monotonic() - start == 10

# Manual advancement
clock.advance(60)
assert clock.monotonic() - start == 70
```

**FakeClock API:**

| Method | Description |
| --- | --- |
| `monotonic()` | Return current monotonic time |
| `utcnow()` | Return current wall-clock time |
| `sleep(seconds)` | Advance time instantly (calls `advance()`) |
| `advance(seconds)` | Advance both clocks by duration (non-negative only) |
| `set_monotonic(value)` | Set monotonic time to absolute value |
| `set_wall(value)` | Set wall-clock time (must be timezone-aware) |

Both `advance()` and `sleep()` advance monotonic and wall-clock time together.
All operations are thread-safe.

## Deadline Pattern

`Deadline` represents an immutable wall-clock expiration. Use it to track when
an operation must complete:

```python nocheck
from datetime import UTC, datetime, timedelta
from weakincentives import Deadline

# Create a deadline 1 hour from now
deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1))

# Check remaining time
remaining = deadline.remaining()  # timedelta

# Check if expired (negative remaining means expired)
if remaining <= timedelta(0):
    raise TimeoutError("Deadline expired")
```

**Deadline validation:**

- `expires_at` must be timezone-aware (UTC)
- `expires_at` must be in the future
- `expires_at` must be at least 1 second in the future

These validations happen at construction time, failing fast on invalid inputs.

### Testing Deadlines

Inject `FakeClock` to control time in tests:

```python nocheck
from datetime import UTC, datetime, timedelta
from weakincentives import Deadline, FakeClock

clock = FakeClock()
clock.set_wall(datetime(2024, 6, 1, 12, 0, tzinfo=UTC))

deadline = Deadline(
    expires_at=datetime(2024, 6, 1, 13, 0, tzinfo=UTC),
    clock=clock,
)

assert deadline.remaining() == timedelta(hours=1)

clock.advance(1800)  # 30 minutes
assert deadline.remaining() == timedelta(minutes=30)

clock.advance(1800)  # Another 30 minutes
assert deadline.remaining() == timedelta(0)  # Exactly at deadline

clock.advance(1)
assert deadline.remaining() < timedelta(0)  # Expired
```

## Budget Pattern

`Budget` combines time and token limits into a single resource envelope. Use it
to constrain agent execution:

```python nocheck
from datetime import UTC, datetime, timedelta
from weakincentives import Budget, Deadline

# Time-only budget
budget = Budget(
    deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=30))
)

# Token-only budget
budget = Budget(max_total_tokens=100_000)

# Combined budget
budget = Budget(
    deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1)),
    max_total_tokens=500_000,
    max_input_tokens=400_000,
    max_output_tokens=100_000,
)
```

**Budget validation:**

- At least one limit must be set
- Token limits must be positive when provided

### BudgetTracker

`BudgetTracker` tracks cumulative token usage against a budget and raises
`BudgetExceededError` when limits are breached:

```python nocheck
from weakincentives import Budget, BudgetTracker, BudgetExceededError
from weakincentives.runtime.events import TokenUsage

budget = Budget(max_total_tokens=10_000)
tracker = BudgetTracker(budget=budget)

# Record usage per evaluation (replaces previous for same ID)
tracker.record_cumulative("eval-1", TokenUsage(input_tokens=3000, output_tokens=1000))
tracker.record_cumulative("eval-2", TokenUsage(input_tokens=2000, output_tokens=500))

# Check total consumption
consumed = tracker.consumed
print(f"Total: {consumed.total_tokens}")  # 6500

# Check if budget exceeded (raises if exceeded)
try:
    tracker.check()
except BudgetExceededError as e:
    print(f"Exceeded: {e.exceeded_dimension}")
```

**BudgetExceededError** includes:

- `budget` - The budget that was exceeded
- `consumed` - Token usage at time of check
- `exceeded_dimension` - Which limit was breached: `"deadline"`, `"total_tokens"`,
  `"input_tokens"`, or `"output_tokens"`

`BudgetTracker` is thread-safe for concurrent execution.

## Common Testing Patterns

### Testing Timeout Logic

```python nocheck
from weakincentives import FakeClock
from weakincentives.runtime.lifecycle import wait_until

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

### Testing Heartbeats

```python nocheck
from weakincentives import FakeClock
from weakincentives.runtime.watchdog import Heartbeat

clock = FakeClock()
hb = Heartbeat(clock=clock)

hb.beat()
assert hb.elapsed() == 0.0

clock.advance(10)
assert hb.elapsed() == 10.0
```

### Testing Throttle Delays

```python nocheck
from datetime import timedelta
from weakincentives import FakeClock
from weakincentives.adapters.throttle import sleep_for

clock = FakeClock()
initial = clock.monotonic()

sleep_for(timedelta(seconds=5), sleeper=clock)

assert clock.monotonic() == initial + 5  # No real delay
```

### Pytest Fixture

The test helpers provide a `fake_clock` fixture:

```python nocheck
from tests.helpers.time import fake_clock
from weakincentives import FakeClock

def test_something(fake_clock: FakeClock) -> None:
    fake_clock.set_wall(datetime(2024, 1, 1, tzinfo=UTC))
    # ...
```

## Design Principles

**Narrow protocols**: Components depend on the narrowest protocol meeting their
needs (`WallClock` vs `MonotonicClock` vs `Clock`).

**Dependency injection**: All time-dependent components accept an injectable
clock parameter with a sensible default (`SYSTEM_CLOCK`).

**Deterministic testing**: `FakeClock` enables instant time advancement without
real delays.

**Thread safety**: `FakeClock`, `Heartbeat`, and `BudgetTracker` are thread-safe.

**Immutability**: `Deadline` and `Budget` are frozen dataclasses.

**UTC exclusive**: All wall-clock times are timezone-aware UTC. No local time
handling.

**Fail fast**: Invalid inputs (naive datetime, negative durations, expired
deadlines) raise immediately at construction.

## Next Steps

- [Lifecycle](lifecycle.md): Health checks, shutdown, and watchdog monitoring
- [Orchestration](orchestration.md): MainLoop and request handling
- [Testing](testing.md): General testing patterns
