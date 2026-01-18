# Clock Specification

## Overview

All production code in `weakincentives` uses a controllable, injectable time source
for deterministic testing and reproducible agent execution. Direct use of standard
library time functions is **prohibited** in production code and enforced via CI.

**Key Benefits:**

- **Deterministic tests**: No flaky timeouts or race conditions
- **Fast tests**: 3× speedup achieved by eliminating real sleep calls
- **Reproducible sessions**: Time is controlled like any other input

## Hard Restriction

**RULE**: Production code in `src/weakincentives/` must NEVER directly call:

- `time.time()`, `time.monotonic()`, `time.perf_counter()`
- `time.sleep()`, `asyncio.sleep()`
- `datetime.now()`, `datetime.utcnow()`

**Enforcement**: The `scripts/validate_clock_usage.py` script scans all production
code using AST analysis. Violations cause CI failures.

**Allowed Exceptions**:

- `datetime.fromtimestamp()` - Converting external timestamps
- `datetime.fromisoformat()` - Parsing timestamp strings

**Validation**:

```bash
make validate-clock  # Included in 'make check'
```

## Clock Protocol

```python
from typing import Protocol
from datetime import datetime

class Clock(Protocol):
    """Controllable time source for deterministic testing."""

    def now(self) -> datetime:
        """Current UTC datetime.

        Production: Wall clock time
        Tests: Simulated time controlled by test harness
        """

    def monotonic(self) -> float:
        """Monotonic timestamp in seconds.

        For measuring intervals and timeouts. Guaranteed never to go
        backwards, even if system clock is adjusted.

        Production: time.monotonic()
        Tests: Simulated monotonic counter
        """

    def sleep(self, seconds: float) -> None:
        """Block for specified duration.

        Production: time.sleep(seconds)
        Tests: Advance simulated time without real delay
        """

    async def asleep(self, seconds: float) -> None:
        """Async sleep for specified duration.

        Production: await asyncio.sleep(seconds)
        Tests: Advance simulated time without real delay
        """
```

## Implementation

### SystemClock (Production)

```python
# src/weakincentives/runtime/clock.py
import time
import asyncio
from datetime import UTC, datetime

class SystemClock:
    """Real clock for production use."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    async def asleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
```

### FakeClock (Testing)

```python
from dataclasses import dataclass, field
from datetime import timedelta

@dataclass
class FakeClock:
    """Controllable clock for deterministic testing.

    Example:
        clock = FakeClock()
        clock.advance(10.0)  # Jump forward 10 seconds
        assert clock.monotonic() == 10.0
    """

    _utc_epoch: datetime = field(
        default_factory=lambda: datetime(2024, 1, 1, tzinfo=UTC)
    )
    _monotonic: float = 0.0

    def now(self) -> datetime:
        return self._utc_epoch + timedelta(seconds=self._monotonic)

    def monotonic(self) -> float:
        return self._monotonic

    def sleep(self, seconds: float) -> None:
        self._monotonic += seconds

    async def asleep(self, seconds: float) -> None:
        self._monotonic += seconds

    def advance(self, seconds: float) -> None:
        """Manually advance simulated time (test helper)."""
        self._monotonic += seconds

    def set_now(self, dt: datetime) -> None:
        """Set current simulated datetime (test helper)."""
        self._utc_epoch = dt
        self._monotonic = 0.0
```

## Dependency Injection Patterns

### Constructor Injection (Preferred)

For classes that need direct time access:

```python
from dataclasses import dataclass, field
from weakincentives.runtime.clock import Clock, SystemClock

@dataclass(slots=True)
class Heartbeat:
    """Thread-safe heartbeat tracker."""

    clock: Clock = field(default_factory=SystemClock)
    _last_beat: float = field(init=False)

    def __post_init__(self) -> None:
        self._last_beat = self.clock.monotonic()

    def beat(self) -> None:
        self._last_beat = self.clock.monotonic()

    def elapsed(self) -> float:
        return self.clock.monotonic() - self._last_beat
```

**Usage in production**:

```python
# Default constructor uses real time
heartbeat = Heartbeat()
```

**Usage in tests**:

```python
# Inject FakeClock for deterministic testing
clock = FakeClock()
heartbeat = Heartbeat(clock=clock)
clock.advance(5.0)
assert heartbeat.elapsed() == 5.0
```

### Function Parameter Injection

For functions that need time control:

```python
def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float,
    poll_interval: float = 0.1,
    clock: Clock | None = None,
) -> bool:
    """Wait until predicate is True or timeout expires."""
    if clock is None:  # pragma: no branch
        clock = SystemClock()

    deadline = clock.monotonic() + timeout
    while clock.monotonic() < deadline:
        if predicate():
            return True
        clock.sleep(poll_interval)
    return predicate()
```

**Usage**:

```python
# Production: uses real time (clock defaults to SystemClock)
result = wait_until(lambda: server.ready, timeout=10.0)

# Test: instant with FakeClock
clock = FakeClock()
result = wait_until(lambda: True, timeout=10.0, clock=clock)
# Returns immediately, no real delay
```

### Resource Registry Integration

For dependency injection frameworks:

```python
from weakincentives.resources import ResourceRegistry, Binding

registry = ResourceRegistry.of(
    Binding(Clock, lambda r: SystemClock()),
)

with registry.open() as ctx:
    clock = ctx.get(Clock)
    now = clock.now()
```

## Testing Patterns

### Basic Time Control

```python
from weakincentives.runtime import FakeClock
from weakincentives.runtime.watchdog import Heartbeat

def test_heartbeat_tracking():
    clock = FakeClock()
    hb = Heartbeat(clock=clock)

    # Time starts at 0
    assert hb.elapsed() == 0.0

    # Advance 5 seconds instantly
    clock.advance(5.0)
    assert hb.elapsed() == 5.0

    # Beat resets
    hb.beat()
    assert hb.elapsed() == 0.0
```

### Timeout Testing

```python
def test_visibility_timeout():
    clock = FakeClock()
    mailbox = InMemoryMailbox(name="test", clock=clock)

    # Send message with 1-second visibility timeout
    mailbox.send("hello")
    messages = mailbox.receive(visibility_timeout=1)
    assert len(messages) == 1

    # Advance past timeout
    clock.advance(1.2)
    mailbox._reap_expired()  # Trigger timeout check

    # Message should be requeued
    messages = mailbox.receive(wait_time_seconds=0)
    assert len(messages) == 1
    assert messages[0].delivery_count == 2
```

### Long Polling Optimization

Eliminate wasted polling time in tests:

```python
def test_eval_loop_processing():
    # Without wait_time_seconds=0, this would wait 20s when queue is empty
    eval_loop.run(max_iterations=5, wait_time_seconds=0)
    # Test completes instantly instead of waiting 20+ seconds
```

**Impact**: This pattern reduced test suite time from 184s to 60s (67% faster).

### Deadline Assertions

```python
def test_deadline_expiry():
    clock = FakeClock()
    deadline = clock.monotonic() + 30.0

    # Before deadline
    assert clock.monotonic() < deadline

    # Jump to deadline
    clock.advance(30.0)
    assert clock.monotonic() >= deadline
```

### Timestamp Control

```python
def test_event_timestamps():
    clock = FakeClock()
    clock.set_now(datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC))

    session = Session(clock=clock)
    assert session.created_at == datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

    # Advance 1 hour
    clock.advance(3600.0)
    event = create_event(clock=clock)
    assert event.created_at == datetime(2024, 6, 15, 13, 0, 0, tzinfo=UTC)
```

### Async Sleep Testing

```python
async def test_async_polling():
    clock = FakeClock()

    # asleep() advances time without real await
    await clock.asleep(5.0)
    assert clock.monotonic() == 5.0

    # Test completes instantly
```

## Clock-Aware Components

These components support Clock injection (complete list):

### Runtime

- `InMemoryMailbox(clock=...)`
- `Heartbeat(clock=...)`
- `LeaseExtender(clock=...)`
- `wait_until(clock=...)`

### Evals

- `EvalLoop(clock=...)`
- `collect_results(clock=...)`

### Adapters

- `sleep_for(clock=...)` (throttle.py)
- `HookContext(clock=...)` (Claude Agent SDK)
- `LogAggregator(clock=...)` (Claude Agent SDK)

### Contrib

- `RedisMailbox(clock=...)`
- `PodmanTool(clock=...)`

## Enforcement Script

`scripts/validate_clock_usage.py` performs AST-based validation:

**Detection**:

- Scans all `.py` files in `src/weakincentives/`
- Detects forbidden patterns: `time.monotonic()`, `datetime.now()`, `time.sleep()`, etc.
- Handles all import styles (`import time`, `from time import monotonic`)

**Output**:

```
✗ Found 2 violation(s):

  src/weakincentives/example.py:42:8: forbidden time.monotonic(): now = time.monotonic()
  src/weakincentives/example.py:43:8: forbidden time.sleep(): time.sleep(1.0)

Production code must use Clock abstraction instead of direct time calls.
See specs/CLOCK.md for details and migration guide.
```

**Exit codes**:

- `0` - No violations found
- `1` - Violations detected
- `2` - Script error

**Integration**:

```bash
# Manual run
uv run python scripts/validate_clock_usage.py

# CI check (included in full validation)
make validate-clock
make check  # Includes validate-clock
```

## Test Performance Impact

Migrating to Clock abstraction achieved significant test performance improvements:

| Metric | Before | After | Improvement |
| ------------------- | ------ | ------ | ----------- |
| Total test time | 184s | 60s | **3.1×** |
| Slowest test | 60s | 5s | **12×** |
| Eval loop tests | 130s | 4.6s | **28×** |
| Visibility timeouts | 2.4s | \<0.01s | **240×** |

**Key optimizations enabled**:

1. **Polling elimination**: `wait_time_seconds=0` for instant queue checks
1. **Instant timeout tests**: `clock.advance()` instead of `time.sleep()`
1. **Deterministic coordination**: No race conditions from real time delays

See `docs/test_performance_optimization.md` for detailed analysis.

## Design Rationale

### Why Protocol Instead of ABC?

`Protocol` enables structural subtyping - tests can use `FakeClock` without
inheritance. Duck typing keeps the interface simple and doesn't require
registration with a base class.

### Why Not Mock `time` Module?

Mocking `time.monotonic()` globally has issues:

1. **Leaky**: Mock affects unrelated code (stdlib, dependencies)
1. **Fragile**: Breaks if code imports differently (`from time import monotonic`)
1. **Confusing**: Real time still passes, causing race conditions
1. **Testing**: Can't control time per test without complex setup/teardown

Dependency injection is explicit, local, and controlled.

### Why Separate `monotonic()` and `now()`?

Different use cases:

- `monotonic()`: Measuring intervals (timeouts, latency, heartbeats)
- `now()`: Wall clock timestamps (events, logs, audit trails)

In production:

- `monotonic()` uses `time.monotonic()` (immune to clock adjustments)
- `now()` uses `datetime.now(UTC)` (real wall clock time)

In tests:

- Both advance together via `FakeClock.advance()`
- `set_now()` allows setting specific timestamps for reproducibility

### Why Block Direct `datetime.now()`?

Every event timestamp, session creation, and message enqueue uses `datetime.now(UTC)`.
Without control, time becomes invisible global state that makes tests
non-deterministic and irreproducible.

## Common Patterns

### Pattern 1: Default to SystemClock

```python
@dataclass
class Worker:
    clock: Clock = field(default_factory=SystemClock)

    def process(self) -> None:
        start = self.clock.monotonic()
        # ... work ...
        duration = self.clock.monotonic() - start
```

**Benefit**: Production code works without explicit injection.

### Pattern 2: Optional Clock Parameter

```python
def collect_results(
    results: Mailbox[Result, None],
    *,
    timeout_seconds: float,
    clock: Clock | None = None,
) -> list[Result]:
    if clock is None:  # pragma: no cover
        clock = SystemClock()

    deadline = clock.monotonic() + timeout_seconds
    # ...
```

**Benefit**: Backward compatible, testable.

### Pattern 3: Clock-Aware Helpers

```python
def _utcnow(clock: Clock) -> datetime:
    """Helper for getting current UTC time from clock."""
    return clock.now()
```

**Anti-pattern**: Don't create `_utcnow()` that calls `datetime.now()` directly.
Always accept `Clock` parameter.

## Migration Checklist

For new code:

- [ ] Add `clock: Clock = field(default_factory=SystemClock)` to dataclasses
- [ ] Add `clock: Clock | None = None` to function parameters
- [ ] Use `clock.monotonic()` for intervals, `clock.now()` for timestamps
- [ ] Use `clock.sleep()` or `clock.asleep()` instead of blocking
- [ ] Inject `FakeClock()` in tests for deterministic time control

For existing code:

- [ ] Run `scripts/validate_clock_usage.py` to find violations
- [ ] Add Clock parameter to classes/functions that use time
- [ ] Replace `time.monotonic()` with `self.clock.monotonic()`
- [ ] Replace `datetime.now(UTC)` with `self.clock.now()`
- [ ] Replace `time.sleep()` with `self.clock.sleep()`
- [ ] Update tests to inject `FakeClock()` and verify time control

## FAQ

### Can I use `time.perf_counter()`?

No. Use `clock.monotonic()` for all interval measurements. `perf_counter()` has
higher resolution but is non-deterministic in tests.

### What about `time.time_ns()` or `monotonic_ns()`?

No. Use `clock.monotonic()` (float seconds). If nanosecond precision is needed,
extend the `Clock` protocol and update the implementations.

### What if external library calls `time.time()`?

External libraries (in `site-packages/`) are not under our control. Only
production code in `src/weakincentives/` must use `Clock`.

If you wrap an external library, inject `Clock` into your wrapper for all time
operations you control.

### Can I add one `time.monotonic()` call in a helper?

No. All time access must go through `Clock`, no exceptions. This ensures tests
can control time everywhere and violations are caught by CI.

### How do I test code with threading coordination?

Use `FakeClock` for time control and `threading.Event` for coordination:

```python
def test_background_worker():
    clock = FakeClock()
    started = threading.Event()

    def worker():
        started.set()
        clock.sleep(5.0)  # Instant advance, no real delay

    thread = threading.Thread(target=worker)
    thread.start()
    started.wait(timeout=1.0)  # Wait for thread to start
    thread.join(timeout=1.0)
```

### What about InMemoryMailbox reaper thread?

The reaper thread uses real time (`threading.Event.wait(timeout=0.1)`) for its
polling loop. After advancing `FakeClock`, manually trigger the reaper:

```python
clock.advance(1.2)
mailbox._reap_expired()  # Manually trigger timeout check
```

This is a test-only helper - production reaper runs continuously.

## Related Specs

- `specs/TESTING.md` - Test harness and fault injection patterns
- `specs/RESOURCE_REGISTRY.md` - Dependency injection and scoped resources
- `specs/SESSIONS.md` - Session lifecycle and deterministic execution
- `specs/THREAD_SAFETY.md` - Concurrency patterns with time-dependent code

## Summary

The Clock abstraction provides:

- ✅ **Deterministic testing** via injectable time source
- ✅ **3× faster tests** by eliminating real sleep calls
- ✅ **Zero violations** enforced by CI validation
- ✅ **100% production coverage** - all time operations use Clock
- ✅ **Simple patterns** - default factory and optional parameters
- ✅ **Comprehensive testing** - FakeClock enables instant time control

All production code in `src/weakincentives/` uses the Clock abstraction.
Violations are caught automatically in CI via `make validate-clock`.
