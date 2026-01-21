# Threading Primitives Specification

Specification for injectable threading primitives that simplify concurrent code
and enable deterministic testing.

## Overview

WINK uses threading for background workers, parallel execution, and coordination.
This spec defines injectable primitives that follow the same dependency injection
pattern as `Clock`—production implementations use real threading, while test
implementations provide deterministic control.

**Core types:**

- `Executor` - Protocol for submitting work to thread pools
- `Gate` - Protocol for signaling between threads (replaces `threading.Event`)
- `Latch` - One-shot synchronization barrier for coordinating threads
- `BackgroundWorker` - Abstraction for managed daemon threads
- `Checkpoint` - Protocol for cooperative yield points
- `CancellationToken` - Cooperative cancellation for long-running tasks
- `Scheduler` - Protocol for cooperative task scheduling
- `CallbackRegistry` - Thread-safe callback registration and invocation
- `SystemExecutor` - Production implementation using `ThreadPoolExecutor`
- `FakeExecutor` - Test implementation that runs tasks synchronously
- `FakeGate` - Test implementation with manual control
- `FakeCheckpoint` - Test implementation with yield/check counting
- `FakeScheduler` - Test implementation with step-by-step execution
- `SYSTEM_EXECUTOR` - Module-level singleton for production use

## Design Principles

1. **Injectable dependencies**: All threading primitives accept injectable
   implementations, defaulting to production behavior.

1. **Testable without real threads**: `FakeExecutor` runs tasks synchronously in
   the calling thread, eliminating race conditions and non-determinism in tests.

1. **Explicit over implicit**: Background threads are managed explicitly via
   `BackgroundWorker` rather than ad-hoc `threading.Thread` creation.

1. **Consistent with Clock pattern**: Follow the same protocols + fake pattern
   established in `CLOCK.md` for familiarity.

## Executor Protocol

The `Executor` protocol abstracts over thread pool submission:

```python
from weakincentives.threading import (
    Executor,
    SystemExecutor,
    FakeExecutor,
    SYSTEM_EXECUTOR,
)
from collections.abc import Callable
from typing import Protocol, TypeVar

T = TypeVar("T")

class Future(Protocol[T]):
    """Minimal future interface for submitted work."""

    def result(self, timeout: float | None = None) -> T: ...
    def done(self) -> bool: ...
    def cancel(self) -> bool: ...

class Executor(Protocol):
    """Protocol for submitting work to be executed."""

    def submit(self, fn: Callable[[], T]) -> Future[T]: ...

    def map(
        self,
        fn: Callable[[A], T],
        items: Iterable[A],
        *,
        timeout: float | None = None,
    ) -> Iterator[T]: ...

    def shutdown(self, *, wait: bool = True) -> None: ...
```

Components depend on `Executor` rather than directly creating thread pools:

| Component | Usage |
| --- | --- |
| `LoopGroup` | Running multiple loops concurrently |
| `ParallelEvaluator` | Running evaluations in parallel |
| `BatchProcessor` | Processing items concurrently |

## SystemExecutor

The production implementation wraps `ThreadPoolExecutor`:

```python
from weakincentives.threading import SystemExecutor, SYSTEM_EXECUTOR

# Use the singleton (creates workers on demand)
future = SYSTEM_EXECUTOR.submit(lambda: expensive_computation())
result = future.result()

# Or create a scoped executor
executor = SystemExecutor(max_workers=4)
try:
    results = list(executor.map(process_item, items))
finally:
    executor.shutdown()

# Context manager support
with SystemExecutor(max_workers=4) as executor:
    futures = [executor.submit(task) for task in tasks]
    results = [f.result() for f in futures]
```

### SystemExecutor API

| Method | Description |
| --- | --- |
| `submit(fn)` | Submit callable, return Future |
| `map(fn, items)` | Apply fn to items concurrently |
| `shutdown(wait=True)` | Shut down executor, optionally waiting |
| `__enter__` / `__exit__` | Context manager for scoped usage |

Constructor parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `max_workers` | `int \| None` | `None` | Max concurrent threads (None = CPU count) |
| `thread_name_prefix` | `str` | `"worker"` | Prefix for thread names |

## FakeExecutor

The test implementation runs tasks synchronously in the calling thread:

```python
from weakincentives.threading import FakeExecutor

executor = FakeExecutor()

# Tasks run immediately in the calling thread
future = executor.submit(lambda: 42)
assert future.done()  # Already complete
assert future.result() == 42

# Deterministic ordering - no race conditions
results = []
executor.submit(lambda: results.append(1))
executor.submit(lambda: results.append(2))
assert results == [1, 2]  # Always this order
```

### FakeExecutor Features

1. **Synchronous execution**: Tasks complete before `submit()` returns.
1. **No real threads**: All work happens in the calling thread.
1. **Exception propagation**: Exceptions are captured and re-raised on
   `result()`.
1. **Deterministic**: Order of execution matches order of submission.

### FakeExecutor API

| Method | Description |
| --- | --- |
| `submit(fn)` | Execute fn immediately, return completed Future |
| `map(fn, items)` | Apply fn to each item sequentially |
| `shutdown(wait=True)` | No-op (nothing to shut down) |
| `submitted` | Property: list of submitted callables (for assertions) |

### Testing with FakeExecutor

```python
from weakincentives.threading import FakeExecutor

def test_batch_processor():
    executor = FakeExecutor()
    processor = BatchProcessor(executor=executor)

    results = processor.process(items=[1, 2, 3])

    # Assertions are deterministic - no flaky tests
    assert results == [2, 4, 6]
    assert len(executor.submitted) == 3
```

## Gate Protocol

The `Gate` protocol abstracts signaling between threads:

```python
from weakincentives.threading import Gate, SystemGate, FakeGate

class Gate(Protocol):
    """Protocol for thread signaling (like threading.Event)."""

    def set(self) -> None:
        """Open the gate, releasing all waiters."""
        ...

    def clear(self) -> None:
        """Close the gate."""
        ...

    def is_set(self) -> bool:
        """Return True if gate is open."""
        ...

    def wait(self, timeout: float | None = None) -> bool:
        """Block until gate opens or timeout expires.

        Returns True if gate opened, False if timeout expired.
        """
        ...
```

### SystemGate

Production implementation using `threading.Event`:

```python
from weakincentives.threading import SystemGate

gate = SystemGate()

# In worker thread
if gate.wait(timeout=5.0):
    print("Gate opened!")
else:
    print("Timeout waiting for gate")

# In control thread
gate.set()  # Release all waiters
```

### FakeGate

Test implementation with manual time control:

```python
from weakincentives.threading import FakeGate
from weakincentives.clock import FakeClock

clock = FakeClock()
gate = FakeGate(clock=clock)

# wait() checks is_set and advances time without blocking
assert gate.wait(timeout=1.0) is False  # Not set, returns immediately
assert clock.monotonic() == 1.0  # Time advanced by timeout

gate.set()
assert gate.wait(timeout=1.0) is True  # Gate is open
```

### FakeGate API

| Method | Description |
| --- | --- |
| `set()` | Open the gate |
| `clear()` | Close the gate |
| `is_set()` | Check if gate is open |
| `wait(timeout)` | Return is_set(), advance clock by timeout if not set |

## Latch

A one-shot synchronization barrier for coordinating threads:

```python
from weakincentives.threading import Latch

class Latch:
    """One-shot barrier that releases when count reaches zero."""

    def __init__(self, count: int) -> None: ...
    def count_down(self) -> None: ...
    def await_(self, timeout: float | None = None) -> bool: ...

    @property
    def count(self) -> int: ...
```

### Latch Usage

```python
from weakincentives.threading import Latch

# Wait for 3 workers to signal completion
latch = Latch(3)

def worker():
    do_work()
    latch.count_down()

# Start workers...

# Block until all workers complete
if latch.await_(timeout=30.0):
    print("All workers done")
else:
    print("Timeout waiting for workers")
```

### FakeLatch

Test implementation that doesn't block:

```python
from weakincentives.threading import FakeLatch

latch = FakeLatch(3)
assert latch.count == 3

latch.count_down()
assert latch.count == 2
assert latch.await_(timeout=1.0) is False  # Not yet released

latch.count_down()
latch.count_down()
assert latch.count == 0
assert latch.await_(timeout=1.0) is True  # Released
```

## BackgroundWorker

Abstraction for managed daemon threads:

```python
from weakincentives.threading import BackgroundWorker

class BackgroundWorker:
    """Managed daemon thread with lifecycle control."""

    def __init__(
        self,
        target: Callable[[], None],
        *,
        name: str = "worker",
        daemon: bool = True,
    ) -> None: ...

    def start(self) -> None: ...
    def stop(self, timeout: float = 5.0) -> bool: ...
    def join(self, timeout: float | None = None) -> bool: ...

    @property
    def running(self) -> bool: ...
    @property
    def alive(self) -> bool: ...
```

### BackgroundWorker Usage

```python
from weakincentives.threading import BackgroundWorker, SystemGate

stop_signal = SystemGate()

def worker_loop():
    while not stop_signal.is_set():
        process_item()
        stop_signal.wait(timeout=1.0)  # Check periodically

worker = BackgroundWorker(worker_loop, name="processor")
worker.start()

# Later...
stop_signal.set()
worker.stop(timeout=10.0)
```

### FakeBackgroundWorker

Test implementation that runs synchronously:

```python
from weakincentives.threading import FakeBackgroundWorker

calls = []

def worker_fn():
    calls.append("called")

worker = FakeBackgroundWorker(worker_fn)
assert len(calls) == 0

worker.start()  # Runs synchronously
assert len(calls) == 1

# Already "stopped" since it ran synchronously
assert worker.running is False
```

## Cooperative Yielding

For long-running tasks that need to yield control cooperatively, WINK provides
checkpoint and cancellation primitives.

### Checkpoint Protocol

A `Checkpoint` is a yield point where tasks can:

1. Check if they should be cancelled
1. Yield control to other tasks
1. Update progress

```python
from weakincentives.threading import Checkpoint, SystemCheckpoint, FakeCheckpoint

class Checkpoint(Protocol):
    """Protocol for cooperative yield points."""

    def yield_(self) -> None:
        """Yield control to other tasks.

        In production, this is a no-op or calls time.sleep(0).
        In tests, this can be intercepted to control execution order.
        """
        ...

    def check(self) -> None:
        """Check if task should continue.

        Raises CancelledException if cancellation was requested.
        """
        ...

    def is_cancelled(self) -> bool:
        """Return True if cancellation was requested."""
        ...

    @property
    def token(self) -> CancellationToken:
        """The cancellation token for this checkpoint."""
        ...
```

### CancellationToken

Cooperative cancellation that tasks check at yield points:

```python
from weakincentives.threading import CancellationToken, CancelledException

class CancellationToken:
    """Token for cooperative cancellation."""

    def __init__(self) -> None: ...

    def cancel(self) -> None:
        """Request cancellation."""
        ...

    def is_cancelled(self) -> bool:
        """Return True if cancellation was requested."""
        ...

    def check(self) -> None:
        """Raise CancelledException if cancelled."""
        ...

    def child(self) -> CancellationToken:
        """Create a child token that cancels when parent cancels."""
        ...
```

### Using Checkpoints in Long-Running Tasks

Tasks should call `checkpoint.check()` or `checkpoint.yield_()` at regular
intervals:

```python
from weakincentives.threading import Checkpoint, SystemCheckpoint

def process_large_dataset(
    items: list[Item],
    checkpoint: Checkpoint | None = None,
) -> list[Result]:
    checkpoint = checkpoint or SystemCheckpoint()
    results = []

    for i, item in enumerate(items):
        # Yield point - check cancellation and let other tasks run
        checkpoint.check()

        result = expensive_processing(item)
        results.append(result)

        # Yield every N items to prevent starvation
        if i % 100 == 0:
            checkpoint.yield_()

    return results
```

### SystemCheckpoint

Production implementation:

```python
from weakincentives.threading import SystemCheckpoint, CancellationToken

# With explicit token
token = CancellationToken()
checkpoint = SystemCheckpoint(token=token)

# Start long task...
executor.submit(lambda: process_items(items, checkpoint=checkpoint))

# Later, request cancellation
token.cancel()  # Task will raise CancelledException at next check()
```

`SystemCheckpoint.yield_()` calls `time.sleep(0)` which releases the GIL and
allows other Python threads to run.

### FakeCheckpoint

Test implementation with manual control:

```python
from weakincentives.threading import FakeCheckpoint

checkpoint = FakeCheckpoint()

# Track yield points
results = []
def task():
    results.append("start")
    checkpoint.yield_()
    results.append("middle")
    checkpoint.yield_()
    results.append("end")

task()

# Verify yield points were hit
assert checkpoint.yield_count == 2
assert checkpoint.check_count == 0
assert results == ["start", "middle", "end"]

# Test cancellation
checkpoint.token.cancel()
with pytest.raises(CancelledException):
    checkpoint.check()
```

### FakeCheckpoint API

| Property/Method | Description |
| --- | --- |
| `yield_count` | Number of times `yield_()` was called |
| `check_count` | Number of times `check()` was called |
| `token` | The `CancellationToken` (can be cancelled for testing) |
| `reset()` | Reset counters to zero |

### Scheduler Protocol

For advanced cooperative scheduling, the `Scheduler` protocol manages task
execution order:

```python
from weakincentives.threading import Scheduler, FifoScheduler, FakeScheduler

class Scheduler(Protocol):
    """Protocol for cooperative task scheduling."""

    def schedule(self, task: Callable[[], T]) -> Future[T]:
        """Schedule a task for execution."""
        ...

    def yield_(self) -> None:
        """Yield control to the scheduler."""
        ...

    def run_until_complete(self) -> None:
        """Run scheduled tasks until all complete."""
        ...

    def run_one(self) -> bool:
        """Run the next scheduled task. Returns False if queue empty."""
        ...
```

### FifoScheduler

Production cooperative scheduler using a thread pool:

```python
from weakincentives.threading import FifoScheduler

scheduler = FifoScheduler(max_workers=4)

# Schedule tasks
f1 = scheduler.schedule(task1)
f2 = scheduler.schedule(task2)

# Tasks run concurrently, yielding at checkpoint.yield_() calls
scheduler.run_until_complete()
```

### FakeScheduler

Test implementation for deterministic execution:

```python
from weakincentives.threading import FakeScheduler

scheduler = FakeScheduler()

execution_order = []

def task_a():
    execution_order.append("a1")
    scheduler.yield_()  # Yield control
    execution_order.append("a2")

def task_b():
    execution_order.append("b1")
    scheduler.yield_()
    execution_order.append("b2")

scheduler.schedule(task_a)
scheduler.schedule(task_b)

# Run tasks step by step
scheduler.run_one()  # Runs task_a until first yield -> "a1"
scheduler.run_one()  # Runs task_b until first yield -> "b1"
scheduler.run_one()  # Resumes task_a -> "a2"
scheduler.run_one()  # Resumes task_b -> "b2"

assert execution_order == ["a1", "b1", "a2", "b2"]
```

### When to Use Each Primitive

| Primitive | Use Case |
| --- | --- |
| `Checkpoint` | Long-running tasks that need cancellation + yield |
| `CancellationToken` | Just need cancellation, no yielding |
| `Scheduler` | Multiple cooperating tasks with controlled interleaving |
| `Gate.wait(timeout)` | Implicit yield while waiting for signal |

### Checkpoint Integration with Other Primitives

Checkpoints integrate with other primitives:

```python
from weakincentives.threading import (
    Checkpoint,
    SystemCheckpoint,
    Gate,
    SystemGate,
    CancellationToken,
)

class Worker:
    def __init__(
        self,
        stop_signal: Gate | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> None:
        self._stop = stop_signal or SystemGate()
        self._checkpoint = checkpoint or SystemCheckpoint()

    def run(self) -> None:
        while not self._stop.is_set():
            # Check cancellation at each iteration
            self._checkpoint.check()

            self._process_batch()

            # Yield after each batch
            self._checkpoint.yield_()

            # Wait for next poll interval (implicit yield)
            self._stop.wait(timeout=1.0)
```

## CallbackRegistry

Thread-safe callback registration and invocation:

```python
from weakincentives.threading import CallbackRegistry

class CallbackRegistry[T]:
    """Thread-safe registry for callbacks."""

    def register(self, callback: Callable[[T], None]) -> None: ...
    def unregister(self, callback: Callable[[T], None]) -> None: ...
    def invoke(self, value: T) -> int: ...
    def invoke_all(self, value: T) -> list[Exception]: ...
    def clear(self) -> None: ...

    @property
    def count(self) -> int: ...
```

### CallbackRegistry Usage

```python
from weakincentives.threading import CallbackRegistry

registry = CallbackRegistry[str]()

def on_message(msg: str) -> None:
    print(f"Received: {msg}")

registry.register(on_message)

# Invoke all callbacks (exceptions don't stop other callbacks)
errors = registry.invoke_all("hello")
if errors:
    for error in errors:
        logger.error("Callback failed", exc_info=error)
```

### Thread Safety Guarantees

1. **Registration under lock**: `register()` and `unregister()` are atomic.
1. **Snapshot before invoke**: Callbacks are copied before invocation.
1. **Invoke outside lock**: Callbacks execute without holding the lock.
1. **Exception isolation**: One callback's exception doesn't affect others.

This pattern is used by `Heartbeat` for beat callbacks.

## Component Integration

All threading-dependent components accept injectable primitives:

```python
# LoopGroup uses Executor
group = LoopGroup(loops=[...], executor=executor)

# Watchdog uses Gate for stop signaling
watchdog = Watchdog(heartbeats, stop_signal=gate)

# MainLoop uses Gate for shutdown
loop = MainLoop(mailbox, prompt, shutdown_signal=gate)
```

The pattern is consistent across all components:

| Component | Parameter | Type | Default |
| --- | --- | --- | --- |
| `LoopGroup` | `executor` | `Executor` | `SYSTEM_EXECUTOR` |
| `Watchdog` | `stop_signal` | `Gate` | `SystemGate()` |
| `MainLoop` | `shutdown_signal` | `Gate` | `SystemGate()` |
| `InMemoryMailbox` | `executor` | `Executor` | `SYSTEM_EXECUTOR` |

## Usage Patterns

### Parallel Processing with Deterministic Tests

```python
from weakincentives.threading import Executor, SYSTEM_EXECUTOR, FakeExecutor

class BatchProcessor:
    def __init__(self, executor: Executor = SYSTEM_EXECUTOR) -> None:
        self._executor = executor

    def process(self, items: list[int]) -> list[int]:
        futures = [self._executor.submit(lambda x=x: x * 2) for x in items]
        return [f.result() for f in futures]

# Production: real parallelism
processor = BatchProcessor()
results = processor.process([1, 2, 3])

# Test: deterministic, no threads
def test_batch_processor():
    executor = FakeExecutor()
    processor = BatchProcessor(executor=executor)
    results = processor.process([1, 2, 3])
    assert results == [2, 4, 6]
```

### Graceful Shutdown

```python
from weakincentives.threading import BackgroundWorker, Gate, SystemGate

class Poller:
    def __init__(self, stop_signal: Gate | None = None) -> None:
        self._stop = stop_signal or SystemGate()
        self._worker = BackgroundWorker(self._poll_loop, name="poller")

    def start(self) -> None:
        self._worker.start()

    def stop(self, timeout: float = 10.0) -> bool:
        self._stop.set()
        return self._worker.stop(timeout=timeout)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            self._do_poll()
            self._stop.wait(timeout=1.0)
```

### Coordinated Multi-Worker Startup

```python
from weakincentives.threading import Latch, Executor

def start_workers(
    count: int,
    executor: Executor,
) -> None:
    ready_latch = Latch(count)

    def worker(worker_id: int) -> None:
        initialize()
        ready_latch.count_down()  # Signal ready
        run_loop()

    # Start all workers
    futures = [executor.submit(lambda i=i: worker(i)) for i in range(count)]

    # Wait for all to initialize
    ready_latch.await_(timeout=30.0)
    print("All workers ready")
```

### Testing Thread Coordination

```python
from weakincentives.threading import FakeGate, FakeLatch
from weakincentives.clock import FakeClock

def test_shutdown_timeout():
    clock = FakeClock()
    gate = FakeGate(clock=clock)
    poller = Poller(stop_signal=gate, clock=clock)

    poller.start()

    # Simulate timeout without real waiting
    result = gate.wait(timeout=5.0)
    assert result is False
    assert clock.monotonic() == 5.0
```

## Test Helpers

The `tests/helpers/threading.py` module re-exports test implementations and
provides pytest fixtures:

```python
# In tests
from tests.helpers.threading import fake_executor, fake_gate

def test_something(fake_executor: FakeExecutor) -> None:
    processor = BatchProcessor(executor=fake_executor)
    # ...
```

### Available Fixtures

| Fixture | Type | Description |
| --- | --- | --- |
| `fake_executor` | `FakeExecutor` | Fresh executor per test |
| `fake_gate` | `FakeGate` | Fresh gate per test |
| `fake_latch` | `Callable[[int], FakeLatch]` | Factory for latches |

## Public Exports

From the main package:

```python
from weakincentives import (
    # Protocols
    Executor,
    Gate,
    Checkpoint,
    Scheduler,

    # Production implementations
    SystemExecutor,
    SystemGate,
    SystemCheckpoint,
    FifoScheduler,
    BackgroundWorker,
    CallbackRegistry,
    Latch,
    CancellationToken,
    CancelledException,
    SYSTEM_EXECUTOR,

    # Test implementations
    FakeExecutor,
    FakeGate,
    FakeLatch,
    FakeBackgroundWorker,
    FakeCheckpoint,
    FakeScheduler,
)
```

From the threading module:

```python
from weakincentives.threading import (
    # Protocols
    Executor,
    Gate,
    Checkpoint,
    Scheduler,

    # Production
    SystemExecutor,
    SystemGate,
    SystemCheckpoint,
    FifoScheduler,
    CancellationToken,
    CancelledException,
    SYSTEM_EXECUTOR,

    # Test
    FakeExecutor,
    FakeGate,
    FakeCheckpoint,
    FakeScheduler,
)
```

## Migration Guide

### Before: Direct ThreadPoolExecutor

```python
from concurrent.futures import ThreadPoolExecutor

class Processor:
    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def process(self, items: list[int]) -> list[int]:
        futures = [self._executor.submit(transform, x) for x in items]
        return [f.result() for f in futures]

# Testing requires real threads - potentially flaky
def test_processor():
    p = Processor()
    assert p.process([1, 2, 3]) == [2, 4, 6]  # May have race conditions
```

### After: Injectable Executor

```python
from weakincentives.threading import Executor, SYSTEM_EXECUTOR, FakeExecutor

class Processor:
    def __init__(self, executor: Executor = SYSTEM_EXECUTOR) -> None:
        self._executor = executor

    def process(self, items: list[int]) -> list[int]:
        futures = [self._executor.submit(lambda x=x: transform(x)) for x in items]
        return [f.result() for f in futures]

# Testing is deterministic - no threads
def test_processor():
    executor = FakeExecutor()
    p = Processor(executor=executor)
    assert p.process([1, 2, 3]) == [2, 4, 6]  # Always passes
```

### Before: Direct threading.Event

```python
import threading

class Worker:
    def __init__(self) -> None:
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

# Testing requires real waiting or fragile sleep()
def test_worker_stop():
    w = Worker()
    w.stop()
    time.sleep(0.1)  # Fragile!
    assert not w.running
```

### After: Injectable Gate

```python
from weakincentives.threading import Gate, SystemGate, FakeGate

class Worker:
    def __init__(self, stop_signal: Gate | None = None) -> None:
        self._stop = stop_signal or SystemGate()

    def stop(self) -> None:
        self._stop.set()

# Testing is instant - no real waiting
def test_worker_stop():
    gate = FakeGate()
    w = Worker(stop_signal=gate)
    w.stop()
    assert gate.is_set()
```

## Non-Goals

This specification does not address:

- **Async/await**: WINK is synchronous at its core; async bridging is handled
  separately in adapters
- **Distributed locking**: Single-process focus; use Redis or other external
  systems for distributed coordination
- **Lock-free data structures**: Standard library locks are sufficient for
  WINK's throughput requirements
- **Thread-local storage**: Use `ResourceRegistry` scopes instead

## Relationship to Other Specs

| Spec | Relationship |
| --- | --- |
| `CLOCK.md` | Same DI pattern; threading primitives complement time control |
| `THREAD_SAFETY.md` | Documents guarantees; this spec provides the primitives |
| `LIFECYCLE.md` | Uses these primitives for shutdown coordination |
| `TESTING.md` | Fake implementations enable deterministic concurrency tests |

## Summary

Threading primitives follow the same injectable dependency pattern as `Clock`:

1. **Protocols** define the interface (`Executor`, `Gate`, `Checkpoint`, `Scheduler`)
1. **System implementations** provide production behavior
1. **Fake implementations** enable deterministic testing
1. **Module singletons** provide convenient defaults

The primitives address three categories of threading needs:

| Category | Primitives | Purpose |
| --- | --- | --- |
| **Execution** | `Executor`, `BackgroundWorker` | Submit and manage work |
| **Coordination** | `Gate`, `Latch`, `CallbackRegistry` | Synchronize between threads |
| **Cooperation** | `Checkpoint`, `CancellationToken`, `Scheduler` | Yield control voluntarily |

This eliminates the primary sources of test flakiness in concurrent code:
non-deterministic scheduling, real time delays, and uncontrolled interleaving.
