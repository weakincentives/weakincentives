# Thread Testing Framework Specification

## Purpose

Provide a thin abstraction layer over Python's native threading that enables
deterministic testing of concurrent code. The framework allows cooperative
control over execution ordering while maintaining minimal overhead in
production.

**Implementation:** `src/weakincentives/threads/`

## Problem Statement

Thread-based code is notoriously difficult to test reliably:

- **Non-determinism**: Thread scheduling is OS-controlled, causing flaky tests
- **Race conditions**: Bugs may appear sporadically, passing CI but failing in production
- **Interleaving coverage**: Cannot systematically test specific execution orderings
- **Debugging difficulty**: Reproducing failures requires exact timing conditions

## Design Goals

1. **Minimal production overhead**: Thin wrappers with near-zero cost when not testing
2. **Deterministic test mode**: Full control over thread interleaving
3. **Cooperative scheduling**: Explicit yield points enable precise ordering control
4. **Drop-in replacement**: Compatible with existing threading patterns
5. **Composable**: Works with existing WINK infrastructure (lifecycle, sessions)

## Core Abstractions

### ThreadContext Protocol

Dependency injection point for threading primitives:

```python
from typing import Protocol
from collections.abc import Callable

class ThreadContext(Protocol):
    """Context providing threading primitives.

    Production code obtains primitives through this context, enabling
    test code to substitute deterministic implementations.
    """

    def spawn(
        self,
        target: Callable[[], None],
        *,
        name: str | None = None,
        daemon: bool = False,
    ) -> ThreadHandle:
        """Spawn a new thread executing the target callable."""
        ...

    def lock(self) -> Lock:
        """Create a new lock."""
        ...

    def rlock(self) -> RLock:
        """Create a new reentrant lock."""
        ...

    def event(self) -> Event:
        """Create a new event."""
        ...

    def condition(self, lock: Lock | RLock | None = None) -> Condition:
        """Create a new condition variable."""
        ...

    def semaphore(self, value: int = 1) -> Semaphore:
        """Create a new semaphore."""
        ...

    def barrier(self, parties: int) -> Barrier:
        """Create a new barrier."""
        ...

    def checkpoint(self, name: str | None = None) -> None:
        """Yield control to scheduler (no-op in production)."""
        ...
```

### ThreadHandle Protocol

Abstraction over thread instances:

```python
class ThreadHandle(Protocol):
    """Handle to a spawned thread."""

    @property
    def name(self) -> str:
        """Thread name."""
        ...

    @property
    def ident(self) -> int | None:
        """Thread identifier (None if not started)."""
        ...

    @property
    def is_alive(self) -> bool:
        """True if thread is running."""
        ...

    def start(self) -> None:
        """Start the thread."""
        ...

    def join(self, timeout: float | None = None) -> None:
        """Wait for thread to complete."""
        ...
```

### Synchronization Primitives

Mirror Python's threading primitives:

| Protocol | Python Equivalent | Key Methods |
|----------|-------------------|-------------|
| `Lock` | `threading.Lock` | `acquire()`, `release()`, `locked()` |
| `RLock` | `threading.RLock` | `acquire()`, `release()` |
| `Event` | `threading.Event` | `set()`, `clear()`, `wait()`, `is_set()` |
| `Condition` | `threading.Condition` | `acquire()`, `release()`, `wait()`, `notify()`, `notify_all()` |
| `Semaphore` | `threading.Semaphore` | `acquire()`, `release()` |
| `Barrier` | `threading.Barrier` | `wait()`, `parties`, `n_waiting` |

All primitives support context manager protocol for `acquire()`/`release()`.

## Implementations

### ProductionThreadContext

Thin wrappers over native `threading` module:

```python
class ProductionThreadContext:
    """Production implementation using native threads.

    All methods delegate directly to threading module with minimal overhead.
    checkpoint() is a no-op in production.
    """

    def spawn(self, target, *, name=None, daemon=False) -> ThreadHandle:
        thread = threading.Thread(target=target, name=name, daemon=daemon)
        return NativeThreadHandle(thread)

    def lock(self) -> Lock:
        return NativeLock(threading.Lock())

    def checkpoint(self, name: str | None = None) -> None:
        pass  # No-op in production
```

### DeterministicThreadContext

Test implementation with full scheduling control:

```python
class DeterministicThreadContext:
    """Test implementation with deterministic scheduling.

    Threads run cooperatively, yielding at checkpoints and synchronization
    operations. The scheduler controls which thread runs next.

    Attributes:
        scheduler: Controls thread execution order.
    """

    scheduler: Scheduler

    def spawn(self, target, *, name=None, daemon=False) -> ThreadHandle:
        return DeterministicThread(target, name=name, scheduler=self.scheduler)

    def lock(self) -> Lock:
        return DeterministicLock(scheduler=self.scheduler)

    def checkpoint(self, name: str | None = None) -> None:
        self.scheduler.yield_control(checkpoint_name=name)
```

## Scheduler

Controls execution order in test mode:

```python
@dataclass
class Scheduler:
    """Deterministic thread scheduler for testing.

    Manages a set of runnable threads and controls which executes next.
    Threads yield control at checkpoints and blocking operations.
    """

    # Scheduling strategies
    strategy: SchedulingStrategy = field(default_factory=RoundRobinStrategy)

    # Execution state
    threads: list[DeterministicThread] = field(default_factory=list)
    current_thread: DeterministicThread | None = None

    def run_until_blocked(self) -> None:
        """Run current thread until it yields or blocks."""
        ...

    def run_until_complete(self) -> None:
        """Run all threads until all complete or deadlock."""
        ...

    def step(self) -> StepResult:
        """Execute one scheduling step, return what happened."""
        ...

    def yield_control(self, *, checkpoint_name: str | None = None) -> None:
        """Called by threads to yield to scheduler."""
        ...

    def select_next(self) -> DeterministicThread | None:
        """Select next thread to run based on strategy."""
        ...
```

### Scheduling Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `RoundRobinStrategy` | Cycle through threads in order | Default, predictable |
| `RandomStrategy` | Random selection (seeded) | Fuzzing, exploration |
| `ScriptedStrategy` | Explicit ordering via script | Testing specific interleavings |
| `ExhaustiveStrategy` | Try all possible orderings | Verification (small state spaces) |

### ScriptedStrategy

For testing specific interleavings:

```python
class ScriptedStrategy:
    """Execute threads in a predetermined order.

    Example::

        strategy = ScriptedStrategy([
            "worker-1",  # Run worker-1 until it yields
            "worker-2",  # Run worker-2 until it yields
            "worker-1",  # Run worker-1 again
            "*",         # Run any runnable thread
        ])
    """

    script: Sequence[str]  # Thread names or "*" for any
    position: int = 0
```

## Checkpoints

Explicit yield points for cooperative scheduling:

```python
def process_item(item: Item, *, ctx: ThreadContext) -> None:
    """Process an item with explicit checkpoints."""

    ctx.checkpoint("before_validate")
    validate(item)

    ctx.checkpoint("before_transform")
    result = transform(item)

    ctx.checkpoint("before_persist")
    persist(result)
```

In production, `checkpoint()` is a no-op. In test mode, it yields control to
the scheduler, enabling precise control over when each thread progresses.

### Decorator-Based Checkpoints

The `@checkpointed` decorator automatically injects checkpoints at function
entry and exit:

```python
from weakincentives.threads import checkpointed, get_context

@checkpointed
def process_item(item: Item) -> Result:
    """Checkpoints injected at entry and exit automatically."""
    return transform(item)

@checkpointed(name="critical_section")
def update_shared_state(value: int) -> None:
    """Named checkpoint for targeted scheduling."""
    shared.value = value
```

The decorator uses the thread-local context to emit checkpoints:

```python
def checkpointed(
    fn: Callable[P, T] | None = None,
    *,
    name: str | None = None,
    entry: bool = True,
    exit: bool = True,
) -> Callable[P, T]:
    """Decorator that injects checkpoints at function boundaries.

    Args:
        fn: Function to wrap.
        name: Base name for checkpoints (defaults to function name).
        entry: Emit checkpoint on entry.
        exit: Emit checkpoint on exit.

    Checkpoint names follow pattern: "{name}:entry" and "{name}:exit"
    """
    ...
```

### Context Manager Checkpoints

For finer-grained control within a function, use `checkpoint_region`:

```python
from weakincentives.threads import checkpoint_region

def complex_operation(data: Data, *, ctx: ThreadContext) -> Result:
    """Use context managers for checkpoints around critical regions."""

    with checkpoint_region(ctx, "validation"):
        # Checkpoint at entry: "validation:enter"
        validate(data)
        # Checkpoint at exit: "validation:exit"

    with checkpoint_region(ctx, "transform", exit_only=True):
        # No entry checkpoint
        result = expensive_transform(data)
        # Checkpoint at exit: "transform:exit"

    return result
```

Context manager signature:

```python
@contextmanager
def checkpoint_region(
    ctx: ThreadContext,
    name: str,
    *,
    entry: bool = True,
    exit: bool = True,
    exit_only: bool = False,
) -> Iterator[None]:
    """Context manager that emits checkpoints on entry and/or exit.

    Args:
        ctx: Thread context for checkpoint emission.
        name: Base name for checkpoints.
        entry: Emit checkpoint on entry (unless exit_only).
        exit: Emit checkpoint on exit.
        exit_only: Convenience flag to set entry=False.
    """
    ...
```

### Thread-Local Context Access

For decorator-based checkpoints that don't take explicit context:

```python
from weakincentives.threads import set_context, get_context, context_scope

# Global default (typically ProductionThreadContext)
set_context(ProductionThreadContext())

# Get current context (thread-local with global fallback)
ctx = get_context()

# Scoped override for testing
with context_scope(DeterministicThreadContext(scheduler)):
    # All @checkpointed calls in this scope use deterministic context
    run_test()
```

### Automatic Checkpoints at Synchronization

The deterministic primitives automatically yield at blocking operations:

| Operation | Yields Control |
|-----------|----------------|
| `lock.acquire()` | Before attempting, after acquiring |
| `event.wait()` | Before waiting, after waking |
| `condition.wait()` | Before waiting, after waking |
| `barrier.wait()` | Before waiting, after all arrive |
| `thread.join()` | Before waiting, after completion |

## Test Utilities

### ThreadTestCase

Base class for thread tests:

```python
class ThreadTestCase:
    """Test case with deterministic thread context.

    Provides a DeterministicThreadContext and Scheduler for each test.
    Automatically detects deadlocks and unfinished threads.
    """

    ctx: DeterministicThreadContext
    scheduler: Scheduler

    def run_threads(
        self,
        *targets: Callable[[], None],
        strategy: SchedulingStrategy | None = None,
    ) -> ExecutionTrace:
        """Run targets as threads, return execution trace."""
        ...

    def assert_no_deadlock(self) -> None:
        """Assert no deadlock occurred during execution."""
        ...

    def assert_invariant(
        self,
        predicate: Callable[[], bool],
        message: str = "Invariant violated",
    ) -> None:
        """Check invariant holds at current scheduler state."""
        ...
```

### Execution Tracing

Record execution history for debugging:

```python
@dataclass(frozen=True)
class ExecutionStep:
    """Record of one scheduling step."""
    thread_name: str
    checkpoint_name: str | None
    operation: str  # "checkpoint", "acquire", "release", etc.
    timestamp: int  # Logical clock

@dataclass
class ExecutionTrace:
    """Complete execution history."""
    steps: tuple[ExecutionStep, ...]

    def find_race(self, *variables: str) -> RaceCondition | None:
        """Analyze trace for races on given variables."""
        ...

    def to_sequence_diagram(self) -> str:
        """Generate ASCII sequence diagram."""
        ...
```

### Deadlock Detection

Automatic detection when all threads are blocked:

```python
@dataclass(frozen=True)
class DeadlockError(Exception):
    """Raised when scheduler detects all threads blocked."""

    threads: tuple[ThreadInfo, ...]
    held_locks: Mapping[Lock, str]  # lock -> holder thread name
    waiting_on: Mapping[str, object]  # thread name -> what it's waiting for
```

## Integration Patterns

### With ResourceRegistry

Register context as a resource:

```python
from weakincentives.resources import Binding, ResourceRegistry, Scope

def create_test_registry() -> ResourceRegistry:
    scheduler = Scheduler()
    ctx = DeterministicThreadContext(scheduler=scheduler)

    return ResourceRegistry.of(
        Binding(ThreadContext, lambda r: ctx),
        Binding(Scheduler, lambda r: scheduler),
    )
```

### With Lifecycle Components

Existing lifecycle code should accept ThreadContext:

```python
class LoopGroup:
    def __init__(
        self,
        loops: Sequence[Runnable],
        *,
        thread_context: ThreadContext | None = None,
        # ... other params
    ) -> None:
        self._ctx = thread_context or ProductionThreadContext()

    def run(self, ...) -> None:
        for loop in self.loops:
            handle = self._ctx.spawn(target=loop.run, name=loop.name)
            handle.start()
```

### With Session

Testing concurrent session access:

```python
def test_concurrent_dispatch():
    ctx, scheduler = create_test_context()
    session = Session(dispatcher=InProcessDispatcher())

    def worker_1():
        ctx.checkpoint("w1_before_dispatch")
        session.dispatch(Event1())
        ctx.checkpoint("w1_after_dispatch")

    def worker_2():
        ctx.checkpoint("w2_before_dispatch")
        session.dispatch(Event2())
        ctx.checkpoint("w2_after_dispatch")

    # Test specific interleaving: w1 dispatches, then w2
    scheduler.strategy = ScriptedStrategy([
        "worker_1",  # w1 reaches before_dispatch
        "worker_1",  # w1 dispatches and reaches after_dispatch
        "worker_2",  # w2 reaches before_dispatch
        "worker_2",  # w2 dispatches and reaches after_dispatch
    ])

    trace = run_threads(ctx, worker_1, worker_2)
    assert session[Plan].all() == (expected_state,)
```

## Exhaustive Testing

For small state spaces, explore all interleavings:

```python
def test_all_interleavings():
    """Verify invariant holds under all possible thread schedules."""

    def invariant(session: Session) -> bool:
        return session[Counter].latest().value >= 0

    for trace in explore_all_interleavings(worker_1, worker_2, worker_3):
        assert invariant(session), f"Violated at: {trace}"
```

**Warning**: Exhaustive exploration has exponential complexity. Use only for
small test cases (2-3 threads, few checkpoints).

### State Space Bounds

| Threads | Checkpoints/Thread | Approximate Interleavings |
|---------|-------------------|---------------------------|
| 2 | 3 | ~20 |
| 2 | 5 | ~252 |
| 3 | 3 | ~1,680 |
| 3 | 5 | ~756,756 |

## Property-Based Testing Integration

Combine with Hypothesis for randomized exploration:

```python
from hypothesis import given, settings
from hypothesis import strategies as st

@given(schedule=st.lists(st.sampled_from(["worker_1", "worker_2", "*"])))
@settings(max_examples=1000)
def test_random_schedules(schedule):
    """Test many random schedules."""
    ctx, scheduler = create_test_context()
    scheduler.strategy = ScriptedStrategy(schedule)

    try:
        trace = run_to_completion(ctx, worker_1, worker_2)
        assert_invariant_holds(trace)
    except DeadlockError:
        pass  # Deadlock is acceptable for some schedules
```

## Usage Example

Complete example testing a concurrent counter:

```python
from weakincentives.threads import (
    DeterministicThreadContext,
    Scheduler,
    ScriptedStrategy,
    ThreadTestCase,
)

class TestConcurrentCounter(ThreadTestCase):
    """Test thread-safe counter implementation."""

    def test_increment_decrement_interleaving(self):
        """Test specific interleaving that could cause race."""
        counter = Counter(value=0, ctx=self.ctx)

        def incrementer():
            for _ in range(3):
                counter.increment()

        def decrementer():
            for _ in range(3):
                counter.decrement()

        # Interleave: inc, dec, inc, dec, inc, dec
        self.scheduler.strategy = ScriptedStrategy([
            "incrementer", "decrementer",
            "incrementer", "decrementer",
            "incrementer", "decrementer",
        ])

        trace = self.run_threads(incrementer, decrementer)

        assert counter.value == 0
        self.assert_no_deadlock()

    def test_no_deadlock_any_schedule(self):
        """Verify no deadlock under random schedules."""
        counter = Counter(value=0, ctx=self.ctx)

        self.scheduler.strategy = RandomStrategy(seed=42)

        for _ in range(100):
            trace = self.run_threads(
                lambda: counter.increment(),
                lambda: counter.decrement(),
            )
            self.assert_no_deadlock()
```

### Example with Decorators and Context Managers

Production code using automatic checkpoints:

```python
from weakincentives.threads import (
    checkpointed,
    checkpoint_region,
    get_context,
    ThreadContext,
)

@checkpointed
def fetch_and_process(url: str) -> Result:
    """Entry/exit checkpoints injected automatically."""
    data = fetch(url)
    return process(data)

@checkpointed(name="db_write")
def persist_result(result: Result) -> None:
    """Named checkpoint for precise test control."""
    ctx = get_context()

    with checkpoint_region(ctx, "validate"):
        validate(result)

    with checkpoint_region(ctx, "write"):
        db.write(result)
```

Test using scripted strategy to target specific interleavings:

```python
from weakincentives.threads import (
    DeterministicThreadContext,
    Scheduler,
    ScriptedStrategy,
    context_scope,
)

def test_concurrent_persist():
    scheduler = Scheduler()
    ctx = DeterministicThreadContext(scheduler=scheduler)

    result_1 = Result(id=1, value="first")
    result_2 = Result(id=2, value="second")

    with context_scope(ctx):
        # Script: r1 validates, r2 validates, r1 writes, r2 writes
        # This tests that validation and write are properly atomic
        scheduler.strategy = ScriptedStrategy([
            "persist_1",  # entry checkpoint
            "persist_1",  # validate:enter
            "persist_2",  # entry checkpoint
            "persist_2",  # validate:enter
            "persist_1",  # validate:exit
            "persist_1",  # write:enter
            "persist_2",  # validate:exit
            "persist_2",  # write:enter
            "persist_1",  # write:exit
            "persist_1",  # exit checkpoint
            "persist_2",  # write:exit
            "persist_2",  # exit checkpoint
        ])

        t1 = ctx.spawn(lambda: persist_result(result_1), name="persist_1")
        t2 = ctx.spawn(lambda: persist_result(result_2), name="persist_2")
        t1.start()
        t2.start()

        scheduler.run_until_complete()

        assert db.get(1).value == "first"
        assert db.get(2).value == "second"
```

## Implementation Notes

### Coroutine-Based Deterministic Threads

Deterministic threads can be implemented using Python generators or
`greenlet` for true cooperative multitasking:

```python
class DeterministicThread:
    """Thread implemented as a coroutine for deterministic control."""

    def __init__(self, target: Callable, scheduler: Scheduler):
        self._target = target
        self._scheduler = scheduler
        self._greenlet: greenlet | None = None

    def start(self) -> None:
        self._greenlet = greenlet(self._run)
        self._scheduler.register(self)

    def switch_to(self) -> None:
        """Resume this thread's execution."""
        self._greenlet.switch()

    def yield_control(self) -> None:
        """Yield back to scheduler."""
        self._scheduler.greenlet.switch()
```

### Alternative: No External Dependencies

For environments where `greenlet` is unavailable, use threading with
explicit synchronization:

```python
class DeterministicThread:
    """Deterministic thread using real threading + barriers."""

    def __init__(self, target: Callable, scheduler: Scheduler):
        self._target = target
        self._scheduler = scheduler
        self._proceed = threading.Event()
        self._yielded = threading.Event()
        self._thread = threading.Thread(target=self._run)

    def _run(self) -> None:
        for step in self._execute_with_yields():
            self._yielded.set()
            self._proceed.wait()
            self._proceed.clear()
```

## Limitations

- **GIL**: Python's GIL means true parallelism is limited; this framework
  focuses on testing logical concurrency, not performance
- **External resources**: Cannot control timing of I/O, network, or
  system calls
- **Exhaustive testing**: Only feasible for small state spaces
- **Greenlet dependency**: Full cooperative scheduling requires greenlet;
  fallback mode has higher overhead
- **Native code**: Cannot intercept threading in C extensions

## Related Specifications

- `specs/THREAD_SAFETY.md` - Thread safety guarantees for WINK components
- `specs/LIFECYCLE.md` - LoopGroup thread coordination
- `specs/TESTING.md` - General testing standards
- `specs/FORMAL_VERIFICATION.md` - TLA+ for verifying concurrent algorithms
