# Thread Testing Framework Specification

## Purpose

Enable deterministic testing of concurrent code by controlling thread
interleaving.

**Implementation:** `src/weakincentives/threads/`

## Core Insight

Thread bugs come from uncontrolled interleaving. The fix: make interleaving
controllable. In production, threads run normally. In tests, a scheduler
decides which thread runs next.

The entire framework reduces to one primitive: **yield points** where threads
can be paused and resumed.

## Minimal Design

### The Checkpoint

A checkpoint is a point where a thread yields control:

```python
def checkpoint(name: str | None = None) -> None:
    """Yield control to scheduler. No-op in production."""
    ctx = _current_context.get()
    if ctx is not None:
        ctx.yield_control(name)
```

That's it. Production code sprinkles `checkpoint()` calls at interesting
points. In production, they do nothing. In tests, they yield to a scheduler.

### Thread Context

```python
@dataclass
class ThreadContext:
    """Test context that controls thread scheduling."""

    scheduler: Scheduler

    def yield_control(self, name: str | None = None) -> None:
        """Pause current thread, let scheduler pick next."""
        self.scheduler.yield_from_thread(name)
```

Production code doesn't need a context—`checkpoint()` is just a no-op.

### Scheduler

```python
@dataclass
class Scheduler:
    """Controls which thread runs next."""

    threads: dict[str, DeterministicThread] = field(default_factory=dict)
    runnable: deque[str] = field(default_factory=deque)
    script: list[str] | None = None  # If set, follow this order
    position: int = 0

    def yield_from_thread(self, checkpoint_name: str | None) -> None:
        """Current thread yields. Switch to next."""
        # Record checkpoint, switch to scheduler greenlet
        ...

    def step(self) -> bool:
        """Run one thread until it yields. Returns False if all done."""
        ...

    def run(self) -> None:
        """Run all threads to completion."""
        while self.step():
            pass
```

### Scripted Execution

The key testing capability: run threads in a specific order.

```python
scheduler = Scheduler(script=[
    "writer",   # Run writer until it yields
    "reader",   # Run reader until it yields
    "writer",   # Run writer again
])
```

This lets you test exact interleavings that trigger bugs.

## Usage

### Production Code

Add checkpoints at state transitions:

```python
def transfer(from_acct: Account, to_acct: Account, amount: int) -> None:
    checkpoint("before_debit")
    from_acct.balance -= amount

    checkpoint("between")  # Interesting point: money is "in flight"

    checkpoint("before_credit")
    to_acct.balance += amount
```

### Test Code

```python
def test_concurrent_transfer_race():
    """Test interleaving that could lose money."""
    ctx = ThreadContext(scheduler=Scheduler(script=[
        "t1",  # t1: debit from A
        "t2",  # t2: debit from A (reads stale balance!)
        "t1",  # t1: credit to B
        "t2",  # t2: credit to B
    ]))

    with thread_context(ctx):
        t1 = spawn(lambda: transfer(A, B, 100), name="t1")
        t2 = spawn(lambda: transfer(A, B, 100), name="t2")

        ctx.scheduler.run()

        # With proper locking: A=800, B=200
        # Without: depends on interleaving
        assert A.balance + B.balance == 1000  # Money conserved?
```

## Checkpoint Helpers

### Decorator

```python
@checkpointed
def process(item: Item) -> Result:
    # Checkpoint at entry and exit
    return transform(item)
```

### Context Manager

```python
with checkpoint_region("critical"):
    # Checkpoints at enter and exit
    do_work()
```

Both are thin wrappers over `checkpoint()`.

## Implementation

Use `greenlet` for cooperative switching:

```python
class DeterministicThread:
    def __init__(self, target: Callable, name: str, scheduler: Scheduler):
        self.greenlet = greenlet(target)
        self.name = name
        self.scheduler = scheduler

    def switch_to(self) -> None:
        self.greenlet.switch()

    def yield_control(self) -> None:
        self.scheduler.main_greenlet.switch()
```

Without greenlet, fall back to real threads with Event synchronization (higher
overhead but no dependencies).

## What This Spec Excludes

Intentionally deferred:

- **Wrapped primitives** (Lock, Event, etc.): Use real ones. Checkpoints
  around lock acquire/release are sufficient for most tests.
- **Exhaustive exploration**: Build on scripted execution if needed.
- **Race detection**: Analyze traces offline if needed.
- **Thread pools**: Spawn individual threads.

These can be added later if proven necessary.

## Limitations

- Cannot control scheduling inside native code or system calls
- Requires explicit checkpoint placement (or decorator instrumentation)
- Greenlet required for efficient implementation

## Related Specifications

- `specs/THREAD_SAFETY.md` - Thread safety guarantees for WINK components
- `specs/LIFECYCLE.md` - LoopGroup thread coordination
