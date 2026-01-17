# Thread Testing Framework Specification

## Purpose

Enable deterministic testing of concurrent code with minimal production impact.

**Implementation:** `src/weakincentives/threads/`

## Design Principles

1. **Zero prod cost**: Checkpoints compile to nothing in production
2. **Easy schedule iteration**: Test many interleavings with one test
3. **Automatic deadlock detection**: Fail fast when threads can't progress

## Minimal Production Impact

### The Checkpoint

```python
# weakincentives/threads/__init__.py

_scheduler: Scheduler | None = None

def checkpoint(name: str | None = None) -> None:
    """Yield point. No-op unless scheduler is active."""
    if _scheduler is not None:
        _scheduler.checkpoint(name)
```

**Cost in production**: One `None` check per call. No context variable lookup,
no function indirection, no allocation.

For zero overhead, strip checkpoints entirely with `-O`:

```python
def checkpoint(name: str | None = None) -> None:
    if __debug__ and _scheduler is not None:
        _scheduler.checkpoint(name)
```

Now `python -O` eliminates checkpoint calls completely.

### Production Code

Checkpoints go at state transition boundaries:

```python
from weakincentives.threads import checkpoint

def transfer(src: Account, dst: Account, amount: int) -> None:
    checkpoint("debit")
    src.balance -= amount
    checkpoint("credit")
    dst.balance += amount
```

No test imports, no context passing, no protocol dependencies.

## Test API

### Running a Single Schedule

```python
from weakincentives.threads import run_with_schedule

def test_specific_interleaving():
    def worker_a():
        transfer(A, B, 100)

    def worker_b():
        transfer(A, C, 100)

    run_with_schedule(
        {"a": worker_a, "b": worker_b},
        schedule=["a", "b", "a", "b"],  # Explicit ordering
    )

    assert A.balance == 800
```

### Testing All Interleavings

```python
from weakincentives.threads import run_all_schedules

def test_no_lost_money():
    def worker_a():
        transfer(A, B, 100)

    def worker_b():
        transfer(A, C, 100)

    for result in run_all_schedules({"a": worker_a, "b": worker_b}):
        assert A.balance + B.balance + C.balance == 1000, f"Failed: {result.schedule}"
```

`run_all_schedules` yields each valid schedule. Deadlocks are skipped (or
optionally raised). Exponential in checkpoints—use for small cases.

### Random Schedule Sampling

```python
from weakincentives.threads import run_random_schedules

def test_random_sampling():
    for result in run_random_schedules(workers, iterations=1000, seed=42):
        assert invariant_holds()
```

Reproducible via seed. Good for fuzzing larger state spaces.

## Deadlock Detection

A deadlock occurs when all threads are blocked and none can proceed.

```python
@dataclass
class Deadlock(Exception):
    """All threads blocked, none can proceed."""
    blocked: dict[str, str]  # thread -> what it's waiting for
    schedule_so_far: list[str]
```

Detection is automatic:

```python
def step(self) -> StepResult:
    runnable = [t for t in self.threads if t.can_proceed]
    if not runnable:
        if any(not t.done for t in self.threads):
            raise Deadlock(...)
        return StepResult.ALL_DONE
    # ... pick and run one
```

### Deadlock Handling Options

```python
# Raise on deadlock (default for run_with_schedule)
run_with_schedule(workers, schedule, on_deadlock="raise")

# Skip deadlocked schedules (default for run_all_schedules)
for result in run_all_schedules(workers, on_deadlock="skip"):
    ...

# Collect deadlocks separately
for result in run_all_schedules(workers, on_deadlock="collect"):
    if result.deadlocked:
        print(f"Deadlock at: {result.schedule}")
```

## Implementation

### Scheduler

```python
@dataclass
class Scheduler:
    threads: dict[str, WorkerThread]
    schedule: Iterator[str] | None = None  # None = round-robin
    trace: list[str] = field(default_factory=list)

    def checkpoint(self, name: str | None) -> None:
        """Called by checkpoint(). Pause current thread."""
        current = self._current_thread()
        current.pause()
        self._switch_to_main()

    def run(self) -> ScheduleResult:
        """Run until all threads complete or deadlock."""
        while True:
            match self.step():
                case StepResult.CONTINUE:
                    pass
                case StepResult.ALL_DONE:
                    return ScheduleResult(schedule=self.trace, deadlocked=False)
                case StepResult.DEADLOCK:
                    return ScheduleResult(schedule=self.trace, deadlocked=True)

    def step(self) -> StepResult:
        runnable = [n for n, t in self.threads.items() if t.can_run]
        if not runnable:
            return StepResult.DEADLOCK if self._any_alive() else StepResult.ALL_DONE

        name = self._pick(runnable)
        self.trace.append(name)
        self.threads[name].resume_until_checkpoint()
        return StepResult.CONTINUE
```

### WorkerThread

```python
class WorkerThread:
    def __init__(self, target: Callable, name: str):
        self.name = name
        self._can_run = threading.Event()
        self._paused = threading.Event()
        self._done = False
        self._thread = threading.Thread(target=self._run, args=(target,))

    def _run(self, target: Callable) -> None:
        self._can_run.wait()
        try:
            target()
        finally:
            self._done = True
            self._paused.set()

    @property
    def can_run(self) -> bool:
        return not self._done and self._paused.is_set()

    def resume_until_checkpoint(self) -> None:
        self._paused.clear()
        self._can_run.set()
        self._paused.wait()

    def pause(self) -> None:
        self._paused.set()
        self._can_run.wait()
        self._can_run.clear()
```

### Schedule Enumeration

```python
def run_all_schedules(
    workers: dict[str, Callable],
    on_deadlock: Literal["raise", "skip", "collect"] = "skip",
) -> Iterator[ScheduleResult]:
    """Enumerate all valid schedules via DFS."""

    def explore(state: SchedulerState) -> Iterator[ScheduleResult]:
        runnable = state.runnable_threads()
        if not runnable:
            if state.all_done():
                yield ScheduleResult(state.trace, deadlocked=False)
            elif on_deadlock == "collect":
                yield ScheduleResult(state.trace, deadlocked=True)
            elif on_deadlock == "raise":
                raise Deadlock(state.trace)
            return

        for thread_name in runnable:
            for result in explore(state.after_step(thread_name)):
                yield result

    yield from explore(SchedulerState.initial(workers))
```

## Checkpoint Helpers

For wrapping existing code without modification:

```python
@checkpointed
def existing_function():
    ...  # Checkpoints at entry/exit

@checkpointed("custom_name")
def another_function():
    ...

with checkpoint_region("critical"):
    ...  # Checkpoints at enter/exit
```

## Complete Example

```python
from weakincentives.threads import checkpoint, run_all_schedules

class Account:
    def __init__(self, balance: int):
        self.balance = balance
        self.lock = threading.Lock()

def transfer(src: Account, dst: Account, amount: int) -> None:
    checkpoint("acquire_src")
    with src.lock:
        checkpoint("acquired_src")
        checkpoint("acquire_dst")
        with dst.lock:
            checkpoint("acquired_dst")
            src.balance -= amount
            dst.balance += amount

def test_transfer_preserves_total():
    """Verify money is never lost under any interleaving."""
    A = Account(1000)
    B = Account(0)
    C = Account(0)

    workers = {
        "ab": lambda: transfer(A, B, 100),
        "ac": lambda: transfer(A, C, 100),
    }

    for result in run_all_schedules(workers, on_deadlock="skip"):
        # Reset state for each schedule
        A.balance, B.balance, C.balance = 1000, 0, 0
        # ... re-run would happen here in real impl
        assert A.balance + B.balance + C.balance == 1000

def test_detects_deadlock():
    """Lock ordering bug causes deadlock."""
    A, B = Account(100), Account(100)

    def ab(): transfer(A, B, 10)
    def ba(): transfer(B, A, 10)  # Opposite order!

    results = list(run_all_schedules({"ab": ab, "ba": ba}, on_deadlock="collect"))
    deadlocks = [r for r in results if r.deadlocked]

    assert len(deadlocks) > 0  # Some schedules deadlock
```

## Limitations

- Cannot control scheduling inside native code or system calls
- Exhaustive enumeration is exponential—use for small state spaces
- Real locks still block; checkpoints around acquire/release enable testing

## Related Specifications

- `specs/THREAD_SAFETY.md` - Thread safety guarantees for WINK components
- `specs/LIFECYCLE.md` - LoopGroup thread coordination
