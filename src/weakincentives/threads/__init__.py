# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyright: reportPrivateUsage=false
# pyright: reportUnusedCallResult=false

"""Thread testing framework for deterministic concurrency testing.

This module provides tools for testing concurrent code by controlling
thread interleaving. In production, checkpoints are no-ops. In tests,
a scheduler controls which thread runs next.

Example:
    Production code adds checkpoints at state transitions::

        from weakincentives.threads import checkpoint

        def transfer(src: Account, dst: Account, amount: int) -> None:
            checkpoint("debit")
            src.balance -= amount
            checkpoint("credit")
            dst.balance += amount

    Test code controls interleaving::

        from weakincentives.threads import run_with_schedule

        def test_transfer():
            run_with_schedule(
                {"t1": lambda: transfer(A, B, 100),
                 "t2": lambda: transfer(A, C, 100)},
                schedule=["t1", "t2", "t1", "t2"],
            )
            assert A.balance == 800
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import wraps
from typing import Literal, ParamSpec, TypeVar, overload

from ._scheduler import Scheduler
from ._types import Deadlock, ScheduleResult, StepResult
from ._worker import WorkerThread

# Thread-local storage for current scheduler
_scheduler_var: threading.local = threading.local()


def _get_scheduler() -> Scheduler | None:
    """Get the current thread's scheduler, if any."""
    return getattr(_scheduler_var, "scheduler", None)


def _set_scheduler(scheduler: Scheduler | None) -> None:
    """Set the current thread's scheduler."""
    _scheduler_var.scheduler = scheduler


def checkpoint(name: str | None = None) -> None:
    """Yield point for deterministic scheduling.

    In production (no scheduler active), this is a no-op.
    In tests, yields control to the scheduler.

    Args:
        name: Optional name for this checkpoint (for debugging).
    """
    scheduler = _get_scheduler()
    if scheduler is not None:
        scheduler.checkpoint(name)


@contextmanager
def _scheduler_context(scheduler: Scheduler) -> Iterator[None]:
    """Context manager that installs a scheduler for the current thread."""
    old = _get_scheduler()
    _set_scheduler(scheduler)
    try:
        yield
    finally:
        _set_scheduler(old)


def run_with_schedule(
    workers: dict[str, Callable[[], None]],
    schedule: Sequence[str] | None = None,
    *,
    on_deadlock: Literal["raise", "return"] = "raise",
) -> ScheduleResult:
    """Run workers with a specific schedule.

    Args:
        workers: Dict mapping thread names to callables.
        schedule: Explicit ordering of thread names. Use "*" for any runnable.
            If None, uses round-robin.
        on_deadlock: How to handle deadlock. "raise" raises Deadlock,
            "return" returns a ScheduleResult with deadlocked=True.

    Returns:
        ScheduleResult with execution trace.

    Raises:
        Deadlock: If on_deadlock="raise" and threads deadlock.

    Example::

        result = run_with_schedule(
            {"a": worker_a, "b": worker_b},
            schedule=["a", "b", "a", "b"],
        )
    """
    scheduler = Scheduler(schedule=schedule)

    # Wrap each worker to install scheduler context
    def make_wrapped(target: Callable[[], None]) -> Callable[[], None]:
        def wrapped() -> None:
            with _scheduler_context(scheduler):
                target()

        return wrapped

    for name, target in workers.items():
        scheduler.register(name, make_wrapped(target))

    return scheduler.run(on_deadlock=on_deadlock)


def run_all_schedules(  # noqa: C901, PLR0912
    workers: dict[str, Callable[[], None]],
    *,
    on_deadlock: Literal["raise", "skip", "collect"] = "skip",
    max_schedules: int | None = None,
) -> Iterator[ScheduleResult]:
    """Enumerate all possible thread interleavings.

    Warning: This is exponential in the number of checkpoints.
    Use only for small test cases.

    Args:
        workers: Dict mapping thread names to callables.
        on_deadlock: How to handle deadlocks.
            "raise": Raise Deadlock exception.
            "skip": Skip deadlocked schedules (don't yield them).
            "collect": Yield deadlocked schedules with deadlocked=True.
        max_schedules: Maximum number of schedules to explore (None = unlimited).

    Yields:
        ScheduleResult for each explored schedule.

    Example::

        for result in run_all_schedules(workers, on_deadlock="skip"):
            assert invariant_holds(), f"Failed: {result.schedule}"
    """
    count = 0

    def should_continue() -> bool:
        nonlocal count
        if max_schedules is not None and count >= max_schedules:
            return False
        count += 1
        return True

    # Use iterative DFS with explicit stack to avoid deep recursion
    # Each stack entry is (trace_so_far, next_thread_index_to_try)
    stack: list[tuple[list[str], int]] = [([], 0)]

    while stack:
        if not should_continue():
            return

        trace, _next_idx = stack.pop()

        # Create fresh scheduler and replay trace
        scheduler = Scheduler()

        def make_wrapped(
            target: Callable[[], None], sched: Scheduler
        ) -> Callable[[], None]:
            def wrapped() -> None:
                with _scheduler_context(sched):
                    target()

            return wrapped

        for name, target in workers.items():
            scheduler.register(name, make_wrapped(target, scheduler))

        # Start all threads
        for worker in scheduler.threads.values():
            worker.start()

        # Replay trace
        for name in trace:
            scheduler._current_thread_name = name
            scheduler._trace.append(name)
            scheduler.threads[name].resume_until_checkpoint()

        # Find runnable threads
        runnable = [n for n, t in scheduler.threads.items() if t.can_run]

        if not runnable:
            all_done = all(t.done for t in scheduler.threads.values())
            if all_done:
                yield ScheduleResult(
                    schedule=tuple(trace),
                    deadlocked=False,
                    checkpoints=tuple(scheduler._checkpoints),
                )
            elif (
                on_deadlock == "collect"
            ):  # pragma: no cover - requires blocked threads
                yield ScheduleResult(
                    schedule=tuple(trace),
                    deadlocked=True,
                    checkpoints=tuple(scheduler._checkpoints),
                )
            elif on_deadlock == "raise":  # pragma: no cover - requires blocked threads
                blocked = {
                    n: "waiting" for n, t in scheduler.threads.items() if not t.done
                }
                # Clean up before raising
                for worker in scheduler.threads.values():
                    worker._can_run.set()
                    if worker._thread:
                        worker._thread.join(timeout=0.1)
                raise Deadlock(blocked=blocked, schedule_so_far=tuple(trace))
            # skip: don't yield

            # Clean up threads
            for worker in scheduler.threads.values():
                worker._can_run.set()
                if worker._thread:  # pragma: no branch
                    worker._thread.join(timeout=0.1)
            continue

        # Push branches for each runnable thread (in reverse order for DFS)
        stack.extend(
            [([*trace, runnable[i]], 0) for i in range(len(runnable) - 1, -1, -1)]
        )

        # Clean up threads for this iteration
        for worker in scheduler.threads.values():
            worker._can_run.set()
            if worker._thread:  # pragma: no branch
                worker._thread.join(timeout=0.1)


def _handle_random_deadlock(
    scheduler: Scheduler,
    trace: list[str],
    checkpoints: list[str | None],
    on_deadlock: Literal["raise", "skip", "collect"],
) -> ScheduleResult | None:
    """Handle deadlock in random schedule. Returns result to yield or None."""
    all_done = all(t.done for t in scheduler.threads.values())
    if all_done:
        return ScheduleResult(
            schedule=tuple(trace),
            deadlocked=False,
            checkpoints=tuple(checkpoints),
        )
    if on_deadlock == "collect":  # pragma: no cover - requires blocked threads
        return ScheduleResult(
            schedule=tuple(trace),
            deadlocked=True,
            checkpoints=tuple(checkpoints),
        )
    if on_deadlock == "raise":  # pragma: no cover - requires blocked threads
        blocked = {n: "waiting" for n, t in scheduler.threads.items() if not t.done}
        _cleanup_threads(scheduler)
        raise Deadlock(blocked=blocked, schedule_so_far=tuple(trace))
    # skip: return None to not yield
    return None  # pragma: no cover - requires blocked threads


def _cleanup_threads(scheduler: Scheduler) -> None:
    """Clean up worker threads."""
    for worker in scheduler.threads.values():
        worker._can_run.set()
        if worker._thread:  # pragma: no branch
            worker._thread.join(timeout=0.1)


def run_random_schedules(  # noqa: C901
    workers: dict[str, Callable[[], None]],
    *,
    iterations: int = 100,
    seed: int | None = None,
    on_deadlock: Literal["raise", "skip", "collect"] = "skip",
) -> Iterator[ScheduleResult]:
    """Run workers with random schedules.

    Good for fuzzing larger state spaces where exhaustive
    enumeration is infeasible.

    Args:
        workers: Dict mapping thread names to callables.
        iterations: Number of random schedules to try.
        seed: Random seed for reproducibility.
        on_deadlock: How to handle deadlocks.

    Yields:
        ScheduleResult for each iteration.

    Example::

        for result in run_random_schedules(workers, iterations=1000, seed=42):
            assert invariant_holds()
    """
    rng = random.Random(seed)  # nosec B311 - not used for security

    for _ in range(iterations):
        scheduler = Scheduler()

        def make_wrapped(
            target: Callable[[], None], sched: Scheduler
        ) -> Callable[[], None]:
            def wrapped() -> None:
                with _scheduler_context(sched):
                    target()

            return wrapped

        for name, target in workers.items():
            scheduler.register(name, make_wrapped(target, scheduler))

        # Start all threads
        for worker in scheduler.threads.values():
            worker.start()

        # Run with random selection
        trace: list[str] = []
        checkpoints: list[str | None] = []

        while True:
            runnable = [n for n, t in scheduler.threads.items() if t.can_run]

            if not runnable:
                result = _handle_random_deadlock(
                    scheduler, trace, checkpoints, on_deadlock
                )
                if result is not None:  # pragma: no branch
                    yield result
                break

            # Random selection
            name = rng.choice(runnable)
            trace.append(name)
            scheduler._current_thread_name = name
            scheduler.threads[name].resume_until_checkpoint()
            checkpoints.append(scheduler.threads[name]._current_checkpoint)

        _cleanup_threads(scheduler)


_P = ParamSpec("_P")
_T = TypeVar("_T")


@overload
def checkpointed(fn: Callable[_P, _T]) -> Callable[_P, _T]: ...  # noqa: UP047


@overload
def checkpointed(fn: str) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


@overload
def checkpointed(
    fn: None = None, *, name: str | None = None
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


def checkpointed(  # noqa: UP047
    fn: Callable[_P, _T] | str | None = None,
    *,
    name: str | None = None,
) -> Callable[_P, _T] | Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Decorator that adds checkpoints at function entry and exit.

    Args:
        fn: Function to wrap, or checkpoint name if used as @checkpointed("name").
        name: Explicit checkpoint name (defaults to function name).

    Example::

        @checkpointed
        def process(item):
            ...  # Checkpoints at entry and exit

        @checkpointed("custom_name")
        def another():
            ...
    """
    # Handle @checkpointed("name") syntax
    if isinstance(fn, str):
        checkpoint_str = fn

        def decorator(f: Callable[_P, _T]) -> Callable[_P, _T]:
            return _wrap_with_checkpoints(f, checkpoint_str)

        return decorator

    # Handle @checkpointed() or @checkpointed(name="...")
    if fn is None:

        def decorator(f: Callable[_P, _T]) -> Callable[_P, _T]:
            return _wrap_with_checkpoints(f, name)

        return decorator

    # Handle @checkpointed (bare decorator)
    return _wrap_with_checkpoints(fn, name)


def _wrap_with_checkpoints(  # noqa: UP047
    fn: Callable[_P, _T], checkpoint_name: str | None
) -> Callable[_P, _T]:
    """Internal helper to wrap a function with checkpoints."""
    actual_name = checkpoint_name or getattr(fn, "__name__", "anonymous")

    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        checkpoint(f"{actual_name}:enter")
        try:
            return fn(*args, **kwargs)
        finally:
            checkpoint(f"{actual_name}:exit")

    return wrapper


@contextmanager
def checkpoint_region(name: str) -> Iterator[None]:
    """Context manager that emits checkpoints on entry and exit.

    Args:
        name: Name for the checkpoint region.

    Example::

        with checkpoint_region("critical"):
            # checkpoint("critical:enter") called here
            do_work()
            # checkpoint("critical:exit") called at end
    """
    checkpoint(f"{name}:enter")
    try:
        yield
    finally:
        checkpoint(f"{name}:exit")


__all__ = [
    "Deadlock",
    "ScheduleResult",
    "Scheduler",
    "StepResult",
    "WorkerThread",
    "checkpoint",
    "checkpoint_region",
    "checkpointed",
    "run_all_schedules",
    "run_random_schedules",
    "run_with_schedule",
]
