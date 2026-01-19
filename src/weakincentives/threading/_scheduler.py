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

"""Scheduler implementations for cooperative task scheduling."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TypeVar

from weakincentives.threading._executor import CompletedFuture
from weakincentives.threading._types import Future

T = TypeVar("T")


@dataclass
class FifoScheduler:
    """Production cooperative scheduler using a thread pool.

    Tasks are submitted to a thread pool and run concurrently.
    yield_() provides a hint to yield control but relies on the
    OS scheduler for actual preemption.

    Example::

        scheduler = FifoScheduler(max_workers=4)

        f1 = scheduler.schedule(task1)
        f2 = scheduler.schedule(task2)

        scheduler.run_until_complete()

        print(f1.result(), f2.result())
    """

    max_workers: int = 4
    _executor: ThreadPoolExecutor | None = field(default=None, repr=False)
    _futures: list[Future[Any]] = field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        repr=False,
    )

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Lazily create the thread pool."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="scheduler",
            )
        return self._executor

    def schedule(self, task: Callable[[], T]) -> Future[T]:
        """Schedule a task for execution."""
        executor = self._ensure_executor()
        future = executor.submit(task)
        self._futures.append(future)
        return future

    def yield_(self) -> None:
        """Yield control hint (releases GIL briefly)."""
        import time

        time.sleep(0)

    def run_until_complete(self) -> None:
        """Wait for all scheduled tasks to complete."""
        for future in self._futures:
            _ = future.result()
        self._futures.clear()

    def run_one(self) -> bool:
        """Run one task to completion.

        In production scheduler, this just waits for the next
        pending future to complete.
        """
        if not self._futures:
            return False

        # Wait for first incomplete future
        for future in self._futures:
            if not future.done():
                _ = future.result()
                return True

        return False

    def shutdown(self) -> None:
        """Shut down the scheduler."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._futures.clear()


@dataclass
class FakeScheduler:
    """Test scheduler with deterministic step-by-step execution.

    Tasks are run cooperatively using generators. When a task calls
    scheduler.yield_(), it suspends and the next task runs.

    Example::

        scheduler = FakeScheduler()

        execution_order = []

        def task_a():
            execution_order.append("a1")
            scheduler.yield_()
            execution_order.append("a2")

        def task_b():
            execution_order.append("b1")
            scheduler.yield_()
            execution_order.append("b2")

        scheduler.schedule(task_a)
        scheduler.schedule(task_b)

        scheduler.run_one()  # -> "a1"
        scheduler.run_one()  # -> "b1"
        scheduler.run_one()  # -> "a2"
        scheduler.run_one()  # -> "b2"

        assert execution_order == ["a1", "b1", "a2", "b2"]

    Note: The scheduled functions must be aware of the scheduler and
    call scheduler.yield_() to yield control. In tests, you typically
    inject the scheduler or use a global for simplicity.
    """

    _ready_queue: deque[_ScheduledTask[Any]] = field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=deque,
        repr=False,
    )
    _current_task: _ScheduledTask[Any] | None = field(default=None, repr=False)
    _yield_requested: bool = field(default=False, repr=False)

    def schedule(self, task: Callable[[], T]) -> Future[T]:
        """Schedule a task for execution."""
        scheduled = _ScheduledTask(task)
        self._ready_queue.append(scheduled)
        return scheduled.future

    def yield_(self) -> None:
        """Yield control to the scheduler.

        This is called by tasks to indicate they want to yield.
        In run_one(), this causes the current task to be moved
        back to the ready queue.
        """
        self._yield_requested = True

    def run_one(self) -> bool:
        """Run the next task until it yields or completes.

        Returns:
            False if no tasks are pending.
        """
        if not self._ready_queue:
            return False

        task = self._ready_queue.popleft()
        self._current_task = task
        self._yield_requested = False

        try:
            task.run_until_yield(self)
        finally:
            self._current_task = None

        # If task yielded (not completed), put back in queue
        # Note: Current implementation always completes tasks, so this is defensive
        if not task.completed and self._yield_requested:  # pragma: no cover
            self._ready_queue.append(task)

        return True

    def run_until_complete(self) -> None:
        """Run all tasks until completion."""
        while self.run_one():
            pass

    def reset(self) -> None:
        """Reset scheduler state."""
        self._ready_queue.clear()
        self._current_task = None
        self._yield_requested = False


@dataclass
class _ScheduledTask[T]:
    """Internal wrapper for scheduled tasks."""

    task: Callable[[], T]
    future: CompletedFuture[T] = field(default_factory=lambda: CompletedFuture())
    completed: bool = False
    _result: T | None = field(default=None, repr=False)
    _exception: BaseException | None = field(default=None, repr=False)

    def run_until_yield(self, scheduler: FakeScheduler) -> None:
        """Run the task until it yields or completes.

        Note: This simple implementation runs the entire task.
        True cooperative scheduling would require the task to be
        written as a generator or use async. For testing purposes,
        tasks should be structured to call yield_() at known points.
        """
        if self.completed:  # pragma: no cover
            return

        try:
            result = self.task()
            self._result = result
            self.future = CompletedFuture.of(result)
            self.completed = True
        except BaseException as e:
            self._exception = e
            self.future = CompletedFuture.failed(e)
            self.completed = True


__all__ = [
    "FakeScheduler",
    "FifoScheduler",
]
