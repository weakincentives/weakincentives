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

"""Core types and protocols for threading primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
A = TypeVar("A")


class CancelledException(Exception):
    """Raised when a task is cancelled via CancellationToken."""

    pass


class Future(Protocol[T_co]):
    """Minimal future interface for submitted work."""

    def result(self, timeout: float | None = None) -> T_co:
        """Wait for and return the result.

        Args:
            timeout: Maximum seconds to wait, or None for no limit.

        Raises:
            TimeoutError: If timeout expires before result is available.
            Exception: If the task raised an exception.
        """
        ...

    def done(self) -> bool:
        """Return True if the task has completed."""
        ...

    def cancel(self) -> bool:
        """Attempt to cancel the task.

        Returns True if cancellation was successful.
        """
        ...


@runtime_checkable
class Executor(Protocol):
    """Protocol for submitting work to be executed.

    Production implementations use thread pools for parallel execution.
    Test implementations run tasks synchronously for deterministic behavior.
    """

    def submit(self, fn: Callable[[], T]) -> Future[T]:
        """Submit a callable for execution.

        Args:
            fn: Zero-argument callable to execute.

        Returns:
            A Future representing the pending result.
        """
        ...

    def map(
        self,
        fn: Callable[[A], T],
        items: Iterable[A],
        *,
        timeout: float | None = None,
    ) -> Iterator[T]:
        """Apply fn to each item, returning results in order.

        Args:
            fn: Function to apply to each item.
            items: Iterable of items to process.
            timeout: Maximum seconds to wait for all results.

        Returns:
            Iterator yielding results in the same order as items.
        """
        ...

    def shutdown(self, *, wait: bool = True) -> None:
        """Shut down the executor.

        Args:
            wait: If True, wait for pending tasks to complete.
        """
        ...


@runtime_checkable
class Gate(Protocol):
    """Protocol for thread signaling (like threading.Event).

    Gates are used for signaling between threads. A gate can be
    opened (set) or closed (clear), and threads can wait for it to open.
    """

    def set(self) -> None:
        """Open the gate, releasing all waiters."""
        ...

    def clear(self) -> None:
        """Close the gate."""
        ...

    def is_set(self) -> bool:
        """Return True if the gate is open."""
        ...

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the gate opens or timeout expires.

        Args:
            timeout: Maximum seconds to wait, or None for no limit.

        Returns:
            True if the gate opened, False if timeout expired.
        """
        ...


@runtime_checkable
class Checkpoint(Protocol):
    """Protocol for cooperative yield points.

    Checkpoints allow long-running tasks to:
    1. Check if they should be cancelled
    2. Yield control to other tasks
    3. Cooperate with the scheduler
    """

    def yield_(self) -> None:
        """Yield control to other tasks.

        In production, this releases the GIL briefly.
        In tests, this can be intercepted to control execution order.
        """
        ...

    def check(self) -> None:
        """Check if the task should continue.

        Raises:
            CancelledException: If cancellation was requested.
        """
        ...

    def is_cancelled(self) -> bool:
        """Return True if cancellation was requested."""
        ...

    @property
    def token(self) -> CancellationToken:
        """The cancellation token for this checkpoint."""
        ...


class CancellationToken(Protocol):
    """Protocol for cooperative cancellation.

    Cancellation tokens allow a parent to request cancellation of a task.
    The task must check the token at yield points to respond to cancellation.
    """

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


@runtime_checkable
class Scheduler(Protocol):
    """Protocol for cooperative task scheduling.

    Schedulers manage the execution of multiple cooperating tasks,
    allowing controlled interleaving for testing.
    """

    def schedule(self, task: Callable[[], T]) -> Future[T]:
        """Schedule a task for execution.

        Args:
            task: Zero-argument callable to execute.

        Returns:
            A Future representing the pending result.
        """
        ...

    def yield_(self) -> None:
        """Yield control to the scheduler."""
        ...

    def run_until_complete(self) -> None:
        """Run scheduled tasks until all complete."""
        ...

    def run_one(self) -> bool:
        """Run the next scheduled task.

        Returns:
            False if the task queue is empty.
        """
        ...


__all__ = [
    "A",
    "CancellationToken",
    "CancelledException",
    "Checkpoint",
    "Executor",
    "Future",
    "Gate",
    "Scheduler",
    "T",
]
