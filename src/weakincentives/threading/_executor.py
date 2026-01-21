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

"""Executor implementations for thread pool abstraction."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TypeVar

from weakincentives.threading._types import Future

T = TypeVar("T")
A = TypeVar("A")


@dataclass
class CompletedFuture[T]:
    """A future that is already complete with a value or exception."""

    _value: T | None = None
    _exception: BaseException | None = None

    @classmethod
    def of(cls, value: T) -> CompletedFuture[T]:
        """Create a completed future with a value."""
        return CompletedFuture[T](_value=value)

    @classmethod
    def failed(cls, exc: BaseException) -> CompletedFuture[T]:
        """Create a completed future with an exception."""
        return CompletedFuture[T](_exception=exc)

    def result(self, timeout: float | None = None) -> T:
        """Return the result or raise the stored exception."""
        del timeout  # unused
        if self._exception is not None:
            raise self._exception
        # Safe because either _value or _exception is set
        return self._value  # type: ignore[return-value]

    def done(self) -> bool:
        """Always returns True since this future is complete."""
        return True

    def cancel(self) -> bool:
        """Cannot cancel a completed future."""
        return False


@dataclass
class SystemExecutor:
    """Production executor using ThreadPoolExecutor.

    Example::

        with SystemExecutor(max_workers=4) as executor:
            futures = [executor.submit(task) for task in tasks]
            results = [f.result() for f in futures]

    Or use the module singleton::

        from weakincentives.threading import SYSTEM_EXECUTOR

        future = SYSTEM_EXECUTOR.submit(lambda: expensive_computation())
    """

    max_workers: int | None = None
    thread_name_prefix: str = "worker"
    _executor: ThreadPoolExecutor | None = field(default=None, repr=False)

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Lazily create the thread pool."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix=self.thread_name_prefix,
            )
        return self._executor

    def submit(self, fn: Callable[[], T]) -> Future[T]:
        """Submit a callable for execution in the thread pool."""
        executor = self._ensure_executor()
        return executor.submit(fn)

    def map(
        self,
        fn: Callable[[A], T],
        items: Iterable[A],
        *,
        timeout: float | None = None,
    ) -> Iterator[T]:
        """Apply fn to each item concurrently."""
        executor = self._ensure_executor()
        return executor.map(fn, items, timeout=timeout)

    def shutdown(self, *, wait: bool = True) -> None:
        """Shut down the executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def __enter__(self) -> SystemExecutor:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit - shuts down executor."""
        self.shutdown(wait=True)


@dataclass
class FakeExecutor:
    """Test executor that runs tasks synchronously.

    Tasks are executed immediately in the calling thread when submitted,
    eliminating race conditions and enabling deterministic testing.

    Example::

        executor = FakeExecutor()

        future = executor.submit(lambda: 42)
        assert future.done()  # Already complete
        assert future.result() == 42

        # Track submissions for assertions
        assert len(executor.submitted) == 1
    """

    submitted: list[Callable[[], object]] = field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
    )
    _shutdown: bool = field(default=False, repr=False)

    def submit(self, fn: Callable[[], T]) -> Future[T]:
        """Execute fn immediately and return a completed Future."""
        if self._shutdown:
            msg = "Executor has been shut down"
            raise RuntimeError(msg)

        self.submitted.append(fn)
        try:
            result = fn()
            return CompletedFuture.of(result)
        except BaseException as e:
            return CompletedFuture.failed(e)

    def map(
        self,
        fn: Callable[[A], T],
        items: Iterable[A],
        *,
        timeout: float | None = None,
    ) -> Iterator[T]:
        """Apply fn to each item sequentially."""
        del timeout  # unused in fake
        if self._shutdown:
            msg = "Executor has been shut down"
            raise RuntimeError(msg)

        for item in items:
            yield fn(item)

    def shutdown(self, *, wait: bool = True) -> None:
        """Mark executor as shut down (no-op for resources)."""
        del wait  # unused
        self._shutdown = True

    def __enter__(self) -> FakeExecutor:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.shutdown(wait=True)

    def reset(self) -> None:
        """Reset state for reuse in tests."""
        self.submitted.clear()
        self._shutdown = False


__all__ = [
    "CompletedFuture",
    "FakeExecutor",
    "SystemExecutor",
]
