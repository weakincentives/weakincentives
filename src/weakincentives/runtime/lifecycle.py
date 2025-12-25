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

"""Lifecycle management for graceful shutdown of loops.

This module provides primitives for coordinating graceful shutdown across
multiple MainLoop and EvalLoop instances running in the same process.

Example::

    from weakincentives.runtime import LoopGroup, ShutdownCoordinator

    # Run multiple loops with coordinated shutdown
    group = LoopGroup(loops=[main_loop, eval_loop])
    group.run()  # Blocks until SIGTERM/SIGINT

    # Or manual coordination
    coordinator = ShutdownCoordinator.install()
    coordinator.register(loop.shutdown)
    loop.run()
"""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol, Self

if TYPE_CHECKING:
    from types import FrameType


class Runnable(Protocol):
    """Protocol for loops that support graceful shutdown.

    Both MainLoop and EvalLoop implement this protocol, enabling them to be
    managed by LoopGroup and ShutdownCoordinator.
    """

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        """Run the loop, processing messages until stopped.

        Exits when:
        - max_iterations reached
        - shutdown() called
        - Mailbox closed

        Args:
            max_iterations: Maximum polling iterations (None = unlimited).
            visibility_timeout: Seconds messages remain invisible during
                processing. Must exceed maximum expected execution time.
            wait_time_seconds: Long poll duration (0-20 seconds).
        """
        ...

    def shutdown(self, *, timeout: float = 30.0) -> bool:
        """Request graceful shutdown.

        Sets the shutdown flag and waits for in-flight work to complete.
        Returns when the loop has stopped or timeout expires.

        Args:
            timeout: Maximum seconds to wait for in-flight work.

        Returns:
            True if loop stopped cleanly, False if timeout expired.
        """
        ...

    @property
    def running(self) -> bool:
        """True if the loop is currently processing messages."""
        ...

    def __enter__(self) -> Self:
        """Context manager entry (no-op, returns self)."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit triggers shutdown."""
        ...


class ShutdownCoordinator:
    """Coordinates graceful shutdown across multiple loops.

    Installs signal handlers for SIGTERM and SIGINT. When a signal arrives,
    all registered callbacks are invoked. Thread-safe for concurrent
    registration and signal delivery.

    Example::

        coordinator = ShutdownCoordinator.install()
        coordinator.register(loop.shutdown)
        loop.run()
    """

    _instance: ShutdownCoordinator | None = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the coordinator.

        Use ShutdownCoordinator.install() to get the singleton instance
        with signal handlers installed.
        """
        super().__init__()
        self._callbacks: list[Callable[[], object]] = []
        self._callbacks_lock = threading.Lock()
        self._triggered = threading.Event()

    @classmethod
    def install(
        cls,
        *,
        signals: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT),
    ) -> ShutdownCoordinator:
        """Install signal handlers and return the coordinator.

        Safe to call multiple times; returns the same instance. Signal
        handlers are installed only once.

        Args:
            signals: Signals to handle. Defaults to SIGTERM and SIGINT.

        Returns:
            The singleton ShutdownCoordinator instance.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
                for sig in signals:
                    _ = signal.signal(sig, cls._instance._handle_signal)
            return cls._instance

    @classmethod
    def get(cls) -> ShutdownCoordinator | None:
        """Return the installed coordinator, or None if not installed."""
        with cls._instance_lock:
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance.

        Primarily for testing. Does not restore original signal handlers.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance._triggered.clear()
                with cls._instance._callbacks_lock:
                    cls._instance._callbacks.clear()
            cls._instance = None

    def register(self, callback: Callable[[], object]) -> None:
        """Register a callback to invoke on shutdown.

        Callbacks are invoked in registration order. If shutdown has
        already been triggered, the callback is invoked immediately.

        Args:
            callback: Zero-argument callable (typically loop.shutdown).
        """
        with self._callbacks_lock:
            self._callbacks.append(callback)
            if self._triggered.is_set():
                _ = callback()

    def unregister(self, callback: Callable[[], object]) -> None:
        """Remove a callback from the shutdown list.

        Args:
            callback: Previously registered callback.
        """
        with self._callbacks_lock, contextlib.suppress(ValueError):
            self._callbacks.remove(callback)

    def trigger(self) -> None:
        """Manually trigger shutdown (for testing or programmatic control)."""
        self._triggered.set()
        with self._callbacks_lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            _ = callback()

    @property
    def triggered(self) -> bool:
        """True if shutdown has been triggered."""
        return self._triggered.is_set()

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Signal handler that triggers shutdown."""
        _ = (signum, frame)
        self.trigger()


class LoopGroup:
    """Coordinates lifecycle of multiple loops.

    Runs each loop in a separate thread and handles coordinated shutdown.
    Integrates with ShutdownCoordinator for signal-driven termination.

    Example::

        group = LoopGroup(loops=[main_loop, eval_loop])
        group.run()  # Blocks until shutdown signal or all loops exit

    Example with context manager::

        with LoopGroup(loops=[main_loop, eval_loop]) as group:
            group.run()
        # Shutdown triggered on context exit
    """

    def __init__(
        self,
        loops: Sequence[Runnable],
        *,
        shutdown_timeout: float = 30.0,
    ) -> None:
        """Initialize the LoopGroup.

        Args:
            loops: Sequence of Runnable loops to coordinate.
            shutdown_timeout: Maximum seconds to wait for each loop during shutdown.
        """
        super().__init__()
        self.loops = loops
        self.shutdown_timeout = shutdown_timeout
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future[None]] = []
        self._shutdown_event = threading.Event()

    def run(
        self,
        *,
        install_signals: bool = True,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        """Run all loops until shutdown.

        Blocks until all loops exit or shutdown is triggered. Each loop
        runs in a dedicated thread.

        Args:
            install_signals: If True, install SIGTERM/SIGINT handlers.
            visibility_timeout: Passed to each loop's run() method.
            wait_time_seconds: Passed to each loop's run() method.
        """
        if install_signals:
            coordinator = ShutdownCoordinator.install()
            coordinator.register(self._trigger_shutdown)

        self._executor = ThreadPoolExecutor(
            max_workers=len(self.loops),
            thread_name_prefix="loop-worker",
        )

        try:
            for loop in self.loops:
                future: Future[None] = self._executor.submit(
                    loop.run,
                    visibility_timeout=visibility_timeout,
                    wait_time_seconds=wait_time_seconds,
                )
                self._futures.append(future)

            # Wait for all loops to complete
            for future in self._futures:
                future.result()

        finally:
            self._executor.shutdown(wait=True)
            self._futures.clear()

    def shutdown(self, *, timeout: float | None = None) -> bool:
        """Shutdown all loops gracefully.

        Args:
            timeout: Maximum seconds to wait per loop. Defaults to shutdown_timeout.

        Returns:
            True if all loops stopped cleanly, False if any timeout expired.
        """
        effective_timeout = timeout if timeout is not None else self.shutdown_timeout
        self._shutdown_event.set()

        results: list[bool] = []
        for loop in self.loops:
            result = loop.shutdown(timeout=effective_timeout)
            results.append(result)

        return all(results)

    def _trigger_shutdown(self) -> bool:
        """Internal callback for ShutdownCoordinator."""
        return self.shutdown()

    def __enter__(self) -> Self:
        """Context manager entry. Returns self for use in with statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit. Triggers shutdown and waits for completion."""
        _ = (exc_type, exc_val, exc_tb)
        _ = self.shutdown()


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float,
    poll_interval: float = 0.1,
) -> bool:
    """Wait until predicate returns True or timeout expires.

    Args:
        predicate: Zero-argument callable that returns True when done.
        timeout: Maximum seconds to wait.
        poll_interval: Seconds between predicate checks.

    Returns:
        True if predicate returned True, False if timeout expired.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return predicate()


__all__ = [
    "LoopGroup",
    "Runnable",
    "ShutdownCoordinator",
    "wait_until",
]
