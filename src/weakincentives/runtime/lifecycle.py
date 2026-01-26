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
multiple AgentLoop and EvalLoop instances running in the same process.

Example::

    from weakincentives.runtime import LoopGroup, ShutdownCoordinator

    # Run multiple loops with coordinated shutdown
    group = LoopGroup(loops=[agent_loop, eval_loop])
    group.run()  # Blocks until SIGTERM/SIGINT

    # With health endpoints and watchdog
    group = LoopGroup(
        loops=[agent_loop],
        health_port=8080,
        watchdog_threshold=720.0,
    )
    group.run()

    # Or manual coordination
    coordinator = ShutdownCoordinator.install()
    coordinator.register(loop.shutdown)
    loop.run()

Health endpoints::

    # Run loops with health endpoint for Kubernetes probes
    group = LoopGroup(loops=[agent_loop], health_port=8080)
    group.run()  # GET /health/live and /health/ready available
"""

from __future__ import annotations

import contextlib
import signal
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol, Self

from ..clock import SYSTEM_CLOCK, Clock
from .watchdog import HealthServer, Heartbeat, Watchdog

if TYPE_CHECKING:
    from types import FrameType


class Runnable(Protocol):
    """Protocol for loops that support graceful shutdown.

    Both AgentLoop and EvalLoop implement this protocol, enabling them to be
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

    @property
    def heartbeat(self) -> Heartbeat | None:
        """Heartbeat tracker for watchdog monitoring.

        Returns None if the loop does not support heartbeat monitoring.
        Loops that support monitoring should return a Heartbeat instance
        and call ``beat()`` at regular intervals during processing.
        """
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

    Features:
        - Health endpoints for Kubernetes liveness/readiness probes
        - Watchdog monitoring to detect and terminate stuck workers
        - Coordinated graceful shutdown via signals

    Example::

        group = LoopGroup(loops=[agent_loop, eval_loop])
        group.run()  # Blocks until shutdown signal or all loops exit

    Example with health and watchdog::

        group = LoopGroup(
            loops=[agent_loop],
            health_port=8080,
            watchdog_threshold=720.0,
        )
        group.run()

    Example with context manager::

        with LoopGroup(loops=[agent_loop, eval_loop]) as group:
            group.run()
        # Shutdown triggered on context exit

    Example with health endpoint::

        group = LoopGroup(loops=[agent_loop], health_port=8080)
        group.run()  # GET /health/live and /health/ready available
    """

    def __init__(  # noqa: PLR0913
        self,
        loops: Sequence[Runnable],
        *,
        shutdown_timeout: float = 30.0,
        health_port: int | None = None,
        health_host: str = "0.0.0.0",  # nosec B104 - bind to all interfaces for k8s
        watchdog_threshold: float | None = 720.0,
        watchdog_interval: float = 60.0,
    ) -> None:
        """Initialize the LoopGroup.

        Args:
            loops: Sequence of Runnable loops to coordinate.
            shutdown_timeout: Maximum seconds to wait for each loop during shutdown.
            health_port: Port for health endpoints. None disables health server.
            health_host: Host for health endpoint. Defaults to all interfaces.
            watchdog_threshold: Seconds without heartbeat before process termination.
                None disables watchdog. Default 720s (12 min) calibrated for
                10-minute prompt evaluations.
            watchdog_interval: Seconds between watchdog checks. Default 60s.
        """
        super().__init__()
        self.loops = loops
        self.shutdown_timeout = shutdown_timeout
        self._health_port = health_port
        self._health_host = health_host
        self._watchdog_threshold = watchdog_threshold
        self._watchdog_interval = watchdog_interval
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future[None]] = []
        self._health_server: HealthServer | None = None
        self._watchdog: Watchdog | None = None

    def run(  # noqa: C901
        self,
        *,
        install_signals: bool = True,
        visibility_timeout: int = 1800,
        wait_time_seconds: int = 30,
    ) -> None:
        """Run all loops until shutdown.

        Blocks until all loops exit or shutdown is triggered. Each loop
        runs in a dedicated thread.

        Args:
            install_signals: If True, install SIGTERM/SIGINT handlers.
            visibility_timeout: Passed to each loop's run() method.
                Default 1800s (30 min) calibrated for 10-min evaluations.
            wait_time_seconds: Passed to each loop's run() method.
                Default 30s for long poll.
        """
        if install_signals:
            coordinator = ShutdownCoordinator.install()
            coordinator.register(self._trigger_shutdown)

        # Collect heartbeats from loops that support them
        heartbeats: list[Heartbeat] = []
        loop_names: list[str] = []
        for i, loop in enumerate(self.loops):
            hb = getattr(loop, "heartbeat", None)
            if hb is not None:
                heartbeats.append(hb)
                loop_names.append(getattr(loop, "name", f"loop-{i}"))

        # Start watchdog if configured and loops support heartbeats
        if self._watchdog_threshold is not None and heartbeats:
            self._watchdog = Watchdog(
                heartbeats,
                stall_threshold=self._watchdog_threshold,
                check_interval=self._watchdog_interval,
                loop_names=loop_names,
            )
            self._watchdog.start()

        # Start health server if configured
        if self._health_port is not None:
            self._health_server = HealthServer(
                host=self._health_host,
                port=self._health_port,
                readiness_check=self._build_readiness_check(heartbeats),
            )
            self._health_server.start()

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
            if self._health_server is not None:
                self._health_server.stop()
                self._health_server = None
            if self._watchdog is not None:
                self._watchdog.stop()
                self._watchdog = None
            self._executor.shutdown(wait=True)
            self._futures.clear()

    def _build_readiness_check(
        self,
        heartbeats: list[Heartbeat],
    ) -> Callable[[], bool]:
        """Build readiness check incorporating heartbeat freshness."""
        threshold = self._watchdog_threshold

        def check() -> bool:
            # All loops must be running
            if not all(loop.running for loop in self.loops):
                return False

            # If watchdog is configured, heartbeats must be fresh
            if threshold is not None:
                for hb in heartbeats:
                    if hb.elapsed() > threshold:
                        return False

            return True

        return check

    def shutdown(self, *, timeout: float | None = None) -> bool:
        """Shutdown all loops gracefully.

        Args:
            timeout: Maximum seconds to wait per loop. Defaults to shutdown_timeout.

        Returns:
            True if all loops stopped cleanly, False if any timeout expired.
        """
        effective_timeout = timeout if timeout is not None else self.shutdown_timeout
        return all(loop.shutdown(timeout=effective_timeout) for loop in self.loops)

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
    clock: Clock = SYSTEM_CLOCK,
) -> bool:
    """Wait until predicate returns True or timeout expires.

    Args:
        predicate: Zero-argument callable that returns True when done.
        timeout: Maximum seconds to wait.
        poll_interval: Seconds between predicate checks.
        clock: Clock for time operations. Defaults to system clock.
            Inject TestClock for deterministic testing.

    Returns:
        True if predicate returned True, False if timeout expired.

    Example (testing)::

        from weakincentives.clock import TestClock

        clock = TestClock()
        calls = []

        def eventually_true() -> bool:
            calls.append(clock.monotonic())
            clock.advance(0.5)  # Each check advances time
            return len(calls) >= 3

        result = wait_until(
            eventually_true,
            timeout=2.0,
            poll_interval=0.5,
            clock=clock,
        )

        assert result is True
        assert len(calls) == 3
    """
    deadline = clock.monotonic() + timeout
    while clock.monotonic() < deadline:
        if predicate():
            return True
        clock.sleep(poll_interval)
    return predicate()


__all__ = [
    "LoopGroup",
    "Runnable",
    "ShutdownCoordinator",
    "wait_until",
]
