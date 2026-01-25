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

Key components:

- ``LoopGroup``: High-level coordinator that runs multiple loops in threads
  with integrated health endpoints and watchdog monitoring.
- ``ShutdownCoordinator``: Low-level signal handler that triggers registered
  callbacks on SIGTERM/SIGINT.
- ``Runnable``: Protocol defining the interface for loops managed by this module.
- ``wait_until``: Utility for polling until a condition becomes true or timeout.

Example::

    from weakincentives.runtime import LoopGroup, ShutdownCoordinator

    # Run multiple loops with coordinated shutdown
    group = LoopGroup(loops=[main_loop, eval_loop])
    group.run()  # Blocks until SIGTERM/SIGINT

    # With health endpoints and watchdog
    group = LoopGroup(
        loops=[main_loop],
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
    group = LoopGroup(loops=[main_loop], health_port=8080)
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

    Both MainLoop and EvalLoop implement this protocol, enabling them to be
    managed by LoopGroup and ShutdownCoordinator.

    Implementers must provide:
    - run(): Start processing, blocking until stopped
    - shutdown(): Signal stop and wait for completion
    - running: Property indicating active state
    - heartbeat: Optional Heartbeat for watchdog monitoring
    - Context manager support (__enter__/__exit__)
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
        """True if the loop is currently processing messages.

        This property transitions to True when run() begins and back to False
        when the loop exits (either normally or due to shutdown). Use this to
        check loop status from other threads.
        """
        ...

    @property
    def heartbeat(self) -> Heartbeat | None:
        """Heartbeat tracker for watchdog monitoring.

        Returns None if the loop does not support heartbeat monitoring.
        Loops that support monitoring should return a Heartbeat instance
        and call ``heartbeat.beat()`` at regular intervals during processing
        (typically at least once per poll iteration).

        The LoopGroup watchdog uses this to detect stuck workers. If no beat
        is recorded within the watchdog_threshold, the process is terminated.
        """
        ...

    def __enter__(self) -> Self:
        """Context manager entry.

        Returns self unchanged. The context manager pattern ensures shutdown()
        is called on exit, even if an exception occurs.
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit triggers shutdown.

        Calls shutdown() to gracefully stop the loop. Does not suppress
        exceptions (always returns None/False).
        """
        ...


class ShutdownCoordinator:
    """Coordinates graceful shutdown across multiple loops.

    Installs signal handlers for SIGTERM and SIGINT. When a signal arrives,
    all registered callbacks are invoked in registration order. Thread-safe
    for concurrent registration and signal delivery.

    This is a singleton: call install() to get the shared instance. Multiple
    install() calls return the same instance (signal handlers are only
    installed once).

    Note:
        Signal handlers can only be installed from the main thread. If you
        need shutdown coordination from non-main threads, use LoopGroup with
        install_signals=False and trigger shutdown programmatically.

    Example::

        coordinator = ShutdownCoordinator.install()
        coordinator.register(loop.shutdown)
        loop.run()

    Example with multiple loops::

        coordinator = ShutdownCoordinator.install()
        coordinator.register(loop1.shutdown)
        coordinator.register(loop2.shutdown)
        # Both loops shut down when SIGTERM/SIGINT received
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
        """Return the installed coordinator, or None if not installed.

        Use this to check if signal handlers have been installed, or to access
        the coordinator without installing handlers (e.g., in library code that
        should not install handlers without the caller's consent).

        Returns:
            The singleton instance if install() has been called, None otherwise.
        """
        with cls._instance_lock:
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for testing.

        Clears all registered callbacks and the triggered flag, then removes
        the singleton instance. The next install() call will create a fresh
        instance.

        Warning:
            Does not restore original signal handlers. After reset(), signals
            will still be delivered to the old (now orphaned) handler until
            install() is called again. Use only in test teardown.
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

        Safe to call with a callback that was never registered or has already
        been unregistered (no-op in both cases).

        Args:
            callback: Previously registered callback to remove.
        """
        with self._callbacks_lock, contextlib.suppress(ValueError):
            self._callbacks.remove(callback)

    def trigger(self) -> None:
        """Manually trigger shutdown.

        Invokes all registered callbacks in registration order. This is useful
        for testing shutdown behavior without sending actual signals, or for
        programmatic shutdown triggered by application logic (e.g., after
        processing a specific number of messages).

        Safe to call multiple times; callbacks are only invoked once per
        registration (subsequent trigger() calls still invoke callbacks, but
        the triggered flag prevents re-registration from causing duplicate
        invocations).
        """
        self._triggered.set()
        with self._callbacks_lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            _ = callback()

    @property
    def triggered(self) -> bool:
        """True if shutdown has been triggered.

        Once set, this remains True until reset() is called. Callbacks
        registered after shutdown has been triggered are invoked immediately
        upon registration.
        """
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

        group = LoopGroup(loops=[main_loop, eval_loop])
        group.run()  # Blocks until shutdown signal or all loops exit

    Example with health and watchdog::

        group = LoopGroup(
            loops=[main_loop],
            health_port=8080,
            watchdog_threshold=720.0,
        )
        group.run()

    Example with context manager::

        with LoopGroup(loops=[main_loop, eval_loop]) as group:
            group.run()
        # Shutdown triggered on context exit

    Example with health endpoint::

        group = LoopGroup(loops=[main_loop], health_port=8080)
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
            loops: Sequence of Runnable loops to coordinate. Each loop runs in
                its own thread when run() is called.
            shutdown_timeout: Maximum seconds to wait for each loop during
                shutdown. If a loop doesn't stop within this time, shutdown()
                returns False but does not forcibly terminate threads.
            health_port: Port for HTTP health endpoints (``/health/live`` and
                ``/health/ready``). None disables the health server. Useful for
                Kubernetes liveness and readiness probes.
            health_host: Interface to bind the health server to. Defaults to
                all interfaces (0.0.0.0) for container environments.
            watchdog_threshold: Seconds without heartbeat before the watchdog
                terminates the process (via os._exit). None disables watchdog.
                Default 720s (12 min) provides headroom for 10-minute evaluations.
                Only effective if loops provide heartbeat support.
            watchdog_interval: Seconds between watchdog health checks. Default 60s.
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

        Blocks until all loops exit or shutdown is triggered. Each loop runs
        in a dedicated thread from a ThreadPoolExecutor.

        When shutdown is triggered (via signal or explicit call), each loop's
        shutdown() method is called. The method returns only after all loops
        have exited.

        Args:
            install_signals: If True, install SIGTERM/SIGINT handlers via
                ShutdownCoordinator. Set False if you manage signals yourself
                or are running in a context where signal handling is not allowed
                (e.g., non-main threads).
            visibility_timeout: Passed to each loop's run() method.
                Default 1800s (30 min) calibrated for 10-min evaluations.
            wait_time_seconds: Passed to each loop's run() method.
                Default 30s for long poll.

        Raises:
            Exception: Re-raises any exception from loop threads after cleanup.
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

        Signals each loop to stop and waits for them to complete. Safe to call
        even if loops are not running (each loop handles this gracefully).

        This method is automatically called by the context manager exit and
        by the ShutdownCoordinator when signals are received.

        Args:
            timeout: Maximum seconds to wait per loop. Defaults to the
                shutdown_timeout provided at construction.

        Returns:
            True if all loops stopped cleanly within their timeouts,
            False if any loop's timeout expired.
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

    Polls the predicate at regular intervals until it returns True or the
    timeout is reached. Performs one final predicate check at timeout to
    avoid missing a just-in-time True result.

    Args:
        predicate: Zero-argument callable that returns True when the waited
            condition is satisfied. Should be side-effect free or idempotent
            since it may be called multiple times.
        timeout: Maximum seconds to wait before returning False.
        poll_interval: Seconds between predicate checks. Lower values provide
            faster response but consume more CPU.
        clock: Clock for time and sleep operations. Defaults to system clock.
            Inject TestClock for deterministic, instant testing.

    Returns:
        True if predicate returned True before or at timeout, False otherwise.

    Example::

        # Wait for a background thread to set a flag
        ready = threading.Event()
        success = wait_until(ready.is_set, timeout=5.0)
        if not success:
            raise TimeoutError("Background task did not complete")

    Example (testing with TestClock)::

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
