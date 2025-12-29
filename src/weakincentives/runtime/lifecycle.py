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

Health endpoints::

    # Run loops with health endpoint for Kubernetes probes
    group = LoopGroup(loops=[main_loop], health_port=8080)
    group.run()  # GET /health/live and /health/ready available
"""

from __future__ import annotations

import contextlib
import json
import signal
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
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


class HealthServer:
    """Minimal HTTP server for Kubernetes health probes.

    Exposes two endpoints:
    - GET /health/live: Liveness probe (200 if process is responsive)
    - GET /health/ready: Readiness probe (200 if all loops running, 503 otherwise)

    Example::

        server = HealthServer(port=8080, readiness_check=lambda: all_loops_running)
        server.start()
        # ... run application ...
        server.stop()
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",  # nosec B104 - bind to all interfaces for k8s
        port: int = 8080,
        readiness_check: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize the health server.

        Args:
            host: Host to bind to. Defaults to all interfaces.
            port: Port to bind to. Use 0 for OS-assigned port.
            readiness_check: Callable returning True if ready. Defaults to always True.
        """
        super().__init__()
        self._host = host
        self._port = port
        self._readiness_check = readiness_check or (lambda: True)
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start health server in a daemon thread.

        Safe to call multiple times; subsequent calls are no-ops if already running.
        """
        if self._server is not None:
            return

        readiness = self._readiness_check

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/health/live":
                    self._send(200, {"status": "healthy"})
                elif self.path == "/health/ready":
                    ok = readiness()
                    self._send(
                        200 if ok else 503,
                        {"status": "healthy" if ok else "unhealthy"},
                    )
                else:
                    self.send_error(404)

            def _send(self, code: int, body: dict[str, str]) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                _ = self.wfile.write(data)

            def log_message(  # pyright: ignore[reportImplicitOverride]
                self,
                format: str,  # noqa: A002
                *args: object,
            ) -> None:
                # Suppress request logging
                pass

        self._server = HTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the health server.

        Safe to call multiple times; subsequent calls are no-ops if not running.
        """
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            self._thread = None

    @property
    def address(self) -> tuple[str, int] | None:
        """Return (host, port) if running, None otherwise.

        Useful when port=0 is passed to get the OS-assigned port.
        """
        if self._server is None:
            return None
        addr = self._server.server_address
        # HTTPServer always uses AF_INET, so address is (host, port)
        return (str(addr[0]), addr[1])


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

    Example with health endpoint::

        group = LoopGroup(loops=[main_loop], health_port=8080)
        group.run()  # GET /health/live and /health/ready available
    """

    def __init__(
        self,
        loops: Sequence[Runnable],
        *,
        shutdown_timeout: float = 30.0,
        health_port: int | None = None,
        health_host: str = "0.0.0.0",  # nosec B104 - bind to all interfaces for k8s
    ) -> None:
        """Initialize the LoopGroup.

        Args:
            loops: Sequence of Runnable loops to coordinate.
            shutdown_timeout: Maximum seconds to wait for each loop during shutdown.
            health_port: Port for health endpoint. If None, no health server is started.
            health_host: Host for health endpoint. Defaults to all interfaces.
        """
        super().__init__()
        self.loops = loops
        self.shutdown_timeout = shutdown_timeout
        self._health_port = health_port
        self._health_host = health_host
        self._health_server: HealthServer | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future[None]] = []

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

        # Start health server if configured
        if self._health_port is not None:
            self._health_server = HealthServer(
                host=self._health_host,
                port=self._health_port,
                readiness_check=lambda: all(loop.running for loop in self.loops),
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
    "HealthServer",
    "LoopGroup",
    "Runnable",
    "ShutdownCoordinator",
    "wait_until",
]
