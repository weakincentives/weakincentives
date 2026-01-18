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

"""Health monitoring for LoopGroup workers.

This module provides primitives for detecting stuck workers and exposing
health endpoints for Kubernetes probes.

Example::

    from weakincentives.runtime import Heartbeat, Watchdog, HealthServer

    # Create heartbeats for each worker
    heartbeats = [Heartbeat() for _ in range(3)]

    # Start watchdog monitoring
    watchdog = Watchdog(heartbeats, stall_threshold=720.0)
    watchdog.start()

    # In each worker loop:
    while running:
        heartbeats[i].beat()
        process_message()
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .clock import Clock, SystemClock

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Heartbeat:
    """Thread-safe heartbeat tracker with observer callbacks.

    Workers call ``beat()`` at regular intervals to prove liveness.
    The watchdog calls ``elapsed()`` to check for stalls. Multiple
    callbacks can be registered to observe beats (lease extension,
    metrics, logging, etc.).

    Example::

        clock = SystemClock()
        hb = Heartbeat(clock=clock)

        # Register callbacks for beat observation
        hb.add_callback(lambda: print("beat!"))

        # In worker loop
        while running:
            process_message()
            hb.beat()  # Records beat AND invokes callbacks

        # Watchdog checks
        if hb.elapsed() > threshold:
            # Worker is stuck
            ...
    """

    clock: Clock = field(default_factory=SystemClock)
    _last_beat: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _callbacks: list[Callable[[], None]] = field(
        default_factory=lambda: list[Callable[[], None]]()
    )

    def __post_init__(self) -> None:
        """Initialize last beat timestamp."""
        self._last_beat = self.clock.monotonic()

    def beat(self) -> None:
        """Record a heartbeat and invoke all registered callbacks.

        Callbacks are invoked outside the lock to avoid deadlock.
        Exceptions in callbacks are logged but do not prevent other
        callbacks from running.
        """
        with self._lock:
            self._last_beat = self.clock.monotonic()
            callbacks = list(self._callbacks)  # Snapshot under lock

        # Invoke outside lock to avoid deadlock
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Heartbeat callback failed")

    def elapsed(self) -> float:
        """Seconds since last heartbeat."""
        with self._lock:
            return self.clock.monotonic() - self._last_beat

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback to be invoked on each beat.

        Callbacks are invoked outside the lock in registration order.
        Exceptions in callbacks are logged but do not prevent other
        callbacks from running.

        Args:
            callback: Callable with no arguments to invoke on beat.
        """
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[], None]) -> None:
        """Remove a previously added callback.

        Args:
            callback: The callback to remove.

        Raises:
            ValueError: If callback was not registered.
        """
        with self._lock:
            self._callbacks.remove(callback)


class Watchdog:
    """Monitors loop heartbeats and terminates on stall.

    The watchdog runs in a daemon thread. If any monitored loop fails to
    heartbeat within the stall threshold, the process is terminated via
    SIGKILL to ensure immediate exit regardless of stuck threads.

    Example::

        clock = SystemClock()
        heartbeats = [Heartbeat(clock=clock) for _ in loops]
        watchdog = Watchdog(heartbeats, stall_threshold=720.0)
        watchdog.start()

        # In each loop's run():
        while running:
            heartbeats[i].beat()
            process_message()
    """

    def __init__(
        self,
        heartbeats: Sequence[Heartbeat],
        *,
        stall_threshold: float = 720.0,
        check_interval: float = 60.0,
        loop_names: Sequence[str] | None = None,
    ) -> None:
        """Initialize the watchdog.

        Args:
            heartbeats: Heartbeat instances to monitor, one per loop.
            stall_threshold: Seconds without heartbeat before termination.
            check_interval: Seconds between watchdog checks.
            loop_names: Optional names for logging. Defaults to indices.
        """
        super().__init__()
        self._heartbeats = heartbeats
        self._stall_threshold = stall_threshold
        self._check_interval = check_interval
        self._loop_names = (
            list(loop_names)
            if loop_names is not None
            else [f"loop-{i}" for i in range(len(heartbeats))]
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def stall_threshold(self) -> float:
        """Seconds without heartbeat before termination."""
        return self._stall_threshold

    def start(self) -> None:
        """Start the watchdog thread."""
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the watchdog thread gracefully."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._check_interval * 2)
            self._thread = None

    def _run(self) -> None:
        """Watchdog loop: check heartbeats, terminate on stall."""
        while not self._stop_event.wait(timeout=self._check_interval):
            stalled = self._check_heartbeats()
            if stalled:
                self._terminate(stalled)  # pragma: no cover - terminates process

    def _check_heartbeats(self) -> list[tuple[str, float]]:
        """Return list of (name, elapsed) for stalled loops."""
        stalled: list[tuple[str, float]] = []
        for name, heartbeat in zip(self._loop_names, self._heartbeats, strict=True):
            elapsed = heartbeat.elapsed()
            if elapsed > self._stall_threshold:
                stalled.append((name, elapsed))
        return stalled

    def _terminate(self, stalled: list[tuple[str, float]]) -> None:
        """Log diagnostics and terminate the process."""
        for name, elapsed in stalled:
            logger.critical(
                "Watchdog: %s stalled for %.1fs (threshold: %.1fs)",
                name,
                elapsed,
                self._stall_threshold,
            )

        logger.critical("Watchdog: terminating process due to stalled workers")

        # SIGKILL ensures termination even if threads are stuck in syscalls
        _ = os.kill(os.getpid(), signal.SIGKILL)


class HealthServer:
    """Minimal HTTP server for Kubernetes health probes.

    Exposes two endpoints:
        - ``GET /health/live``: Returns 200 if process is responsive
        - ``GET /health/ready``: Returns 200 if readiness check passes, 503 otherwise

    Example::

        server = HealthServer(port=8080, readiness_check=lambda: all_loops_healthy())
        server.start()

        # Later
        server.stop()
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",  # nosec B104 - intentional bind to all interfaces
        port: int = 8080,
        readiness_check: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize the health server.

        Args:
            host: Host to bind to. Defaults to all interfaces.
            port: Port to listen on.
            readiness_check: Callable returning True if ready. Defaults to always ready.
        """
        super().__init__()
        self._host = host
        self._port = port
        self._readiness_check = readiness_check or (lambda: True)
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start health server in a daemon thread."""
        if self._server is not None:
            return

        readiness = self._readiness_check

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/health/live":
                    self._send(200, {"status": "healthy"})
                elif self.path == "/health/ready":
                    ok = readiness()
                    status = "healthy" if ok else "unhealthy"
                    self._send(200 if ok else 503, {"status": status})
                else:
                    self.send_error(404)

            def _send(self, code: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                _ = self.wfile.write(data)

            def log_message(  # pyright: ignore[reportImplicitOverride]  # noqa: PLR6301
                self,
                format: str,  # noqa: A002
                *args: object,
            ) -> None:
                _ = (format, args)
                # Suppress request logging

        self._server = HTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the health server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            self._thread = None

    @property
    def address(self) -> tuple[str, int] | None:
        """Return (host, port) if running, None otherwise."""
        if self._server is None:
            return None
        result: tuple[str, int] = self._server.socket.getsockname()
        return result


__all__ = [
    "HealthServer",
    "Heartbeat",
    "Watchdog",
]
