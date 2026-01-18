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

"""Background worker implementations."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class BackgroundWorker:
    """Managed daemon thread with lifecycle control.

    Provides a higher-level abstraction over threading.Thread for
    background workers that need clean startup and shutdown.

    Example::

        stop_signal = SystemGate()

        def worker_loop():
            while not stop_signal.is_set():
                process_item()
                stop_signal.wait(timeout=1.0)

        worker = BackgroundWorker(worker_loop, name="processor")
        worker.start()

        # Later...
        stop_signal.set()
        worker.stop(timeout=10.0)
    """

    target: Callable[[], None]
    name: str = "worker"
    daemon: bool = True
    _thread: threading.Thread | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self) -> None:
        """Start the background worker thread.

        Raises:
            RuntimeError: If the worker was already started.
        """
        with self._lock:
            if self._started:
                msg = "Worker already started"
                raise RuntimeError(msg)

            self._thread = threading.Thread(
                target=self.target,
                name=self.name,
                daemon=self.daemon,
            )
            self._thread.start()
            self._started = True

    def stop(self, timeout: float = 5.0) -> bool:
        """Wait for the worker to stop.

        Note: This method does NOT signal the worker to stop. You must
        use a separate mechanism (like a Gate) to signal shutdown.

        Args:
            timeout: Maximum seconds to wait for the thread to finish.

        Returns:
            True if the thread finished within timeout, False otherwise.
        """
        with self._lock:
            thread = self._thread

        if thread is None:
            return True

        thread.join(timeout=timeout)
        return not thread.is_alive()

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the worker thread to finish.

        Args:
            timeout: Maximum seconds to wait, or None for no limit.

        Returns:
            True if the thread finished, False if still running.
        """
        with self._lock:
            thread = self._thread

        if thread is None:
            return True

        thread.join(timeout=timeout)
        return not thread.is_alive()

    @property
    def running(self) -> bool:
        """Return True if the worker was started and is still running."""
        with self._lock:
            if self._thread is None:
                return False
            return self._thread.is_alive()

    @property
    def alive(self) -> bool:
        """Alias for running property."""
        return self.running


@dataclass
class FakeBackgroundWorker:
    """Test worker that runs synchronously.

    The target function is executed immediately when start() is called,
    in the calling thread. Useful for testing worker logic without
    real threading.

    Example::

        calls = []

        def worker_fn():
            calls.append("called")

        worker = FakeBackgroundWorker(worker_fn)
        worker.start()  # Runs synchronously
        assert calls == ["called"]
    """

    target: Callable[[], None]
    name: str = "worker"
    daemon: bool = True
    _started: bool = field(default=False, repr=False)
    _ran: bool = field(default=False, repr=False)

    def start(self) -> None:
        """Execute the target function synchronously.

        Raises:
            RuntimeError: If the worker was already started.
        """
        if self._started:
            msg = "Worker already started"
            raise RuntimeError(msg)

        self._started = True
        self.target()
        self._ran = True

    def stop(self, timeout: float = 5.0) -> bool:
        """No-op for fake worker. Always returns True."""
        del timeout  # unused
        return True

    def join(self, timeout: float | None = None) -> bool:
        """No-op for fake worker. Always returns True."""
        del timeout  # unused
        return True

    @property
    def running(self) -> bool:
        """Fake worker is never 'running' since it completes synchronously."""
        return False

    @property
    def alive(self) -> bool:
        """Alias for running property."""
        return self.running

    def reset(self) -> None:
        """Reset state for reuse in tests."""
        self._started = False
        self._ran = False


__all__ = [
    "BackgroundWorker",
    "FakeBackgroundWorker",
]
