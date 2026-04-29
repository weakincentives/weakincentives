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

"""Coordinated startup and shutdown for long-running services.

A :class:`ShutdownCoordinator` is the standard mechanism for graceful
termination: register cleanup callbacks, optionally watch a sentinel
event, and call :meth:`shutdown` (or signal it) to run them in
last-registered-first-run order. :class:`LoopGroup` owns a set of
worker threads that share that lifecycle.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import final

from weakincentives.logging import get_logger

__all__ = [
    "LoopGroup",
    "LoopGroupError",
    "ShutdownCoordinator",
]

_LOG = get_logger("weakincentives.lifecycle")


class LoopGroupError(Exception):
    """Raised when a managed worker fails fatally."""


@final
class ShutdownCoordinator:
    """Run registered cleanup callbacks once on shutdown.

    Callbacks fire in reverse registration order. A coordinator is
    "shut down" exactly once; subsequent calls to :meth:`shutdown` are
    no-ops. The :attr:`event` attribute is a :class:`threading.Event`
    workers can wait on between iterations to react quickly.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._callbacks: list[Callable[[], None]] = []
        self._event = threading.Event()
        self._completed = threading.Event()

    @property
    def event(self) -> threading.Event:
        return self._event

    @property
    def is_shutting_down(self) -> bool:
        return self._event.is_set()

    @property
    def is_complete(self) -> bool:
        return self._completed.is_set()

    def register(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback. Last in, first out."""
        with self._lock:
            if self._completed.is_set():
                callback()
                return
            self._callbacks.append(callback)

    def shutdown(self) -> None:
        """Trigger shutdown. Idempotent."""
        with self._lock:
            already = self._event.is_set()
            self._event.set()
            if already:
                return
            callbacks = list(reversed(self._callbacks))
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception as error:
                _LOG.exception("shutdown.callback_failed", error=str(error))
        self._completed.set()


@dataclass(slots=True)
class LoopGroup:
    """Coordinates a fixed pool of worker threads.

    Each worker is a callable that takes the coordinator's
    :class:`threading.Event` and returns when it's set. ``join``
    triggers shutdown and waits for every worker to finish.
    """

    coordinator: ShutdownCoordinator
    name: str = "loopgroup"
    _threads: list[threading.Thread] = field(default_factory=lambda: [], repr=False)
    _failures: list[BaseException] = field(default_factory=lambda: [], repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def spawn(
        self,
        target: Callable[[threading.Event], None],
        *,
        name: str | None = None,
    ) -> None:
        thread = threading.Thread(
            target=self._run,
            args=(target,),
            name=name or f"{self.name}-{len(self._threads)}",
            daemon=True,
        )
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def _run(self, target: Callable[[threading.Event], None]) -> None:
        try:
            target(self.coordinator.event)
        except BaseException as error:
            with self._lock:
                self._failures.append(error)
            self.coordinator.shutdown()

    def join(self, timeout: float | None = None) -> None:
        self.coordinator.shutdown()
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=timeout)
        with self._lock:
            failures = list(self._failures)
            self._failures.clear()
        if failures:
            raise LoopGroupError(
                f"{len(failures)} worker(s) failed: {failures[0]!r}"
            ) from failures[0]

    def alive_workers(self) -> Iterable[str]:
        with self._lock:
            return [t.name for t in self._threads if t.is_alive()]
