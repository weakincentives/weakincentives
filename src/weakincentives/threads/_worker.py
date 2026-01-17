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

# pyright: reportUnusedCallResult=false

"""Worker thread implementation for deterministic scheduling."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._scheduler import Scheduler


@dataclass(slots=True)
class WorkerThread:
    """A thread that yields control at checkpoints.

    Uses real OS threads with Event synchronization to achieve
    deterministic scheduling. Only one thread runs at a time.
    """

    name: str
    _target: Callable[[], None]
    _scheduler: Scheduler

    _can_run: threading.Event = field(default_factory=threading.Event)
    _paused: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _done: bool = field(default=False)
    _error: BaseException | None = field(default=None, repr=False)
    _started: bool = field(default=False)
    _current_checkpoint: str | None = field(default=None)

    def start(self) -> None:
        """Start the worker thread."""
        if self._started:
            msg = f"Thread {self.name!r} already started"
            raise RuntimeError(msg)

        self._started = True
        self._thread = threading.Thread(
            target=self._run,
            name=f"worker-{self.name}",
            daemon=True,
        )
        self._thread.start()
        # Wait for thread to reach first pause point (initial ready state)
        _ = self._paused.wait()

    def _run(self) -> None:
        """Thread entry point. Waits for permission then runs target."""
        # Signal we're ready and waiting
        self._paused.set()
        _ = self._can_run.wait()
        self._can_run.clear()

        try:
            self._target()
        except BaseException as e:
            self._error = e
        finally:
            self._done = True
            self._paused.set()

    @property
    def can_run(self) -> bool:
        """True if thread is ready to be scheduled."""
        return self._started and not self._done and self._paused.is_set()

    @property
    def done(self) -> bool:
        """True if thread has completed."""
        return self._done

    @property
    def error(self) -> BaseException | None:
        """Exception raised by thread, if any."""
        return self._error

    def resume_until_checkpoint(self) -> None:
        """Let thread run until it hits a checkpoint or finishes."""
        if self._done:
            return

        self._paused.clear()
        self._can_run.set()
        _ = self._paused.wait()

    def pause_at_checkpoint(self, name: str | None) -> None:
        """Called from within thread to pause at a checkpoint.

        This is called by checkpoint() when this thread hits a yield point.
        """
        self._current_checkpoint = name
        self._paused.set()
        _ = self._can_run.wait()
        self._can_run.clear()

    def join(self, timeout: float | None = None) -> None:
        """Wait for thread to complete."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)


__all__ = ["WorkerThread"]
