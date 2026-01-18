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

"""Message lease extension tied to heartbeat activity.

LeaseExtender prevents message visibility timeout during long-running request
processing by extending the lease whenever a heartbeat occurs. This approach
ties lease extension directly to proof-of-work: if the worker is actively
processing (beating), the lease extends; if the worker is stuck (no beats),
the lease expires naturally and the message becomes visible for another worker.

Example::

    from weakincentives.runtime import LeaseExtender, LeaseExtenderConfig, Heartbeat

    extender = LeaseExtender(config=LeaseExtenderConfig(interval=30))
    heartbeat = Heartbeat()

    with extender.attach(msg, heartbeat):
        # Pass heartbeat through adapter to tools
        adapter.evaluate(prompt, session=session, heartbeat=heartbeat)
        # Tools call heartbeat.beat() during execution
        # Each beat potentially extends the message lease
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..clock import SYSTEM_CLOCK, MonotonicClock
from .mailbox import ReceiptHandleExpiredError

if TYPE_CHECKING:
    from .mailbox import Message
    from .watchdog import Heartbeat

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LeaseExtenderConfig:
    """Configuration for heartbeat-triggered lease extension.

    Attributes:
        interval: Minimum seconds between extensions. Extensions are triggered
            by heartbeats but rate-limited to avoid excessive API calls.
        extension: Visibility timeout to request on each extension (seconds).
            Relative to current time, not original receive time.
        enabled: Whether to enable automatic extension. Defaults to True.
    """

    interval: float = 60.0
    extension: int = 300
    enabled: bool = True


@dataclass(slots=True)
class LeaseExtender:
    """Extends message visibility when heartbeats occur.

    Attaches to a heartbeat for the duration of message processing. When
    the heartbeat's ``beat()`` is called, checks if enough time has elapsed
    since the last extension and extends the message visibility if so.

    This approach ensures lease extension only happens when actual work is
    being done. If the worker stalls (no heartbeats), the lease expires
    naturally and the message becomes visible for reprocessing.

    Example::

        extender = LeaseExtender(config=LeaseExtenderConfig(interval=30))
        heartbeat = Heartbeat()

        with extender.attach(msg, heartbeat):
            # Pass heartbeat through adapter to tools
            adapter.evaluate(prompt, session=session, heartbeat=heartbeat)
            # Tools call heartbeat.beat() during execution
            # Each beat potentially extends the message lease

    For testing, inject a controllable clock::

        from weakincentives.clock import TestClock

        clock = TestClock()
        extender = LeaseExtender(
            config=LeaseExtenderConfig(interval=30),
            clock=clock,
        )

        # ... attach to message and heartbeat ...
        # clock.advance(30)  # Advance past the interval
        # heartbeat.beat()   # Will trigger extension
    """

    config: LeaseExtenderConfig = field(default_factory=LeaseExtenderConfig)
    clock: MonotonicClock = field(default=SYSTEM_CLOCK, repr=False)
    """Clock for time operations. Defaults to system clock."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _msg: Message[Any, Any] | None = field(default=None, repr=False)
    _heartbeat: Heartbeat | None = field(default=None, repr=False)
    _last_extension: float = field(default=0.0, repr=False)

    @contextmanager
    def attach(
        self,
        msg: Message[Any, Any],
        heartbeat: Heartbeat,
    ) -> Generator[None, None, None]:
        """Attach to a heartbeat for message lease extension.

        While attached, each ``heartbeat.beat()`` call may trigger a lease
        extension if the configured interval has elapsed.

        Args:
            msg: The message to extend visibility for.
            heartbeat: The heartbeat to observe for beats.

        Yields:
            Control to the caller while attached.
        """
        if not self.config.enabled:
            yield
            return

        self._attach(msg, heartbeat)
        try:
            yield
        finally:
            self._detach()

    def _attach(self, msg: Message[Any, Any], heartbeat: Heartbeat) -> None:
        """Attach lease extension callback to heartbeat."""
        with self._lock:
            if self._msg is not None:
                raise RuntimeError("LeaseExtender already attached")

            self._msg = msg
            self._heartbeat = heartbeat
            # Initialize to 0.0 so first beat always extends immediately
            self._last_extension = 0.0

        # Register callback (outside lock - add_callback has its own lock)
        heartbeat.add_callback(self._on_beat)

    def _detach(self) -> None:
        """Detach lease extension callback from heartbeat."""
        with self._lock:
            heartbeat = self._heartbeat
            self._msg = None
            self._heartbeat = None

        # Unregister callback (outside lock - remove_callback has its own lock)
        if heartbeat is not None:
            heartbeat.remove_callback(self._on_beat)

    def _on_beat(self) -> None:
        """Called when heartbeat.beat() is invoked."""
        with self._lock:
            if self._msg is None:
                return

            now = self.clock.monotonic()
            elapsed = now - self._last_extension

            if elapsed < self.config.interval:
                return  # Rate limit

            msg = self._msg
            extension = self.config.extension

        # Perform extension outside lock to avoid holding lock during I/O
        try:
            msg.extend_visibility(extension)
            with self._lock:
                self._last_extension = self.clock.monotonic()
            _logger.debug(
                "Extended visibility for message %s by %d seconds",
                msg.id,
                extension,
            )
        except ReceiptHandleExpiredError:
            _logger.warning(
                "Lease extension failed for message %s: receipt handle expired",
                msg.id,
            )
            # Don't detach - let processing continue, it will handle gracefully
        except Exception:
            _logger.exception(
                "Lease extension failed for message %s",
                msg.id,
            )


__all__ = [
    "LeaseExtender",
    "LeaseExtenderConfig",
]
