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

"""Dead Letter Queue types and consumer.

DLQs capture messages that cannot be processed after repeated attempts,
preventing poison messages from blocking queues while preserving them
for inspection and remediation.

Example::

    from weakincentives.runtime import DLQPolicy, DeadLetter

    # Configure AgentLoop with DLQ
    agent_loop = MyAgentLoop(
        adapter=adapter,
        requests=requests,
        dlq=DLQPolicy(
            mailbox=dead_letters,
            max_delivery_count=5,
        ),
    )

    # Messages that fail 5 times are sent to dead_letters instead of retrying
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Self
from uuid import UUID

from .lifecycle import wait_until
from .mailbox import Mailbox, Message, ReceiptHandleExpiredError
from .watchdog import Heartbeat

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class DeadLetter[T]:
    """Dead-lettered message with failure metadata.

    Preserves the original message body along with context about why
    and when it was dead-lettered.

    Type parameters:
        T: Original message body type.
    """

    message_id: str
    """Original message ID."""

    body: T
    """Original message body."""

    source_mailbox: str
    """Name of the mailbox the message came from."""

    delivery_count: int
    """Number of delivery attempts before dead-lettering."""

    last_error: str
    """String representation of the final error."""

    last_error_type: str
    """Fully qualified type name of the final error."""

    dead_lettered_at: datetime
    """Timestamp when the message was dead-lettered."""

    first_received_at: datetime
    """Timestamp of the first delivery attempt."""

    request_id: UUID | None = None
    """Request ID if the body is a AgentLoopRequest or EvalRequest."""

    reply_to: str | None = None
    """Original reply_to mailbox name, if any."""

    trace_id: str | None = None
    """Trace ID for distributed tracing correlation.

    Populated from RunContext.trace_id when available (AgentLoop).
    May be None for contexts without distributed tracing (e.g., EvalLoop).
    """


@dataclass(slots=True, frozen=True)
class DLQPolicy[T, R]:
    """Dead letter queue policy.

    Combines destination mailbox with decision logic for when to
    dead-letter failed messages. Subclass to customize behavior.

    Type parameters:
        T: Original message body type.
        R: Original reply type.
    """

    mailbox: Mailbox[DeadLetter[T], None]
    """Destination for dead-lettered messages."""

    max_delivery_count: int = 5
    """Maximum delivery attempts before dead-lettering.

    After this many receive() calls without acknowledge(), the message
    is sent to the DLQ and acknowledged from the source queue.
    """

    include_errors: frozenset[type[Exception]] | None = None
    """Exception types that trigger immediate dead-lettering.

    If set, these exceptions bypass retry and dead-letter immediately.
    None means all exceptions follow the delivery count threshold.

    Uses exact type matching, not isinstance(). Subclasses of listed
    types are not automatically included.
    """

    exclude_errors: frozenset[type[Exception]] | None = None
    """Exception types that never dead-letter.

    These exceptions always retry (respecting visibility backoff).
    Useful for transient network errors that should keep retrying.

    Uses exact type matching, not isinstance(). Subclasses of listed
    types are not automatically excluded.
    """

    def should_dead_letter(self, message: Message[T, Any], error: Exception) -> bool:
        """Determine if the message should be dead-lettered.

        Override this method for custom dead-letter logic.

        Args:
            message: The failed message.
            error: The exception that caused the failure.

        Returns:
            True to dead-letter, False to retry with backoff.
        """
        error_type = type(error)

        # Excluded errors never dead-letter
        if self.exclude_errors and error_type in self.exclude_errors:
            return False

        # Included errors always dead-letter immediately
        if self.include_errors and error_type in self.include_errors:
            return True

        # Otherwise, check delivery count threshold
        return message.delivery_count >= self.max_delivery_count


class DLQConsumer[T]:
    """Runnable consumer for dead letter queues.

    Processes dead-lettered messages with a custom handler. Designed
    to run alongside AgentLoop workers in a LoopGroup.

    Example::

        from weakincentives.runtime import LoopGroup, DLQConsumer

        dlq_consumer = DLQConsumer(
            mailbox=dead_letters,
            handler=alert_handler,
        )

        group = LoopGroup(
            loops=[agent_loop, eval_loop, dlq_consumer],
            health_port=8080,
        )
        group.run()
    """

    _mailbox: Mailbox[DeadLetter[T], None]
    _handler: Callable[[DeadLetter[T]], None]
    _visibility_timeout: int
    _running: bool
    _shutdown_event: threading.Event
    _lock: threading.Lock
    _heartbeat: Heartbeat

    def __init__(
        self,
        *,
        mailbox: Mailbox[DeadLetter[T], None],
        handler: Callable[[DeadLetter[T]], None],
        visibility_timeout: int = 300,
    ) -> None:
        """Initialize the DLQ consumer.

        Args:
            mailbox: Mailbox containing dead-lettered messages.
            handler: Callback to process each dead letter. Should not raise;
                exceptions are logged and the message is nacked with long backoff.
            visibility_timeout: Default visibility timeout for messages.
        """
        super().__init__()
        self._mailbox = mailbox
        self._handler = handler
        self._visibility_timeout = visibility_timeout
        self._running = False
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        self._heartbeat = Heartbeat()

    @property
    def heartbeat(self) -> Heartbeat:
        """Heartbeat tracker for watchdog monitoring."""
        return self._heartbeat

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int | None = None,
        wait_time_seconds: int = 20,
    ) -> None:
        """Process dead letters until shutdown.

        Args:
            max_iterations: Maximum polling iterations. None for unlimited.
            visibility_timeout: Seconds messages remain invisible during
                processing. Defaults to the value passed to __init__.
            wait_time_seconds: Long poll duration for receiving messages.
        """
        with self._lock:
            self._running = True
            self._shutdown_event.clear()

        iterations = 0
        vt = (
            visibility_timeout
            if visibility_timeout is not None
            else self._visibility_timeout
        )

        try:
            while max_iterations is None or iterations < max_iterations:
                # Check shutdown before blocking on receive
                if self._shutdown_event.is_set():
                    break

                # Exit if mailbox closed
                if self._mailbox.closed:
                    break

                self._heartbeat.beat()

                messages = self._mailbox.receive(
                    visibility_timeout=vt,
                    wait_time_seconds=wait_time_seconds,
                )

                for msg in messages:
                    # Check shutdown between messages
                    if self._shutdown_event.is_set():  # pragma: no cover
                        with contextlib.suppress(ReceiptHandleExpiredError):
                            msg.nack(visibility_timeout=0)
                        break

                    try:
                        self._handler(msg.body)
                        msg.acknowledge()
                    except Exception:
                        _logger.exception(
                            "DLQ handler failed",
                            extra={
                                "message_id": msg.id,
                            },
                        )
                        # Long backoff for DLQ failures
                        with contextlib.suppress(ReceiptHandleExpiredError):
                            msg.nack(visibility_timeout=3600)

                    self._heartbeat.beat()

                iterations += 1
        finally:
            with self._lock:
                self._running = False

    def shutdown(self, *, timeout: float = 30.0) -> bool:
        """Signal shutdown and wait for completion.

        Args:
            timeout: Maximum seconds to wait for the loop to stop.

        Returns:
            True if loop stopped cleanly, False if timeout expired.
        """
        self._shutdown_event.set()
        return wait_until(lambda: not self.running, timeout=timeout)

    @property
    def running(self) -> bool:
        """True if the consumer is currently processing messages."""
        with self._lock:
            return self._running

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


__all__ = [
    "DLQConsumer",
    "DLQPolicy",
    "DeadLetter",
]
