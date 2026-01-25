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

    # Configure MainLoop with DLQ
    main_loop = MyMainLoop(
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

    Wraps a message that exhausted retry attempts or matched an immediate
    dead-letter condition. Preserves the original message body along with
    diagnostic context for inspection, alerting, or manual remediation.

    Use the ``request_id`` and ``trace_id`` fields to correlate dead letters
    with logs and distributed traces. The ``last_error`` and ``last_error_type``
    fields help identify root causes without needing to reproduce the failure.

    Type Parameters:
        T: Original message body type (e.g., ``MainLoopRequest``).

    Example::

        def alert_on_dead_letter(dl: DeadLetter[MyRequest]) -> None:
            logger.error(
                "Dead letter: %s failed %d times with %s",
                dl.message_id,
                dl.delivery_count,
                dl.last_error_type,
            )
            metrics.increment("dlq.messages", tags={"error": dl.last_error_type})
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
    """Request ID if the body is a MainLoopRequest or EvalRequest."""

    reply_to: str | None = None
    """Original reply_to mailbox name, if any."""

    trace_id: str | None = None
    """Trace ID for distributed tracing correlation.

    Populated from RunContext.trace_id when available (MainLoop).
    May be None for contexts without distributed tracing (e.g., EvalLoop).
    """


@dataclass(slots=True, frozen=True)
class DLQPolicy[T, R]:
    """Dead letter queue policy for failed message handling.

    Configures when and where to route messages that fail processing.
    Pass an instance to ``MainLoop`` or ``EvalLoop`` via the ``dlq``
    parameter to enable automatic dead-lettering.

    The default behavior dead-letters after ``max_delivery_count`` failures.
    Use ``include_errors`` for immediate dead-lettering of non-retryable
    errors (e.g., validation failures), and ``exclude_errors`` for errors
    that should always retry (e.g., transient network issues).

    For custom logic, subclass and override ``should_dead_letter()``.

    Type Parameters:
        T: Original message body type (e.g., ``MainLoopRequest``).
        R: Reply type from the source mailbox (usually ``MainLoopResponse``).

    Example::

        # Basic policy: dead-letter after 5 failures
        policy = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=5)

        # Immediate dead-letter for validation errors, never for timeouts
        policy = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=3,
            include_errors=frozenset({ValidationError}),
            exclude_errors=frozenset({TimeoutError}),
        )
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
        """Determine whether a failed message should be dead-lettered.

        Called by the loop after each processing failure. The default
        implementation checks ``exclude_errors``, then ``include_errors``,
        then ``max_delivery_count``. Override for custom logic such as
        error message pattern matching or per-request retry budgets.

        Args:
            message: The failed message, including ``delivery_count``.
            error: The exception raised during processing.

        Returns:
            ``True`` to move the message to the DLQ immediately.
            ``False`` to return it to the source queue for retry.

        Example::

            @dataclass(frozen=True)
            class CustomDLQPolicy(DLQPolicy[MyRequest, MyResponse]):
                def should_dead_letter(
                    self, message: Message[MyRequest, Any], error: Exception
                ) -> bool:
                    # Custom logic: dead-letter after 10 attempts for specific errors
                    if isinstance(error, RateLimitError):
                        return message.delivery_count >= 10
                    return super().should_dead_letter(message, error)
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
    """Runnable consumer for processing dead-lettered messages.

    Polls a dead letter mailbox and invokes a handler for each message.
    Designed for alerting, logging, metrics collection, or manual review
    workflows. Integrates with ``LoopGroup`` for coordinated lifecycle
    management alongside ``MainLoop`` and ``EvalLoop`` workers.

    The consumer is thread-safe: call ``run()`` from a worker thread and
    ``shutdown()`` from any thread to request graceful termination. The
    ``heartbeat`` property enables watchdog monitoring in production.

    Handler exceptions are logged and the message is returned to the queue
    with a 1-hour visibility timeout (long backoff). Handlers should avoid
    raising exceptions for expected conditions.

    Type Parameters:
        T: Original message body type wrapped in ``DeadLetter[T]``.

    Example::

        from weakincentives.runtime import LoopGroup, DLQConsumer

        def alert_handler(dl: DeadLetter[MyRequest]) -> None:
            send_alert(f"Message {dl.message_id} failed: {dl.last_error}")

        dlq_consumer = DLQConsumer(
            mailbox=dead_letters,
            handler=alert_handler,
        )

        # Run alongside main application loops
        group = LoopGroup(
            loops=[main_loop, eval_loop, dlq_consumer],
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
        """Initialize a DLQ consumer.

        Args:
            mailbox: Source mailbox containing dead-lettered messages.
                Must be a ``Mailbox[DeadLetter[T], None]`` (no reply type).
            handler: Callback invoked for each dead letter. Receives the
                ``DeadLetter[T]`` body (not the raw ``Message``). Should
                handle its own errors; uncaught exceptions are logged and
                the message returns to the queue with 1-hour backoff.
            visibility_timeout: Default seconds messages stay invisible
                during processing. Can be overridden per ``run()`` call.
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
        """Heartbeat tracker for watchdog monitoring.

        Updated automatically during polling and after each message is
        processed. Use with ``LoopGroup`` watchdog to detect stuck consumers.
        """
        return self._heartbeat

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int | None = None,
        wait_time_seconds: int = 20,
    ) -> None:
        """Process dead letters until shutdown or iteration limit.

        Blocks the calling thread, polling the mailbox and invoking the
        handler for each dead letter. Exits when ``shutdown()`` is called,
        the mailbox closes, or ``max_iterations`` is reached.

        Args:
            max_iterations: Maximum polling iterations before returning.
                ``None`` (default) runs until shutdown. Useful for testing.
            visibility_timeout: Seconds messages remain invisible while
                being processed. Defaults to the constructor value (300s).
                Set high enough for handler execution plus buffer time.
            wait_time_seconds: Long-poll duration in seconds. Higher values
                reduce API calls but increase shutdown latency.

        Note:
            Call from a dedicated thread. This method blocks until shutdown.
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
        """Signal shutdown and wait for the run loop to exit.

        Thread-safe. Can be called from any thread, including signal
        handlers. The consumer finishes processing the current message
        (if any) before exiting.

        Args:
            timeout: Maximum seconds to wait for ``run()`` to return.
                If exceeded, returns ``False`` but the shutdown signal
                remains set (the consumer will still stop eventually).

        Returns:
            ``True`` if the consumer stopped within the timeout,
            ``False`` if still running when timeout expired.
        """
        self._shutdown_event.set()
        return wait_until(lambda: not self.running, timeout=timeout)

    @property
    def running(self) -> bool:
        """Whether the consumer's run loop is currently active.

        Thread-safe. Returns ``True`` between the start of ``run()``
        and its return. Use to check if shutdown completed.
        """
        with self._lock:
            return self._running

    def __enter__(self) -> Self:
        """Enter context manager, returning self.

        Does not start the consumer; call ``run()`` separately.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, triggering graceful shutdown.

        Calls ``shutdown()`` and waits for the run loop to complete.
        Exceptions from the ``with`` block are not suppressed.
        """
        _ = (exc_type, exc_val, exc_tb)
        _ = self.shutdown()


__all__ = [
    "DLQConsumer",
    "DLQPolicy",
    "DeadLetter",
]
