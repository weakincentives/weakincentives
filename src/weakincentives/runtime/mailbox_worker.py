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

"""Base class for mailbox-driven processing loops.

MailboxWorker provides the common infrastructure for polling a mailbox,
processing messages, and handling graceful shutdown. Both AgentLoop and
EvalLoop extend this class.
"""

from __future__ import annotations

import contextlib
import threading
from abc import ABC, abstractmethod
from typing import Self

from .lease_extender import LeaseExtender, LeaseExtenderConfig
from .lifecycle import wait_until
from .mailbox import Mailbox, Message, ReceiptHandleExpiredError
from .watchdog import Heartbeat


class MailboxWorker[RequestT, ResponseT](ABC):
    """Abstract base class for mailbox-driven processing loops.

    Provides common infrastructure for:
    - Polling messages from a mailbox
    - Graceful shutdown with in-flight message completion
    - Lease extension to prevent message timeout
    - Heartbeat for liveness monitoring

    Subclasses implement ``_process_message()`` to define message handling logic.

    Example::

        class MyWorker(MailboxWorker[MyRequest, MyResponse]):
            def _process_message(
                self, msg: Message[MyRequest, MyResponse]
            ) -> None:
                result = process(msg.body)
                msg.reply(result)
                msg.acknowledge()

    Lifecycle:
        - Use ``run()`` to start processing messages
        - Call ``shutdown()`` to request graceful stop
        - In-flight messages complete before exit
        - Supports context manager protocol for automatic cleanup
    """

    _requests: Mailbox[RequestT, ResponseT]
    _heartbeat: Heartbeat
    _lease_extender: LeaseExtender
    _shutdown_event: threading.Event
    _running: bool
    _lock: threading.Lock

    def __init__(
        self,
        *,
        requests: Mailbox[RequestT, ResponseT],
        lease_extender_config: LeaseExtenderConfig | None = None,
    ) -> None:
        """Initialize the mailbox worker.

        Args:
            requests: Mailbox to receive messages from.
            lease_extender_config: Optional configuration for lease extension.
        """
        super().__init__()
        self._requests = requests
        self._shutdown_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        self._heartbeat = Heartbeat()
        lease_config = (
            lease_extender_config
            if lease_extender_config is not None
            else LeaseExtenderConfig()
        )
        self._lease_extender = LeaseExtender(config=lease_config)

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        """Run the processing loop.

        Polls the mailbox for messages and processes each one via
        ``_process_message()``. Messages are processed with lease extension
        to prevent visibility timeout during long operations.

        The loop exits when:
        - max_iterations is reached
        - shutdown() is called
        - The mailbox is closed

        In-flight messages complete before exit. Unprocessed messages from
        the current batch are nacked for redelivery.

        Args:
            max_iterations: Maximum polling iterations. None for unlimited.
            visibility_timeout: Seconds messages remain invisible during
                processing. Should exceed maximum expected execution time.
            wait_time_seconds: Long poll duration for receiving messages.
        """
        with self._lock:
            self._running = True
            self._shutdown_event.clear()

        iterations = 0
        try:
            while max_iterations is None or iterations < max_iterations:
                # Check shutdown before blocking on receive
                if self._shutdown_event.is_set():
                    break

                # Exit if mailbox closed
                if self._requests.closed:
                    break

                messages = self._requests.receive(
                    visibility_timeout=visibility_timeout,
                    wait_time_seconds=wait_time_seconds,
                )

                # Beat after receive (proves we're not stuck waiting)
                self._heartbeat.beat()

                for msg in messages:
                    # Check shutdown between messages
                    if self._shutdown_event.is_set():
                        # Nack unprocessed message for redelivery
                        with contextlib.suppress(ReceiptHandleExpiredError):
                            msg.nack(visibility_timeout=0)
                        break

                    # Attach lease extender to heartbeat for this message
                    with self._lease_extender.attach(msg, self._heartbeat):
                        self._process_message(msg)

                    # Beat after processing (proves message handling completes)
                    self._heartbeat.beat()

                iterations += 1
        finally:
            with self._lock:
                self._running = False

    @abstractmethod
    def _process_message(self, msg: Message[RequestT, ResponseT]) -> None:
        """Process a single message from the mailbox.

        Subclasses must implement this method to define message handling logic.
        The method should handle all outcomes including success, failure, and
        error responses. It's responsible for calling msg.reply() and
        msg.acknowledge() or msg.nack() as appropriate.

        This method is called with the lease extender attached, so heartbeat
        beats will extend the message visibility.

        Args:
            msg: The message to process.
        """
        ...

    def shutdown(self, *, timeout: float = 30.0) -> bool:
        """Request graceful shutdown and wait for completion.

        Sets the shutdown flag. If the loop is running, waits up to timeout
        seconds for it to stop.

        Args:
            timeout: Maximum seconds to wait for the loop to stop.

        Returns:
            True if loop stopped cleanly, False if timeout expired.
        """
        self._shutdown_event.set()
        return wait_until(lambda: not self.running, timeout=timeout)

    @property
    def running(self) -> bool:
        """True if the loop is currently processing messages."""
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
    "MailboxWorker",
]
