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

"""Ticket primitive for asynchronous request/response patterns.

Tickets provide a clean interface for cross-thread communication without
shared mutable state. A ticket represents a single future result that will
be provided by another thread.

Example::

    # Requester side
    ticket: Ticket[HealthStatus] = Ticket()
    worker.send(HealthCheckRequest(ticket=ticket))

    # ... do other work ...

    # Block until result available
    status = ticket.wait(timeout=5.0)

    # Or check without blocking
    if ticket.is_ready():
        status = ticket.result()


    # Worker side
    def handle_health_check(request: HealthCheckRequest) -> None:
        try:
            status = gather_health_status()
            request.ticket.complete(status)
        except Exception as e:
            request.ticket.fail(e)

The ticket pattern is preferred over directly reading shared state because:

1. **Explicit ownership**: One writer, one reader. No concurrent access.
2. **Clear lifecycle**: Create, complete, read. No dangling references.
3. **Testable**: No timing dependencies. Complete ticket, check result.
4. **Type-safe**: Generic over result type.

Tickets should NOT be reused. Create a new ticket for each request.
"""

from __future__ import annotations

import threading
from typing import cast


class TicketError(Exception):
    """Base class for ticket-related errors."""


class TicketTimeoutError(TicketError):
    """Raised when wait() times out before result is available."""


class TicketAlreadyCompletedError(TicketError):
    """Raised when complete() or fail() called on already-completed ticket."""


class TicketNotReadyError(TicketError):
    """Raised when result() called before ticket is completed."""


class Ticket[T]:
    """Single-use container for asynchronous results.

    A ticket is created by a requester, passed to a worker, and completed
    with either a result or an error. The requester can then retrieve the
    result synchronously.

    Thread safety: Tickets are safe for exactly one writer and one reader.
    The writer calls complete() or fail() exactly once. The reader calls
    wait() or result() to retrieve the value.

    Type parameters:
        T: The type of the result value.

    Example::

        # Create ticket and send to worker
        ticket: Ticket[int] = Ticket()
        queue.put(ComputeRequest(n=42, ticket=ticket))

        # Wait for result with timeout
        try:
            result = ticket.wait(timeout=5.0)
            print(f"Got result: {result}")
        except TicketTimeoutError:
            print("Worker didn't respond in time")
        except ComputeError as e:
            print(f"Worker failed: {e}")

    """

    __slots__ = ("_completed", "_error", "_event", "_result")

    def __init__(self) -> None:
        """Create a new ticket awaiting a result."""
        super().__init__()
        self._event = threading.Event()
        self._result: T | None = None
        self._error: BaseException | None = None
        self._completed = False

    def complete(self, result: T) -> None:
        """Complete the ticket with a successful result.

        This method should be called exactly once by the worker thread.
        After completion, the requester's wait() call will return the result.

        Args:
            result: The result value to provide to the requester.

        Raises:
            TicketAlreadyCompletedError: If complete() or fail() was already called.
        """
        if self._completed:
            raise TicketAlreadyCompletedError(
                "Ticket has already been completed or failed"
            )
        self._result = result
        self._completed = True
        self._event.set()

    def fail(self, error: BaseException) -> None:
        """Complete the ticket with an error.

        This method should be called exactly once by the worker thread
        if the operation fails. After completion, the requester's wait()
        call will raise the error.

        Args:
            error: The exception to raise to the requester.

        Raises:
            TicketAlreadyCompletedError: If complete() or fail() was already called.
        """
        if self._completed:
            raise TicketAlreadyCompletedError(
                "Ticket has already been completed or failed"
            )
        self._error = error
        self._completed = True
        self._event.set()

    def wait(self, timeout: float | None = None) -> T:
        """Wait for the result to become available.

        Blocks until the worker calls complete() or fail(), or until
        the timeout expires.

        Args:
            timeout: Maximum seconds to wait. None waits indefinitely.

        Returns:
            The result value provided via complete().

        Raises:
            TicketTimeoutError: If timeout expires before completion.
            BaseException: The error provided via fail(), if any.
        """
        if not self._event.wait(timeout=timeout):
            raise TicketTimeoutError(f"Ticket not completed within {timeout}s timeout")
        if self._error is not None:
            raise self._error
        return cast(T, self._result)

    def result(self) -> T:
        """Return the result without waiting.

        Returns:
            The result value provided via complete().

        Raises:
            TicketNotReadyError: If ticket has not been completed.
            BaseException: The error provided via fail(), if any.
        """
        if not self._completed:
            raise TicketNotReadyError("Ticket has not been completed yet")
        if self._error is not None:
            raise self._error
        return cast(T, self._result)

    def is_ready(self) -> bool:
        """Check if the result is available without blocking.

        Returns:
            True if complete() or fail() has been called.
        """
        return self._event.is_set()

    def is_success(self) -> bool:
        """Check if the ticket completed successfully.

        Returns:
            True if complete() was called (not fail()).
            False if not ready or if fail() was called.
        """
        return self._completed and self._error is None

    def is_failure(self) -> bool:
        """Check if the ticket completed with an error.

        Returns:
            True if fail() was called.
            False if not ready or if complete() was called.
        """
        return self._completed and self._error is not None


__all__ = [
    "Ticket",
    "TicketAlreadyCompletedError",
    "TicketError",
    "TicketNotReadyError",
    "TicketTimeoutError",
]
