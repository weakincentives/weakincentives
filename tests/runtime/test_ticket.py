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

"""Tests for the Ticket primitive."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from weakincentives.runtime import (
    Ticket,
    TicketAlreadyCompletedError,
    TicketNotReadyError,
    TicketTimeoutError,
)


class TestTicketBasicOperations:
    """Test basic ticket operations."""

    def test_complete_and_wait(self) -> None:
        """Ticket can be completed with a result and waited on."""
        ticket: Ticket[int] = Ticket()
        ticket.complete(42)

        result = ticket.wait()

        assert result == 42

    def test_complete_with_none(self) -> None:
        """Ticket can be completed with None value."""
        ticket: Ticket[None] = Ticket()
        ticket.complete(None)

        result = ticket.wait()

        assert result is None

    def test_complete_with_complex_type(self) -> None:
        """Ticket works with complex types."""
        ticket: Ticket[dict[str, list[int]]] = Ticket()
        value = {"numbers": [1, 2, 3]}
        ticket.complete(value)

        result = ticket.wait()

        assert result == {"numbers": [1, 2, 3]}

    def test_fail_and_wait_raises_error(self) -> None:
        """Ticket can be failed and wait() raises the error."""
        ticket: Ticket[int] = Ticket()
        error = ValueError("something went wrong")
        ticket.fail(error)

        with pytest.raises(ValueError, match="something went wrong"):
            ticket.wait()

    def test_fail_with_custom_exception(self) -> None:
        """Ticket preserves custom exception types."""

        class CustomError(Exception):
            pass

        ticket: Ticket[str] = Ticket()
        ticket.fail(CustomError("custom"))

        with pytest.raises(CustomError, match="custom"):
            ticket.wait()


class TestTicketReadyState:
    """Test ticket ready state checking."""

    def test_is_ready_before_completion(self) -> None:
        """Ticket is not ready before complete() or fail()."""
        ticket: Ticket[int] = Ticket()

        assert not ticket.is_ready()

    def test_is_ready_after_complete(self) -> None:
        """Ticket is ready after complete()."""
        ticket: Ticket[int] = Ticket()
        ticket.complete(42)

        assert ticket.is_ready()

    def test_is_ready_after_fail(self) -> None:
        """Ticket is ready after fail()."""
        ticket: Ticket[int] = Ticket()
        ticket.fail(ValueError("error"))

        assert ticket.is_ready()

    def test_is_success_after_complete(self) -> None:
        """is_success() returns True after complete()."""
        ticket: Ticket[int] = Ticket()
        ticket.complete(42)

        assert ticket.is_success()
        assert not ticket.is_failure()

    def test_is_failure_after_fail(self) -> None:
        """is_failure() returns True after fail()."""
        ticket: Ticket[int] = Ticket()
        ticket.fail(ValueError("error"))

        assert ticket.is_failure()
        assert not ticket.is_success()

    def test_is_success_before_completion(self) -> None:
        """is_success() returns False before completion."""
        ticket: Ticket[int] = Ticket()

        assert not ticket.is_success()

    def test_is_failure_before_completion(self) -> None:
        """is_failure() returns False before completion."""
        ticket: Ticket[int] = Ticket()

        assert not ticket.is_failure()


class TestTicketResult:
    """Test non-blocking result() method."""

    def test_result_after_complete(self) -> None:
        """result() returns value after complete()."""
        ticket: Ticket[str] = Ticket()
        ticket.complete("hello")

        assert ticket.result() == "hello"

    def test_result_after_fail(self) -> None:
        """result() raises error after fail()."""
        ticket: Ticket[str] = Ticket()
        ticket.fail(RuntimeError("failed"))

        with pytest.raises(RuntimeError, match="failed"):
            ticket.result()

    def test_result_before_completion_raises(self) -> None:
        """result() raises TicketNotReadyError if not completed."""
        ticket: Ticket[str] = Ticket()

        with pytest.raises(TicketNotReadyError):
            ticket.result()


class TestTicketTimeout:
    """Test timeout behavior."""

    def test_wait_with_timeout_success(self) -> None:
        """wait() with timeout returns result if available."""
        ticket: Ticket[int] = Ticket()
        ticket.complete(42)

        result = ticket.wait(timeout=1.0)

        assert result == 42

    def test_wait_timeout_expires(self) -> None:
        """wait() raises TicketTimeoutError if timeout expires."""
        ticket: Ticket[int] = Ticket()

        with pytest.raises(TicketTimeoutError):
            ticket.wait(timeout=0.01)

    def test_wait_indefinitely(self) -> None:
        """wait() with None timeout waits indefinitely (completes quickly here)."""
        ticket: Ticket[int] = Ticket()
        ticket.complete(42)

        # This would block forever if ticket wasn't completed
        result = ticket.wait(timeout=None)

        assert result == 42


class TestTicketDoubleCompletion:
    """Test that tickets can only be completed once."""

    def test_double_complete_raises(self) -> None:
        """Calling complete() twice raises TicketAlreadyCompletedError."""
        ticket: Ticket[int] = Ticket()
        ticket.complete(1)

        with pytest.raises(TicketAlreadyCompletedError):
            ticket.complete(2)

    def test_double_fail_raises(self) -> None:
        """Calling fail() twice raises TicketAlreadyCompletedError."""
        ticket: Ticket[int] = Ticket()
        ticket.fail(ValueError("first"))

        with pytest.raises(TicketAlreadyCompletedError):
            ticket.fail(ValueError("second"))

    def test_complete_then_fail_raises(self) -> None:
        """Calling fail() after complete() raises TicketAlreadyCompletedError."""
        ticket: Ticket[int] = Ticket()
        ticket.complete(42)

        with pytest.raises(TicketAlreadyCompletedError):
            ticket.fail(ValueError("error"))

    def test_fail_then_complete_raises(self) -> None:
        """Calling complete() after fail() raises TicketAlreadyCompletedError."""
        ticket: Ticket[int] = Ticket()
        ticket.fail(ValueError("error"))

        with pytest.raises(TicketAlreadyCompletedError):
            ticket.complete(42)


class TestTicketThreadSafety:
    """Test ticket behavior across threads."""

    def test_complete_in_another_thread(self) -> None:
        """Ticket can be completed from another thread."""
        ticket: Ticket[int] = Ticket()

        def worker() -> None:
            time.sleep(0.01)  # Small delay to ensure wait() is called first
            ticket.complete(42)

        thread = threading.Thread(target=worker)
        thread.start()

        result = ticket.wait(timeout=1.0)
        thread.join()

        assert result == 42

    def test_fail_in_another_thread(self) -> None:
        """Ticket can be failed from another thread."""
        ticket: Ticket[int] = Ticket()

        def worker() -> None:
            time.sleep(0.01)
            ticket.fail(ValueError("from worker"))

        thread = threading.Thread(target=worker)
        thread.start()

        with pytest.raises(ValueError, match="from worker"):
            ticket.wait(timeout=1.0)

        thread.join()

    def test_multiple_waiters_get_same_result(self) -> None:
        """Multiple threads waiting on the same ticket get the same result."""
        ticket: Ticket[int] = Ticket()
        results: list[int] = []
        errors: list[Exception] = []

        def waiter() -> None:
            try:
                results.append(ticket.wait(timeout=1.0))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=waiter) for _ in range(5)]
        for t in threads:
            t.start()

        time.sleep(0.01)  # Let waiters start waiting
        ticket.complete(42)

        for t in threads:
            t.join()

        assert results == [42, 42, 42, 42, 42]
        assert errors == []

    def test_concurrent_request_response_pattern(self) -> None:
        """Demonstrates typical request/response pattern."""
        results: list[int] = []

        def worker(requests: list[tuple[int, Ticket[int]]]) -> None:
            for value, ticket in requests:
                time.sleep(0.001)  # Simulate work
                ticket.complete(value * 2)

        # Create requests with tickets
        requests: list[tuple[int, Ticket[int]]] = []
        for i in range(10):
            ticket: Ticket[int] = Ticket()
            requests.append((i, ticket))

        # Start worker
        thread = threading.Thread(target=worker, args=(requests,))
        thread.start()

        # Collect results
        for value, ticket in requests:
            result = ticket.wait(timeout=1.0)
            results.append(result)
            assert result == value * 2

        thread.join()

    def test_high_concurrency_stress_test(self) -> None:
        """Stress test with many concurrent tickets."""
        num_tickets = 100
        tickets: list[Ticket[int]] = [Ticket() for _ in range(num_tickets)]

        def completer(index: int) -> None:
            time.sleep(0.001 * (index % 10))  # Stagger completions
            tickets[index].complete(index)

        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all completions
            futures = [executor.submit(completer, i) for i in range(num_tickets)]

            # Wait for all completions
            for f in futures:
                f.result()

        # Verify all results
        for i, ticket in enumerate(tickets):
            assert ticket.is_ready()
            assert ticket.result() == i


class TestTicketEdgeCases:
    """Test edge cases and special scenarios."""

    def test_wait_already_completed_returns_immediately(self) -> None:
        """wait() returns immediately if ticket already completed."""
        ticket: Ticket[int] = Ticket()
        ticket.complete(42)

        start = time.monotonic()
        result = ticket.wait(timeout=10.0)
        elapsed = time.monotonic() - start

        assert result == 42
        assert elapsed < 0.1  # Should be nearly instant

    def test_wait_already_failed_raises_immediately(self) -> None:
        """wait() raises immediately if ticket already failed."""
        ticket: Ticket[int] = Ticket()
        ticket.fail(ValueError("error"))

        start = time.monotonic()
        with pytest.raises(ValueError):
            ticket.wait(timeout=10.0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1  # Should be nearly instant

    def test_exception_with_traceback_preserved(self) -> None:
        """Exception traceback is preserved through fail()."""
        ticket: Ticket[int] = Ticket()

        def create_error() -> ValueError:
            return ValueError("with context")

        error = create_error()
        ticket.fail(error)

        with pytest.raises(ValueError) as exc_info:
            ticket.wait()

        # Verify it's the same exception object
        assert exc_info.value is error


class TestTicketImports:
    """Test that all ticket types are properly exported."""

    def test_imports_from_runtime_module(self) -> None:
        """All ticket types can be imported from runtime module."""
        from weakincentives.runtime import (
            Ticket,
            TicketAlreadyCompletedError,
            TicketError,
            TicketNotReadyError,
            TicketTimeoutError,
        )

        assert Ticket is not None
        assert issubclass(TicketTimeoutError, TicketError)
        assert issubclass(TicketAlreadyCompletedError, TicketError)
        assert issubclass(TicketNotReadyError, TicketError)

    def test_ticket_module_exported(self) -> None:
        """Ticket module is accessible from runtime."""
        from weakincentives.runtime import ticket

        assert hasattr(ticket, "Ticket")
        assert hasattr(ticket, "TicketTimeoutError")
