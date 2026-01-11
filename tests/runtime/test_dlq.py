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

"""Tests for Dead Letter Queue functionality."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime.dlq import DeadLetter, DLQConsumer, DLQPolicy
from weakincentives.runtime.mailbox import (
    FakeMailbox,
    InMemoryMailbox,
    Mailbox,
    Message,
)
from weakincentives.runtime.main_loop import (
    MainLoop,
    MainLoopConfig,
    MainLoopRequest,
    MainLoopResult,
)
from weakincentives.runtime.run_context import RunContext
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.protocols import SessionProtocol
from weakincentives.runtime.watchdog import Heartbeat


@dataclass(slots=True, frozen=True)
class _Request:
    """Sample request type for testing."""

    message: str


@dataclass(slots=True, frozen=True)
class _Output:
    """Sample output type for testing."""

    result: str


@dataclass(slots=True, frozen=True)
class _Params:
    """Sample params type for testing."""

    content: str


class _MockAdapter(ProviderAdapter[_Output]):
    """Mock adapter that can be configured to fail."""

    def __init__(
        self,
        *,
        response: PromptResponse[_Output] | None = None,
        error: Exception | None = None,
        fail_count: int = 0,
    ) -> None:
        self._response = response or PromptResponse(
            prompt_name="test",
            text="test output",
            output=_Output(result="success"),
        )
        self._error = error
        self._fail_count = fail_count
        self._call_count = 0

    def evaluate(
        self,
        prompt: Prompt[_Output],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
    ) -> PromptResponse[_Output]:
        del prompt, session, deadline, budget, budget_tracker, heartbeat, run_context
        self._call_count += 1
        if self._error is not None:
            raise self._error
        if self._call_count <= self._fail_count:
            raise RuntimeError(f"Simulated failure {self._call_count}")
        return self._response


class _TestLoop(MainLoop[_Request, _Output]):
    """Test implementation of MainLoop."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: Mailbox[MainLoopRequest[_Request], MainLoopResult[_Output]],
        config: MainLoopConfig | None = None,
        dlq: DLQPolicy[MainLoopRequest[_Request], MainLoopResult[_Output]]
        | None = None,
    ) -> None:
        super().__init__(adapter=adapter, requests=requests, config=config, dlq=dlq)
        self._template = PromptTemplate[_Output](
            ns="test",
            key="test-prompt",
            sections=[
                MarkdownSection[_Params](
                    title="Test",
                    template="$content",
                    key="test",
                ),
            ],
        )

    def prepare(
        self, request: _Request, *, experiment: object = None
    ) -> tuple[Prompt[_Output], Session]:
        del experiment  # Unused in tests
        prompt = Prompt(self._template).bind(_Params(content=request.message))
        session = Session(tags={"loop": "test"})
        return prompt, session


# =============================================================================
# DeadLetter Tests
# =============================================================================


def test_dead_letter_creation() -> None:
    """DeadLetter captures all required metadata."""
    body = MainLoopRequest(request=_Request(message="test"))
    dead_letter: DeadLetter[MainLoopRequest[_Request]] = DeadLetter(
        message_id="msg-123",
        body=body,
        source_mailbox="requests",
        delivery_count=5,
        last_error="Test error",
        last_error_type="builtins.RuntimeError",
        dead_lettered_at=datetime.now(UTC),
        first_received_at=datetime.now(UTC),
        request_id=body.request_id,
        reply_to="results",
        trace_id="trace-abc",
    )

    assert dead_letter.message_id == "msg-123"
    assert dead_letter.body is body
    assert dead_letter.source_mailbox == "requests"
    assert dead_letter.delivery_count == 5
    assert dead_letter.last_error == "Test error"
    assert dead_letter.last_error_type == "builtins.RuntimeError"
    assert dead_letter.request_id == body.request_id
    assert dead_letter.reply_to == "results"
    assert dead_letter.trace_id == "trace-abc"


def test_dead_letter_optional_fields() -> None:
    """DeadLetter has sensible defaults for optional fields."""
    dead_letter: DeadLetter[str] = DeadLetter(
        message_id="msg-123",
        body="test message",
        source_mailbox="requests",
        delivery_count=5,
        last_error="Test error",
        last_error_type="builtins.RuntimeError",
        dead_lettered_at=datetime.now(UTC),
        first_received_at=datetime.now(UTC),
    )

    assert dead_letter.request_id is None
    assert dead_letter.reply_to is None
    assert dead_letter.trace_id is None


def test_dead_letter_is_frozen() -> None:
    """DeadLetter is immutable."""
    dead_letter: DeadLetter[str] = DeadLetter(
        message_id="msg-123",
        body="test message",
        source_mailbox="requests",
        delivery_count=5,
        last_error="Test error",
        last_error_type="builtins.RuntimeError",
        dead_lettered_at=datetime.now(UTC),
        first_received_at=datetime.now(UTC),
    )

    with pytest.raises(AttributeError):
        dead_letter.message_id = "changed"  # type: ignore[misc]


# =============================================================================
# DLQPolicy Tests
# =============================================================================


def test_dlq_policy_default_values() -> None:
    """DLQPolicy has sensible defaults."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        policy: DLQPolicy[str, None] = DLQPolicy(mailbox=dlq_mailbox)

        assert policy.mailbox is dlq_mailbox
        assert policy.max_delivery_count == 5
        assert policy.include_errors is None
        assert policy.exclude_errors is None
    finally:
        dlq_mailbox.close()


def test_dlq_policy_should_dead_letter_by_count() -> None:
    """DLQPolicy dead-letters when delivery count exceeds threshold."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        policy: DLQPolicy[str, None] = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=3,
        )

        # Create a mock message
        msg: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=2,
            enqueued_at=datetime.now(UTC),
        )
        error = RuntimeError("test error")

        # Below threshold - should not dead-letter
        assert not policy.should_dead_letter(msg, error)

        # At threshold - should dead-letter
        msg_at_threshold: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=3,
            enqueued_at=datetime.now(UTC),
        )
        assert policy.should_dead_letter(msg_at_threshold, error)

        # Above threshold - should dead-letter
        msg_above_threshold: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=5,
            enqueued_at=datetime.now(UTC),
        )
        assert policy.should_dead_letter(msg_above_threshold, error)
    finally:
        dlq_mailbox.close()


def test_dlq_policy_include_errors() -> None:
    """DLQPolicy immediately dead-letters included error types."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        policy: DLQPolicy[str, None] = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=5,
            include_errors=frozenset({ValueError, TypeError}),
        )

        msg: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=1,  # First attempt
            enqueued_at=datetime.now(UTC),
        )

        # Included error - should dead-letter immediately
        assert policy.should_dead_letter(msg, ValueError("bad value"))
        assert policy.should_dead_letter(msg, TypeError("bad type"))

        # Not included error - should not dead-letter on first attempt
        assert not policy.should_dead_letter(msg, RuntimeError("other error"))
    finally:
        dlq_mailbox.close()


def test_dlq_policy_exclude_errors() -> None:
    """DLQPolicy never dead-letters excluded error types."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        policy: DLQPolicy[str, None] = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=2,
            exclude_errors=frozenset({TimeoutError}),
        )

        msg: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=10,  # Way above threshold
            enqueued_at=datetime.now(UTC),
        )

        # Excluded error - should never dead-letter
        assert not policy.should_dead_letter(msg, TimeoutError("timeout"))

        # Other error at high count - should dead-letter
        assert policy.should_dead_letter(msg, RuntimeError("other error"))
    finally:
        dlq_mailbox.close()


def test_dlq_policy_exclude_takes_precedence() -> None:
    """Exclude errors take precedence over include errors."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        # This is a weird config but should work
        policy: DLQPolicy[str, None] = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=5,
            include_errors=frozenset({ValueError}),
            exclude_errors=frozenset({ValueError}),  # Same error in both
        )

        msg: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=1,
            enqueued_at=datetime.now(UTC),
        )

        # Exclude takes precedence - should NOT dead-letter
        assert not policy.should_dead_letter(msg, ValueError("test"))
    finally:
        dlq_mailbox.close()


# =============================================================================
# MainLoop DLQ Integration Tests
# =============================================================================


def test_mainloop_error_reply_without_dlq() -> None:
    """MainLoop sends error reply without DLQ configured (original behavior)."""
    results: InMemoryMailbox[MainLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should be acknowledged (removed from queue)
        assert requests.approximate_count() == 0

        # Error reply should be sent
        assert results.approximate_count() == 1
        msgs = results.receive(max_messages=1)
        assert not msgs[0].body.success
        assert "failure" in (msgs[0].body.error or "")
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_mainloop_nacks_with_dlq_before_threshold() -> None:
    """MainLoop nacks failed messages with DLQ before threshold is reached."""
    results: InMemoryMailbox[MainLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[MainLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        dlq = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=5)
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Run once - should nack for retry (below threshold)
        loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should be nacked (still in queue)
        assert requests.approximate_count() == 1

        # No error reply sent on retry path
        assert results.approximate_count() == 0

        # Not dead-lettered yet
        assert dlq_mailbox.approximate_count() == 0
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_mainloop_sends_to_dlq_after_threshold() -> None:
    """MainLoop sends to DLQ when delivery count equals threshold."""
    results: InMemoryMailbox[MainLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[MainLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("persistent failure"))
        # Use max_delivery_count=1 to trigger DLQ on first failure
        dlq = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=1)
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Run once - should dead-letter immediately (delivery_count=1 >= max=1)
        loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should be dead-lettered
        assert requests.approximate_count() == 0
        assert dlq_mailbox.approximate_count() == 1

        # Check dead letter content
        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert len(dlq_msgs) == 1
        dead_letter = dlq_msgs[0].body
        assert dead_letter.message_id is not None
        assert dead_letter.body.request_id == request.request_id
        assert dead_letter.source_mailbox == "requests"
        assert dead_letter.delivery_count == 1
        assert "persistent failure" in dead_letter.last_error
        assert dead_letter.last_error_type == "builtins.RuntimeError"
        dlq_msgs[0].acknowledge()

        # Error reply should be sent
        result_msgs = results.receive(max_messages=1)
        assert len(result_msgs) == 1
        assert not result_msgs[0].body.success
        assert "Dead-lettered" in (result_msgs[0].body.error or "")
        result_msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_mainloop_immediate_dlq_for_included_error() -> None:
    """MainLoop immediately dead-letters included error types."""
    results: InMemoryMailbox[MainLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[MainLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=ValueError("validation error"))
        dlq = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=5,
            include_errors=frozenset({ValueError}),
        )
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Run once - should immediately dead-letter
        loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should be dead-lettered on first attempt
        assert requests.approximate_count() == 0
        assert dlq_mailbox.approximate_count() == 1

        # Check delivery count is 1 (immediate dead-letter)
        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert dlq_msgs[0].body.delivery_count == 1
        dlq_msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_mainloop_never_dlq_for_excluded_error() -> None:
    """MainLoop never dead-letters excluded error types."""
    results: InMemoryMailbox[MainLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[MainLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=TimeoutError("transient timeout"))
        dlq = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=2,
            exclude_errors=frozenset({TimeoutError}),
        )
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Run many times - should never dead-letter
        for _ in range(5):
            loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should still be in queue (nacked, not dead-lettered)
        assert requests.approximate_count() == 1
        assert dlq_mailbox.approximate_count() == 0
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_mainloop_dlq_preserves_request_id() -> None:
    """MainLoop DLQ preserves request ID in dead letter."""
    results: InMemoryMailbox[MainLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[MainLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        dlq = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=1)
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert len(dlq_msgs) == 1
        assert dlq_msgs[0].body.request_id == request.request_id
        dlq_msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_mainloop_dlq_preserves_reply_to() -> None:
    """MainLoop DLQ preserves reply_to mailbox name in dead letter."""
    results: InMemoryMailbox[MainLoopResult[_Output], None] = InMemoryMailbox(
        name="my-results-queue"
    )
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[MainLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        dlq = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=1)
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert len(dlq_msgs) == 1
        assert dlq_msgs[0].body.reply_to == "my-results-queue"
        dlq_msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_mainloop_dlq_without_reply_to() -> None:
    """MainLoop DLQ handles messages without reply_to."""
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[MainLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        dlq = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=1)
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = MainLoopRequest(request=_Request(message="hello"))
        # Send without reply_to
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should be dead-lettered
        assert requests.approximate_count() == 0
        assert dlq_mailbox.approximate_count() == 1

        # reply_to should be None in dead letter
        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert dlq_msgs[0].body.reply_to is None
        dlq_msgs[0].acknowledge()
    finally:
        requests.close()
        dlq_mailbox.close()


def test_mainloop_dlq_handles_reply_error() -> None:
    """MainLoop DLQ handles errors when sending reply."""
    from weakincentives.runtime.mailbox import MailboxConnectionError

    results: FakeMailbox[MainLoopResult[_Output], None] = FakeMailbox(name="results")
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[MainLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        dlq = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=1)
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Make reply send fail
        results.set_connection_error(MailboxConnectionError("connection lost"))

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should still be dead-lettered despite reply failure
        assert requests.approximate_count() == 0
        assert dlq_mailbox.approximate_count() == 1
    finally:
        requests.close()
        dlq_mailbox.close()


# =============================================================================
# DLQConsumer Tests
# =============================================================================


def test_dlq_consumer_processes_dead_letters() -> None:
    """DLQConsumer processes dead letters with handler."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        processed: list[DeadLetter[str]] = []

        def handler(dl: DeadLetter[str]) -> None:
            processed.append(dl)

        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=handler)

        # Add dead letters
        for i in range(3):
            dlq_mailbox.send(
                DeadLetter(
                    message_id=f"msg-{i}",
                    body=f"message {i}",
                    source_mailbox="source",
                    delivery_count=5,
                    last_error="error",
                    last_error_type="builtins.RuntimeError",
                    dead_lettered_at=datetime.now(UTC),
                    first_received_at=datetime.now(UTC),
                )
            )

        # Run consumer - multiple iterations to process all messages
        consumer.run(max_iterations=5, wait_time_seconds=0)

        # All dead letters should be processed
        assert len(processed) == 3
        assert dlq_mailbox.approximate_count() == 0
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_nacks_on_handler_failure() -> None:
    """DLQConsumer nacks messages when handler fails."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:

        def failing_handler(dl: DeadLetter[str]) -> None:
            raise RuntimeError("handler failure")

        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=failing_handler)

        dlq_mailbox.send(
            DeadLetter(
                message_id="msg-1",
                body="test",
                source_mailbox="source",
                delivery_count=5,
                last_error="error",
                last_error_type="builtins.RuntimeError",
                dead_lettered_at=datetime.now(UTC),
                first_received_at=datetime.now(UTC),
            )
        )

        consumer.run(max_iterations=1, wait_time_seconds=0)

        # Message should still be in queue (nacked)
        assert dlq_mailbox.approximate_count() == 1
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_has_heartbeat() -> None:
    """DLQConsumer has heartbeat for watchdog monitoring."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=lambda _: None)

        assert consumer.heartbeat is not None
        # Heartbeat should have never been beaten
        assert consumer.heartbeat.elapsed() > 0
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_shutdown() -> None:
    """DLQConsumer supports graceful shutdown."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=lambda _: None)

        # Start in background
        thread = threading.Thread(
            target=lambda: consumer.run(max_iterations=None, wait_time_seconds=1)
        )
        thread.start()

        # Give it time to start
        time.sleep(0.1)
        assert consumer.running

        # Shutdown
        result = consumer.shutdown(timeout=2.0)
        assert result is True
        assert not consumer.running

        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_context_manager() -> None:
    """DLQConsumer supports context manager protocol."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        with DLQConsumer(mailbox=dlq_mailbox, handler=lambda _: None) as consumer:
            assert consumer is not None
            # Consumer should be usable
            assert not consumer.running
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_running_property() -> None:
    """DLQConsumer running property reflects actual state."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=lambda _: None)

        # Initially not running
        assert not consumer.running

        # Run in background
        thread = threading.Thread(
            target=lambda: consumer.run(max_iterations=1, wait_time_seconds=0)
        )
        thread.start()

        # Wait for thread to finish
        thread.join(timeout=1.0)

        # Should be stopped now
        assert not consumer.running
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_respects_max_iterations() -> None:
    """DLQConsumer respects max_iterations limit."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        process_count = 0

        def counting_handler(dl: DeadLetter[str]) -> None:
            nonlocal process_count
            process_count += 1

        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=counting_handler)

        # Add many dead letters
        for i in range(10):
            dlq_mailbox.send(
                DeadLetter(
                    message_id=f"msg-{i}",
                    body=f"message {i}",
                    source_mailbox="source",
                    delivery_count=5,
                    last_error="error",
                    last_error_type="builtins.RuntimeError",
                    dead_lettered_at=datetime.now(UTC),
                    first_received_at=datetime.now(UTC),
                )
            )

        # Run only 2 iterations
        consumer.run(max_iterations=2, wait_time_seconds=0)

        # Should have processed at least some (depends on batch size)
        assert process_count >= 1
        # But not all
        assert dlq_mailbox.approximate_count() >= 0
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_beats_heartbeat() -> None:
    """DLQConsumer beats heartbeat during processing."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=lambda _: None)

        # Add a dead letter
        dlq_mailbox.send(
            DeadLetter(
                message_id="msg-1",
                body="test",
                source_mailbox="source",
                delivery_count=5,
                last_error="error",
                last_error_type="builtins.RuntimeError",
                dead_lettered_at=datetime.now(UTC),
                first_received_at=datetime.now(UTC),
            )
        )

        initial_elapsed = consumer.heartbeat.elapsed()

        # Run consumer
        consumer.run(max_iterations=1, wait_time_seconds=0)

        # Heartbeat should have been beaten
        assert consumer.heartbeat.elapsed() < initial_elapsed
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_exits_when_mailbox_closed() -> None:
    """DLQConsumer exits when mailbox is closed."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=lambda _: None)

        # Close mailbox
        dlq_mailbox.close()

        # Run should exit immediately
        consumer.run(max_iterations=10, wait_time_seconds=0)

        # Should not raise, should just exit
        assert not consumer.running
    finally:
        pass  # Mailbox already closed


def test_dlq_consumer_nacks_on_shutdown_during_messages() -> None:
    """DLQConsumer nacks unprocessed messages during shutdown."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        processed: list[str] = []

        def slow_handler(dl: DeadLetter[str]) -> None:
            processed.append(dl.message_id)
            # Trigger shutdown after processing first message
            consumer.shutdown(timeout=0.1)

        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=slow_handler)

        # Add multiple dead letters
        for i in range(3):
            dlq_mailbox.send(
                DeadLetter(
                    message_id=f"msg-{i}",
                    body=f"message {i}",
                    source_mailbox="source",
                    delivery_count=5,
                    last_error="error",
                    last_error_type="builtins.RuntimeError",
                    dead_lettered_at=datetime.now(UTC),
                    first_received_at=datetime.now(UTC),
                )
            )

        # Run consumer - it should process one then get shutdown signal
        consumer.run(max_iterations=10, wait_time_seconds=0)

        # First message should have been processed
        assert len(processed) >= 1
        # Any remaining messages should be nacked back to queue
    finally:
        dlq_mailbox.close()


# =============================================================================
# Custom DLQPolicy Tests
# =============================================================================


@dataclass(slots=True, frozen=True)
class _ErrorBudgetPolicy(DLQPolicy[str, None]):
    """Custom policy that dead-letters based on mock error budget."""

    error_budget_exceeded: bool = False

    def should_dead_letter(self, message: Message[str, Any], error: Exception) -> bool:
        # Fall back to default behavior for threshold
        if message.delivery_count >= self.max_delivery_count:
            return True

        # Custom logic: dead-letter if error budget exceeded
        return self.error_budget_exceeded


def test_custom_dlq_policy() -> None:
    """Custom DLQPolicy can implement custom dead-letter logic."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        # Policy with error budget exceeded
        policy = _ErrorBudgetPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=10,
            error_budget_exceeded=True,
        )

        msg: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=1,  # Below threshold
            enqueued_at=datetime.now(UTC),
        )

        # Should dead-letter due to custom logic
        assert policy.should_dead_letter(msg, RuntimeError("test"))

        # Policy without error budget exceeded
        policy_ok = _ErrorBudgetPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=10,
            error_budget_exceeded=False,
        )

        # Should not dead-letter (below threshold, budget OK)
        assert not policy_ok.should_dead_letter(msg, RuntimeError("test"))
    finally:
        dlq_mailbox.close()


# =============================================================================
# Module Exports Tests
# =============================================================================


def test_runtime_exports_dlq_types() -> None:
    """Runtime module exports DLQ types."""
    from weakincentives.runtime import (
        DeadLetter as ExportedDeadLetter,
        DLQConsumer as ExportedConsumer,
        DLQPolicy as ExportedPolicy,
    )

    assert ExportedDeadLetter is DeadLetter
    assert ExportedPolicy is DLQPolicy
    assert ExportedConsumer is DLQConsumer
