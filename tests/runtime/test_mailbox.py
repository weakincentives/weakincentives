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

"""Tests for mailbox implementations."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC

import pytest

from weakincentives.runtime.mailbox import (
    CollectingMailbox,
    CompositeResolver,
    FakeMailbox,
    InMemoryMailbox,
    InMemoryMailboxFactory,
    Mailbox,
    MailboxConnectionError,
    MailboxFactory,
    MailboxFullError,
    MailboxResolutionError,
    Message,
    MessageFinalizedError,
    NullMailbox,
    ReceiptHandleExpiredError,
    RegistryResolver,
    ReplyNotAvailableError,
)


@dataclass(slots=True, frozen=True)
class _Event:
    """Sample event type for testing."""

    data: str


@dataclass(slots=True, frozen=True)
class _Result:
    """Sample result type for testing."""

    status: str


# =============================================================================
# Message Tests
# =============================================================================


def test_message_is_mutable_for_finalized_state() -> None:
    """Message instances have mutable _finalized state but frozen public fields."""
    msg: Message[str, str] = Message(
        id="msg-1",
        body="hello",
        receipt_handle="handle-1",
        delivery_count=1,
        enqueued_at=time.time(),  # type: ignore[arg-type]
    )
    # Public fields should not be directly modifiable
    assert msg.id == "msg-1"
    assert msg.is_finalized is False


def test_message_lifecycle_methods_delegate_to_callbacks() -> None:
    """Message lifecycle methods call bound callbacks."""
    ack_called = []
    nack_called = []
    extend_called = []

    msg: Message[str, str] = Message(
        id="msg-1",
        body="hello",
        receipt_handle="handle-1",
        delivery_count=1,
        enqueued_at=time.time(),  # type: ignore[arg-type]
        _acknowledge_fn=lambda: ack_called.append(True),
        _nack_fn=lambda t: nack_called.append(t),
        _extend_fn=lambda t: extend_called.append(t),
    )

    msg.acknowledge()
    assert ack_called == [True]
    assert msg.is_finalized is True

    msg._finalized = False  # Reset for testing nack
    msg.nack(visibility_timeout=30)
    assert nack_called == [30]
    assert msg.is_finalized is True

    msg._finalized = False  # Reset for testing extend
    msg.extend_visibility(60)
    assert extend_called == [60]


def test_message_reply_without_reply_to_raises() -> None:
    """Message.reply() raises ReplyNotAvailableError when no reply_to."""
    msg: Message[str, str] = Message(
        id="msg-1",
        body="hello",
        receipt_handle="handle-1",
        delivery_count=1,
        enqueued_at=time.time(),  # type: ignore[arg-type]
        reply_to=None,
    )

    with pytest.raises(ReplyNotAvailableError):
        msg.reply("response")


def test_message_reply_after_acknowledge_raises() -> None:
    """Message.reply() raises MessageFinalizedError after acknowledge."""
    responses: CollectingMailbox[str, None] = CollectingMailbox(name="responses")
    msg: Message[str, str] = Message(
        id="msg-1",
        body="hello",
        receipt_handle="handle-1",
        delivery_count=1,
        enqueued_at=time.time(),  # type: ignore[arg-type]
        reply_to=responses,
    )

    msg.acknowledge()

    with pytest.raises(MessageFinalizedError):
        msg.reply("response")


def test_message_reply_after_nack_raises() -> None:
    """Message.reply() raises MessageFinalizedError after nack."""
    responses: CollectingMailbox[str, None] = CollectingMailbox(name="responses")
    msg: Message[str, str] = Message(
        id="msg-1",
        body="hello",
        receipt_handle="handle-1",
        delivery_count=1,
        enqueued_at=time.time(),  # type: ignore[arg-type]
        reply_to=responses,
    )

    msg.nack(visibility_timeout=0)

    with pytest.raises(MessageFinalizedError):
        msg.reply("response")


def test_message_reply_success() -> None:
    """Message.reply() sends to reply_to destination."""
    responses: CollectingMailbox[str, None] = CollectingMailbox(name="responses")

    msg: Message[str, str] = Message(
        id="msg-1",
        body="hello",
        receipt_handle="handle-1",
        delivery_count=1,
        enqueued_at=time.time(),  # type: ignore[arg-type]
        reply_to=responses,
    )

    result_id = msg.reply("response-body")
    assert isinstance(result_id, str)
    assert len(result_id) > 0
    assert responses.sent == ["response-body"]


# =============================================================================
# InMemoryMailbox Tests
# =============================================================================


class TestInMemoryMailbox:  # noqa: PLR0904
    """Tests for InMemoryMailbox implementation."""

    def test_send_returns_message_id(self) -> None:
        """send() returns a unique message ID."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            msg_id = mailbox.send("hello")
            assert isinstance(msg_id, str)
            assert len(msg_id) > 0
        finally:
            mailbox.close()

    def test_send_and_receive_basic(self) -> None:
        """Basic send and receive workflow."""
        mailbox: InMemoryMailbox[_Event, None] = InMemoryMailbox(name="test")
        try:
            event = _Event(data="test-data")
            mailbox.send(event)

            messages = mailbox.receive(max_messages=1)
            assert len(messages) == 1
            assert messages[0].body == event
            assert messages[0].delivery_count == 1
        finally:
            mailbox.close()

    def test_send_with_reply_to(self) -> None:
        """send() accepts reply_to parameter."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox(name="test")
        responses: InMemoryMailbox[str, None] = InMemoryMailbox(name="responses")
        try:
            mailbox.send("hello", reply_to=responses)
            messages = mailbox.receive(max_messages=1)
            assert len(messages) == 1
            assert messages[0].reply_to is responses
        finally:
            mailbox.close()
            responses.close()

    def test_receive_empty_queue(self) -> None:
        """receive() returns empty list when queue is empty."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            messages = mailbox.receive(max_messages=1)
            assert messages == []
        finally:
            mailbox.close()

    def test_receive_max_messages(self) -> None:
        """receive() respects max_messages limit."""
        mailbox: InMemoryMailbox[int, None] = InMemoryMailbox(name="test")
        try:
            for i in range(5):
                mailbox.send(i)

            messages = mailbox.receive(max_messages=3)
            assert len(messages) == 3
            assert [m.body for m in messages] == [0, 1, 2]
        finally:
            mailbox.close()

    def test_receive_fifo_order(self) -> None:
        """Messages are received in FIFO order."""
        mailbox: InMemoryMailbox[int, None] = InMemoryMailbox(name="test")
        try:
            for i in range(3):
                mailbox.send(i)

            messages = mailbox.receive(max_messages=3)
            assert [m.body for m in messages] == [0, 1, 2]
        finally:
            mailbox.close()

    def test_acknowledge_removes_message(self) -> None:
        """acknowledge() removes message from queue."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive(max_messages=1)
            assert len(messages) == 1

            messages[0].acknowledge()
            assert mailbox.approximate_count() == 0
        finally:
            mailbox.close()

    def test_acknowledge_expired_handle_raises(self) -> None:
        """acknowledge() raises ReceiptHandleExpiredError for invalid handle."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive(max_messages=1)
            messages[0].acknowledge()

            with pytest.raises(ReceiptHandleExpiredError):
                messages[0].acknowledge()
        finally:
            mailbox.close()

    def test_nack_returns_message_to_queue(self) -> None:
        """nack() returns message to queue for redelivery."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive(max_messages=1)
            messages[0].nack(visibility_timeout=0)

            # Message should be available again
            messages = mailbox.receive(max_messages=1)
            assert len(messages) == 1
            assert messages[0].body == "hello"
            assert messages[0].delivery_count == 2
        finally:
            mailbox.close()

    def test_nack_expired_handle_raises(self) -> None:
        """nack() raises ReceiptHandleExpiredError for invalid handle."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive(max_messages=1)
            messages[0].acknowledge()

            with pytest.raises(ReceiptHandleExpiredError):
                messages[0].nack(visibility_timeout=0)
        finally:
            mailbox.close()

    def test_extend_visibility_expired_handle_raises(self) -> None:
        """extend_visibility() raises ReceiptHandleExpiredError for invalid handle."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive(max_messages=1)
            messages[0].acknowledge()

            with pytest.raises(ReceiptHandleExpiredError):
                messages[0].extend_visibility(60)
        finally:
            mailbox.close()

    def test_visibility_timeout_requeues_message(self) -> None:
        """Message is requeued after visibility timeout expires."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive(max_messages=1, visibility_timeout=1)
            assert len(messages) == 1

            # Wait for visibility timeout to expire
            time.sleep(1.2)

            # Message should be available again
            messages = mailbox.receive(max_messages=1)
            assert len(messages) == 1
            assert messages[0].body == "hello"
            assert messages[0].delivery_count == 2
        finally:
            mailbox.close()

    def test_purge_removes_all_messages(self) -> None:
        """purge() removes all messages from queue."""
        mailbox: InMemoryMailbox[int, None] = InMemoryMailbox(name="test")
        try:
            for i in range(5):
                mailbox.send(i)

            count = mailbox.purge()
            assert count == 5
            assert mailbox.approximate_count() == 0
        finally:
            mailbox.close()

    def test_approximate_count(self) -> None:
        """approximate_count() returns correct message count."""
        mailbox: InMemoryMailbox[int, None] = InMemoryMailbox(name="test")
        try:
            assert mailbox.approximate_count() == 0

            for i in range(3):
                mailbox.send(i)
            assert mailbox.approximate_count() == 3

            mailbox.receive(max_messages=1)[0].acknowledge()
            assert mailbox.approximate_count() == 2
        finally:
            mailbox.close()

    def test_max_size_enforced(self) -> None:
        """MailboxFullError raised when max_size exceeded."""
        mailbox: InMemoryMailbox[int, None] = InMemoryMailbox(name="test", max_size=2)
        try:
            mailbox.send(1)
            mailbox.send(2)

            with pytest.raises(MailboxFullError):
                mailbox.send(3)
        finally:
            mailbox.close()

    def test_long_poll_wait_time(self) -> None:
        """wait_time_seconds blocks until message arrives."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:

            def sender() -> None:
                time.sleep(0.2)
                mailbox.send("hello")

            thread = threading.Thread(target=sender)
            thread.start()

            start = time.monotonic()
            messages = mailbox.receive(max_messages=1, wait_time_seconds=1)
            elapsed = time.monotonic() - start

            thread.join()
            assert len(messages) == 1
            assert messages[0].body == "hello"
            # Should have waited ~0.2s, not the full 1s
            assert elapsed < 0.5
        finally:
            mailbox.close()

    def test_long_poll_timeout(self) -> None:
        """wait_time_seconds returns empty on timeout."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            start = time.monotonic()
            messages = mailbox.receive(max_messages=1, wait_time_seconds=0.2)
            elapsed = time.monotonic() - start

            assert len(messages) == 0
            assert elapsed >= 0.2
        finally:
            mailbox.close()

    def test_thread_safety(self) -> None:
        """Mailbox is thread-safe for concurrent access."""
        mailbox: InMemoryMailbox[int, None] = InMemoryMailbox(name="test")
        try:
            num_messages = 100
            num_threads = 4

            def sender(start: int) -> None:
                for i in range(num_messages // num_threads):
                    mailbox.send(start + i)

            def receiver() -> int:
                count = 0
                while True:
                    messages = mailbox.receive(max_messages=10, visibility_timeout=30)
                    if not messages:
                        break
                    for msg in messages:
                        msg.acknowledge()
                        count += 1
                return count

            # Start sender threads
            sender_threads = [
                threading.Thread(
                    target=sender, args=(i * (num_messages // num_threads),)
                )
                for i in range(num_threads)
            ]
            for t in sender_threads:
                t.start()
            for t in sender_threads:
                t.join()

            # Wait for messages to be available
            time.sleep(0.1)

            # Receive all messages
            total_received = 0
            while mailbox.approximate_count() > 0:
                messages = mailbox.receive(max_messages=10)
                for msg in messages:
                    msg.acknowledge()
                    total_received += 1

            assert total_received == num_messages
        finally:
            mailbox.close()

    def test_message_enqueued_at(self) -> None:
        """Message enqueued_at is set correctly."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive(max_messages=1)
            assert messages[0].enqueued_at.tzinfo == UTC
        finally:
            mailbox.close()

    def test_reply_with_mailbox_reference(self) -> None:
        """Message.reply() sends directly to reply_to mailbox."""
        responses: InMemoryMailbox[str, None] = InMemoryMailbox(name="responses")
        requests: InMemoryMailbox[str, str] = InMemoryMailbox(name="requests")
        try:
            requests.send("hello", reply_to=responses)
            messages = requests.receive(max_messages=1)
            assert len(messages) == 1

            reply_id = messages[0].reply("reply-body")
            assert isinstance(reply_id, str)

            # Verify reply was sent to responses mailbox
            reply_messages = responses.receive(max_messages=1)
            assert len(reply_messages) == 1
            assert reply_messages[0].body == "reply-body"

            messages[0].acknowledge()
        finally:
            requests.close()
            responses.close()

    def test_reply_sends_directly_to_mailbox(self) -> None:
        """Reply sends directly to the reply_to mailbox reference."""
        requests: InMemoryMailbox[str, str] = InMemoryMailbox(name="requests")
        responses: InMemoryMailbox[str, None] = InMemoryMailbox(name="responses")
        try:
            # Send with reply_to mailbox reference
            requests.send("hello", reply_to=responses)
            messages = requests.receive(max_messages=1)

            # Reply should send directly to responses mailbox
            messages[0].reply("response")
            messages[0].acknowledge()

            # Response should be in the responses mailbox
            response_msgs = responses.receive(max_messages=1)
            assert len(response_msgs) == 1
            assert response_msgs[0].body == "response"
            response_msgs[0].acknowledge()
        finally:
            requests.close()
            responses.close()

    def test_reply_without_reply_to_raises(self) -> None:
        """Message.reply() raises when no reply_to specified."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox(name="test")
        try:
            # Send without reply_to
            mailbox.send("hello")
            messages = mailbox.receive(max_messages=1)

            with pytest.raises(ReplyNotAvailableError, match="no reply_to"):
                messages[0].reply("response")
        finally:
            mailbox.close()


# =============================================================================
# NullMailbox Tests
# =============================================================================


class TestNullMailbox:
    """Tests for NullMailbox implementation."""

    def test_send_returns_id(self) -> None:
        """send() returns a message ID even though message is dropped."""
        mailbox: NullMailbox[str, None] = NullMailbox()
        msg_id = mailbox.send("hello")
        assert isinstance(msg_id, str)
        assert len(msg_id) > 0

    def test_receive_always_empty(self) -> None:
        """receive() always returns empty list."""
        mailbox: NullMailbox[str, None] = NullMailbox()
        mailbox.send("hello")
        mailbox.send("world")
        assert mailbox.receive(max_messages=10) == []

    def test_purge_returns_zero(self) -> None:
        """purge() returns zero (nothing to purge)."""
        mailbox: NullMailbox[str, None] = NullMailbox()
        mailbox.send("hello")
        assert mailbox.purge() == 0

    def test_approximate_count_zero(self) -> None:
        """approximate_count() always returns zero."""
        mailbox: NullMailbox[str, None] = NullMailbox()
        mailbox.send("hello")
        assert mailbox.approximate_count() == 0


# =============================================================================
# CollectingMailbox Tests
# =============================================================================


class TestCollectingMailbox:
    """Tests for CollectingMailbox implementation."""

    def test_send_collects_messages(self) -> None:
        """send() stores messages in sent list."""
        mailbox: CollectingMailbox[_Event, None] = CollectingMailbox()
        event1 = _Event(data="first")
        event2 = _Event(data="second")
        mailbox.send(event1)
        mailbox.send(event2)

        assert len(mailbox.sent) == 2
        assert mailbox.sent[0] == event1
        assert mailbox.sent[1] == event2

    def test_receive_always_empty(self) -> None:
        """receive() returns empty (collecting only)."""
        mailbox: CollectingMailbox[str, None] = CollectingMailbox()
        mailbox.send("hello")
        assert mailbox.receive(max_messages=10) == []

    def test_purge_clears_collected(self) -> None:
        """purge() clears all collected messages."""
        mailbox: CollectingMailbox[str, None] = CollectingMailbox()
        mailbox.send("hello")
        mailbox.send("world")

        count = mailbox.purge()
        assert count == 2
        assert mailbox.sent == []

    def test_approximate_count(self) -> None:
        """approximate_count() returns number of collected messages."""
        mailbox: CollectingMailbox[str, None] = CollectingMailbox()
        assert mailbox.approximate_count() == 0
        mailbox.send("hello")
        assert mailbox.approximate_count() == 1


# =============================================================================
# FakeMailbox Tests
# =============================================================================


class TestFakeMailbox:
    """Tests for FakeMailbox implementation."""

    def test_basic_send_receive(self) -> None:
        """Basic send and receive workflow."""
        mailbox: FakeMailbox[_Event, None] = FakeMailbox()
        event = _Event(data="test")
        mailbox.send(event)

        messages = mailbox.receive(max_messages=1)
        assert len(messages) == 1
        assert messages[0].body == event
        assert messages[0].delivery_count == 1

    def test_acknowledge(self) -> None:
        """acknowledge() removes message."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)
        messages[0].acknowledge()

        assert mailbox.approximate_count() == 0

    def test_nack_requeues(self) -> None:
        """nack() returns message to queue."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)
        messages[0].nack(visibility_timeout=0)

        messages = mailbox.receive(max_messages=1)
        assert len(messages) == 1
        assert messages[0].delivery_count == 2

    def test_expire_handle(self) -> None:
        """expire_handle() causes ReceiptHandleExpiredError on acknowledge."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)

        mailbox.expire_handle(messages[0].receipt_handle)

        with pytest.raises(ReceiptHandleExpiredError):
            messages[0].acknowledge()

    def test_expire_handle_affects_nack(self) -> None:
        """expire_handle() causes ReceiptHandleExpiredError on nack."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)

        mailbox.expire_handle(messages[0].receipt_handle)

        with pytest.raises(ReceiptHandleExpiredError):
            messages[0].nack()

    def test_expire_handle_affects_extend(self) -> None:
        """expire_handle() causes ReceiptHandleExpiredError on extend_visibility."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)

        mailbox.expire_handle(messages[0].receipt_handle)

        with pytest.raises(ReceiptHandleExpiredError):
            messages[0].extend_visibility(60)

    def test_set_connection_error_affects_send(self) -> None:
        """set_connection_error() causes error on send."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        error = MailboxConnectionError("Redis down")
        mailbox.set_connection_error(error)

        with pytest.raises(MailboxConnectionError, match="Redis down"):
            mailbox.send("hello")

    def test_set_connection_error_affects_receive(self) -> None:
        """set_connection_error() causes error on receive."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")
        mailbox.set_connection_error(MailboxConnectionError("Redis down"))

        with pytest.raises(MailboxConnectionError):
            mailbox.receive(max_messages=1)

    def test_set_connection_error_affects_purge(self) -> None:
        """set_connection_error() causes error on purge."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.set_connection_error(MailboxConnectionError("Redis down"))

        with pytest.raises(MailboxConnectionError):
            mailbox.purge()

    def test_set_connection_error_affects_count(self) -> None:
        """set_connection_error() causes error on approximate_count."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.set_connection_error(MailboxConnectionError("Redis down"))

        with pytest.raises(MailboxConnectionError):
            mailbox.approximate_count()

    def test_clear_connection_error(self) -> None:
        """clear_connection_error() removes pending error."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.set_connection_error(MailboxConnectionError("Redis down"))
        mailbox.clear_connection_error()

        # Should work now
        mailbox.send("hello")
        assert mailbox.approximate_count() == 1

    def test_inject_message(self) -> None:
        """inject_message() adds message directly to queue."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        msg_id = mailbox.inject_message("injected", delivery_count=3)

        messages = mailbox.receive(max_messages=1)
        assert len(messages) == 1
        assert messages[0].body == "injected"
        assert messages[0].id == msg_id
        assert messages[0].delivery_count == 4  # 3 + 1 for this receive

    def test_inject_message_with_custom_id(self) -> None:
        """inject_message() respects custom message ID."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.inject_message("test", msg_id="custom-id")

        messages = mailbox.receive(max_messages=1)
        assert messages[0].id == "custom-id"

    def test_acknowledge_unknown_handle_raises(self) -> None:
        """acknowledge() raises for unknown (not expired) handle."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)
        messages[0].acknowledge()  # First acknowledge succeeds

        # Second acknowledge with same handle fails (handle no longer in invisible)
        with pytest.raises(ReceiptHandleExpiredError, match="not found"):
            messages[0].acknowledge()

    def test_nack_unknown_handle_raises(self) -> None:
        """nack() raises for unknown (not expired) handle."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)
        messages[0].acknowledge()  # Remove from invisible

        with pytest.raises(ReceiptHandleExpiredError, match="not found"):
            messages[0].nack()

    def test_extend_unknown_handle_raises(self) -> None:
        """extend_visibility() raises for unknown (not expired) handle."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)
        messages[0].acknowledge()  # Remove from invisible

        with pytest.raises(ReceiptHandleExpiredError, match="not found"):
            messages[0].extend_visibility(60)

    def test_extend_visibility_success(self) -> None:
        """extend_visibility() succeeds for valid handle."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)

        # Extend visibility should not raise
        messages[0].extend_visibility(60)

        # Message should still be in invisible state
        assert mailbox.approximate_count() == 1
        messages[0].acknowledge()
        assert mailbox.approximate_count() == 0

    def test_purge_without_connection_error(self) -> None:
        """purge() works normally without connection error."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")
        mailbox.send("world")

        count = mailbox.purge()
        assert count == 2
        assert mailbox.approximate_count() == 0

    def test_send_with_reply_to(self) -> None:
        """send() accepts reply_to parameter."""
        mailbox: FakeMailbox[str, str] = FakeMailbox()
        responses: CollectingMailbox[str, None] = CollectingMailbox(name="responses")
        mailbox.send("hello", reply_to=responses)
        messages = mailbox.receive(max_messages=1)
        assert messages[0].reply_to is responses

    def test_inject_message_with_reply_to(self) -> None:
        """inject_message() accepts reply_to parameter."""
        mailbox: FakeMailbox[str, str] = FakeMailbox()
        responses: CollectingMailbox[str, None] = CollectingMailbox(name="responses")
        mailbox.inject_message("test", reply_to=responses)
        messages = mailbox.receive(max_messages=1)
        assert messages[0].reply_to is responses


# =============================================================================
# FakeMailbox Reply Tests
# =============================================================================


def test_fake_mailbox_reply_without_reply_to_raises() -> None:
    """FakeMailbox Message.reply() raises when no reply_to specified."""
    mailbox: FakeMailbox[str, str] = FakeMailbox(name="test")
    # Send without reply_to
    mailbox.send("hello")
    messages = mailbox.receive(max_messages=1)

    with pytest.raises(ReplyNotAvailableError, match="no reply_to"):
        messages[0].reply("response")


def test_fake_mailbox_reply_sends_to_mailbox() -> None:
    """FakeMailbox reply sends to the reply_to mailbox."""
    mailbox: FakeMailbox[str, str] = FakeMailbox(name="test")
    responses: CollectingMailbox[str, None] = CollectingMailbox(name="responses")
    mailbox.send("hello", reply_to=responses)
    messages = mailbox.receive(max_messages=1)

    # Reply should send directly to responses mailbox
    messages[0].reply("response")
    messages[0].acknowledge()

    # Response should be in the CollectingMailbox
    assert responses.sent == ["response"]


# =============================================================================
# Additional InMemoryMailbox Tests for Coverage
# =============================================================================


class TestInMemoryMailboxCoverage:
    """Additional tests for InMemoryMailbox coverage."""

    def test_extend_visibility_success(self) -> None:
        """extend_visibility() extends timeout for valid handle."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive(max_messages=1, visibility_timeout=1)
            assert len(messages) == 1

            # Extend visibility
            messages[0].extend_visibility(60)

            # Wait for original timeout to pass
            time.sleep(1.2)

            # Message should still be invisible (not requeued)
            assert mailbox.approximate_count() == 1
            new_messages = mailbox.receive(max_messages=1)
            assert len(new_messages) == 0  # Still invisible

            # Acknowledge to clean up
            messages[0].acknowledge()
        finally:
            mailbox.close()


# =============================================================================
# Protocol Tests
# =============================================================================


def test_in_memory_mailbox_is_mailbox() -> None:
    """InMemoryMailbox implements Mailbox protocol."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox()
    try:
        assert isinstance(mailbox, Mailbox)
    finally:
        mailbox.close()


def test_null_mailbox_is_mailbox() -> None:
    """NullMailbox implements Mailbox protocol."""
    mailbox: NullMailbox[str, None] = NullMailbox()
    assert isinstance(mailbox, Mailbox)


def test_collecting_mailbox_is_mailbox() -> None:
    """CollectingMailbox implements Mailbox protocol."""
    mailbox: CollectingMailbox[str, None] = CollectingMailbox()
    assert isinstance(mailbox, Mailbox)


def test_fake_mailbox_is_mailbox() -> None:
    """FakeMailbox implements Mailbox protocol."""
    mailbox: FakeMailbox[str, None] = FakeMailbox()
    assert isinstance(mailbox, Mailbox)


# =============================================================================
# Closed State Tests
# =============================================================================


def test_in_memory_mailbox_closed_initially_false() -> None:
    """InMemoryMailbox.closed is False initially."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox()
    try:
        assert mailbox.closed is False
    finally:
        mailbox.close()


def test_in_memory_mailbox_closed_after_close() -> None:
    """InMemoryMailbox.closed is True after close()."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox()
    mailbox.close()
    assert mailbox.closed is True


def test_in_memory_mailbox_receive_returns_empty_when_closed() -> None:
    """InMemoryMailbox.receive() returns empty when closed."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox()
    mailbox.send("test")
    mailbox.close()

    # Should return empty even though there's a message
    messages = mailbox.receive(max_messages=1)
    assert len(messages) == 0


def test_in_memory_mailbox_close_wakes_blocked_receivers() -> None:
    """InMemoryMailbox.close() wakes receivers blocked on wait."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox()
    received: list[bool] = []

    def receiver() -> None:
        # This would block for 10 seconds normally
        _ = mailbox.receive(max_messages=1, wait_time_seconds=10)
        received.append(True)

    thread = threading.Thread(target=receiver)
    thread.start()

    # Give thread time to start waiting
    time.sleep(0.1)

    # Close should wake the receiver
    mailbox.close()

    # Thread should exit quickly
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert len(received) == 1


def test_null_mailbox_closed_initially_false() -> None:
    """NullMailbox.closed is False initially."""
    mailbox: NullMailbox[str, None] = NullMailbox()
    assert mailbox.closed is False


def test_null_mailbox_closed_after_close() -> None:
    """NullMailbox.closed is True after close()."""
    mailbox: NullMailbox[str, None] = NullMailbox()
    mailbox.close()
    assert mailbox.closed is True


def test_collecting_mailbox_closed_initially_false() -> None:
    """CollectingMailbox.closed is False initially."""
    mailbox: CollectingMailbox[str, None] = CollectingMailbox()
    assert mailbox.closed is False


def test_collecting_mailbox_closed_after_close() -> None:
    """CollectingMailbox.closed is True after close()."""
    mailbox: CollectingMailbox[str, None] = CollectingMailbox()
    mailbox.close()
    assert mailbox.closed is True


def test_fake_mailbox_closed_initially_false() -> None:
    """FakeMailbox.closed is False initially."""
    mailbox: FakeMailbox[str, None] = FakeMailbox()
    assert mailbox.closed is False


def test_fake_mailbox_closed_after_close() -> None:
    """FakeMailbox.closed is True after close()."""
    mailbox: FakeMailbox[str, None] = FakeMailbox()
    mailbox.close()
    assert mailbox.closed is True


# =============================================================================
# Resolver Tests
# =============================================================================


def test_registry_resolver_resolve() -> None:
    """RegistryResolver.resolve() returns registered mailbox."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = RegistryResolver[str]({"test": mailbox})
        resolved = resolver.resolve("test")
        assert resolved is mailbox
    finally:
        mailbox.close()


def test_registry_resolver_resolve_optional() -> None:
    """RegistryResolver.resolve_optional() returns None for unknown."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = RegistryResolver[str]({"test": mailbox})
        assert resolver.resolve_optional("unknown") is None
        assert resolver.resolve_optional("test") is mailbox
    finally:
        mailbox.close()


def test_registry_resolver_resolve_raises_on_unknown() -> None:
    """RegistryResolver.resolve() raises MailboxResolutionError for unknown identifier."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = RegistryResolver[str]({"test": mailbox})
        with pytest.raises(MailboxResolutionError) as exc_info:
            resolver.resolve("unknown")
        assert exc_info.value.identifier == "unknown"
        assert "unknown" in str(exc_info.value)
    finally:
        mailbox.close()


def test_composite_resolver_resolve_from_registry() -> None:
    """CompositeResolver.resolve() returns mailbox from registry."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = CompositeResolver[str](registry={"test": mailbox}, factory=None)
        resolved = resolver.resolve("test")
        assert resolved is mailbox
    finally:
        mailbox.close()


def test_composite_resolver_resolve_raises_without_factory() -> None:
    """CompositeResolver.resolve() raises when identifier not in registry and no factory."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = CompositeResolver[str](registry={"test": mailbox}, factory=None)
        with pytest.raises(MailboxResolutionError) as exc_info:
            resolver.resolve("unknown")
        assert exc_info.value.identifier == "unknown"
    finally:
        mailbox.close()


def test_composite_resolver_resolve_uses_factory() -> None:
    """CompositeResolver.resolve() uses factory for unknown identifiers."""
    created_mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="created")

    class _TestFactory(MailboxFactory[str]):
        def create(self, identifier: str) -> Mailbox[str, None]:
            return created_mailbox

    try:
        resolver = CompositeResolver[str](registry={}, factory=_TestFactory())
        resolved = resolver.resolve("dynamic")
        assert resolved is created_mailbox
    finally:
        created_mailbox.close()


def test_composite_resolver_resolve_optional_from_registry() -> None:
    """CompositeResolver.resolve_optional() returns mailbox from registry."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = CompositeResolver[str](registry={"test": mailbox}, factory=None)
        resolved = resolver.resolve_optional("test")
        assert resolved is mailbox
    finally:
        mailbox.close()


def test_composite_resolver_resolve_optional_none_without_factory() -> None:
    """CompositeResolver.resolve_optional() returns None when no factory and not in registry."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = CompositeResolver[str](registry={"test": mailbox}, factory=None)
        assert resolver.resolve_optional("unknown") is None
    finally:
        mailbox.close()


def test_composite_resolver_resolve_optional_uses_factory() -> None:
    """CompositeResolver.resolve_optional() uses factory for unknown identifiers."""
    created_mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="created")

    class _TestFactory(MailboxFactory[str]):
        def create(self, identifier: str) -> Mailbox[str, None]:
            return created_mailbox

    try:
        resolver = CompositeResolver[str](registry={}, factory=_TestFactory())
        resolved = resolver.resolve_optional("dynamic")
        assert resolved is created_mailbox
    finally:
        created_mailbox.close()


def test_composite_resolver_resolve_optional_catches_resolution_error() -> None:
    """CompositeResolver.resolve_optional() returns None when factory raises MailboxResolutionError."""

    class _FailingFactory(MailboxFactory[str]):
        def create(self, identifier: str) -> Mailbox[str, None]:
            raise MailboxResolutionError(identifier)

    resolver = CompositeResolver[str](registry={}, factory=_FailingFactory())
    assert resolver.resolve_optional("dynamic") is None


# =============================================================================
# Parameter Validation Tests
# =============================================================================


from weakincentives.runtime.mailbox import InvalidParameterError  # noqa: E402


class TestParameterValidation:
    """Tests for timeout parameter validation."""

    def test_receive_negative_visibility_timeout_raises(self) -> None:
        """receive() raises InvalidParameterError for negative visibility_timeout."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            with pytest.raises(
                InvalidParameterError, match="visibility_timeout must be non-negative"
            ):
                mailbox.receive(visibility_timeout=-1)
        finally:
            mailbox.close()

    def test_receive_excessive_visibility_timeout_raises(self) -> None:
        """receive() raises InvalidParameterError for visibility_timeout > 43200."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            with pytest.raises(InvalidParameterError, match="must be at most 43200"):
                mailbox.receive(visibility_timeout=43201)
        finally:
            mailbox.close()

    def test_receive_negative_wait_time_raises(self) -> None:
        """receive() raises InvalidParameterError for negative wait_time_seconds."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            with pytest.raises(
                InvalidParameterError, match="wait_time_seconds must be non-negative"
            ):
                mailbox.receive(wait_time_seconds=-1)
        finally:
            mailbox.close()

    def test_receive_zero_visibility_timeout_allowed(self) -> None:
        """receive() allows visibility_timeout=0."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive(visibility_timeout=0)
            assert len(messages) == 1
        finally:
            mailbox.close()

    def test_receive_max_visibility_timeout_allowed(self) -> None:
        """receive() allows visibility_timeout=43200 (max value)."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive(visibility_timeout=43200)
            assert len(messages) == 1
            messages[0].acknowledge()
        finally:
            mailbox.close()

    def test_nack_negative_visibility_timeout_raises(self) -> None:
        """nack() raises InvalidParameterError for negative visibility_timeout."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive()
            with pytest.raises(
                InvalidParameterError, match="visibility_timeout must be non-negative"
            ):
                messages[0].nack(visibility_timeout=-1)
        finally:
            mailbox.close()

    def test_nack_excessive_visibility_timeout_raises(self) -> None:
        """nack() raises InvalidParameterError for visibility_timeout > 43200."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive()
            with pytest.raises(InvalidParameterError, match="must be at most 43200"):
                messages[0].nack(visibility_timeout=43201)
        finally:
            mailbox.close()

    def test_extend_visibility_negative_timeout_raises(self) -> None:
        """extend_visibility() raises InvalidParameterError for negative timeout."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive()
            with pytest.raises(
                InvalidParameterError, match="timeout must be non-negative"
            ):
                messages[0].extend_visibility(-1)
        finally:
            mailbox.close()

    def test_extend_visibility_excessive_timeout_raises(self) -> None:
        """extend_visibility() raises InvalidParameterError for timeout > 43200."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive()
            with pytest.raises(InvalidParameterError, match="must be at most 43200"):
                messages[0].extend_visibility(43201)
        finally:
            mailbox.close()

    def test_fake_mailbox_receive_validates_parameters(self) -> None:
        """FakeMailbox.receive() also validates parameters."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")

        with pytest.raises(InvalidParameterError):
            mailbox.receive(visibility_timeout=-1)

        with pytest.raises(InvalidParameterError):
            mailbox.receive(wait_time_seconds=-1)


# =============================================================================
# InMemoryMailboxFactory Tests
# =============================================================================


class TestInMemoryMailboxFactory:
    """Tests for InMemoryMailboxFactory."""

    def test_factory_creates_mailbox(self) -> None:
        """Factory creates an InMemoryMailbox instance."""
        factory: InMemoryMailboxFactory[str] = InMemoryMailboxFactory()
        mailbox = factory.create("test-queue")
        try:
            assert mailbox.name == "test-queue"
            # Verify it's functional
            mailbox.send("hello")
            msgs = mailbox.receive()
            assert len(msgs) == 1
            assert msgs[0].body == "hello"
            msgs[0].acknowledge()
        finally:
            mailbox.close()

    def test_factory_caches_mailbox_with_shared_registry(self) -> None:
        """Factory caches mailbox when shared registry is provided."""
        registry: dict[str, Mailbox[str, None]] = {}
        factory: InMemoryMailboxFactory[str] = InMemoryMailboxFactory(registry=registry)

        mailbox1 = factory.create("test-queue")
        mailbox2 = factory.create("test-queue")

        try:
            # Same mailbox instance is returned
            assert mailbox1 is mailbox2
            assert "test-queue" in registry
            assert registry["test-queue"] is mailbox1
        finally:
            mailbox1.close()
