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

"""Tests for NullMailbox, CollectingMailbox, FakeMailbox variants, and InMemoryMailbox coverage."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.helpers.time import ControllableClock
from weakincentives.runtime.mailbox import (
    CollectingMailbox,
    FakeMailbox,
    InMemoryMailbox,
    MailboxConnectionError,
    NullMailbox,
    ReceiptHandleExpiredError,
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
# Additional InMemoryMailbox Coverage Tests
# =============================================================================


class TestInMemoryMailboxCoverage:
    """Additional tests for InMemoryMailbox coverage."""

    def test_extend_visibility_success(self) -> None:
        """extend_visibility() extends timeout for valid handle."""
        clock = ControllableClock()
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test", clock=clock)
        try:
            mailbox.send("hello")
            messages = mailbox.receive(max_messages=1, visibility_timeout=1)
            assert len(messages) == 1

            # Extend visibility
            messages[0].extend_visibility(60)

            # Advance clock past original timeout and trigger reap synchronously
            clock.advance(2)
            mailbox._reap_expired()

            # Message should still be invisible (not requeued)
            assert mailbox.approximate_count() == 1
            new_messages = mailbox.receive(max_messages=1)
            assert len(new_messages) == 0  # Still invisible

            # Acknowledge to clean up
            messages[0].acknowledge()
        finally:
            mailbox.close()
