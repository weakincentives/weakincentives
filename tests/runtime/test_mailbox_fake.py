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

"""Tests for FakeMailbox implementation."""

from __future__ import annotations

import pytest

from tests.runtime.conftest import SampleEvent
from weakincentives.runtime.mailbox import (
    CollectingMailbox,
    FakeMailbox,
    InvalidParameterError,
    MailboxConnectionError,
    ReceiptHandleExpiredError,
    ReplyNotAvailableError,
)

# =============================================================================
# FakeMailbox Tests
# =============================================================================


class TestFakeMailbox:
    """Tests for FakeMailbox implementation."""

    def test_basic_send_receive(self) -> None:
        """Basic send and receive workflow."""
        mailbox: FakeMailbox[SampleEvent, None] = FakeMailbox()
        event = SampleEvent(data="test")
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
# FakeMailbox Parameter Validation Tests
# =============================================================================


def test_fake_mailbox_receive_validates_parameters() -> None:
    """FakeMailbox.receive() also validates parameters."""
    mailbox: FakeMailbox[str, None] = FakeMailbox()
    mailbox.send("hello")

    with pytest.raises(InvalidParameterError):
        mailbox.receive(visibility_timeout=-1)

    with pytest.raises(InvalidParameterError):
        mailbox.receive(wait_time_seconds=-1)
