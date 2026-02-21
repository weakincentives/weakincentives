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

"""Tests for Message lifecycle, protocol compliance, and closed state."""

from __future__ import annotations

import threading
import time

import pytest

from weakincentives.runtime.mailbox import (
    CollectingMailbox,
    FakeMailbox,
    InMemoryMailbox,
    Mailbox,
    Message,
    MessageFinalizedError,
    NullMailbox,
    ReplyNotAvailableError,
)

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
