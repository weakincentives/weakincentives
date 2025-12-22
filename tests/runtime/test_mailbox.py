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

"""Tests for mailbox module."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from weakincentives.runtime.mailbox import (
    ImmediateReply,
    InMemoryMailbox,
    InMemoryReply,
    InMemoryReplyChannel,
    InMemoryReplyStore,
    MailboxFullError,
    NeverResolvingReply,
    NoReplyChannelError,
    NullMailbox,
    RecordingMailbox,
    ReplyAlreadySentError,
    ReplyCancelledError,
    ReplyExpectedError,
    ReplyState,
    ReplyTimeoutError,
)

# =============================================================================
# ReplyState Tests
# =============================================================================


class TestReplyState:
    """Tests for ReplyState enum."""

    def test_all_states_defined(self) -> None:
        """ReplyState has all expected states."""
        assert ReplyState.PENDING.value == "pending"
        assert ReplyState.RESOLVED.value == "resolved"
        assert ReplyState.CANCELLED.value == "cancelled"
        assert ReplyState.EXPIRED.value == "expired"


# =============================================================================
# InMemoryReplyStore Tests
# =============================================================================


class TestInMemoryReplyStore:
    """Tests for InMemoryReplyStore."""

    def test_create_new_entry(self) -> None:
        """create() returns True for new entry."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        assert store.create("test-id", ttl=60) is True

    def test_create_duplicate_fails(self) -> None:
        """create() returns False for duplicate ID."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        assert store.create("test-id", ttl=60) is True
        assert store.create("test-id", ttl=60) is False

    def test_get_returns_pending_entry(self) -> None:
        """get() returns pending entry with correct state."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)

        entry = store.get("test-id")
        assert entry is not None
        assert entry.id == "test-id"
        assert entry.state == ReplyState.PENDING
        assert entry.value is None

    def test_get_missing_returns_none(self) -> None:
        """get() returns None for missing entry."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        assert store.get("missing") is None

    def test_resolve_pending_entry(self) -> None:
        """resolve() updates pending entry to resolved."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)

        assert store.resolve("test-id", "result") is True

        entry = store.get("test-id")
        assert entry is not None
        assert entry.state == ReplyState.RESOLVED
        assert entry.value == "result"
        assert entry.resolved_at is not None

    def test_resolve_missing_returns_false(self) -> None:
        """resolve() returns False for missing entry."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        assert store.resolve("missing", "result") is False

    def test_resolve_resolved_returns_false(self) -> None:
        """resolve() returns False for already resolved entry."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        _ = store.resolve("test-id", "first")

        assert store.resolve("test-id", "second") is False

    def test_cancel_pending_entry(self) -> None:
        """cancel() updates pending entry to cancelled."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)

        assert store.cancel("test-id") is True

        entry = store.get("test-id")
        assert entry is not None
        assert entry.state == ReplyState.CANCELLED

    def test_cancel_resolved_returns_false(self) -> None:
        """cancel() returns False for resolved entry."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        _ = store.resolve("test-id", "result")

        assert store.cancel("test-id") is False

    def test_delete_entry(self) -> None:
        """delete() removes entry and returns True."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)

        assert store.delete("test-id") is True
        assert store.get("test-id") is None

    def test_delete_missing_returns_false(self) -> None:
        """delete() returns False for missing entry."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        assert store.delete("missing") is False

    def test_consume_removes_entry(self) -> None:
        """consume() returns and removes entry atomically."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        _ = store.resolve("test-id", "result")

        entry = store.consume("test-id")
        assert entry is not None
        assert entry.value == "result"
        assert store.get("test-id") is None

    def test_consume_missing_returns_none(self) -> None:
        """consume() returns None for missing entry."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        assert store.consume("missing") is None

    def test_expired_entry_detected_on_get(self) -> None:
        """get() updates state to EXPIRED for expired pending entries."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=0)  # Immediate expiry

        # Allow time to expire
        time.sleep(0.01)

        entry = store.get("test-id")
        assert entry is not None
        assert entry.state == ReplyState.EXPIRED

    def test_scan_expired_finds_expired_entries(self) -> None:
        """scan_expired() returns expired entry IDs."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("expired-1", ttl=0)
        _ = store.create("expired-2", ttl=0)
        _ = store.create("valid", ttl=3600)

        time.sleep(0.01)

        expired = store.scan_expired()
        assert "expired-1" in expired
        assert "expired-2" in expired
        assert "valid" not in expired

    def test_scan_expired_respects_limit(self) -> None:
        """scan_expired() respects limit parameter."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        for i in range(5):
            _ = store.create(f"expired-{i}", ttl=0)

        time.sleep(0.01)

        expired = store.scan_expired(limit=2)
        assert len(expired) == 2

    def test_cleanup_expired_deletes_entries(self) -> None:
        """cleanup_expired() deletes expired entries."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("expired-1", ttl=0)
        _ = store.create("expired-2", ttl=0)
        _ = store.create("valid", ttl=3600)

        time.sleep(0.01)

        deleted = store.cleanup_expired()
        assert deleted == 2
        assert store.get("expired-1") is None
        assert store.get("expired-2") is None
        assert store.get("valid") is not None


# =============================================================================
# InMemoryReply Tests
# =============================================================================


class TestInMemoryReply:
    """Tests for InMemoryReply."""

    def test_id_property(self) -> None:
        """id property returns reply identifier."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)

        assert reply.id == "test-id"

    def test_poll_returns_none_when_pending(self) -> None:
        """poll() returns None for pending reply."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)

        assert reply.poll() is None

    def test_poll_returns_value_when_resolved(self) -> None:
        """poll() returns value for resolved reply."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        _ = store.resolve("test-id", "result")
        reply = InMemoryReply("test-id", store)

        assert reply.poll() == "result"

    def test_is_ready_false_when_pending(self) -> None:
        """is_ready() returns False for pending reply."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)

        assert reply.is_ready() is False

    def test_is_ready_true_when_resolved(self) -> None:
        """is_ready() returns True for resolved reply."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        _ = store.resolve("test-id", "result")
        reply = InMemoryReply("test-id", store)

        assert reply.is_ready() is True

    def test_is_cancelled_false_when_pending(self) -> None:
        """is_cancelled() returns False for pending reply."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)

        assert reply.is_cancelled() is False

    def test_is_cancelled_true_when_cancelled(self) -> None:
        """is_cancelled() returns True for cancelled reply."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        _ = store.cancel("test-id")
        reply = InMemoryReply("test-id", store)

        assert reply.is_cancelled() is True

    def test_cancel_returns_true_when_pending(self) -> None:
        """cancel() returns True for pending reply."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)

        assert reply.cancel() is True
        assert reply.is_cancelled() is True

    def test_cancel_returns_false_when_resolved(self) -> None:
        """cancel() returns False for resolved reply."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        _ = store.resolve("test-id", "result")
        reply = InMemoryReply("test-id", store)

        assert reply.cancel() is False

    def test_wait_returns_immediately_when_resolved(self) -> None:
        """wait() returns immediately for resolved reply."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        _ = store.resolve("test-id", "result")
        reply = InMemoryReply("test-id", store)

        result = reply.wait(timeout=1)
        assert result == "result"

    def test_wait_raises_on_timeout(self) -> None:
        """wait() raises ReplyTimeoutError on timeout."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)

        with pytest.raises(ReplyTimeoutError):
            reply.wait(timeout=0.01)

    def test_wait_raises_when_cancelled(self) -> None:
        """wait() raises ReplyCancelledError for cancelled reply."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        _ = store.cancel("test-id")
        reply = InMemoryReply("test-id", store)

        with pytest.raises(ReplyCancelledError):
            reply.wait(timeout=1)

    def test_wait_concurrent_resolution(self) -> None:
        """wait() returns when reply is resolved concurrently."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)

        def resolver() -> None:
            time.sleep(0.05)
            _ = store.resolve("test-id", "concurrent-result")
            reply.notify()

        thread = threading.Thread(target=resolver)
        thread.start()

        result = reply.wait(timeout=1)
        assert result == "concurrent-result"
        thread.join()


# =============================================================================
# InMemoryReplyChannel Tests
# =============================================================================


class TestInMemoryReplyChannel:
    """Tests for InMemoryReplyChannel."""

    def test_is_open_true_when_pending(self) -> None:
        """is_open() returns True for pending entry."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)
        channel = InMemoryReplyChannel("test-id", store, reply)

        assert channel.is_open() is True

    def test_is_open_false_after_send(self) -> None:
        """is_open() returns False after send()."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)
        channel = InMemoryReplyChannel("test-id", store, reply)

        channel.send("result")
        assert channel.is_open() is False

    def test_send_resolves_reply(self) -> None:
        """send() resolves the backing reply entry."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)
        channel = InMemoryReplyChannel("test-id", store, reply)

        channel.send("result")

        entry = store.get("test-id")
        assert entry is not None
        assert entry.state == ReplyState.RESOLVED
        assert entry.value == "result"

    def test_send_twice_raises(self) -> None:
        """send() twice raises ReplyAlreadySentError."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)
        channel = InMemoryReplyChannel("test-id", store, reply)

        channel.send("first")

        with pytest.raises(ReplyAlreadySentError):
            channel.send("second")

    def test_send_notifies_waiters(self) -> None:
        """send() notifies threads waiting on reply."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)
        channel = InMemoryReplyChannel("test-id", store, reply)

        received: list[str] = []

        def waiter() -> None:
            result = reply.wait(timeout=1)
            received.append(result)

        thread = threading.Thread(target=waiter)
        thread.start()

        time.sleep(0.05)  # Allow waiter to start
        channel.send("notified-result")

        thread.join()
        assert received == ["notified-result"]


# =============================================================================
# InMemoryMailbox Tests
# =============================================================================


class TestInMemoryMailbox:
    """Tests for InMemoryMailbox."""

    def test_send_returns_message_id(self) -> None:
        """send() returns a unique message ID."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()

        id1 = mailbox.send("hello")
        id2 = mailbox.send("world")

        assert id1 != id2
        assert len(id1) > 0

    def test_send_with_delay(self) -> None:
        """send() with delay makes message invisible."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send("delayed", delay_seconds=1)

        messages = mailbox.receive()
        assert len(messages) == 0

    def test_send_respects_max_size(self) -> None:
        """send() raises MailboxFullError when full."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox(max_size=2)
        _ = mailbox.send("one")
        _ = mailbox.send("two")

        with pytest.raises(MailboxFullError):
            mailbox.send("three")

    def test_receive_returns_messages(self) -> None:
        """receive() returns sent messages."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send("hello")

        messages = mailbox.receive()
        assert len(messages) == 1
        assert messages[0].body == "hello"

    def test_receive_max_messages(self) -> None:
        """receive() respects max_messages parameter."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        for i in range(5):
            _ = mailbox.send(f"message-{i}")

        messages = mailbox.receive(max_messages=2)
        assert len(messages) == 2

    def test_receive_makes_messages_invisible(self) -> None:
        """receive() makes messages invisible to other receivers."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send("exclusive")

        messages1 = mailbox.receive(visibility_timeout=60)
        messages2 = mailbox.receive()

        assert len(messages1) == 1
        assert len(messages2) == 0

    def test_receive_long_poll(self) -> None:
        """receive() with wait_time_seconds blocks for messages."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()

        def sender() -> None:
            time.sleep(0.05)
            _ = mailbox.send("delayed-arrival")

        thread = threading.Thread(target=sender)
        thread.start()

        messages = mailbox.receive(wait_time_seconds=1)
        assert len(messages) == 1
        assert messages[0].body == "delayed-arrival"
        thread.join()

    def test_receive_long_poll_timeout(self) -> None:
        """receive() with wait_time_seconds returns empty on timeout."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()

        start = time.time()
        messages = mailbox.receive(wait_time_seconds=0.1)
        elapsed = time.time() - start

        assert len(messages) == 0
        assert elapsed >= 0.1

    def test_message_acknowledge(self) -> None:
        """acknowledge() removes message from queue."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send("ack-me")

        messages = mailbox.receive(visibility_timeout=1)
        assert messages[0].acknowledge() is True
        assert mailbox.approximate_count() == 0

    def test_message_nack(self) -> None:
        """nack() returns message to queue."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send("nack-me")

        messages = mailbox.receive(visibility_timeout=60)
        assert messages[0].nack(visibility_timeout=0) is True

        # Message should be redeliverable
        messages2 = mailbox.receive()
        assert len(messages2) == 1
        assert messages2[0].delivery_count == 2

    def test_message_extend_visibility(self) -> None:
        """extend_visibility() extends message invisibility."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send("extend-me")

        messages = mailbox.receive(visibility_timeout=1)
        assert messages[0].extend_visibility(60) is True

        # Give original timeout time to expire
        time.sleep(0.1)

        # Message should still be invisible
        messages2 = mailbox.receive()
        assert len(messages2) == 0

    def test_visibility_timeout_expiry(self) -> None:
        """Messages return to queue after visibility timeout."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send("timeout-me")

        messages1 = mailbox.receive(visibility_timeout=0)
        assert len(messages1) == 1

        # Allow visibility timeout to expire
        time.sleep(0.05)

        messages2 = mailbox.receive()
        assert len(messages2) == 1
        assert messages2[0].delivery_count == 2

    def test_purge_removes_all_messages(self) -> None:
        """purge() removes all messages."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        for i in range(5):
            _ = mailbox.send(f"message-{i}")

        count = mailbox.purge()
        assert count == 5
        assert mailbox.approximate_count() == 0

    def test_approximate_count(self) -> None:
        """approximate_count() returns total message count."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send("one")
        _ = mailbox.send("two")
        _ = mailbox.receive(visibility_timeout=60)

        # One in queue, one in-flight
        assert mailbox.approximate_count() == 2


# =============================================================================
# Request-Reply Pattern Tests
# =============================================================================


class TestRequestReplyPattern:
    """Tests for request-reply pattern."""

    def test_send_expecting_reply_returns_reply(self) -> None:
        """send_expecting_reply() returns a Reply handle."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        reply = mailbox.send_expecting_reply("request")

        assert reply.id is not None
        assert reply.is_ready() is False

    def test_message_expects_reply(self) -> None:
        """Message.expects_reply() returns True for reply messages."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send_expecting_reply("request")

        messages = mailbox.receive()
        assert len(messages) == 1
        assert messages[0].expects_reply() is True

    def test_message_reply_resolves_reply(self) -> None:
        """Message.reply() resolves the Reply handle."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        reply = mailbox.send_expecting_reply("request")

        messages = mailbox.receive()
        messages[0].reply("response")

        assert reply.is_ready() is True
        assert reply.wait() == "response"

    def test_message_reply_acknowledges(self) -> None:
        """Message.reply() also acknowledges the message."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send_expecting_reply("request")

        messages = mailbox.receive(visibility_timeout=60)
        messages[0].reply("response")

        # Message should be gone
        assert mailbox.approximate_count() == 0

    def test_message_acknowledge_raises_when_reply_expected(self) -> None:
        """acknowledge() raises ReplyExpectedError when reply expected."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send_expecting_reply("request")

        messages = mailbox.receive()
        with pytest.raises(ReplyExpectedError):
            messages[0].acknowledge()

    def test_message_reply_raises_without_channel(self) -> None:
        """reply() raises NoReplyChannelError for non-reply messages."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send("no-reply-expected")

        messages = mailbox.receive()
        with pytest.raises(NoReplyChannelError):
            messages[0].reply("unwanted")

    def test_concurrent_request_reply(self) -> None:
        """Request-reply works correctly with concurrent sender/receiver."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        results: list[str] = []

        def consumer() -> None:
            for msg in mailbox.receive(wait_time_seconds=1):
                msg.reply(f"reply-to-{msg.body}")

        def producer() -> None:
            reply = mailbox.send_expecting_reply("async-request")
            result = reply.wait(timeout=2)
            results.append(result)

        consumer_thread = threading.Thread(target=consumer)
        producer_thread = threading.Thread(target=producer)

        consumer_thread.start()
        time.sleep(0.05)  # Ensure consumer is ready
        producer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        assert results == ["reply-to-async-request"]


# =============================================================================
# Testing Utilities Tests
# =============================================================================


class TestImmediateReply:
    """Tests for ImmediateReply."""

    def test_wait_returns_preset_value(self) -> None:
        """wait() returns the preset value immediately."""
        reply: ImmediateReply[str] = ImmediateReply("instant")
        assert reply.wait() == "instant"

    def test_poll_returns_preset_value(self) -> None:
        """poll() returns the preset value."""
        reply: ImmediateReply[str] = ImmediateReply("instant")
        assert reply.poll() == "instant"

    def test_is_ready_true(self) -> None:
        """is_ready() returns True."""
        reply: ImmediateReply[str] = ImmediateReply("instant")
        assert reply.is_ready() is True

    def test_cancel_succeeds(self) -> None:
        """cancel() returns True and sets cancelled state."""
        reply: ImmediateReply[str] = ImmediateReply("instant")
        assert reply.cancel() is True
        assert reply.is_cancelled() is True


class TestNeverResolvingReply:
    """Tests for NeverResolvingReply."""

    def test_wait_raises_timeout(self) -> None:
        """wait() always raises ReplyTimeoutError."""
        reply: NeverResolvingReply[str] = NeverResolvingReply()
        with pytest.raises(ReplyTimeoutError):
            reply.wait()

    def test_poll_returns_none(self) -> None:
        """poll() returns None."""
        reply: NeverResolvingReply[str] = NeverResolvingReply()
        assert reply.poll() is None

    def test_is_ready_false(self) -> None:
        """is_ready() returns False."""
        reply: NeverResolvingReply[str] = NeverResolvingReply()
        assert reply.is_ready() is False

    def test_cancel_succeeds(self) -> None:
        """cancel() returns True."""
        reply: NeverResolvingReply[str] = NeverResolvingReply()
        assert reply.cancel() is True


class TestNullMailbox:
    """Tests for NullMailbox."""

    def test_send_returns_id(self) -> None:
        """send() returns a message ID."""
        mailbox: NullMailbox[str, str] = NullMailbox()
        msg_id = mailbox.send("hello")
        assert len(msg_id) > 0

    def test_send_records_message(self) -> None:
        """send() records message for inspection."""
        mailbox: NullMailbox[str, str] = NullMailbox()
        _ = mailbox.send("recorded")
        assert mailbox.messages_sent == ["recorded"]

    def test_send_expecting_reply_returns_never_resolving(self) -> None:
        """send_expecting_reply() returns NeverResolvingReply."""
        mailbox: NullMailbox[str, str] = NullMailbox()
        reply = mailbox.send_expecting_reply("request")

        with pytest.raises(ReplyTimeoutError):
            reply.wait()

    def test_receive_returns_empty(self) -> None:
        """receive() returns empty sequence."""
        mailbox: NullMailbox[str, str] = NullMailbox()
        _ = mailbox.send("ignored")
        assert mailbox.receive() == []

    def test_approximate_count_zero(self) -> None:
        """approximate_count() returns 0."""
        mailbox: NullMailbox[str, str] = NullMailbox()
        assert mailbox.approximate_count() == 0


class TestRecordingMailbox:
    """Tests for RecordingMailbox."""

    def test_records_sent_messages(self) -> None:
        """RecordingMailbox records all sent messages."""
        inner: InMemoryMailbox[str, str] = InMemoryMailbox()
        mailbox: RecordingMailbox[str, str] = RecordingMailbox(inner)

        _ = mailbox.send("first")
        _ = mailbox.send("second")

        assert mailbox.sent_messages == ["first", "second"]

    def test_records_received_messages(self) -> None:
        """RecordingMailbox records all received messages."""
        inner: InMemoryMailbox[str, str] = InMemoryMailbox()
        mailbox: RecordingMailbox[str, str] = RecordingMailbox(inner)

        _ = mailbox.send("to-receive")
        _ = mailbox.receive()

        assert len(mailbox.received_messages) == 1
        assert mailbox.received_messages[0].body == "to-receive"

    def test_delegates_to_inner(self) -> None:
        """RecordingMailbox delegates to inner mailbox."""
        inner: InMemoryMailbox[str, str] = InMemoryMailbox()
        mailbox: RecordingMailbox[str, str] = RecordingMailbox(inner)

        _ = mailbox.send("test")
        assert inner.approximate_count() == 1


# =============================================================================
# Edge Cases and Additional Coverage Tests
# =============================================================================


class TestMailboxEdgeCases:
    """Tests for mailbox-specific edge cases."""

    def test_send_expecting_reply_max_size(self) -> None:
        """send_expecting_reply raises MailboxFullError when full."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox(max_size=1)
        _ = mailbox.send("first")

        with pytest.raises(MailboxFullError):
            mailbox.send_expecting_reply("second")

    def test_send_expecting_reply_with_max_size_not_exceeded(self) -> None:
        """send_expecting_reply works when max_size not exceeded."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox(max_size=5)
        # Send expecting reply when there's room
        reply = mailbox.send_expecting_reply("request")
        assert reply is not None
        assert mailbox.approximate_count() == 1

    def test_message_acknowledge_expired_handle(self) -> None:
        """acknowledge returns False for expired handle."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send("test")

        messages = mailbox.receive(visibility_timeout=0)
        # Allow visibility to expire
        time.sleep(0.05)
        # Trigger expiry by trying to receive again
        _ = mailbox.receive()

        # Now the original handle should be expired
        assert messages[0].acknowledge() is False

    def test_message_nack_expired_handle(self) -> None:
        """nack returns False for expired handle."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send("test")

        messages = mailbox.receive(visibility_timeout=0)
        # Allow visibility to expire and get redelivered
        time.sleep(0.05)
        _ = mailbox.receive()

        # Now the original handle should be expired
        assert messages[0].nack() is False

    def test_message_extend_visibility_expired_handle(self) -> None:
        """extend_visibility returns False for expired handle."""
        mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()
        _ = mailbox.send("test")

        messages = mailbox.receive(visibility_timeout=0)
        # Allow visibility to expire
        time.sleep(0.05)
        _ = mailbox.receive()

        # Now the original handle should be expired
        assert messages[0].extend_visibility(60) is False


class TestReplyEdgeCases:
    """Tests for reply-specific edge cases."""

    def test_reply_check_entry_state_cancelled(self) -> None:
        """_check_entry_state raises ReplyCancelledError for cancelled."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)

        # Cancel and check
        _ = store.cancel("test-id")
        with pytest.raises(ReplyCancelledError):
            reply.wait(timeout=0.01)

    def test_reply_check_entry_state_expired(self) -> None:
        """_check_entry_state raises ReplyTimeoutError for expired."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=0)

        time.sleep(0.01)

        reply = InMemoryReply("test-id", store)
        with pytest.raises(ReplyTimeoutError):
            reply.wait(timeout=0.01)

    def test_reply_channel_send_to_expired(self) -> None:
        """send raises ReplyAlreadySentError when entry expired."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=0)
        reply = InMemoryReply("test-id", store)
        channel = InMemoryReplyChannel("test-id", store, reply)

        time.sleep(0.01)

        with pytest.raises(ReplyAlreadySentError):
            channel.send("value")

    def test_reply_channel_send_to_cancelled(self) -> None:
        """send() raises ReplyAlreadySentError when entry cancelled."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)
        channel = InMemoryReplyChannel("test-id", store, reply)

        # Cancel the reply
        _ = store.cancel("test-id")

        with pytest.raises(ReplyAlreadySentError, match="cancelled"):
            channel.send("value")

    def test_reply_wait_entry_not_found(self) -> None:
        """wait() raises ReplyTimeoutError when entry not in store."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        # Create reply for non-existent entry
        reply = InMemoryReply("non-existent", store)

        with pytest.raises(ReplyTimeoutError, match="not found"):
            reply.wait(timeout=0.01)

    def test_reply_wait_entry_deleted_during_wait(self) -> None:
        """wait() raises ReplyTimeoutError when entry deleted during wait."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)

        def delete_entry() -> None:
            time.sleep(0.02)
            _ = store.delete("test-id")
            reply.notify()

        thread = threading.Thread(target=delete_entry)
        thread.start()

        with pytest.raises(ReplyTimeoutError):
            reply.wait(timeout=1)
        thread.join()

    def test_reply_wait_spurious_wakeup(self) -> None:
        """wait() loops on spurious wakeup when entry still pending."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)
        reply = InMemoryReply("test-id", store)

        def spurious_then_resolve() -> None:
            # Spurious wakeup - notify without resolving
            time.sleep(0.02)
            reply.notify()
            # Then actually resolve
            time.sleep(0.02)
            _ = store.resolve("test-id", "result")
            reply.notify()

        thread = threading.Thread(target=spurious_then_resolve)
        thread.start()

        result = reply.wait(timeout=1)
        assert result == "result"
        thread.join()


class TestReplyStoreEdgeCases:
    """Tests for reply store edge cases."""

    def test_reply_store_resolve_expired(self) -> None:
        """resolve returns False when entry expired."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=0)

        time.sleep(0.01)

        assert store.resolve("test-id", "value") is False
        entry = store.get("test-id")
        assert entry is not None
        assert entry.state == ReplyState.EXPIRED

    def test_reply_store_cancel_expired(self) -> None:
        """cancel returns False when entry expired."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=0)

        time.sleep(0.01)

        assert store.cancel("test-id") is False

    def test_reply_store_consume_expired(self) -> None:
        """consume returns expired entry."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=0)

        time.sleep(0.01)

        entry = store.consume("test-id")
        assert entry is not None
        assert entry.state == ReplyState.EXPIRED

    def test_reply_store_consume_pending_not_expired(self) -> None:
        """consume() returns pending non-expired entry."""
        store: InMemoryReplyStore[str] = InMemoryReplyStore()
        _ = store.create("test-id", ttl=60)

        # Consume while still pending
        entry = store.consume("test-id")
        assert entry is not None
        assert entry.state == ReplyState.PENDING

    def test_reply_store_cleanup_delete_returns_false(self) -> None:
        """cleanup_expired() handles when delete returns False."""

        # Create a store subclass that makes delete return False for specific ID
        class RacyReplyStore(InMemoryReplyStore[str]):
            def delete(self, entry_id: str) -> bool:
                # Always return False to simulate race condition
                _ = super().delete(entry_id)
                return False

        store: RacyReplyStore = RacyReplyStore()
        _ = store.create("test-id", ttl=0)
        time.sleep(0.01)

        deleted = store.cleanup_expired()
        assert deleted == 0


class TestTestingUtilitiesEdgeCases:
    """Tests for testing utilities edge cases."""

    def test_immediate_reply_id_property(self) -> None:
        """ImmediateReply.id returns the reply identifier."""
        reply: ImmediateReply[str] = ImmediateReply("value", reply_id="custom-id")
        assert reply.id == "custom-id"

    def test_immediate_reply_timeout_ignored(self) -> None:
        """ImmediateReply.wait ignores timeout parameter."""
        reply: ImmediateReply[str] = ImmediateReply("value")
        # Should not raise even with tiny timeout
        result = reply.wait(timeout=0.001)
        assert result == "value"

    def test_never_resolving_reply_id_property(self) -> None:
        """NeverResolvingReply.id returns the reply identifier."""
        reply: NeverResolvingReply[str] = NeverResolvingReply(reply_id="custom-id")
        assert reply.id == "custom-id"

    def test_never_resolving_reply_is_cancelled(self) -> None:
        """NeverResolvingReply.is_cancelled returns False until cancelled."""
        reply: NeverResolvingReply[str] = NeverResolvingReply()
        assert reply.is_cancelled() is False
        _ = reply.cancel()
        assert reply.is_cancelled() is True

    def test_never_resolving_reply_timeout_ignored(self) -> None:
        """NeverResolvingReply.wait ignores timeout parameter."""
        reply: NeverResolvingReply[str] = NeverResolvingReply()
        with pytest.raises(ReplyTimeoutError):
            reply.wait(timeout=0.001)

    def test_null_mailbox_purge(self) -> None:
        """NullMailbox.purge returns count and clears."""
        mailbox: NullMailbox[str, str] = NullMailbox()
        _ = mailbox.send("one")
        _ = mailbox.send("two")

        count = mailbox.purge()
        assert count == 2
        assert len(mailbox.messages_sent) == 0

    def test_recording_mailbox_send_expecting_reply(self) -> None:
        """RecordingMailbox records send_expecting_reply messages."""
        inner: InMemoryMailbox[str, str] = InMemoryMailbox()
        mailbox: RecordingMailbox[str, str] = RecordingMailbox(inner)

        _ = mailbox.send_expecting_reply("request")

        assert mailbox.sent_messages == ["request"]

    def test_recording_mailbox_purge(self) -> None:
        """RecordingMailbox delegates purge."""
        inner: InMemoryMailbox[str, str] = InMemoryMailbox()
        mailbox: RecordingMailbox[str, str] = RecordingMailbox(inner)

        _ = mailbox.send("test")
        count = mailbox.purge()

        assert count == 1

    def test_recording_mailbox_approximate_count(self) -> None:
        """RecordingMailbox.approximate_count delegates to inner."""
        inner: InMemoryMailbox[str, str] = InMemoryMailbox()
        mailbox: RecordingMailbox[str, str] = RecordingMailbox(inner)

        _ = mailbox.send("one")
        _ = mailbox.send("two")

        assert mailbox.approximate_count() == 2

    def test_module_dir(self) -> None:
        """Mailbox module __dir__ returns expected entries."""
        from weakincentives.runtime import mailbox

        dir_entries = dir(mailbox)
        assert "InMemoryMailbox" in dir_entries
        assert "Reply" in dir_entries
        assert "MailboxError" in dir_entries


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of mailbox operations."""

    def test_concurrent_sends(self) -> None:
        """Concurrent sends don't lose messages."""
        mailbox: InMemoryMailbox[int, str] = InMemoryMailbox()
        num_messages = 100

        def sender(start: int) -> None:
            for i in range(10):
                _ = mailbox.send(start + i)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(sender, i * 10) for i in range(10)]
            for f in futures:
                f.result()

        assert mailbox.approximate_count() == num_messages

    def test_concurrent_receives(self) -> None:
        """Concurrent receives don't deliver same message twice."""
        mailbox: InMemoryMailbox[int, str] = InMemoryMailbox()
        num_messages = 100

        for i in range(num_messages):
            _ = mailbox.send(i)

        received: list[int] = []
        lock = threading.Lock()

        def receiver() -> None:
            while True:
                messages = mailbox.receive(max_messages=1, visibility_timeout=60)
                if not messages:
                    break
                with lock:
                    received.append(messages[0].body)
                messages[0].acknowledge()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(receiver) for _ in range(10)]
            for f in futures:
                f.result()

        # All messages received exactly once
        assert len(received) == num_messages
        assert len(set(received)) == num_messages

    def test_concurrent_reply_store_operations(self) -> None:
        """Concurrent reply store operations are thread-safe."""
        store: InMemoryReplyStore[int] = InMemoryReplyStore()
        num_entries = 100

        def creator(start: int) -> None:
            for i in range(10):
                _ = store.create(f"entry-{start + i}", ttl=60)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(creator, i * 10) for i in range(10)]
            for f in futures:
                f.result()

        # Verify all entries exist
        for i in range(num_entries):
            assert store.get(f"entry-{i}") is not None
