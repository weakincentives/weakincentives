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

"""Tests for InMemoryMailbox implementation and parameter validation."""

from __future__ import annotations

import threading
import time
from datetime import UTC

import pytest

from tests.helpers.time import ControllableClock
from tests.runtime.conftest import SampleEvent
from weakincentives.runtime.mailbox import (
    InMemoryMailbox,
    InvalidParameterError,
    MailboxFullError,
    ReceiptHandleExpiredError,
    ReplyNotAvailableError,
)

# =============================================================================
# InMemoryMailbox Tests
# =============================================================================


class TestInMemoryMailbox:
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
        mailbox: InMemoryMailbox[SampleEvent, None] = InMemoryMailbox(name="test")
        try:
            event = SampleEvent(data="test-data")
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
        clock = ControllableClock()
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test", clock=clock)
        try:
            mailbox.send("hello")
            messages = mailbox.receive(max_messages=1, visibility_timeout=1)
            assert len(messages) == 1

            # Advance clock past visibility timeout and trigger reap synchronously
            clock.advance(2)
            mailbox._reap_expired()

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
# Additional InMemoryMailbox Tests for Coverage
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


# =============================================================================
# Parameter Validation Tests (InMemoryMailbox)
# =============================================================================


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
