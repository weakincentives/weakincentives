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

"""Integration tests for SQS mailbox using LocalStack.

These tests require LocalStack to be running on localhost:4566.
Run with: docker run -d -p 4566:4566 localstack/localstack

To run these tests:
    pytest tests/contrib/mailbox/test_sqs_mailbox.py -v -m sqs
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from mypy_boto3_sqs import SQSClient

    from weakincentives.contrib.mailbox import SQSMailbox

pytestmark = pytest.mark.sqs


class TestSQSMailboxBasic:
    """Basic send/receive tests."""

    def test_send_and_receive_string(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Send and receive a simple string message."""
        msg_id = sqs_mailbox.send("hello world")
        assert msg_id is not None

        messages = sqs_mailbox.receive(visibility_timeout=30)
        assert len(messages) == 1
        assert messages[0].body == "hello world"
        assert messages[0].id == msg_id

        messages[0].acknowledge()

    def test_send_and_receive_dict(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Send and receive a dictionary message."""
        data = {"key": "value", "count": 42}
        sqs_mailbox.send(data)

        messages = sqs_mailbox.receive(visibility_timeout=30)
        assert len(messages) == 1
        assert messages[0].body == data
        messages[0].acknowledge()

    def test_send_and_receive_dataclass(
        self, sqs_client: SQSClient, sqs_queue_url: str
    ) -> None:
        """Send and receive a dataclass message with type hint."""
        from weakincentives.contrib.mailbox import SQSMailbox

        @dataclass
        class TestEvent:
            name: str
            count: int

        mb: SQSMailbox[TestEvent] = SQSMailbox(
            queue_url=sqs_queue_url,
            client=sqs_client,
            body_type=TestEvent,
        )

        try:
            event = TestEvent(name="test", count=123)
            mb.send(event)

            messages = mb.receive(visibility_timeout=30)
            assert len(messages) == 1
            assert messages[0].body.name == "test"
            assert messages[0].body.count == 123
            messages[0].acknowledge()
        finally:
            mb.close()

    def test_receive_empty_queue(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Receive from empty queue returns empty list."""
        messages = sqs_mailbox.receive(visibility_timeout=30, wait_time_seconds=0)
        assert messages == []

    def test_receive_multiple_messages(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Receive multiple messages in one call."""
        for i in range(5):
            sqs_mailbox.send(f"msg-{i}")

        messages = sqs_mailbox.receive(max_messages=10, visibility_timeout=30)
        assert len(messages) == 5

        for msg in messages:
            msg.acknowledge()

    def test_max_messages_clamped(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """max_messages is clamped to 1-10 range."""
        for i in range(15):
            sqs_mailbox.send(f"msg-{i}")

        # Request more than 10
        messages = sqs_mailbox.receive(max_messages=100, visibility_timeout=30)
        assert len(messages) <= 10

        for msg in messages:
            msg.acknowledge()


class TestSQSMailboxVisibility:
    """Visibility timeout tests."""

    def test_message_invisible_after_receive(
        self, sqs_mailbox: SQSMailbox[Any]
    ) -> None:
        """Received message is invisible to other consumers."""
        sqs_mailbox.send("test")

        # First receive
        msgs1 = sqs_mailbox.receive(visibility_timeout=30)
        assert len(msgs1) == 1

        # Second receive should be empty
        msgs2 = sqs_mailbox.receive(visibility_timeout=30, wait_time_seconds=0)
        assert len(msgs2) == 0

        msgs1[0].acknowledge()

    def test_message_visible_after_timeout(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Message becomes visible after visibility timeout expires."""
        sqs_mailbox.send("test")

        # Receive with short timeout
        msgs1 = sqs_mailbox.receive(visibility_timeout=1)
        assert len(msgs1) == 1
        msg_id = msgs1[0].id

        # Wait for timeout (SQS has minimum 1 second precision)
        time.sleep(2)

        # Should be visible again
        msgs2 = sqs_mailbox.receive(visibility_timeout=30)
        assert len(msgs2) == 1
        assert msgs2[0].id == msg_id
        msgs2[0].acknowledge()

    def test_extend_visibility(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Extend visibility timeout prevents requeue."""
        sqs_mailbox.send("test")

        msgs = sqs_mailbox.receive(visibility_timeout=2)
        msg = msgs[0]

        # Extend before timeout
        time.sleep(1)
        msg.extend_visibility(30)

        # Wait past original timeout
        time.sleep(2)

        # Should still be invisible
        msgs2 = sqs_mailbox.receive(visibility_timeout=30, wait_time_seconds=0)
        assert len(msgs2) == 0

        msg.acknowledge()

    def test_nack_immediate_requeue(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Nack with timeout=0 makes message immediately visible."""
        sqs_mailbox.send("test")

        msgs1 = sqs_mailbox.receive(visibility_timeout=60)
        msg_id = msgs1[0].id

        # Nack with immediate visibility
        msgs1[0].nack(visibility_timeout=0)

        # Should be immediately available
        msgs2 = sqs_mailbox.receive(visibility_timeout=30)
        assert len(msgs2) == 1
        assert msgs2[0].id == msg_id
        msgs2[0].acknowledge()

    def test_nack_delayed_requeue(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Nack with timeout delays message visibility."""
        sqs_mailbox.send("test")

        msgs1 = sqs_mailbox.receive(visibility_timeout=60)
        msg_id = msgs1[0].id

        # Nack with delay
        msgs1[0].nack(visibility_timeout=2)

        # Should not be immediately available
        msgs2 = sqs_mailbox.receive(visibility_timeout=30, wait_time_seconds=0)
        assert len(msgs2) == 0

        # Wait for nack delay
        time.sleep(3)

        # Should now be available
        msgs3 = sqs_mailbox.receive(visibility_timeout=30)
        assert len(msgs3) == 1
        assert msgs3[0].id == msg_id
        msgs3[0].acknowledge()


class TestSQSMailboxDelay:
    """Delayed message tests."""

    def test_delayed_message_not_immediately_visible(
        self, sqs_mailbox: SQSMailbox[Any]
    ) -> None:
        """Delayed message is not immediately visible."""
        sqs_mailbox.send("delayed", delay_seconds=3)

        # Should not be available immediately
        msgs = sqs_mailbox.receive(visibility_timeout=30, wait_time_seconds=0)
        assert len(msgs) == 0

    def test_delayed_message_visible_after_delay(
        self, sqs_mailbox: SQSMailbox[Any]
    ) -> None:
        """Delayed message becomes visible after delay expires."""
        sqs_mailbox.send("delayed", delay_seconds=2)

        # Wait for delay
        time.sleep(3)

        # Should now be available
        msgs = sqs_mailbox.receive(visibility_timeout=30)
        assert len(msgs) == 1
        assert msgs[0].body == "delayed"
        msgs[0].acknowledge()


class TestSQSMailboxDeliveryCount:
    """Delivery count tracking tests."""

    def test_first_delivery_count_is_one(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """First delivery has count of 1."""
        sqs_mailbox.send("test")

        msgs = sqs_mailbox.receive(visibility_timeout=30)
        assert msgs[0].delivery_count == 1
        msgs[0].acknowledge()

    def test_delivery_count_increments_on_redelivery(
        self, sqs_mailbox: SQSMailbox[Any]
    ) -> None:
        """Delivery count increments on each redelivery."""
        sqs_mailbox.send("test")

        # First delivery
        msgs1 = sqs_mailbox.receive(visibility_timeout=1)
        assert msgs1[0].delivery_count == 1

        # Wait for timeout
        time.sleep(2)

        # Second delivery
        msgs2 = sqs_mailbox.receive(visibility_timeout=1)
        assert msgs2[0].delivery_count == 2

        # Wait for timeout
        time.sleep(2)

        # Third delivery
        msgs3 = sqs_mailbox.receive(visibility_timeout=30)
        assert msgs3[0].delivery_count == 3
        msgs3[0].acknowledge()


class TestSQSMailboxPurgeAndCount:
    """Purge and count tests."""

    def test_approximate_count(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Approximate count tracks messages."""
        assert sqs_mailbox.approximate_count() == 0

        for i in range(5):
            sqs_mailbox.send(f"msg-{i}")

        # SQS count is eventually consistent, give it a moment
        time.sleep(1)
        count = sqs_mailbox.approximate_count()
        assert count == 5

    def test_purge_queue(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Purge removes all messages."""
        for i in range(5):
            sqs_mailbox.send(f"msg-{i}")

        # Give SQS time to register messages
        time.sleep(1)

        # Purge
        count = sqs_mailbox.purge()
        assert count >= 0  # Approximate count before purge

        # Give purge time to complete
        time.sleep(2)

        # Should be empty
        msgs = sqs_mailbox.receive(visibility_timeout=30, wait_time_seconds=0)
        assert len(msgs) == 0


class TestSQSMailboxErrors:
    """Error handling tests."""

    def test_acknowledge_after_timeout_raises(
        self, sqs_mailbox: SQSMailbox[Any]
    ) -> None:
        """Acknowledge after visibility timeout raises error."""
        from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

        sqs_mailbox.send("test")

        msgs = sqs_mailbox.receive(visibility_timeout=1)
        msg = msgs[0]

        # Wait for timeout
        time.sleep(2)

        # Let another consumer receive (invalidates our handle)
        msgs2 = sqs_mailbox.receive(visibility_timeout=30)
        if msgs2:
            msgs2[0].acknowledge()

        # Our old handle should be invalid
        with pytest.raises(ReceiptHandleExpiredError):
            msg.acknowledge()

    def test_nack_after_acknowledge_raises(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Nack after acknowledge raises error."""
        from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

        sqs_mailbox.send("test")

        msgs = sqs_mailbox.receive(visibility_timeout=30)
        msg = msgs[0]

        msg.acknowledge()

        with pytest.raises(ReceiptHandleExpiredError):
            msg.nack(visibility_timeout=0)

    def test_extend_after_acknowledge_raises(
        self, sqs_mailbox: SQSMailbox[Any]
    ) -> None:
        """Extend after acknowledge raises error."""
        from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

        sqs_mailbox.send("test")

        msgs = sqs_mailbox.receive(visibility_timeout=30)
        msg = msgs[0]

        msg.acknowledge()

        with pytest.raises(ReceiptHandleExpiredError):
            msg.extend_visibility(60)

    def test_serialization_error_on_invalid_body(
        self, sqs_client: SQSClient, sqs_queue_url: str
    ) -> None:
        """Serialization error for non-serializable body."""
        from weakincentives.contrib.mailbox import SQSMailbox
        from weakincentives.runtime.mailbox import SerializationError

        mb: SQSMailbox[Any] = SQSMailbox(
            queue_url=sqs_queue_url,
            client=sqs_client,
        )

        # Lambda functions are not JSON serializable
        with pytest.raises(SerializationError):
            mb.send(lambda x: x)  # type: ignore[arg-type]

        mb.close()


class TestSQSMailboxConcurrency:
    """Concurrent access tests."""

    def test_concurrent_receive_no_duplicates(
        self, sqs_mailbox: SQSMailbox[Any]
    ) -> None:
        """Concurrent receives don't return the same message twice."""
        # Send 20 messages
        for i in range(20):
            sqs_mailbox.send(f"msg-{i}")

        # Give SQS time to register
        time.sleep(1)

        received: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            local_received = []
            for _ in range(10):
                msgs = sqs_mailbox.receive(visibility_timeout=60)
                for m in msgs:
                    local_received.append(m.id)
                    m.acknowledge()
            with lock:
                received.extend(local_received)

        # Run concurrent workers
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No duplicates
        assert len(received) == len(set(received)), "Duplicate messages received"


class TestSQSMailboxLongPoll:
    """Long polling tests."""

    def test_long_poll_returns_when_message_arrives(
        self, sqs_mailbox: SQSMailbox[Any]
    ) -> None:
        """Long poll returns as soon as message arrives."""
        # Start long poll in background
        result: list[Any] = []

        def poll() -> None:
            msgs = sqs_mailbox.receive(visibility_timeout=30, wait_time_seconds=10)
            result.extend(msgs)

        t = threading.Thread(target=poll)
        t.start()

        # Give thread time to start polling
        time.sleep(0.5)

        # Send a message
        sqs_mailbox.send("hello")

        # Wait for poll to complete
        t.join(timeout=15)

        assert len(result) == 1
        assert result[0].body == "hello"
        result[0].acknowledge()

    def test_long_poll_times_out(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Long poll returns empty after timeout."""
        start = time.time()
        msgs = sqs_mailbox.receive(visibility_timeout=30, wait_time_seconds=1)
        elapsed = time.time() - start

        assert len(msgs) == 0
        assert elapsed >= 0.5  # Should have waited at least some time


class TestSQSMailboxClose:
    """Close behavior tests."""

    def test_receive_returns_empty_after_close(
        self, sqs_mailbox: SQSMailbox[Any]
    ) -> None:
        """Receive returns empty after mailbox is closed."""
        sqs_mailbox.send("test")
        sqs_mailbox.close()

        msgs = sqs_mailbox.receive(visibility_timeout=30)
        assert len(msgs) == 0

    def test_closed_property(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Closed property reflects state."""
        assert sqs_mailbox.closed is False
        sqs_mailbox.close()
        assert sqs_mailbox.closed is True


class TestSQSMailboxOrdering:
    """Message ordering tests (best-effort for standard SQS)."""

    def test_fifo_best_effort(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """Standard SQS provides best-effort FIFO ordering."""
        # Send in order
        for i in range(5):
            sqs_mailbox.send(f"msg-{i}")

        # Give SQS time
        time.sleep(1)

        # Receive all
        received = []
        for _ in range(10):  # Extra iterations for safety
            msgs = sqs_mailbox.receive(visibility_timeout=60)
            if not msgs:
                break
            for m in msgs:
                received.append(m.body)
                m.acknowledge()

        # Should have received all messages (order may vary for standard queues)
        assert len(received) == 5
        assert set(received) == {f"msg-{i}" for i in range(5)}


class TestSQSMailboxMessageAttributes:
    """Message attributes tests."""

    def test_enqueued_at_populated(self, sqs_mailbox: SQSMailbox[Any]) -> None:
        """enqueued_at is populated from SentTimestamp."""
        before = time.time()
        sqs_mailbox.send("test")
        after = time.time()

        # Give SQS time
        time.sleep(0.5)

        msgs = sqs_mailbox.receive(visibility_timeout=30)
        assert len(msgs) == 1

        enqueued_ts = msgs[0].enqueued_at.timestamp()
        assert before <= enqueued_ts <= after + 1  # Allow some slack

        msgs[0].acknowledge()
