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

"""Integration tests for RedisMailbox.

These tests require a container runtime (Docker or Podman) to spin up Redis
instances. Tests are skipped if prerequisites are not met.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING

import pytest
from redis_utils import (
    REDIS_CLUSTER_TESTS_ENABLED,
    is_redis_available,
    redis_cluster,
    redis_standalone,
    skip_if_no_redis,
)

from weakincentives.contrib.mailbox import RedisMailbox
from weakincentives.runtime.mailbox import (
    Mailbox,
    MailboxFullError,
    ReceiptHandleExpiredError,
)

if TYPE_CHECKING:
    pass


@dataclass(slots=True, frozen=True)
class _Event:
    """Sample event type for testing."""

    data: str
    count: int = 0


# Skip all tests in this module if Redis is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.redis,
    pytest.mark.skipif(not is_redis_available(), reason=skip_if_no_redis()),
    pytest.mark.timeout(60),
]


# =============================================================================
# Standalone Redis Tests
# =============================================================================


@pytest.mark.redis_standalone
class TestRedisMailboxStandalone:
    """Tests for RedisMailbox with standalone Redis."""

    def test_send_returns_message_id(self) -> None:
        """send() returns a unique message ID."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test", client=client, body_type=str
            )
            try:
                msg_id = mailbox.send("hello")
                assert isinstance(msg_id, str)
                assert len(msg_id) > 0
            finally:
                mailbox.close()

    def test_send_and_receive_basic(self) -> None:
        """Basic send and receive workflow."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[_Event] = RedisMailbox(
                name="test", client=client, body_type=_Event
            )
            try:
                event = _Event(data="test-data", count=42)
                mailbox.send(event)

                messages = mailbox.receive(max_messages=1)
                assert len(messages) == 1
                assert messages[0].body == event
                assert messages[0].delivery_count == 1
            finally:
                mailbox.close()

    def test_receive_empty_queue(self) -> None:
        """receive() returns empty list when queue is empty."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-empty", client=client, body_type=str
            )
            try:
                messages = mailbox.receive(max_messages=1)
                assert messages == []
            finally:
                mailbox.close()

    def test_receive_max_messages(self) -> None:
        """receive() respects max_messages limit."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[int] = RedisMailbox(
                name="test-max", client=client, body_type=int
            )
            try:
                for i in range(5):
                    mailbox.send(i)

                messages = mailbox.receive(max_messages=3)
                assert len(messages) == 3
                # FIFO order
                assert [m.body for m in messages] == [0, 1, 2]
            finally:
                mailbox.close()

    def test_receive_fifo_order(self) -> None:
        """Messages are received in FIFO order."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[int] = RedisMailbox(
                name="test-fifo", client=client, body_type=int
            )
            try:
                for i in range(3):
                    mailbox.send(i)

                messages = mailbox.receive(max_messages=3)
                assert [m.body for m in messages] == [0, 1, 2]
            finally:
                mailbox.close()

    def test_acknowledge_removes_message(self) -> None:
        """acknowledge() removes message from queue."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-ack", client=client, body_type=str
            )
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
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-ack-expired", client=client, body_type=str
            )
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
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-nack", client=client, body_type=str
            )
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
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-nack-expired", client=client, body_type=str
            )
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
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-extend-expired", client=client, body_type=str
            )
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
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-visibility",
                client=client,
                body_type=str,
                reaper_interval=0.1,
            )
            try:
                mailbox.send("hello")
                messages = mailbox.receive(max_messages=1, visibility_timeout=1)
                assert len(messages) == 1

                # Wait for visibility timeout to expire
                time.sleep(1.5)

                # Message should be available again
                messages = mailbox.receive(max_messages=1)
                assert len(messages) == 1
                assert messages[0].body == "hello"
                assert messages[0].delivery_count == 2
            finally:
                mailbox.close()

    def test_stale_receipt_handle_rejected_after_redelivery(self) -> None:
        """Stale receipt handles are rejected after message is redelivered.

        This tests the critical invariant that after a visibility timeout expires
        and a message is redelivered, the old receipt handle becomes invalid.
        Without this, a slow consumer could acknowledge a message that's been
        handed to a different consumer.
        """
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-stale-handle",
                client=client,
                body_type=str,
                reaper_interval=0.1,
            )
            try:
                mailbox.send("hello")

                # First consumer receives message
                first_delivery = mailbox.receive(max_messages=1, visibility_timeout=1)
                assert len(first_delivery) == 1
                old_msg = first_delivery[0]

                # Wait for visibility timeout to expire
                time.sleep(1.5)

                # Second consumer receives the redelivered message
                second_delivery = mailbox.receive(max_messages=1)
                assert len(second_delivery) == 1
                new_msg = second_delivery[0]
                assert new_msg.delivery_count == 2

                # Old receipt handle should be rejected
                with pytest.raises(ReceiptHandleExpiredError):
                    old_msg.acknowledge()

                # New receipt handle should work
                new_msg.acknowledge()
                assert mailbox.approximate_count() == 0
            finally:
                mailbox.close()

    def test_stale_receipt_handle_rejected_for_nack(self) -> None:
        """Stale receipt handles are rejected for nack after redelivery."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-stale-nack",
                client=client,
                body_type=str,
                reaper_interval=0.1,
            )
            try:
                mailbox.send("hello")

                # First consumer receives message
                first_delivery = mailbox.receive(max_messages=1, visibility_timeout=1)
                old_msg = first_delivery[0]

                # Wait for visibility timeout to expire
                time.sleep(1.5)

                # Second consumer receives the redelivered message
                second_delivery = mailbox.receive(max_messages=1)
                new_msg = second_delivery[0]

                # Old receipt handle should be rejected for nack
                with pytest.raises(ReceiptHandleExpiredError):
                    old_msg.nack(visibility_timeout=0)

                # New message can still be processed
                new_msg.acknowledge()
            finally:
                mailbox.close()

    def test_stale_receipt_handle_rejected_for_extend(self) -> None:
        """Stale receipt handles are rejected for extend_visibility after redelivery."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-stale-extend",
                client=client,
                body_type=str,
                reaper_interval=0.1,
            )
            try:
                mailbox.send("hello")

                # First consumer receives message
                first_delivery = mailbox.receive(max_messages=1, visibility_timeout=1)
                old_msg = first_delivery[0]

                # Wait for visibility timeout to expire
                time.sleep(1.5)

                # Second consumer receives the redelivered message
                second_delivery = mailbox.receive(max_messages=1)
                new_msg = second_delivery[0]

                # Old receipt handle should be rejected for extend
                with pytest.raises(ReceiptHandleExpiredError):
                    old_msg.extend_visibility(60)

                # New message can still be processed
                new_msg.acknowledge()
            finally:
                mailbox.close()

    def test_purge_removes_all_messages(self) -> None:
        """purge() removes all messages from queue."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[int] = RedisMailbox(
                name="test-purge", client=client, body_type=int
            )
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
        with redis_standalone() as client:
            mailbox: RedisMailbox[int] = RedisMailbox(
                name="test-count", client=client, body_type=int
            )
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
        with redis_standalone() as client:
            mailbox: RedisMailbox[int] = RedisMailbox(
                name="test-maxsize", client=client, body_type=int, max_size=2
            )
            try:
                mailbox.send(1)
                mailbox.send(2)

                with pytest.raises(MailboxFullError):
                    mailbox.send(3)
            finally:
                mailbox.close()

    def test_delay_seconds(self) -> None:
        """delay_seconds delays message visibility."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-delay", client=client, body_type=str, reaper_interval=0.1
            )
            try:
                mailbox.send("delayed", delay_seconds=1)

                # Message should not be visible immediately
                messages = mailbox.receive(max_messages=1)
                assert len(messages) == 0

                # Wait for delay to expire (reaper moves it to pending)
                time.sleep(1.5)

                messages = mailbox.receive(max_messages=1)
                assert len(messages) == 1
                assert messages[0].body == "delayed"
            finally:
                mailbox.close()

    def test_long_poll_wait_time(self) -> None:
        """wait_time_seconds blocks until message arrives."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-longpoll", client=client, body_type=str
            )
            try:

                def sender() -> None:
                    time.sleep(0.3)
                    mailbox.send("hello")

                thread = threading.Thread(target=sender)
                thread.start()

                start = time.monotonic()
                messages = mailbox.receive(max_messages=1, wait_time_seconds=2)
                elapsed = time.monotonic() - start

                thread.join()
                assert len(messages) == 1
                assert messages[0].body == "hello"
                # Should have waited ~0.3s, not the full 2s
                assert elapsed < 1.0
            finally:
                mailbox.close()

    def test_long_poll_timeout(self) -> None:
        """wait_time_seconds returns empty on timeout."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-longpoll-timeout", client=client, body_type=str
            )
            try:
                start = time.monotonic()
                messages = mailbox.receive(max_messages=1, wait_time_seconds=1)
                elapsed = time.monotonic() - start

                assert len(messages) == 0
                assert elapsed >= 0.9  # Allow some tolerance
            finally:
                mailbox.close()

    def test_thread_safety(self) -> None:
        """Mailbox is thread-safe for concurrent access."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[int] = RedisMailbox(
                name="test-threadsafe", client=client, body_type=int
            )
            try:
                num_messages = 100
                num_threads = 4

                def sender(start: int) -> None:
                    for i in range(num_messages // num_threads):
                        mailbox.send(start + i)

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
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-enqueued", client=client, body_type=str
            )
            try:
                mailbox.send("hello")
                messages = mailbox.receive(max_messages=1)
                assert messages[0].enqueued_at.tzinfo == UTC
            finally:
                mailbox.close()

    def test_extend_visibility_success(self) -> None:
        """extend_visibility() extends timeout for valid handle."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-extend", client=client, body_type=str, reaper_interval=0.1
            )
            try:
                mailbox.send("hello")
                messages = mailbox.receive(max_messages=1, visibility_timeout=1)
                assert len(messages) == 1

                # Extend visibility
                messages[0].extend_visibility(60)

                # Wait for original timeout to pass
                time.sleep(1.5)

                # Message should still be invisible (not requeued)
                assert mailbox.approximate_count() == 1
                new_messages = mailbox.receive(max_messages=1)
                assert len(new_messages) == 0  # Still invisible

                # Acknowledge to clean up
                messages[0].acknowledge()
            finally:
                mailbox.close()

    def test_nack_with_delay(self) -> None:
        """nack() with visibility_timeout delays redelivery."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-nack-delay",
                client=client,
                body_type=str,
                reaper_interval=0.1,
            )
            try:
                mailbox.send("hello")
                messages = mailbox.receive(max_messages=1)
                messages[0].nack(visibility_timeout=1)

                # Message should not be visible immediately
                immediate = mailbox.receive(max_messages=1)
                assert len(immediate) == 0

                # Wait for delay to expire
                time.sleep(1.5)

                # Message should be available again
                messages = mailbox.receive(max_messages=1)
                assert len(messages) == 1
                assert messages[0].delivery_count == 2
            finally:
                mailbox.close()

    def test_implements_mailbox_protocol(self) -> None:
        """RedisMailbox implements Mailbox protocol."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-protocol", client=client, body_type=str
            )
            try:
                assert isinstance(mailbox, Mailbox)
            finally:
                mailbox.close()

    def test_closed_initially_false(self) -> None:
        """RedisMailbox.closed is False initially."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-closed", client=client, body_type=str
            )
            try:
                assert mailbox.closed is False
            finally:
                mailbox.close()

    def test_closed_after_close(self) -> None:
        """RedisMailbox.closed is True after close()."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-closed-after", client=client, body_type=str
            )
            mailbox.close()
            assert mailbox.closed is True

    def test_receive_returns_empty_when_closed(self) -> None:
        """RedisMailbox.receive() returns empty when closed."""
        with redis_standalone() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-closed-receive", client=client, body_type=str
            )
            mailbox.send("test")
            mailbox.close()

            # Should return empty even though there's a message
            messages = mailbox.receive(max_messages=1)
            assert len(messages) == 0

    def test_multiple_queues_isolated(self) -> None:
        """Messages in different queues are isolated."""
        with redis_standalone() as client:
            mailbox1: RedisMailbox[str] = RedisMailbox(
                name="queue1", client=client, body_type=str
            )
            mailbox2: RedisMailbox[str] = RedisMailbox(
                name="queue2", client=client, body_type=str
            )
            try:
                mailbox1.send("message1")
                mailbox2.send("message2")

                msgs1 = mailbox1.receive(max_messages=1)
                msgs2 = mailbox2.receive(max_messages=1)

                assert len(msgs1) == 1
                assert len(msgs2) == 1
                assert msgs1[0].body == "message1"
                assert msgs2[0].body == "message2"
            finally:
                mailbox1.close()
                mailbox2.close()


# =============================================================================
# Redis Cluster Tests
# =============================================================================


@pytest.mark.redis_cluster
@pytest.mark.skipif(
    not REDIS_CLUSTER_TESTS_ENABLED,
    reason="Redis Cluster tests disabled (set REDIS_CLUSTER_TESTS=0 to disable)",
)
class TestRedisMailboxCluster:
    """Tests for RedisMailbox with Redis Cluster.

    These tests are enabled by default. Cluster setup takes ~60s per test.
    Disable with REDIS_CLUSTER_TESTS=0 environment variable.
    """

    def test_send_and_receive_cluster(self) -> None:
        """Basic send and receive workflow on cluster."""
        with redis_cluster() as client:
            mailbox: RedisMailbox[_Event] = RedisMailbox(
                name="test-cluster", client=client, body_type=_Event
            )
            try:
                event = _Event(data="cluster-test", count=123)
                mailbox.send(event)

                messages = mailbox.receive(max_messages=1)
                assert len(messages) == 1
                assert messages[0].body == event
            finally:
                mailbox.close()

    def test_cluster_hash_tags_work(self) -> None:
        """Verify hash tags keep queue keys on same slot."""
        with redis_cluster() as client:
            # Send multiple messages - if hash tags don't work,
            # Lua scripts would fail with CROSSSLOT error
            mailbox: RedisMailbox[int] = RedisMailbox(
                name="test-hashtag", client=client, body_type=int
            )
            try:
                for i in range(10):
                    mailbox.send(i)

                messages = mailbox.receive(max_messages=10)
                assert len(messages) == 10

                # Acknowledge all
                for msg in messages:
                    msg.acknowledge()

                assert mailbox.approximate_count() == 0
            finally:
                mailbox.close()

    def test_cluster_visibility_timeout(self) -> None:
        """Visibility timeout works on cluster."""
        with redis_cluster() as client:
            mailbox: RedisMailbox[str] = RedisMailbox(
                name="test-cluster-vis",
                client=client,
                body_type=str,
                reaper_interval=0.1,
            )
            try:
                mailbox.send("hello")
                messages = mailbox.receive(max_messages=1, visibility_timeout=1)
                assert len(messages) == 1

                # Wait for visibility timeout
                time.sleep(1.5)

                # Message should be requeued
                messages = mailbox.receive(max_messages=1)
                assert len(messages) == 1
                assert messages[0].delivery_count == 2
            finally:
                mailbox.close()

    def test_cluster_purge(self) -> None:
        """purge() works on cluster."""
        with redis_cluster() as client:
            mailbox: RedisMailbox[int] = RedisMailbox(
                name="test-cluster-purge", client=client, body_type=int
            )
            try:
                for i in range(5):
                    mailbox.send(i)

                count = mailbox.purge()
                assert count == 5
                assert mailbox.approximate_count() == 0
            finally:
                mailbox.close()
