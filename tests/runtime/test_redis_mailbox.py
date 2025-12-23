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

"""Integration tests for RedisMailbox implementation.

These tests require a Redis server to be available. They will be skipped
if redis-server is not installed on the system.

Run with: pytest tests/runtime/test_redis_mailbox.py -v --no-cov
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING

import pytest

from tests.helpers import skip_if_no_redis, skip_if_no_redis_cluster

if TYPE_CHECKING:
    from redis import Redis
    from redis.cluster import RedisCluster

    from tests.helpers import RedisClusterManager, RedisStandalone
    from weakincentives.runtime.mailbox import RedisMailbox

# Skip entire module if Redis is not available
pytestmark = pytest.mark.skipif(
    not skip_if_no_redis(),
    reason="Redis not installed",
)


@dataclass(slots=True, frozen=True)
class _Event:
    """Sample event type for testing."""

    data: str


@pytest.fixture
def redis_standalone() -> Iterator[RedisStandalone]:
    """Provide a standalone Redis server for testing."""
    from tests.helpers import RedisStandalone

    server = RedisStandalone()
    try:
        server.start()
    except (FileNotFoundError, RuntimeError) as e:
        pytest.skip(f"Could not start Redis server: {e}")
    yield server
    server.stop()


@pytest.fixture
def redis_cluster() -> Iterator[RedisClusterManager]:
    """Provide a Redis Cluster for testing."""
    from tests.helpers import RedisClusterManager

    manager = RedisClusterManager()
    try:
        manager.start()
    except RuntimeError as e:
        pytest.skip(f"Could not start Redis cluster: {e}")
    yield manager
    manager.stop()


@pytest.fixture
def redis_client(redis_standalone: RedisStandalone) -> Iterator[Redis[bytes]]:
    """Provide a Redis client connected to standalone server."""
    client = redis_standalone.client()
    yield client
    client.close()


@pytest.fixture
def cluster_client(redis_cluster: RedisClusterManager) -> Iterator[RedisCluster[bytes]]:
    """Provide a Redis Cluster client."""
    client = redis_cluster.client()
    yield client
    client.close()


@pytest.fixture
def mailbox(redis_client: Redis[bytes]) -> Iterator[RedisMailbox[str]]:
    """Provide a RedisMailbox instance for testing."""
    from weakincentives.runtime.mailbox import RedisMailbox

    mailbox: RedisMailbox[str] = RedisMailbox(name="test", client=redis_client)
    yield mailbox
    mailbox.close()
    # Clean up Redis keys
    redis_client.flushdb()


@pytest.fixture
def event_mailbox(redis_client: Redis[bytes]) -> Iterator[RedisMailbox[dict]]:
    """Provide a RedisMailbox for _Event type."""
    from weakincentives.runtime.mailbox import RedisMailbox

    mailbox: RedisMailbox[dict] = RedisMailbox(name="events", client=redis_client)
    yield mailbox
    mailbox.close()
    redis_client.flushdb()


# =============================================================================
# Basic Operations Tests
# =============================================================================


class TestRedisMailboxBasicOperations:
    """Tests for basic RedisMailbox operations."""

    def test_send_returns_message_id(self, mailbox) -> None:
        """send() returns a unique message ID."""
        msg_id = mailbox.send("hello")
        assert isinstance(msg_id, str)
        assert len(msg_id) > 0

    def test_send_and_receive_basic(self, mailbox) -> None:
        """Basic send and receive workflow."""
        mailbox.send("test-data")
        messages = mailbox.receive(max_messages=1)

        assert len(messages) == 1
        assert messages[0].body == "test-data"
        assert messages[0].delivery_count == 1

    def test_send_and_receive_dict(self, event_mailbox) -> None:
        """Send and receive dict payloads."""
        event = {"data": "test-data", "count": 42}
        event_mailbox.send(event)

        messages = event_mailbox.receive(max_messages=1)
        assert len(messages) == 1
        assert messages[0].body == event

    def test_receive_empty_queue(self, mailbox) -> None:
        """receive() returns empty list when queue is empty."""
        messages = mailbox.receive(max_messages=1)
        assert messages == []

    def test_receive_max_messages(self, mailbox) -> None:
        """receive() respects max_messages limit."""
        for i in range(5):
            mailbox.send(str(i))

        messages = mailbox.receive(max_messages=3)
        assert len(messages) == 3
        assert [m.body for m in messages] == ["0", "1", "2"]

    def test_receive_max_messages_clamped(self, mailbox) -> None:
        """receive() clamps max_messages to 1-10 range."""
        for i in range(15):
            mailbox.send(str(i))

        # max_messages > 10 should be clamped to 10
        messages = mailbox.receive(max_messages=15)
        assert len(messages) == 10

    def test_receive_fifo_order(self, mailbox) -> None:
        """Messages are received in FIFO order."""
        for i in range(3):
            mailbox.send(str(i))

        messages = mailbox.receive(max_messages=3)
        assert [m.body for m in messages] == ["0", "1", "2"]

    def test_message_enqueued_at(self, mailbox) -> None:
        """Message enqueued_at is set correctly with UTC timezone."""
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)
        assert messages[0].enqueued_at.tzinfo == UTC


# =============================================================================
# Acknowledge Tests
# =============================================================================


class TestRedisMailboxAcknowledge:
    """Tests for acknowledge operations."""

    def test_acknowledge_removes_message(self, mailbox) -> None:
        """acknowledge() removes message from queue."""
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)
        assert len(messages) == 1

        messages[0].acknowledge()
        assert mailbox.approximate_count() == 0

    def test_acknowledge_expired_handle_raises(self, mailbox) -> None:
        """acknowledge() raises ReceiptHandleExpiredError for invalid handle."""
        from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)
        messages[0].acknowledge()

        with pytest.raises(ReceiptHandleExpiredError):
            messages[0].acknowledge()

    def test_acknowledge_after_redelivery_raises(self, mailbox) -> None:
        """acknowledge() with old handle raises after redelivery."""
        from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

        mailbox.send("hello")
        msg1 = mailbox.receive(max_messages=1, visibility_timeout=1)[0]

        # Wait for timeout and redelivery
        time.sleep(2.5)

        # Get redelivered message
        msg2 = mailbox.receive(max_messages=1)[0]
        assert msg2.delivery_count == 2

        # Old receipt handle should be invalid
        with pytest.raises(ReceiptHandleExpiredError):
            msg1.acknowledge()

        # New receipt handle should work
        msg2.acknowledge()


# =============================================================================
# Nack Tests
# =============================================================================


class TestRedisMailboxNack:
    """Tests for nack operations."""

    def test_nack_returns_message_to_queue(self, mailbox) -> None:
        """nack() returns message to queue for immediate redelivery."""
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)
        messages[0].nack(visibility_timeout=0)

        # Message should be available again immediately
        messages = mailbox.receive(max_messages=1)
        assert len(messages) == 1
        assert messages[0].body == "hello"
        assert messages[0].delivery_count == 2

    def test_nack_with_delay(self, mailbox) -> None:
        """nack() with visibility_timeout delays redelivery."""
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)
        messages[0].nack(visibility_timeout=1)

        # Message should not be immediately available
        messages = mailbox.receive(max_messages=1)
        assert len(messages) == 0

        # Wait for visibility to expire (1s) plus reaper interval (1s) plus margin
        time.sleep(2.5)

        messages = mailbox.receive(max_messages=1)
        assert len(messages) == 1
        assert messages[0].delivery_count == 2

    def test_nack_expired_handle_raises(self, mailbox) -> None:
        """nack() raises ReceiptHandleExpiredError for invalid handle."""
        from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)
        messages[0].acknowledge()

        with pytest.raises(ReceiptHandleExpiredError):
            messages[0].nack(visibility_timeout=0)


# =============================================================================
# Extend Visibility Tests
# =============================================================================


class TestRedisMailboxExtendVisibility:
    """Tests for extend_visibility operations."""

    def test_extend_visibility_success(self, mailbox) -> None:
        """extend_visibility() extends timeout for valid handle."""
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1, visibility_timeout=1)
        assert len(messages) == 1

        # Extend visibility before it expires
        messages[0].extend_visibility(60)

        # Wait past original timeout
        time.sleep(2.5)

        # Message should still be invisible (not requeued)
        assert mailbox.approximate_count() == 1
        new_messages = mailbox.receive(max_messages=1)
        assert len(new_messages) == 0

        # Acknowledge to clean up
        messages[0].acknowledge()

    def test_extend_visibility_expired_handle_raises(self, mailbox) -> None:
        """extend_visibility() raises ReceiptHandleExpiredError for invalid handle."""
        from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1)
        messages[0].acknowledge()

        with pytest.raises(ReceiptHandleExpiredError):
            messages[0].extend_visibility(60)


# =============================================================================
# Visibility Timeout Tests
# =============================================================================


class TestRedisMailboxVisibilityTimeout:
    """Tests for visibility timeout behavior."""

    def test_visibility_timeout_requeues_message(self, mailbox) -> None:
        """Message is requeued after visibility timeout expires."""
        mailbox.send("hello")
        messages = mailbox.receive(max_messages=1, visibility_timeout=1)
        assert len(messages) == 1

        # Wait for visibility timeout to expire (1s) plus reaper interval (1s) plus margin
        time.sleep(2.5)

        # Message should be available again
        messages = mailbox.receive(max_messages=1)
        assert len(messages) == 1
        assert messages[0].body == "hello"
        assert messages[0].delivery_count == 2

    def test_message_invisible_before_timeout(self, mailbox) -> None:
        """Message is not visible to other receive() calls before timeout."""
        mailbox.send("hello")
        mailbox.receive(max_messages=1, visibility_timeout=10)

        # Should not find another message
        messages = mailbox.receive(max_messages=1)
        assert len(messages) == 0


# =============================================================================
# Delay Seconds Tests
# =============================================================================


class TestRedisMailboxDelay:
    """Tests for delay_seconds functionality."""

    def test_delay_seconds(self, mailbox) -> None:
        """delay_seconds delays message visibility."""
        mailbox.send("delayed", delay_seconds=1)

        # Message should not be visible immediately
        messages = mailbox.receive(max_messages=1)
        assert len(messages) == 0

        # Wait for delay to expire
        time.sleep(2.5)

        messages = mailbox.receive(max_messages=1)
        assert len(messages) == 1
        assert messages[0].body == "delayed"


# =============================================================================
# Purge Tests
# =============================================================================


class TestRedisMailboxPurge:
    """Tests for purge operations."""

    def test_purge_removes_all_messages(self, mailbox) -> None:
        """purge() removes all messages from queue."""
        for i in range(5):
            mailbox.send(str(i))

        count = mailbox.purge()
        assert count == 5
        assert mailbox.approximate_count() == 0

    def test_purge_includes_invisible_messages(self, mailbox) -> None:
        """purge() removes both pending and invisible messages."""
        mailbox.send("1")
        mailbox.send("2")

        # Make one message invisible
        mailbox.receive(max_messages=1, visibility_timeout=60)

        # purge should remove both
        count = mailbox.purge()
        assert count == 2
        assert mailbox.approximate_count() == 0


# =============================================================================
# Approximate Count Tests
# =============================================================================


class TestRedisMailboxApproximateCount:
    """Tests for approximate_count operations."""

    def test_approximate_count(self, mailbox) -> None:
        """approximate_count() returns correct message count."""
        assert mailbox.approximate_count() == 0

        for i in range(3):
            mailbox.send(str(i))
        assert mailbox.approximate_count() == 3

        mailbox.receive(max_messages=1)[0].acknowledge()
        assert mailbox.approximate_count() == 2

    def test_approximate_count_includes_invisible(self, mailbox) -> None:
        """approximate_count() includes invisible messages."""
        mailbox.send("1")
        mailbox.send("2")

        # Make one invisible
        mailbox.receive(max_messages=1, visibility_timeout=60)

        # Count should still be 2 (1 pending + 1 invisible)
        assert mailbox.approximate_count() == 2


# =============================================================================
# Long Poll Tests
# =============================================================================


class TestRedisMailboxLongPoll:
    """Tests for long polling functionality."""

    def test_long_poll_returns_immediately_with_messages(self, mailbox) -> None:
        """wait_time_seconds returns immediately when messages are available."""
        mailbox.send("hello")

        start = time.monotonic()
        messages = mailbox.receive(max_messages=1, wait_time_seconds=5)
        elapsed = time.monotonic() - start

        assert len(messages) == 1
        assert elapsed < 1.0  # Should return much faster than 5 seconds

    def test_long_poll_timeout(self, mailbox) -> None:
        """wait_time_seconds returns empty on timeout."""
        start = time.monotonic()
        messages = mailbox.receive(max_messages=1, wait_time_seconds=0.5)
        elapsed = time.monotonic() - start

        assert len(messages) == 0
        assert elapsed >= 0.5

    def test_long_poll_wakes_on_message(self, mailbox) -> None:
        """wait_time_seconds wakes when message arrives."""

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
        # Should have received within ~0.5s (sender delay + polling)
        assert elapsed < 1.0


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestRedisMailboxThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_send(self, mailbox) -> None:
        """Concurrent sends are handled correctly."""
        num_threads = 4
        messages_per_thread = 12
        num_messages = num_threads * messages_per_thread

        def sender(start: int) -> None:
            for i in range(messages_per_thread):
                mailbox.send(str(start + i))

        threads = [
            threading.Thread(target=sender, args=(i * messages_per_thread,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert mailbox.approximate_count() == num_messages

    def test_concurrent_receive(self, mailbox) -> None:
        """Concurrent receives don't return duplicates."""
        num_messages = 50

        for i in range(num_messages):
            mailbox.send(str(i))

        received: list[str] = []
        lock = threading.Lock()

        def receiver() -> None:
            while True:
                messages = mailbox.receive(max_messages=5, visibility_timeout=30)
                if not messages:
                    break
                for msg in messages:
                    with lock:
                        received.append(msg.body)
                    msg.acknowledge()

        threads = [threading.Thread(target=receiver) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All messages should be received exactly once
        assert len(received) == num_messages
        assert len(set(received)) == num_messages  # No duplicates


# =============================================================================
# Closed State Tests
# =============================================================================


class TestRedisMailboxClosedState:
    """Tests for closed state behavior."""

    def test_closed_initially_false(self, mailbox) -> None:
        """closed is False initially."""
        assert mailbox.closed is False

    def test_closed_after_close(self, mailbox) -> None:
        """closed is True after close()."""
        mailbox.close()
        assert mailbox.closed is True

    def test_receive_returns_empty_when_closed(self, mailbox) -> None:
        """receive() returns empty when closed."""
        mailbox.send("test")
        mailbox.close()

        messages = mailbox.receive(max_messages=1)
        assert len(messages) == 0


# =============================================================================
# Protocol Tests
# =============================================================================


def test_redis_mailbox_is_mailbox(mailbox) -> None:
    """RedisMailbox implements Mailbox protocol."""
    from weakincentives.runtime.mailbox import Mailbox

    assert isinstance(mailbox, Mailbox)


# =============================================================================
# Redis Cluster Tests
# =============================================================================


@pytest.mark.skipif(
    not skip_if_no_redis_cluster(),
    reason="Redis Cluster requires redis-cli",
)
class TestRedisMailboxCluster:
    """Tests for RedisMailbox with Redis Cluster."""

    @pytest.fixture
    def cluster_mailbox(self, cluster_client):
        """Provide a RedisMailbox connected to cluster."""
        from weakincentives.runtime.mailbox import RedisMailbox

        mailbox: RedisMailbox[str] = RedisMailbox(
            name="cluster-test", client=cluster_client
        )
        yield mailbox
        mailbox.close()

    def test_cluster_send_receive(self, cluster_mailbox) -> None:
        """Basic send/receive works with cluster."""
        cluster_mailbox.send("hello")
        messages = cluster_mailbox.receive(max_messages=1)

        assert len(messages) == 1
        assert messages[0].body == "hello"

    def test_cluster_fifo_order(self, cluster_mailbox) -> None:
        """FIFO order preserved in cluster."""
        for i in range(5):
            cluster_mailbox.send(str(i))

        messages = cluster_mailbox.receive(max_messages=5)
        assert [m.body for m in messages] == ["0", "1", "2", "3", "4"]

    def test_cluster_acknowledge(self, cluster_mailbox) -> None:
        """Acknowledge works in cluster."""
        cluster_mailbox.send("hello")
        messages = cluster_mailbox.receive(max_messages=1)
        messages[0].acknowledge()

        assert cluster_mailbox.approximate_count() == 0

    def test_cluster_nack(self, cluster_mailbox) -> None:
        """Nack works in cluster."""
        cluster_mailbox.send("hello")
        messages = cluster_mailbox.receive(max_messages=1)
        messages[0].nack(visibility_timeout=0)

        messages = cluster_mailbox.receive(max_messages=1)
        assert messages[0].delivery_count == 2

    def test_cluster_purge(self, cluster_mailbox) -> None:
        """Purge works in cluster."""
        for i in range(3):
            cluster_mailbox.send(str(i))

        count = cluster_mailbox.purge()
        assert count == 3


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestRedisMailboxErrorHandling:
    """Tests for error handling in RedisMailbox."""

    def test_serialization_error_on_non_serializable(self, redis_client) -> None:
        """SerializationError raised for non-serializable body."""
        from weakincentives.runtime.mailbox import RedisMailbox, SerializationError

        mailbox: RedisMailbox[object] = RedisMailbox(name="test", client=redis_client)
        try:
            with pytest.raises(SerializationError):
                mailbox.send(lambda x: x)  # type: ignore[arg-type]
        finally:
            mailbox.close()
            redis_client.flushdb()


# =============================================================================
# Key Format Tests
# =============================================================================


class TestRedisMailboxKeyFormat:
    """Tests for Redis key format and hash tags."""

    def test_key_format_uses_hash_tags(self, mailbox, redis_client) -> None:
        """Keys use hash tags for cluster compatibility."""
        mailbox.send("test")

        # Verify keys use hash tag format
        keys = redis_client.keys("*")
        key_names = [k.decode() if isinstance(k, bytes) else k for k in keys]

        # All keys should have the {queue:name} hash tag
        for key in key_names:
            assert "{queue:test}" in key


# =============================================================================
# Reaper Tests
# =============================================================================


class TestRedisMailboxReaper:
    """Tests for visibility reaper functionality."""

    def test_reaper_requeues_expired_messages(self, redis_client) -> None:
        """Reaper thread requeues messages with expired visibility."""
        from weakincentives.runtime.mailbox import RedisMailbox

        # Use fast reaper interval for testing
        mailbox: RedisMailbox[str] = RedisMailbox(
            name="reaper-test", client=redis_client, reaper_interval=0.1
        )
        try:
            mailbox.send("hello")
            mailbox.receive(max_messages=1, visibility_timeout=1)

            # Wait for visibility to expire and reaper to run
            time.sleep(2.5)

            # Message should be back in queue
            messages = mailbox.receive(max_messages=1)
            assert len(messages) == 1
            assert messages[0].delivery_count == 2
        finally:
            mailbox.close()
            redis_client.flushdb()


# =============================================================================
# Multiple Queue Tests
# =============================================================================


class TestRedisMailboxMultipleQueues:
    """Tests for multiple queue isolation."""

    def test_queues_are_isolated(self, redis_client) -> None:
        """Different queues are isolated from each other."""
        from weakincentives.runtime.mailbox import RedisMailbox

        mailbox1: RedisMailbox[str] = RedisMailbox(name="queue1", client=redis_client)
        mailbox2: RedisMailbox[str] = RedisMailbox(name="queue2", client=redis_client)

        try:
            mailbox1.send("msg1")
            mailbox2.send("msg2")

            assert mailbox1.approximate_count() == 1
            assert mailbox2.approximate_count() == 1

            msgs1 = mailbox1.receive(max_messages=1)
            assert msgs1[0].body == "msg1"

            msgs2 = mailbox2.receive(max_messages=1)
            assert msgs2[0].body == "msg2"
        finally:
            mailbox1.close()
            mailbox2.close()
            redis_client.flushdb()
