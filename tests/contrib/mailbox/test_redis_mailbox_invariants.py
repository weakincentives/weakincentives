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

# ruff: noqa: PERF401
"""
Targeted tests for specific mailbox invariants.

These tests focus on edge cases and race conditions that are
critical for correctness. Each test class corresponds to an
invariant from specs/VERIFICATION.md.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from redis import Redis

    from weakincentives.contrib.mailbox import RedisMailbox

try:
    from hypothesis import HealthCheck, given, settings, strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

pytestmark = pytest.mark.redis_standalone


class TestReceiptHandleFreshness:
    """Tests for INV-2: Receipt Handle Freshness."""

    def test_redelivery_generates_new_handle(self, mailbox: RedisMailbox[Any]) -> None:
        """Each delivery of the same message gets a unique handle."""
        mailbox.send("test")

        # First receive
        msgs1 = mailbox.receive(visibility_timeout=1)
        handle1 = msgs1[0].receipt_handle

        # Wait for timeout
        time.sleep(1.5)

        # Second receive (redelivery)
        msgs2 = mailbox.receive(visibility_timeout=30)
        handle2 = msgs2[0].receipt_handle

        assert handle1 != handle2, "Redelivery must generate new handle"
        assert msgs1[0].id == msgs2[0].id, "Same message ID"

    def test_old_handle_rejected_after_redelivery(
        self, mailbox: RedisMailbox[Any]
    ) -> None:
        """Stale handles from previous delivery are rejected."""
        from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

        mailbox.send("test")

        # First consumer receives
        msgs1 = mailbox.receive(visibility_timeout=1)
        old_handle = msgs1[0].receipt_handle
        msg_id = msgs1[0].id

        # Wait for timeout and redelivery
        time.sleep(1.5)

        # Second consumer receives
        msgs2 = mailbox.receive(visibility_timeout=30)
        assert len(msgs2) == 1

        # First consumer tries to ack with old handle
        old_suffix = old_handle.split(":", 1)[1]
        with pytest.raises(ReceiptHandleExpiredError):
            mailbox._acknowledge(msg_id, old_suffix)

        # Second consumer can still ack
        msgs2[0].acknowledge()

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(st.integers(min_value=2, max_value=5))  # type: ignore[misc]
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )  # type: ignore[misc]
    def test_handle_unique_across_n_deliveries(
        self, redis_client: Redis[bytes], n: int
    ) -> None:
        """Handles are unique across N deliveries of the same message."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mailbox: RedisMailbox[Any] = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.05,
        )

        try:
            mailbox.send("test")
            handles = []

            for _ in range(n):
                msgs = mailbox.receive(visibility_timeout=1)
                if msgs:
                    handles.append(msgs[0].receipt_handle)
                time.sleep(0.2)  # Allow reaper to run

            # All handles should be unique
            assert len(handles) == len(set(handles)), (
                f"Duplicate handles found: {handles}"
            )
        finally:
            mailbox.close()
            mailbox.purge()


class TestMessageStateExclusivity:
    """Tests for INV-1: Message State Exclusivity."""

    def test_receive_atomic_transition(
        self, mailbox: RedisMailbox[Any], redis_client: Redis[bytes]
    ) -> None:
        """Receive atomically moves message from pending to invisible."""
        mailbox.send("test")

        # Receive
        msgs = mailbox.receive(visibility_timeout=30)
        msg_id = msgs[0].id

        # Check state
        in_pending = redis_client.lrange(mailbox._keys.pending, 0, -1)
        in_invisible = redis_client.zscore(mailbox._keys.invisible, msg_id)

        assert msg_id.encode() not in in_pending
        assert in_invisible is not None

    def test_concurrent_receive_no_duplicates(self, mailbox: RedisMailbox[Any]) -> None:
        """Parallel receives never return the same message twice."""
        # Send 100 messages
        for i in range(100):
            mailbox.send(f"msg-{i}")

        received: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            local_received = []
            for _ in range(50):
                msgs = mailbox.receive(visibility_timeout=60)
                for m in msgs:
                    local_received.append(m.id)
            with lock:
                received.extend(local_received)

        # Run 4 concurrent workers
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No duplicates
        assert len(received) == len(set(received)), "Duplicate messages received"

    def test_reap_and_receive_race(
        self, mailbox: RedisMailbox[Any], redis_client: Redis[bytes]
    ) -> None:
        """
        Reaper and receive don't cause duplicate state.

        This tests a potential race where:
        1. Message expires in invisible
        2. Reaper starts moving it to pending
        3. Consumer calls receive
        """
        # Send and receive with short timeout
        mailbox.send("test")
        msgs = mailbox.receive(visibility_timeout=1)
        msg_id = msgs[0].id

        # Wait for expiry
        time.sleep(1.2)

        # Check invariant: message in exactly one place
        in_pending = msg_id.encode() in redis_client.lrange(
            mailbox._keys.pending, 0, -1
        )
        in_invisible = redis_client.zscore(mailbox._keys.invisible, msg_id) is not None

        assert in_pending != in_invisible, (
            f"Message in invalid state: pending={in_pending}, invisible={in_invisible}"
        )


class TestNoMessageLoss:
    """Tests for INV-5: No Message Loss."""

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(  # type: ignore[misc]
        st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20)
    )
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )  # type: ignore[misc]
    def test_all_messages_accounted_for(
        self, redis_client: Redis[bytes], bodies: list[str]
    ) -> None:
        """Every sent message is acked or remains in queue."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mailbox: RedisMailbox[Any] = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,
        )

        try:
            # Send all messages
            for body in bodies:
                mailbox.send(body)

            # Receive and ack half
            acked: set[str] = set()
            for _ in range(len(bodies) // 2):
                msgs = mailbox.receive(visibility_timeout=60)
                if msgs:
                    msgs[0].acknowledge()
                    acked.add(msgs[0].id)

            # Verify: acked + remaining = total
            remaining = mailbox.approximate_count()
            assert len(acked) + remaining == len(bodies), (
                f"Message loss: sent={len(bodies)}, acked={len(acked)}, remaining={remaining}"
            )
        finally:
            mailbox.close()
            mailbox.purge()

    def test_crash_recovery_no_loss(self, redis_client: Redis[bytes]) -> None:
        """Messages survive mailbox close/reopen."""
        from weakincentives.contrib.mailbox import RedisMailbox

        name = f"test-{uuid4().hex[:8]}"

        # First mailbox instance sends messages
        mb1: RedisMailbox[Any] = RedisMailbox(name=name, client=redis_client)
        for i in range(10):
            mb1.send(f"msg-{i}")
        mb1.close()

        # Second instance sees all messages
        mb2: RedisMailbox[Any] = RedisMailbox(name=name, client=redis_client)
        try:
            assert mb2.approximate_count() == 10
        finally:
            mb2.close()
            mb2.purge()


class TestDeliveryCountMonotonicity:
    """Tests for INV-4: Delivery Count Monotonicity."""

    def test_delivery_count_increments(self, mailbox: RedisMailbox[Any]) -> None:
        """Each delivery increments the count."""
        mailbox.send("test")

        counts = []
        for _ in range(3):
            msgs = mailbox.receive(visibility_timeout=1)
            if msgs:
                counts.append(msgs[0].delivery_count)
            time.sleep(0.2)  # Allow reaper

        # Counts should be strictly increasing
        assert counts == sorted(counts), f"Counts not monotonic: {counts}"
        assert len(set(counts)) == len(counts), f"Duplicate counts: {counts}"

    def test_delivery_count_survives_redelivery(
        self, mailbox: RedisMailbox[Any]
    ) -> None:
        """Delivery count persists across timeout and requeue."""
        mailbox.send("test")

        # First delivery
        msgs1 = mailbox.receive(visibility_timeout=1)
        assert msgs1[0].delivery_count == 1

        # Let it timeout and get requeued
        time.sleep(1.5)

        # Second delivery - count should be 2, not reset to 1
        msgs2 = mailbox.receive(visibility_timeout=1)
        assert msgs2[0].delivery_count == 2, "Delivery count was reset after requeue!"

        # Let it timeout again
        time.sleep(1.5)

        # Third delivery
        msgs3 = mailbox.receive(visibility_timeout=30)
        assert msgs3[0].delivery_count == 3, (
            "Delivery count was reset after second requeue!"
        )

    def test_delivery_count_survives_nack(self, mailbox: RedisMailbox[Any]) -> None:
        """Delivery count persists across nack and requeue."""
        mailbox.send("test")

        # First delivery
        msgs1 = mailbox.receive(visibility_timeout=30)
        assert msgs1[0].delivery_count == 1
        msgs1[0].nack(visibility_timeout=0)  # Immediate requeue

        # Second delivery after nack
        msgs2 = mailbox.receive(visibility_timeout=30)
        assert msgs2[0].delivery_count == 2, "Delivery count was reset after nack!"


class TestVisibilityTimeout:
    """Tests for INV-6: Visibility Timeout Correctness."""

    def test_message_requeued_after_timeout(self, mailbox: RedisMailbox[Any]) -> None:
        """Expired messages return to pending."""
        mailbox.send("test")

        # Receive with short timeout
        msgs = mailbox.receive(visibility_timeout=1)
        assert len(msgs) == 1

        # Immediately try to receive again - should be empty
        msgs2 = mailbox.receive(visibility_timeout=30, wait_time_seconds=0)
        assert len(msgs2) == 0

        # Wait for timeout
        time.sleep(1.5)

        # Now should be available
        msgs3 = mailbox.receive(visibility_timeout=30)
        assert len(msgs3) == 1
        assert msgs3[0].id == msgs[0].id

    def test_extend_prevents_requeue(self, mailbox: RedisMailbox[Any]) -> None:
        """Extended visibility prevents timeout requeue."""
        mailbox.send("test")

        msgs = mailbox.receive(visibility_timeout=1)
        msg = msgs[0]

        # Extend before timeout
        time.sleep(0.5)
        msg.extend_visibility(10)

        # Wait past original timeout
        time.sleep(1.0)

        # Should still be invisible (not requeued)
        msgs2 = mailbox.receive(visibility_timeout=30, wait_time_seconds=0)
        assert len(msgs2) == 0, "Message was requeued despite extension"

        # Original handle should still work
        msg.acknowledge()


class TestEventualRequeue:
    """
    Liveness tests for INV-6: EventualRequeue.

    Note: The TLA+ EventualRequeue temporal property cannot be checked by TLC
    because it quantifies over state variable domains. These property-based
    tests verify the liveness property through the actual implementation.
    """

    def test_expired_message_eventually_requeued(
        self, mailbox: RedisMailbox[Any]
    ) -> None:
        """Expired messages eventually return to pending (liveness)."""
        mailbox.send("test")

        # Receive with very short timeout
        msgs = mailbox.receive(visibility_timeout=1)
        assert len(msgs) == 1
        msg_id = msgs[0].id

        # Wait for expiry + reaper interval + buffer
        max_wait = 3.0  # seconds
        start = time.time()
        requeued = False

        while time.time() - start < max_wait:
            # Try to receive the requeued message
            msgs2 = mailbox.receive(visibility_timeout=30, wait_time_seconds=0)
            if msgs2 and msgs2[0].id == msg_id:
                requeued = True
                msgs2[0].acknowledge()
                break
            time.sleep(0.1)

        assert requeued, f"Message not requeued within {max_wait}s (liveness violation)"

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(st.integers(min_value=1, max_value=5))  # type: ignore[misc]
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )  # type: ignore[misc]
    def test_multiple_messages_eventually_requeued(
        self, redis_client: Redis[bytes], n: int
    ) -> None:
        """All expired messages eventually return to pending."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mailbox: RedisMailbox[Any] = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,  # Fast reaper for testing
        )

        try:
            # Send n messages
            sent_ids = [mailbox.send(f"msg-{i}") for i in range(n)]

            # Receive all with short timeout
            received = []
            for _ in range(n):
                msgs = mailbox.receive(visibility_timeout=1, wait_time_seconds=1)
                if msgs:
                    received.append(msgs[0].id)

            assert len(received) == n

            # Wait for all to expire and requeue
            time.sleep(1.5)

            # All should be requeued and receivable again
            requeued = []
            for _ in range(n * 2):  # Extra iterations for safety
                msgs = mailbox.receive(visibility_timeout=30, wait_time_seconds=0)
                if msgs:
                    requeued.append(msgs[0].id)
                    msgs[0].acknowledge()
                if len(requeued) == n:
                    break
                time.sleep(0.1)

            assert set(requeued) == set(sent_ids), (
                f"Not all messages requeued: expected {set(sent_ids)}, got {set(requeued)}"
            )
        finally:
            mailbox.close()
            mailbox.purge()


class TestFIFOOrdering:
    """Tests for INV-7: FIFO Ordering."""

    def test_messages_received_in_send_order(self, mailbox: RedisMailbox[Any]) -> None:
        """Messages are delivered in FIFO order."""
        # Send in order
        ids = []
        for i in range(10):
            msg_id = mailbox.send(f"msg-{i}")
            ids.append(msg_id)

        # Receive and verify order
        received_ids = []
        while True:
            msgs = mailbox.receive(visibility_timeout=60)
            if not msgs:
                break
            received_ids.append(msgs[0].id)
            msgs[0].acknowledge()

        assert received_ids == ids, "Messages received out of order"
