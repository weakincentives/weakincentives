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

"""
Stress tests for concurrent access patterns.

These tests verify correctness under high concurrency and are marked
as slow tests that run only when explicitly requested.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from weakincentives.runtime import wait_until
from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

if TYPE_CHECKING:
    from redis import Redis

pytestmark = [pytest.mark.redis_standalone, pytest.mark.slow]


class TestConcurrentStress:
    """High-concurrency stress tests."""

    def test_producer_consumer_stress(self, redis_client: Redis[bytes]) -> None:
        """
        Multiple producers and consumers operating concurrently.

        Verifies:
        - No message loss
        - No duplicate processing
        - Correct final count
        """
        from weakincentives.contrib.mailbox import RedisMailbox

        mailbox: RedisMailbox[Any] = RedisMailbox(
            name=f"stress-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,
        )

        num_producers = 4
        num_consumers = 4
        messages_per_producer = 100
        total_messages = num_producers * messages_per_producer

        sent: list[str] = []
        received: list[str] = []
        acked: list[str] = []
        sent_lock = threading.Lock()
        received_lock = threading.Lock()
        acked_lock = threading.Lock()
        stop_consumers = threading.Event()

        def producer(producer_id: int) -> None:
            for i in range(messages_per_producer):
                body = f"p{producer_id}-m{i}"
                msg_id = mailbox.send(body)
                with sent_lock:
                    sent.append(msg_id)

        def consumer(consumer_id: int) -> None:
            while not stop_consumers.is_set():
                msgs = mailbox.receive(
                    visibility_timeout=30,
                    wait_time_seconds=1,
                )
                for msg in msgs:
                    with received_lock:
                        received.append(msg.id)
                    try:
                        msg.acknowledge()
                        with acked_lock:
                            acked.append(msg.id)
                    except ReceiptHandleExpiredError:
                        pass  # Expected if redelivered

        try:
            # Start consumers first
            consumer_threads = [
                threading.Thread(target=consumer, args=(i,))
                for i in range(num_consumers)
            ]
            for t in consumer_threads:
                t.start()

            # Then producers
            producer_threads = [
                threading.Thread(target=producer, args=(i,))
                for i in range(num_producers)
            ]
            for t in producer_threads:
                t.start()
            for t in producer_threads:
                t.join()

            wait_until(
                lambda: mailbox.approximate_count() == 0,
                timeout=2.0,
                poll_interval=0.05,
            )
            stop_consumers.set()
            for t in consumer_threads:
                t.join(timeout=5)

            # Verify
            remaining = mailbox.approximate_count()

            # All messages accounted for
            assert len(sent) == total_messages
            assert len(set(acked)) + remaining == total_messages, (
                f"Message loss: sent={total_messages}, acked={len(set(acked))}, remaining={remaining}"
            )

            # No duplicate acks (set size equals list size)
            assert len(acked) == len(set(acked)), (
                f"Duplicate acks: {len(acked)} total, {len(set(acked))} unique"
            )

        finally:
            stop_consumers.set()
            mailbox.close()
            mailbox.purge()

    def test_reaper_under_load(self, redis_client: Redis[bytes]) -> None:
        """
        Reaper correctly handles messages expiring under load.

        Simulates consumers that are slower than visibility timeout.
        """
        from weakincentives.contrib.mailbox import RedisMailbox

        mailbox: RedisMailbox[Any] = RedisMailbox(
            name=f"reaper-stress-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,
        )

        num_messages = 50
        visibility_timeout = 0  # Immediate timeout

        try:
            # Send messages
            for i in range(num_messages):
                mailbox.send(f"msg-{i}")

            # Receive but don't ack (let them expire)
            received_handles = []
            for _ in range(num_messages):
                msgs = mailbox.receive(visibility_timeout=visibility_timeout)
                if msgs:
                    received_handles.append((msgs[0].id, msgs[0].receipt_handle))

            mailbox._reap_expired()

            # All messages should be back in pending
            count = mailbox.approximate_count()
            assert count == num_messages, (
                f"Expected {num_messages} messages, got {count}"
            )

            # Old handles should be rejected
            for msg_id, old_handle in received_handles[:5]:
                suffix = old_handle.split(":", 1)[1]
                with pytest.raises(ReceiptHandleExpiredError):
                    mailbox._acknowledge(msg_id, suffix)

        finally:
            mailbox.close()
            mailbox.purge()

    def test_concurrent_ack_same_message(self, redis_client: Redis[bytes]) -> None:
        """
        Only one consumer can successfully ack a message.

        When multiple consumers try to ack the same message (due to
        redelivery), only the one with the valid handle should succeed.
        """
        from weakincentives.contrib.mailbox import RedisMailbox

        mailbox: RedisMailbox[Any] = RedisMailbox(
            name=f"ack-race-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.05,
        )

        try:
            mailbox.send("test")

            # Consumer 1 receives
            msgs1 = mailbox.receive(visibility_timeout=0)
            assert len(msgs1) == 1
            handle1 = msgs1[0].receipt_handle
            msg_id = msgs1[0].id

            mailbox._reap_expired()

            # Consumer 2 receives the redelivered message
            msgs2 = mailbox.receive(visibility_timeout=30)
            assert len(msgs2) == 1
            handle2 = msgs2[0].receipt_handle

            # Verify handles are different
            assert handle1 != handle2

            # Race: both try to ack simultaneously
            results: dict[str, str | None] = {"c1": None, "c2": None}

            def ack1() -> None:
                try:
                    suffix = handle1.split(":", 1)[1]
                    mailbox._acknowledge(msg_id, suffix)
                    results["c1"] = "success"
                except ReceiptHandleExpiredError:
                    results["c1"] = "expired"

            def ack2() -> None:
                try:
                    suffix = handle2.split(":", 1)[1]
                    mailbox._acknowledge(msg_id, suffix)
                    results["c2"] = "success"
                except ReceiptHandleExpiredError:
                    results["c2"] = "expired"

            t1 = threading.Thread(target=ack1)
            t2 = threading.Thread(target=ack2)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # Exactly one should succeed (the one with valid handle)
            success_count = sum(1 for r in results.values() if r == "success")
            assert success_count == 1, f"Expected exactly 1 success, got {results}"

            # Consumer 1 should fail (stale handle)
            assert results["c1"] == "expired", f"Old handle should fail: {results}"
            # Consumer 2 should succeed (current handle)
            assert results["c2"] == "success", f"New handle should succeed: {results}"

        finally:
            mailbox.close()
            mailbox.purge()
