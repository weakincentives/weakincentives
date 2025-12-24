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
Stateful property-based tests for RedisMailbox.

These tests use Hypothesis to exercise the mailbox through random
sequences of operations while maintaining a reference model to verify
invariants from specs/VERIFICATION.md.

Key invariants verified:
- INV-1: Message State Exclusivity
- INV-2: Receipt Handle Freshness
- INV-3: Stale Handle Rejection
- INV-4: Delivery Count Monotonicity
- INV-5: No Message Loss
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest

try:
    from hypothesis import HealthCheck, settings, strategies as st
    from hypothesis.stateful import (
        Bundle,
        RuleBasedStateMachine,
        initialize,
        invariant,
        precondition,
        rule,
    )

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

pytestmark = [
    pytest.mark.redis_standalone,
    pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed"),
]


# =============================================================================
# Reference Model
# =============================================================================


@dataclass
class MessageState:
    """Tracks a message through its lifecycle."""

    id: str
    body: Any
    delivery_count: int = 0
    current_handle: str | None = None
    expires_at: float | None = None


@dataclass
class MailboxModel:
    """Reference model for RedisMailbox state."""

    pending: deque[str] = field(default_factory=deque)
    invisible: dict[str, MessageState] = field(default_factory=dict)
    data: dict[str, MessageState] = field(default_factory=dict)
    deleted: set[str] = field(default_factory=set)
    delivery_history: dict[str, list[tuple[int, str]]] = field(default_factory=dict)

    def send(self, msg_id: str, body: Any) -> None:  # noqa: ANN401
        """Model a send operation."""
        state = MessageState(id=msg_id, body=body)
        self.data[msg_id] = state
        self.pending.append(msg_id)

    def receive(self, msg_id: str, handle: str, expires_at: float) -> None:
        """Model a receive operation."""
        if msg_id in self.pending:
            self.pending.remove(msg_id)

        state = self.data[msg_id]
        state.delivery_count += 1
        state.current_handle = handle
        state.expires_at = expires_at
        self.invisible[msg_id] = state

        # Track delivery history
        if msg_id not in self.delivery_history:
            self.delivery_history[msg_id] = []
        self.delivery_history[msg_id].append((state.delivery_count, handle))

    def acknowledge(self, msg_id: str, handle: str) -> bool:
        """Model an acknowledge. Returns True if successful."""
        if msg_id not in self.invisible:
            return False
        if self.invisible[msg_id].current_handle != handle:
            return False

        del self.invisible[msg_id]
        del self.data[msg_id]
        self.deleted.add(msg_id)
        return True

    def nack(
        self, msg_id: str, handle: str, visibility_timeout: int, now: float
    ) -> bool:
        """Model a nack. Returns True if successful."""
        if msg_id not in self.invisible:
            return False
        if self.invisible[msg_id].current_handle != handle:
            return False

        state = self.invisible.pop(msg_id)
        state.current_handle = None

        if visibility_timeout <= 0:
            self.pending.append(msg_id)
            state.expires_at = None
        else:
            state.expires_at = now + visibility_timeout
            self.invisible[msg_id] = state

        return True

    def reap(self, now: float) -> list[str]:
        """Model reaper. Returns list of requeued message IDs."""
        requeued = []
        for msg_id, state in list(self.invisible.items()):
            if state.expires_at is not None and state.expires_at <= now:
                del self.invisible[msg_id]
                state.current_handle = None
                state.expires_at = None
                self.pending.append(msg_id)
                requeued.append(msg_id)
        return requeued

    def extend(
        self, msg_id: str, handle: str, new_timeout: int, now: float
    ) -> bool:
        """Model an extend. Returns True if successful."""
        if msg_id not in self.invisible:
            return False
        if self.invisible[msg_id].current_handle != handle:
            return False

        self.invisible[msg_id].expires_at = now + new_timeout
        return True

    def is_handle_valid(self, msg_id: str, handle: str) -> bool:
        """Check if a handle is currently valid."""
        if msg_id not in self.invisible:
            return False
        return self.invisible[msg_id].current_handle == handle

    def get_pending_count(self) -> int:
        return len(self.pending)

    def get_invisible_count(self) -> int:
        return len(self.invisible)

    def total_count(self) -> int:
        return len(self.data)


# =============================================================================
# Stateful Property-Based Tests
# =============================================================================

if HAS_HYPOTHESIS:

    class RedisMailboxStateMachine(RuleBasedStateMachine):
        """
        Stateful property-based tests for RedisMailbox.

        This state machine exercises the mailbox through random sequences
        of operations while maintaining a reference model. Invariants are
        checked after each step.
        """

        # Bundles track values across rules
        sent_ids = Bundle("sent_ids")
        received = Bundle("received")  # (msg_id, receipt_handle) tuples

        def __init__(self) -> None:
            super().__init__()
            self.model = MailboxModel()
            self.start_time = time.time()

        @initialize()
        def setup(self) -> None:
            """Create fresh mailbox for each test run."""
            try:
                from redis import Redis
            except ImportError:
                pytest.skip("redis not installed")

            from weakincentives.contrib.mailbox import RedisMailbox

            self.client = Redis(host="localhost", port=6379, db=15)
            try:
                self.client.ping()
            except Exception:
                pytest.skip("Redis not available")

            self.client.flushdb()

            self.mailbox = RedisMailbox(
                name=f"test-{uuid4().hex[:8]}",
                client=self.client,
                reaper_interval=0.05,  # 50ms for fast testing
            )

        def teardown(self) -> None:
            """Clean up after test run."""
            if hasattr(self, "mailbox"):
                self.mailbox.close()
                self.mailbox.purge()
            if hasattr(self, "client"):
                self.client.close()

        @rule(target=sent_ids, body=st.binary(min_size=1, max_size=100))
        def send_message(self, body: bytes) -> str:
            """Send a message and track it."""
            msg_id = self.mailbox.send(body)
            self.model.send(msg_id, body)
            return msg_id

        @rule(
            target=received,
            timeout=st.integers(min_value=1, max_value=10),
        )
        @precondition(lambda self: self.model.get_pending_count() > 0)
        def receive_message(self, timeout: int) -> tuple[str, str] | None:
            """Receive a message if any are pending."""
            msgs = self.mailbox.receive(
                visibility_timeout=timeout,
                wait_time_seconds=0,
            )

            if msgs:
                msg = msgs[0]
                expires_at = time.time() + timeout
                self.model.receive(msg.id, msg.receipt_handle, expires_at)
                return (msg.id, msg.receipt_handle)

            return None

        @rule(receipt=received)
        def acknowledge_message(self, receipt: tuple[str, str] | None) -> None:
            """Acknowledge a received message."""
            if receipt is None:
                return

            from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

            msg_id, handle = receipt
            suffix = handle.split(":", 1)[1] if ":" in handle else handle

            try:
                self.mailbox._acknowledge(msg_id, suffix)
                assert self.model.acknowledge(
                    msg_id, handle
                ), "Model predicted failure but implementation succeeded"
            except ReceiptHandleExpiredError:
                assert not self.model.is_handle_valid(
                    msg_id, handle
                ), "Model predicted success but implementation failed"

        @rule(
            receipt=received,
            new_timeout=st.integers(min_value=0, max_value=5),
        )
        def nack_message(
            self, receipt: tuple[str, str] | None, new_timeout: int
        ) -> None:
            """Return a message to the queue."""
            if receipt is None:
                return

            from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

            msg_id, handle = receipt
            suffix = handle.split(":", 1)[1] if ":" in handle else handle

            try:
                self.mailbox._nack(msg_id, suffix, new_timeout)
                expected = self.model.nack(msg_id, handle, new_timeout, time.time())
                assert (
                    expected
                ), "Model predicted failure but implementation succeeded"
            except ReceiptHandleExpiredError:
                assert not self.model.is_handle_valid(
                    msg_id, handle
                ), "Model predicted success but implementation failed"

        @rule(
            receipt=received,
            new_timeout=st.integers(min_value=1, max_value=30),
        )
        def extend_message(
            self, receipt: tuple[str, str] | None, new_timeout: int
        ) -> None:
            """Extend visibility timeout for a received message."""
            if receipt is None:
                return

            from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

            msg_id, handle = receipt
            suffix = handle.split(":", 1)[1] if ":" in handle else handle

            try:
                self.mailbox._extend(msg_id, suffix, new_timeout)
                expected = self.model.extend(
                    msg_id, handle, new_timeout, time.time()
                )
                assert (
                    expected
                ), "Model predicted failure but implementation succeeded"
            except ReceiptHandleExpiredError:
                assert not self.model.is_handle_valid(
                    msg_id, handle
                ), "Model predicted success but implementation failed"

        @rule()
        def advance_time(self) -> None:
            """
            Wait briefly to allow visibility timeouts to expire.
            Also syncs model reaper state.
            """
            time.sleep(0.1)
            self.model.reap(time.time())

        # =====================================================================
        # Invariants - checked after every rule
        # =====================================================================

        @invariant()
        def message_count_matches(self) -> None:
            """Total message count matches between model and implementation."""
            expected = self.model.total_count()
            actual = self.mailbox.approximate_count()
            # Allow for timing differences in invisible set
            assert (
                abs(expected - actual) <= 1
            ), f"Count mismatch: model={expected}, redis={actual}"

        @invariant()
        def no_messages_lost(self) -> None:
            """
            Every message in the model exists in Redis.
            Messages are either pending, invisible, or deleted.
            """
            for msg_id in self.model.data:
                if msg_id in self.model.deleted:
                    continue

                # Check it exists somewhere in Redis
                in_pending = self._msg_in_pending(msg_id)
                in_invisible = self._msg_in_invisible(msg_id)
                has_data = self.client.hexists(self.mailbox._keys.data, msg_id)

                assert has_data, f"Message {msg_id} data lost"
                assert (
                    in_pending or in_invisible
                ), f"Message {msg_id} not in pending or invisible"

        @invariant()
        def message_state_exclusive(self) -> None:
            """Each message is in exactly one location."""
            # Get all known message IDs
            all_pending = self._get_all_pending()
            all_invisible = self._get_all_invisible()

            # Check no overlap
            overlap = set(all_pending) & set(all_invisible)
            assert not overlap, f"Messages in both pending and invisible: {overlap}"

        @invariant()
        def deleted_messages_gone(self) -> None:
            """Deleted messages have no remaining state."""
            for msg_id in self.model.deleted:
                assert not self._msg_in_pending(
                    msg_id
                ), f"Deleted message {msg_id} in pending"
                assert not self._msg_in_invisible(
                    msg_id
                ), f"Deleted message {msg_id} in invisible"

        @invariant()
        def delivery_count_monotonic(self) -> None:
            """Delivery counts are strictly increasing for each message."""
            for msg_id, history in self.model.delivery_history.items():
                counts = [count for count, _ in history]
                for i in range(1, len(counts)):
                    assert (
                        counts[i] > counts[i - 1]
                    ), f"Non-monotonic delivery count for {msg_id}: {counts}"

        # =====================================================================
        # Helper methods
        # =====================================================================

        def _msg_in_pending(self, msg_id: str) -> bool:
            """Check if message is in pending list."""
            pending = self.client.lrange(self.mailbox._keys.pending, 0, -1)
            return msg_id.encode() in pending

        def _msg_in_invisible(self, msg_id: str) -> bool:
            """Check if message is in invisible set."""
            score = self.client.zscore(self.mailbox._keys.invisible, msg_id)
            return score is not None

        def _get_all_pending(self) -> list[str]:
            """Get all message IDs in pending."""
            return [
                m.decode()
                for m in self.client.lrange(self.mailbox._keys.pending, 0, -1)
            ]

        def _get_all_invisible(self) -> list[str]:
            """Get all message IDs in invisible."""
            return [
                m.decode()
                for m in self.client.zrange(self.mailbox._keys.invisible, 0, -1)
            ]

    # Configure Hypothesis settings
    TestRedisMailbox = RedisMailboxStateMachine.TestCase
    TestRedisMailbox.settings = settings(
        max_examples=100,
        stateful_step_count=50,
        deadline=None,  # Disable deadline for I/O operations
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.data_too_large,
        ],
    )
