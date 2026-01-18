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

"""Deterministic thread tests for concurrent library components.

These tests verify thread safety properties of production components.
The checkpoints in production code serve as documentation of yield points
and can be used for controlled testing of lock-free code paths.

For code with OS-level locks, we use regular concurrent testing since
the deterministic scheduler cannot control OS lock acquisition order.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from weakincentives.runtime.lease_extender import LeaseExtender, LeaseExtenderConfig
from weakincentives.runtime.lifecycle import ShutdownCoordinator
from weakincentives.runtime.watchdog import Heartbeat

# =============================================================================
# Mock Message for LeaseExtender tests
# =============================================================================


@dataclass
class MockMessage:
    """Mock message that tracks extend_visibility calls."""

    id: str = "msg-1"
    body: str = "test"
    receipt_handle: str = "handle-1"
    delivery_count: int = 1
    enqueued_at: datetime | None = None
    reply_to: Any = None
    extend_count: int = 0
    _lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        if self.enqueued_at is None:
            self.enqueued_at = datetime.now(UTC)
        if self._lock is None:
            self._lock = threading.Lock()

    def extend_visibility(self, timeout: int) -> None:
        """Track extension calls (thread-safe)."""
        assert self._lock is not None
        with self._lock:
            self.extend_count += 1

    def acknowledge(self) -> None:
        """No-op for mock."""

    def nack(self, visibility_timeout: int = 0) -> None:
        """No-op for mock."""


# =============================================================================
# Heartbeat Tests - Real Component
# =============================================================================


class TestHeartbeatConcurrency:
    """Test thread safety of real Heartbeat component."""

    def test_concurrent_beats_no_crash(self) -> None:
        """Multiple concurrent beat() calls don't crash."""
        heartbeat = Heartbeat()
        call_count = 0
        count_lock = threading.Lock()

        def callback() -> None:
            nonlocal call_count
            with count_lock:
                call_count += 1

        heartbeat.add_callback(callback)

        def beat_many_times() -> None:
            for _ in range(100):
                heartbeat.beat()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(beat_many_times) for _ in range(4)]
            for f in futures:
                f.result()

        # All beats should have invoked the callback
        assert call_count == 400

    def test_concurrent_add_and_beat_no_crash(self) -> None:
        """Concurrent add_callback and beat don't crash."""
        heartbeat = Heartbeat()
        invocations: list[int] = []
        inv_lock = threading.Lock()

        def beater() -> None:
            for _ in range(50):
                heartbeat.beat()

        def adder() -> None:
            for i in range(50):

                def cb(n: int = i) -> None:
                    with inv_lock:
                        invocations.append(n)

                heartbeat.add_callback(cb)

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(beater)
            f2 = executor.submit(adder)
            f1.result()
            f2.result()

        # No assertion on count - just verifying no crash/deadlock

    def test_snapshot_pattern_prevents_deadlock_on_self_modify(self) -> None:
        """Callback that modifies callback list doesn't deadlock.

        This verifies the snapshot pattern: beat() copies callbacks under lock,
        then invokes outside lock, so callbacks can safely modify the list.
        """
        heartbeat = Heartbeat()
        invocations: list[str] = []
        removed = False

        def self_removing_callback() -> None:
            nonlocal removed
            invocations.append("called")
            if not removed:
                removed = True
                # This would deadlock if beat() held lock during callback
                heartbeat.remove_callback(self_removing_callback)

        heartbeat.add_callback(self_removing_callback)

        # First beat: runs callback, callback removes itself
        heartbeat.beat()
        assert invocations == ["called"]

        # Second beat: callback was removed
        heartbeat.beat()
        assert invocations == ["called"]  # No additional call


# =============================================================================
# LeaseExtender Tests - Real Component
# =============================================================================


class TestLeaseExtenderConcurrency:
    """Test thread safety of real LeaseExtender component."""

    def test_attach_beat_detach_sequence(self) -> None:
        """Basic attach-beat-detach sequence works correctly."""
        msg = MockMessage()
        heartbeat = Heartbeat()
        extender = LeaseExtender(config=LeaseExtenderConfig(interval=0.0))

        # Attach
        extender._attach(msg, heartbeat)

        # Beat triggers extension
        heartbeat.beat()
        assert msg.extend_count == 1

        # Detach
        extender._detach()

        # Beat after detach should NOT extend
        heartbeat.beat()
        assert msg.extend_count == 1

    def test_concurrent_beats_with_attached_extender(self) -> None:
        """Concurrent beats with attached extender don't crash."""
        msg = MockMessage()
        heartbeat = Heartbeat()
        # interval=0 means every beat can extend
        extender = LeaseExtender(config=LeaseExtenderConfig(interval=0.0))

        extender._attach(msg, heartbeat)

        def beat_many() -> None:
            for _ in range(100):
                heartbeat.beat()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(beat_many) for _ in range(4)]
            for f in futures:
                f.result()

        extender._detach()

        # All beats should extend (no rate limiting with interval=0)
        assert msg.extend_count == 400

    def test_rate_limiting_prevents_rapid_extensions(self) -> None:
        """Rate limiting prevents multiple extensions within interval.

        With interval=0.0, every beat extends. With high interval,
        only the first beat extends (since _last_extension starts at 0.0).
        """
        # Test 1: no rate limiting (interval=0)
        msg_no_limit = MockMessage(id="no-limit")
        heartbeat_no_limit = Heartbeat()
        extender_no_limit = LeaseExtender(config=LeaseExtenderConfig(interval=0.0))
        extender_no_limit._attach(msg_no_limit, heartbeat_no_limit)
        for _ in range(5):
            heartbeat_no_limit.beat()
        extender_no_limit._detach()

        # All 5 beats should extend
        assert msg_no_limit.extend_count == 5

        # Test 2: with rate limiting (high interval) - use fresh heartbeat
        msg_limited = MockMessage(id="limited")
        heartbeat_limited = Heartbeat()
        extender_limited = LeaseExtender(config=LeaseExtenderConfig(interval=10000.0))
        extender_limited._attach(msg_limited, heartbeat_limited)
        for _ in range(5):
            heartbeat_limited.beat()
        extender_limited._detach()

        # Only first beat should extend (due to rate limiting)
        assert msg_limited.extend_count == 1


# =============================================================================
# ShutdownCoordinator Tests - Real Component
# =============================================================================


class TestShutdownCoordinatorConcurrency:
    """Test thread safety of real ShutdownCoordinator component."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        ShutdownCoordinator.reset()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        ShutdownCoordinator.reset()

    def test_concurrent_register_no_crash(self) -> None:
        """Concurrent register calls don't crash."""
        coordinator = ShutdownCoordinator()
        call_count = 0
        count_lock = threading.Lock()

        def register_many() -> None:
            for _ in range(50):

                def cb() -> None:
                    nonlocal call_count
                    with count_lock:
                        call_count += 1

                coordinator.register(cb)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(register_many) for _ in range(4)]
            for f in futures:
                f.result()

        # Trigger and verify all callbacks run
        coordinator.trigger()
        assert call_count == 200

    def test_concurrent_trigger_no_crash(self) -> None:
        """Concurrent trigger calls don't crash."""
        coordinator = ShutdownCoordinator()
        call_count = 0
        count_lock = threading.Lock()

        def callback() -> None:
            nonlocal call_count
            with count_lock:
                call_count += 1

        coordinator.register(callback)

        def trigger() -> None:
            coordinator.trigger()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(trigger) for _ in range(4)]
            for f in futures:
                f.result()

        # Callback invoked once per trigger that runs its iteration
        # (exact count depends on timing but shouldn't crash)
        assert call_count >= 1

    def test_register_after_trigger_invokes_immediately(self) -> None:
        """Callback registered after trigger is invoked immediately."""
        coordinator = ShutdownCoordinator()
        invocations: list[str] = []

        # Trigger first
        coordinator.trigger()
        assert coordinator.triggered

        # Register after trigger
        coordinator.register(lambda: invocations.append("late"))

        # Should have been invoked immediately
        assert invocations == ["late"]

    def test_snapshot_pattern_prevents_deadlock(self) -> None:
        """Callback that registers new callback doesn't deadlock."""
        coordinator = ShutdownCoordinator()
        invocations: list[str] = []

        def callback_that_registers() -> None:
            invocations.append("first")
            # This would deadlock if trigger() held lock during callback
            coordinator.register(lambda: invocations.append("second"))

        coordinator.register(callback_that_registers)
        coordinator.trigger()

        # "first" was invoked, "second" was invoked immediately on register
        # (because triggered flag was already set)
        assert "first" in invocations
        assert "second" in invocations


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_heartbeat_lease_extender_integration(self) -> None:
        """Heartbeat + LeaseExtender work together correctly."""
        msg = MockMessage()
        heartbeat = Heartbeat()
        extender = LeaseExtender(config=LeaseExtenderConfig(interval=0.0))

        # Simulate message processing with heartbeats
        extender._attach(msg, heartbeat)

        # Worker beats periodically
        heartbeat.beat()
        assert msg.extend_count == 1

        heartbeat.beat()
        assert msg.extend_count == 2

        # Processing complete, detach
        extender._detach()

        # No more extensions after detach
        heartbeat.beat()
        assert msg.extend_count == 2

    def test_multiple_extenders_same_heartbeat(self) -> None:
        """Multiple extenders can share a heartbeat (one at a time)."""
        msg1 = MockMessage(id="msg-1")
        msg2 = MockMessage(id="msg-2")
        heartbeat = Heartbeat()

        extender1 = LeaseExtender(config=LeaseExtenderConfig(interval=0.0))
        extender2 = LeaseExtender(config=LeaseExtenderConfig(interval=0.0))

        # Process first message
        extender1._attach(msg1, heartbeat)
        heartbeat.beat()
        assert msg1.extend_count == 1
        extender1._detach()

        # Process second message
        extender2._attach(msg2, heartbeat)
        heartbeat.beat()
        assert msg2.extend_count == 1
        extender2._detach()

        # msg1 wasn't affected by second attach
        assert msg1.extend_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
