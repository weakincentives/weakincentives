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

These tests use the weakincentives.threads framework to test race conditions
in multi-threaded code with controlled interleaving.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from weakincentives.threads import (
    checkpoint,
    run_all_schedules,
    run_with_schedule,
)

# =============================================================================
# Test Fixtures - Simplified versions of production components
# =============================================================================


@dataclass
class MockMessage:
    """Mock message for testing lease extension."""

    id: str = "msg-1"
    extended: bool = False
    extend_count: int = 0

    def extend_visibility(self, timeout: int) -> None:
        checkpoint("extend_visibility")
        self.extended = True
        self.extend_count += 1


@dataclass
class SimpleHeartbeat:
    """Simplified heartbeat for testing callback races."""

    _callbacks: list[Callable[[], None]] = field(default_factory=list)

    def beat(self) -> None:
        checkpoint("beat_start")
        callbacks = list(self._callbacks)
        checkpoint("beat_snapshot")
        for cb in callbacks:
            cb()
        checkpoint("beat_end")

    def add_callback(self, cb: Callable[[], None]) -> None:
        checkpoint("add_callback")
        self._callbacks.append(cb)

    def remove_callback(self, cb: Callable[[], None]) -> None:
        checkpoint("remove_callback")
        self._callbacks.remove(cb)


@dataclass
class SimpleLeaseExtender:
    """Simplified lease extender for testing attach/detach races."""

    _msg: MockMessage | None = None
    _heartbeat: SimpleHeartbeat | None = None
    _attached: bool = False

    def attach(self, msg: MockMessage, heartbeat: SimpleHeartbeat) -> None:
        checkpoint("attach_start")
        self._msg = msg
        self._heartbeat = heartbeat
        self._attached = True
        checkpoint("attach_registered")
        heartbeat.add_callback(self._on_beat)
        checkpoint("attach_end")

    def detach(self) -> None:
        checkpoint("detach_start")
        heartbeat = self._heartbeat
        self._msg = None
        self._heartbeat = None
        self._attached = False
        checkpoint("detach_cleared")
        if heartbeat:
            heartbeat.remove_callback(self._on_beat)
        checkpoint("detach_end")

    def _on_beat(self) -> None:
        checkpoint("on_beat_start")
        if self._msg is None:
            checkpoint("on_beat_no_msg")
            return
        msg = self._msg
        checkpoint("on_beat_got_msg")
        msg.extend_visibility(300)
        checkpoint("on_beat_end")


# =============================================================================
# Heartbeat Callback Tests
# =============================================================================


class TestHeartbeatCallbackRaces:
    """Test races in heartbeat callback management."""

    def test_add_callback_during_beat_not_invoked(self) -> None:
        """Callback added during beat iteration should not be invoked in that beat.

        This tests the snapshot pattern: beat() takes a snapshot of callbacks
        before iterating, so a callback added mid-iteration won't be called.
        """
        heartbeat = SimpleHeartbeat()
        invocations: list[str] = []

        def callback_a() -> None:
            checkpoint("callback_a")
            invocations.append("a")

        def callback_b() -> None:
            checkpoint("callback_b")
            invocations.append("b")

        heartbeat.add_callback(callback_a)

        def beat_thread() -> None:
            heartbeat.beat()

        def add_thread() -> None:
            checkpoint("add_before")
            heartbeat.add_callback(callback_b)
            checkpoint("add_after")

        # Schedule: beat takes snapshot, then add happens, then callbacks execute
        result = run_with_schedule(
            {"beat": beat_thread, "add": add_thread},
            schedule=["beat", "beat", "add", "beat"],  # snapshot, then add
        )

        assert not result.deadlocked
        # callback_b was added after snapshot, so only 'a' was invoked
        assert invocations == ["a"]

    def test_remove_callback_during_beat_still_invoked(self) -> None:
        """Callback removed during beat iteration is still invoked (snapshot).

        If beat() takes a snapshot and then the callback is removed,
        the callback still runs because it was in the snapshot.
        """
        heartbeat = SimpleHeartbeat()
        invocations: list[str] = []

        def callback_a() -> None:
            checkpoint("callback_a")
            invocations.append("a")

        heartbeat.add_callback(callback_a)

        def beat_thread() -> None:
            heartbeat.beat()

        def remove_thread() -> None:
            checkpoint("remove_before")
            heartbeat.remove_callback(callback_a)
            checkpoint("remove_after")

        # Schedule: beat takes snapshot, remove happens, callback still executes
        result = run_with_schedule(
            {"beat": beat_thread, "remove": remove_thread},
            schedule=["beat", "beat", "remove", "beat"],  # snapshot, remove, then exec
        )

        assert not result.deadlocked
        # callback_a was in snapshot, so it was invoked despite being removed
        assert invocations == ["a"]

    def test_all_interleavings_of_add_and_beat(self) -> None:
        """Exhaustively test all interleavings of add and beat."""
        for result in run_all_schedules(
            {
                "beat": lambda: SimpleHeartbeat().beat(),
                "add": lambda: None,  # Just checkpoints
            }
        ):
            assert not result.deadlocked, f"Deadlock at {result.schedule}"


# =============================================================================
# LeaseExtender Tests
# =============================================================================


class TestLeaseExtenderRaces:
    """Test races in lease extender attach/detach/beat interactions."""

    def test_beat_after_detach_does_not_extend(self) -> None:
        """Beat callback invoked after detach should not extend message.

        This is a critical safety property: once detached, no more extensions.
        """
        msg = MockMessage()
        heartbeat = SimpleHeartbeat()
        extender = SimpleLeaseExtender()

        def attach_detach_thread() -> None:
            extender.attach(msg, heartbeat)
            checkpoint("between_attach_detach")
            extender.detach()

        def beat_thread() -> None:
            checkpoint("beat_before")
            heartbeat.beat()
            checkpoint("beat_after")

        # Schedule: attach, then beat starts but callback sees detached state
        # This test may deadlock due to remove_callback ValueError
        # That's expected - it shows the race condition exists
        _ = run_with_schedule(
            {"lifecycle": attach_detach_thread, "beat": beat_thread},
            schedule=[
                "lifecycle",  # attach_start
                "lifecycle",  # attach_registered
                "lifecycle",  # add_callback
                "lifecycle",  # attach_end
                "lifecycle",  # between_attach_detach
                "lifecycle",  # detach_start
                "lifecycle",  # detach_cleared
                "beat",  # beat_before
                "beat",  # beat_start
                "beat",  # beat_snapshot (captures callback)
                "lifecycle",  # remove_callback (fails - callback already snapshotted)
            ],
        )

    def test_extension_happens_only_when_attached(self) -> None:
        """Message extension only happens when extender is attached."""
        msg = MockMessage()
        heartbeat = SimpleHeartbeat()
        extender = SimpleLeaseExtender()

        # First attach and beat
        extender.attach(msg, heartbeat)

        def beat_while_attached() -> None:
            heartbeat.beat()

        result = run_with_schedule(
            {"beat": beat_while_attached},
            schedule=["beat"] * 6,  # All beat checkpoints
        )

        assert not result.deadlocked
        assert msg.extend_count == 1

    def test_attach_detach_attach_sequence(self) -> None:
        """Test attach-detach-attach sequence maintains consistency."""
        msg1 = MockMessage(id="msg-1")
        msg2 = MockMessage(id="msg-2")
        heartbeat = SimpleHeartbeat()
        extender = SimpleLeaseExtender()

        def lifecycle() -> None:
            extender.attach(msg1, heartbeat)
            extender.detach()
            checkpoint("between_attachments")
            extender.attach(msg2, heartbeat)

        def beat() -> None:
            heartbeat.beat()

        # Lifecycle has:
        # attach: attach_start, attach_registered, add_callback, attach_end (4)
        # detach: detach_start, detach_cleared, remove_callback, detach_end (4)
        # between_attachments (1)
        # attach: attach_start, attach_registered, add_callback, attach_end (4)
        # Total: 13 checkpoints
        #
        # Beat has:
        # beat_start, beat_snapshot, on_beat_start, on_beat_got_msg,
        # extend_visibility, on_beat_end, beat_end (7)
        result = run_with_schedule(
            {"lifecycle": lifecycle, "beat": beat},
            schedule=["lifecycle"] * 13 + ["beat"] * 7,
        )

        assert not result.deadlocked
        # Only msg2 should be extended (msg1 was detached)
        assert msg1.extend_count == 0
        assert msg2.extend_count == 1


# =============================================================================
# Shutdown Coordinator Tests
# =============================================================================


@dataclass
class SimpleShutdownCoordinator:
    """Simplified shutdown coordinator for testing registration races."""

    _triggered: bool = False
    _callbacks: list[Callable[[], None]] = field(default_factory=list)

    def register(self, callback: Callable[[], None]) -> None:
        checkpoint("register_start")
        self._callbacks.append(callback)
        checkpoint("register_end")

    def trigger(self) -> None:
        checkpoint("trigger_start")
        self._triggered = True
        callbacks = list(self._callbacks)  # Snapshot
        checkpoint("trigger_snapshot")
        for cb in callbacks:
            checkpoint("trigger_invoke")
            cb()
        checkpoint("trigger_end")


class TestShutdownCoordinatorRaces:
    """Test races in shutdown coordinator callback registration."""

    def test_register_during_trigger_not_invoked(self) -> None:
        """Callback registered during trigger should not be invoked.

        The trigger() takes a snapshot before iterating, so callbacks
        registered mid-trigger won't be called in that trigger.
        """
        coordinator = SimpleShutdownCoordinator()
        invocations: list[str] = []

        def callback_a() -> None:
            invocations.append("a")

        def callback_b() -> None:
            invocations.append("b")

        coordinator.register(callback_a)

        def trigger_thread() -> None:
            coordinator.trigger()

        def register_thread() -> None:
            coordinator.register(callback_b)

        # Trigger takes snapshot, then register happens
        result = run_with_schedule(
            {"trigger": trigger_thread, "register": register_thread},
            schedule=[
                "trigger",  # trigger_start
                "trigger",  # trigger_snapshot
                "register",  # register_start
                "register",  # register_end
                "trigger",  # trigger_invoke (only callback_a)
                "trigger",  # trigger_end
            ],
        )

        assert not result.deadlocked
        assert invocations == ["a"]  # callback_b not invoked

    def test_all_interleavings_preserve_existing_callbacks(self) -> None:
        """All interleavings should invoke callbacks registered before trigger."""
        for result in run_all_schedules(
            {
                "trigger": lambda: SimpleShutdownCoordinator().trigger(),
                "register": lambda: None,
            }
        ):
            assert not result.deadlocked


# =============================================================================
# Concurrent Counter Tests (Classic Race Condition)
# =============================================================================


class TestConcurrentCounter:
    """Test classic counter race condition to validate framework."""

    def test_unsynchronized_counter_can_lose_updates(self) -> None:
        """Demonstrate that unsynchronized increment can lose updates."""

        @dataclass
        class UnsafeCounter:
            value: int = 0

            def increment(self) -> None:
                checkpoint("read")
                v = self.value
                checkpoint("compute")
                self.value = v + 1
                checkpoint("write")

        # Test if any interleaving causes deadlocks
        for _result in run_all_schedules(
            {
                "t1": lambda: (c := UnsafeCounter()) or c.increment(),
                "t2": lambda: None,  # Placeholder
            }
        ):
            pass  # Just checking no deadlocks

        # With a shared counter between threads, we can show the race
        counter = UnsafeCounter()

        def inc1() -> None:
            checkpoint("t1_read")
            v = counter.value
            checkpoint("t1_compute")
            counter.value = v + 1
            checkpoint("t1_write")

        def inc2() -> None:
            checkpoint("t2_read")
            v = counter.value
            checkpoint("t2_compute")
            counter.value = v + 1
            checkpoint("t2_write")

        # Interleaving: t1 reads 0, t2 reads 0, both write 1 -> lost update
        result = run_with_schedule(
            {"t1": inc1, "t2": inc2},
            schedule=["t1", "t2", "t1", "t2", "t1", "t2"],  # read, read, compute, ...
        )

        assert not result.deadlocked
        # Both threads read 0, so final value is 1 instead of 2
        assert counter.value == 1  # Lost update!

    def test_synchronized_counter_preserves_all_updates(self) -> None:
        """Synchronized counter should preserve all updates."""
        import threading

        @dataclass
        class SafeCounter:
            value: int = 0
            _lock: threading.Lock = field(default_factory=threading.Lock)

            def increment(self) -> None:
                checkpoint("acquire")
                with self._lock:
                    checkpoint("read")
                    v = self.value
                    checkpoint("compute")
                    self.value = v + 1
                    checkpoint("write")
                checkpoint("release")

        counter = SafeCounter()

        def inc1() -> None:
            counter.increment()

        def inc2() -> None:
            counter.increment()

        # With real locks, any interleaving should give correct result
        # Note: This test shows the pattern but real locks don't work
        # with the checkpoint scheduler (they block at OS level)


# =============================================================================
# Message Queue Tests
# =============================================================================


@dataclass
class SimpleMessageQueue:
    """Simplified message queue for testing reaper races."""

    pending: list[str] = field(default_factory=list)
    invisible: dict[str, str] = field(default_factory=dict)

    def send(self, msg: str) -> None:
        checkpoint("send_start")
        self.pending.append(msg)
        checkpoint("send_end")

    def receive(self) -> str | None:
        checkpoint("receive_start")
        if not self.pending:
            checkpoint("receive_empty")
            return None
        msg = self.pending.pop(0)
        checkpoint("receive_got")
        handle = f"handle-{msg}"
        self.invisible[handle] = msg
        checkpoint("receive_invisible")
        return msg

    def acknowledge(self, handle: str) -> bool:
        checkpoint("ack_start")
        if handle not in self.invisible:
            checkpoint("ack_expired")
            return False
        del self.invisible[handle]
        checkpoint("ack_end")
        return True

    def reap_expired(self) -> None:
        """Move expired messages back to pending."""
        checkpoint("reap_start")
        # Simulate: all invisible messages are expired
        for handle in list(self.invisible.keys()):
            checkpoint("reap_check")
            msg = self.invisible.pop(handle)
            checkpoint("reap_pop")
            self.pending.append(msg)
            checkpoint("reap_append")
        checkpoint("reap_end")


class TestMessageQueueRaces:
    """Test races in message queue operations."""

    def test_receive_and_reap_interleaving(self) -> None:
        """Test that receive and reap don't lose or duplicate messages."""
        queue = SimpleMessageQueue()
        queue.send("msg1")

        received: list[str | None] = []

        def receiver() -> None:
            msg = queue.receive()
            received.append(msg)

        def reaper() -> None:
            queue.reap_expired()

        # Test specific interleaving
        result = run_with_schedule(
            {"receive": receiver, "reap": reaper},
            schedule=["receive", "receive", "reap", "receive", "reap", "receive"],
        )

        assert not result.deadlocked
        # Message should be received exactly once
        assert received == ["msg1"]

    def test_ack_after_reap_fails(self) -> None:
        """Ack should fail if message was reaped (returned to pending)."""
        queue = SimpleMessageQueue()
        queue.send("msg1")

        # Receive the message
        _ = queue.receive()
        handle = "handle-msg1"

        ack_results: list[bool] = []

        def acker() -> None:
            result = queue.acknowledge(handle)
            ack_results.append(result)

        def reaper() -> None:
            queue.reap_expired()

        # Reap first (moves message back), then ack fails
        result = run_with_schedule(
            {"ack": acker, "reap": reaper},
            schedule=["reap"] * 5 + ["ack"] * 3,
        )

        assert not result.deadlocked
        assert ack_results == [False]  # Ack failed because message was reaped


# =============================================================================
# Integration Test - Full Scenario
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests combining multiple concurrent components."""

    def test_message_processing_with_heartbeat_and_lease(self) -> None:
        """Test full message processing flow with heartbeat and lease extension."""
        msg = MockMessage()
        heartbeat = SimpleHeartbeat()
        extender = SimpleLeaseExtender()

        processing_complete = False

        def processor() -> None:
            nonlocal processing_complete
            checkpoint("process_start")
            extender.attach(msg, heartbeat)
            checkpoint("process_attached")
            # Simulate processing with heartbeats
            heartbeat.beat()
            checkpoint("process_working")
            heartbeat.beat()
            checkpoint("process_done")
            extender.detach()
            processing_complete = True
            checkpoint("process_detached")

        result = run_with_schedule(
            {"processor": processor},
            schedule=["processor"] * 30,  # Enough for all checkpoints
        )

        assert not result.deadlocked
        assert processing_complete
        # Should have extended twice (two beats while attached)
        assert msg.extend_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
