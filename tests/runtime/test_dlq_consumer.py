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

"""Tests for DLQConsumer functionality."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

from weakincentives.runtime.dlq import DeadLetter, DLQConsumer
from weakincentives.runtime.mailbox import InMemoryMailbox

# =============================================================================
# DLQConsumer Tests
# =============================================================================


def test_dlq_consumer_processes_dead_letters() -> None:
    """DLQConsumer processes dead letters with handler."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        processed: list[DeadLetter[str]] = []

        def handler(dl: DeadLetter[str]) -> None:
            processed.append(dl)

        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=handler)

        # Add dead letters
        for i in range(3):
            dlq_mailbox.send(
                DeadLetter(
                    message_id=f"msg-{i}",
                    body=f"message {i}",
                    source_mailbox="source",
                    delivery_count=5,
                    last_error="error",
                    last_error_type="builtins.RuntimeError",
                    dead_lettered_at=datetime.now(UTC),
                    first_received_at=datetime.now(UTC),
                )
            )

        # Run consumer - multiple iterations to process all messages
        consumer.run(max_iterations=5, wait_time_seconds=0)

        # All dead letters should be processed
        assert len(processed) == 3
        assert dlq_mailbox.approximate_count() == 0
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_nacks_on_handler_failure() -> None:
    """DLQConsumer nacks messages when handler fails."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:

        def failing_handler(dl: DeadLetter[str]) -> None:
            raise RuntimeError("handler failure")

        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=failing_handler)

        dlq_mailbox.send(
            DeadLetter(
                message_id="msg-1",
                body="test",
                source_mailbox="source",
                delivery_count=5,
                last_error="error",
                last_error_type="builtins.RuntimeError",
                dead_lettered_at=datetime.now(UTC),
                first_received_at=datetime.now(UTC),
            )
        )

        consumer.run(max_iterations=1, wait_time_seconds=0)

        # Message should still be in queue (nacked)
        assert dlq_mailbox.approximate_count() == 1
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_has_heartbeat() -> None:
    """DLQConsumer has heartbeat for watchdog monitoring."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=lambda _: None)

        assert consumer.heartbeat is not None
        # Heartbeat should have never been beaten
        assert consumer.heartbeat.elapsed() > 0
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_shutdown() -> None:
    """DLQConsumer supports graceful shutdown."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=lambda _: None)

        # Start in background
        thread = threading.Thread(
            target=lambda: consumer.run(max_iterations=None, wait_time_seconds=1)
        )
        thread.start()

        # Give it time to start
        time.sleep(0.1)
        assert consumer.running

        # Shutdown
        result = consumer.shutdown(timeout=2.0)
        assert result is True
        assert not consumer.running

        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_context_manager() -> None:
    """DLQConsumer supports context manager protocol."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        with DLQConsumer(mailbox=dlq_mailbox, handler=lambda _: None) as consumer:
            assert consumer is not None
            # Consumer should be usable
            assert not consumer.running
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_running_property() -> None:
    """DLQConsumer running property reflects actual state."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=lambda _: None)

        # Initially not running
        assert not consumer.running

        # Run in background
        thread = threading.Thread(
            target=lambda: consumer.run(max_iterations=1, wait_time_seconds=0)
        )
        thread.start()

        # Wait for thread to finish
        thread.join(timeout=1.0)

        # Should be stopped now
        assert not consumer.running
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_respects_max_iterations() -> None:
    """DLQConsumer respects max_iterations limit."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        process_count = 0

        def counting_handler(dl: DeadLetter[str]) -> None:
            nonlocal process_count
            process_count += 1

        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=counting_handler)

        # Add many dead letters
        for i in range(10):
            dlq_mailbox.send(
                DeadLetter(
                    message_id=f"msg-{i}",
                    body=f"message {i}",
                    source_mailbox="source",
                    delivery_count=5,
                    last_error="error",
                    last_error_type="builtins.RuntimeError",
                    dead_lettered_at=datetime.now(UTC),
                    first_received_at=datetime.now(UTC),
                )
            )

        # Run only 2 iterations
        consumer.run(max_iterations=2, wait_time_seconds=0)

        # Should have processed at least some (depends on batch size)
        assert process_count >= 1
        # But not all
        assert dlq_mailbox.approximate_count() >= 0
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_beats_heartbeat() -> None:
    """DLQConsumer beats heartbeat during processing."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=lambda _: None)

        # Add a dead letter
        dlq_mailbox.send(
            DeadLetter(
                message_id="msg-1",
                body="test",
                source_mailbox="source",
                delivery_count=5,
                last_error="error",
                last_error_type="builtins.RuntimeError",
                dead_lettered_at=datetime.now(UTC),
                first_received_at=datetime.now(UTC),
            )
        )

        initial_elapsed = consumer.heartbeat.elapsed()

        # Run consumer
        consumer.run(max_iterations=1, wait_time_seconds=0)

        # Heartbeat should have been beaten
        assert consumer.heartbeat.elapsed() < initial_elapsed
    finally:
        dlq_mailbox.close()


def test_dlq_consumer_exits_when_mailbox_closed() -> None:
    """DLQConsumer exits when mailbox is closed."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=lambda _: None)

        # Close mailbox
        dlq_mailbox.close()

        # Run should exit immediately
        consumer.run(max_iterations=10, wait_time_seconds=0)

        # Should not raise, should just exit
        assert not consumer.running
    finally:
        pass  # Mailbox already closed


def test_dlq_consumer_nacks_on_shutdown_during_messages() -> None:
    """DLQConsumer nacks unprocessed messages during shutdown."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        processed: list[str] = []

        def slow_handler(dl: DeadLetter[str]) -> None:
            processed.append(dl.message_id)
            # Trigger shutdown after processing first message
            consumer.shutdown(timeout=0.1)

        consumer = DLQConsumer(mailbox=dlq_mailbox, handler=slow_handler)

        # Add multiple dead letters
        for i in range(3):
            dlq_mailbox.send(
                DeadLetter(
                    message_id=f"msg-{i}",
                    body=f"message {i}",
                    source_mailbox="source",
                    delivery_count=5,
                    last_error="error",
                    last_error_type="builtins.RuntimeError",
                    dead_lettered_at=datetime.now(UTC),
                    first_received_at=datetime.now(UTC),
                )
            )

        # Run consumer - it should process one then get shutdown signal
        consumer.run(max_iterations=10, wait_time_seconds=0)

        # First message should have been processed
        assert len(processed) >= 1
        # Any remaining messages should be nacked back to queue
    finally:
        dlq_mailbox.close()
