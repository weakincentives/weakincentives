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

"""Tests for DeadLetter, DLQPolicy, custom policies, and module exports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from weakincentives.runtime.agent_loop import AgentLoopRequest
from weakincentives.runtime.dlq import DeadLetter, DLQConsumer, DLQPolicy
from weakincentives.runtime.mailbox import InMemoryMailbox, Message


@dataclass(slots=True, frozen=True)
class _Request:
    """Sample request type for testing."""

    message: str


# =============================================================================
# DeadLetter Tests
# =============================================================================


def test_dead_letter_creation() -> None:
    """DeadLetter captures all required metadata."""
    body = AgentLoopRequest(request=_Request(message="test"))
    dead_letter: DeadLetter[AgentLoopRequest[_Request]] = DeadLetter(
        message_id="msg-123",
        body=body,
        source_mailbox="requests",
        delivery_count=5,
        last_error="Test error",
        last_error_type="builtins.RuntimeError",
        dead_lettered_at=datetime.now(UTC),
        first_received_at=datetime.now(UTC),
        request_id=body.request_id,
        reply_to="results",
        trace_id="trace-abc",
    )

    assert dead_letter.message_id == "msg-123"
    assert dead_letter.body is body
    assert dead_letter.source_mailbox == "requests"
    assert dead_letter.delivery_count == 5
    assert dead_letter.last_error == "Test error"
    assert dead_letter.last_error_type == "builtins.RuntimeError"
    assert dead_letter.request_id == body.request_id
    assert dead_letter.reply_to == "results"
    assert dead_letter.trace_id == "trace-abc"


def test_dead_letter_optional_fields() -> None:
    """DeadLetter has sensible defaults for optional fields."""
    dead_letter: DeadLetter[str] = DeadLetter(
        message_id="msg-123",
        body="test message",
        source_mailbox="requests",
        delivery_count=5,
        last_error="Test error",
        last_error_type="builtins.RuntimeError",
        dead_lettered_at=datetime.now(UTC),
        first_received_at=datetime.now(UTC),
    )

    assert dead_letter.request_id is None
    assert dead_letter.reply_to is None
    assert dead_letter.trace_id is None


def test_dead_letter_is_frozen() -> None:
    """DeadLetter is immutable."""
    dead_letter: DeadLetter[str] = DeadLetter(
        message_id="msg-123",
        body="test message",
        source_mailbox="requests",
        delivery_count=5,
        last_error="Test error",
        last_error_type="builtins.RuntimeError",
        dead_lettered_at=datetime.now(UTC),
        first_received_at=datetime.now(UTC),
    )

    with pytest.raises(AttributeError):
        dead_letter.message_id = "changed"  # type: ignore[misc]


# =============================================================================
# DLQPolicy Tests
# =============================================================================


def test_dlq_policy_default_values() -> None:
    """DLQPolicy has sensible defaults."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        policy: DLQPolicy[str, None] = DLQPolicy(mailbox=dlq_mailbox)

        assert policy.mailbox is dlq_mailbox
        assert policy.max_delivery_count == 5
        assert policy.include_errors is None
        assert policy.exclude_errors is None
    finally:
        dlq_mailbox.close()


def test_dlq_policy_should_dead_letter_by_count() -> None:
    """DLQPolicy dead-letters when delivery count exceeds threshold."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        policy: DLQPolicy[str, None] = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=3,
        )

        # Create a mock message
        msg: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=2,
            enqueued_at=datetime.now(UTC),
        )
        error = RuntimeError("test error")

        # Below threshold - should not dead-letter
        assert not policy.should_dead_letter(msg, error)

        # At threshold - should dead-letter
        msg_at_threshold: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=3,
            enqueued_at=datetime.now(UTC),
        )
        assert policy.should_dead_letter(msg_at_threshold, error)

        # Above threshold - should dead-letter
        msg_above_threshold: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=5,
            enqueued_at=datetime.now(UTC),
        )
        assert policy.should_dead_letter(msg_above_threshold, error)
    finally:
        dlq_mailbox.close()


def test_dlq_policy_include_errors() -> None:
    """DLQPolicy immediately dead-letters included error types."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        policy: DLQPolicy[str, None] = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=5,
            include_errors=frozenset({ValueError, TypeError}),
        )

        msg: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=1,  # First attempt
            enqueued_at=datetime.now(UTC),
        )

        # Included error - should dead-letter immediately
        assert policy.should_dead_letter(msg, ValueError("bad value"))
        assert policy.should_dead_letter(msg, TypeError("bad type"))

        # Not included error - should not dead-letter on first attempt
        assert not policy.should_dead_letter(msg, RuntimeError("other error"))
    finally:
        dlq_mailbox.close()


def test_dlq_policy_exclude_errors() -> None:
    """DLQPolicy never dead-letters excluded error types."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        policy: DLQPolicy[str, None] = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=2,
            exclude_errors=frozenset({TimeoutError}),
        )

        msg: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=10,  # Way above threshold
            enqueued_at=datetime.now(UTC),
        )

        # Excluded error - should never dead-letter
        assert not policy.should_dead_letter(msg, TimeoutError("timeout"))

        # Other error at high count - should dead-letter
        assert policy.should_dead_letter(msg, RuntimeError("other error"))
    finally:
        dlq_mailbox.close()


def test_dlq_policy_exclude_takes_precedence() -> None:
    """Exclude errors take precedence over include errors."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        # This is a weird config but should work
        policy: DLQPolicy[str, None] = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=5,
            include_errors=frozenset({ValueError}),
            exclude_errors=frozenset({ValueError}),  # Same error in both
        )

        msg: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=1,
            enqueued_at=datetime.now(UTC),
        )

        # Exclude takes precedence - should NOT dead-letter
        assert not policy.should_dead_letter(msg, ValueError("test"))
    finally:
        dlq_mailbox.close()


# =============================================================================
# Custom DLQPolicy Tests
# =============================================================================


@dataclass(slots=True, frozen=True)
class _ErrorBudgetPolicy(DLQPolicy[str, None]):
    """Custom policy that dead-letters based on mock error budget."""

    error_budget_exceeded: bool = False

    def should_dead_letter(self, message: Message[str, Any], error: Exception) -> bool:
        # Fall back to default behavior for threshold
        if message.delivery_count >= self.max_delivery_count:
            return True

        # Custom logic: dead-letter if error budget exceeded
        return self.error_budget_exceeded


def test_custom_dlq_policy() -> None:
    """Custom DLQPolicy can implement custom dead-letter logic."""
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    try:
        # Policy with error budget exceeded
        policy = _ErrorBudgetPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=10,
            error_budget_exceeded=True,
        )

        msg: Message[str, Any] = Message(
            id="msg-123",
            body="test",
            receipt_handle="handle",
            delivery_count=1,  # Below threshold
            enqueued_at=datetime.now(UTC),
        )

        # Should dead-letter due to custom logic
        assert policy.should_dead_letter(msg, RuntimeError("test"))

        # Policy without error budget exceeded
        policy_ok = _ErrorBudgetPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=10,
            error_budget_exceeded=False,
        )

        # Should not dead-letter (below threshold, budget OK)
        assert not policy_ok.should_dead_letter(msg, RuntimeError("test"))
    finally:
        dlq_mailbox.close()


# =============================================================================
# Module Exports Tests
# =============================================================================


def test_runtime_exports_dlq_types() -> None:
    """Runtime module exports DLQ types."""
    from weakincentives.runtime import (
        DeadLetter as ExportedDeadLetter,
        DLQConsumer as ExportedConsumer,
        DLQPolicy as ExportedPolicy,
    )

    assert ExportedDeadLetter is DeadLetter
    assert ExportedPolicy is DLQPolicy
    assert ExportedConsumer is DLQConsumer
