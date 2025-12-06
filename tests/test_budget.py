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

"""Tests for the :mod:`weakincentives.budget` module."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

import pytest

from weakincentives.budget import (
    Budget,
    BudgetExceededError,
    BudgetTracker,
)
from weakincentives.deadlines import Deadline
from weakincentives.runtime.events import TokenUsage

if TYPE_CHECKING:
    from tests.helpers import FrozenUtcNow

# Budget dataclass tests


def test_budget_requires_at_least_one_limit() -> None:
    """Budget without any limits should raise ValueError."""
    with pytest.raises(ValueError, match="at least one limit"):
        Budget()


def test_budget_accepts_deadline_only(frozen_utcnow: FrozenUtcNow) -> None:
    """Budget can be constructed with only a deadline."""
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(seconds=30))

    budget = Budget(deadline=deadline)

    assert budget.deadline is deadline
    assert budget.max_total_tokens is None


def test_budget_accepts_total_tokens_only() -> None:
    """Budget can be constructed with only max_total_tokens."""
    budget = Budget(max_total_tokens=1000)

    assert budget.max_total_tokens == 1000
    assert budget.max_input_tokens is None
    assert budget.max_output_tokens is None
    assert budget.deadline is None


def test_budget_accepts_input_tokens_only() -> None:
    """Budget can be constructed with only max_input_tokens."""
    budget = Budget(max_input_tokens=500)

    assert budget.max_input_tokens == 500


def test_budget_accepts_output_tokens_only() -> None:
    """Budget can be constructed with only max_output_tokens."""
    budget = Budget(max_output_tokens=500)

    assert budget.max_output_tokens == 500


def test_budget_accepts_combined_limits(frozen_utcnow: FrozenUtcNow) -> None:
    """Budget can have both deadline and token limits."""
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(seconds=30))

    budget = Budget(
        deadline=deadline,
        max_total_tokens=1000,
        max_input_tokens=800,
        max_output_tokens=200,
    )

    assert budget.deadline is deadline
    assert budget.max_total_tokens == 1000
    assert budget.max_input_tokens == 800
    assert budget.max_output_tokens == 200


def test_budget_rejects_zero_total_tokens() -> None:
    """max_total_tokens must be positive."""
    with pytest.raises(ValueError, match="max_total_tokens must be positive"):
        Budget(max_total_tokens=0)


def test_budget_rejects_negative_total_tokens() -> None:
    """max_total_tokens must be positive."""
    with pytest.raises(ValueError, match="max_total_tokens must be positive"):
        Budget(max_total_tokens=-100)


def test_budget_rejects_zero_input_tokens() -> None:
    """max_input_tokens must be positive."""
    with pytest.raises(ValueError, match="max_input_tokens must be positive"):
        Budget(max_input_tokens=0)


def test_budget_rejects_negative_input_tokens() -> None:
    """max_input_tokens must be positive."""
    with pytest.raises(ValueError, match="max_input_tokens must be positive"):
        Budget(max_input_tokens=-100)


def test_budget_rejects_zero_output_tokens() -> None:
    """max_output_tokens must be positive."""
    with pytest.raises(ValueError, match="max_output_tokens must be positive"):
        Budget(max_output_tokens=0)


def test_budget_rejects_negative_output_tokens() -> None:
    """max_output_tokens must be positive."""
    with pytest.raises(ValueError, match="max_output_tokens must be positive"):
        Budget(max_output_tokens=-100)


def test_budget_is_immutable() -> None:
    """Budget should be frozen."""
    budget = Budget(max_total_tokens=1000)

    with pytest.raises(AttributeError):
        budget.max_total_tokens = 2000  # pyright: ignore[reportAttributeAccessIssue]


# BudgetTracker tests


def test_tracker_records_cumulative_usage() -> None:
    """record_cumulative should store usage per conversation."""
    budget = Budget(max_total_tokens=1000)
    tracker = BudgetTracker(budget=budget)

    usage = TokenUsage(input_tokens=100, output_tokens=50)
    tracker.record_cumulative("eval-1", usage)

    consumed = tracker.consumed
    assert consumed.input_tokens == 100
    assert consumed.output_tokens == 50


def test_tracker_replaces_usage_for_same_conversation() -> None:
    """record_cumulative replaces previous usage for the same conversation."""
    budget = Budget(max_total_tokens=1000)
    tracker = BudgetTracker(budget=budget)

    usage1 = TokenUsage(input_tokens=100, output_tokens=50)
    usage2 = TokenUsage(input_tokens=200, output_tokens=100)

    tracker.record_cumulative("eval-1", usage1)
    tracker.record_cumulative("eval-1", usage2)

    consumed = tracker.consumed
    assert consumed.input_tokens == 200
    assert consumed.output_tokens == 100


def test_tracker_sums_across_conversations() -> None:
    """consumed should sum usage across all conversations."""
    budget = Budget(max_total_tokens=1000)
    tracker = BudgetTracker(budget=budget)

    tracker.record_cumulative("eval-1", TokenUsage(input_tokens=100, output_tokens=50))
    tracker.record_cumulative("eval-2", TokenUsage(input_tokens=150, output_tokens=75))

    consumed = tracker.consumed
    assert consumed.input_tokens == 250
    assert consumed.output_tokens == 125
    assert consumed.total_tokens == 375


def test_tracker_check_passes_within_budget() -> None:
    """check() should not raise when within budget limits."""
    budget = Budget(max_total_tokens=1000)
    tracker = BudgetTracker(budget=budget)

    tracker.record_cumulative("eval-1", TokenUsage(input_tokens=400, output_tokens=200))

    # Should not raise
    tracker.check()


def test_tracker_check_raises_when_total_exceeded() -> None:
    """check() should raise BudgetExceededError when total tokens exceeded."""
    budget = Budget(max_total_tokens=500)
    tracker = BudgetTracker(budget=budget)

    tracker.record_cumulative("eval-1", TokenUsage(input_tokens=400, output_tokens=200))

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check()

    error = cast("BudgetExceededError", exc_info.value)
    assert error.exceeded_dimension == "total_tokens"
    assert error.budget is budget
    assert error.consumed.total_tokens == 600


def test_tracker_check_raises_when_input_exceeded() -> None:
    """check() should raise BudgetExceededError when input tokens exceeded."""
    budget = Budget(max_input_tokens=300)
    tracker = BudgetTracker(budget=budget)

    tracker.record_cumulative("eval-1", TokenUsage(input_tokens=400, output_tokens=100))

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check()

    error = cast("BudgetExceededError", exc_info.value)
    assert error.exceeded_dimension == "input_tokens"


def test_tracker_check_raises_when_output_exceeded() -> None:
    """check() should raise BudgetExceededError when output tokens exceeded."""
    budget = Budget(max_output_tokens=150)
    tracker = BudgetTracker(budget=budget)

    tracker.record_cumulative("eval-1", TokenUsage(input_tokens=100, output_tokens=200))

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check()

    error = cast("BudgetExceededError", exc_info.value)
    assert error.exceeded_dimension == "output_tokens"


def test_tracker_check_raises_when_deadline_expired(
    frozen_utcnow: FrozenUtcNow,
) -> None:
    """check() should raise BudgetExceededError when deadline passed."""
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(seconds=30))
    budget = Budget(deadline=deadline)
    tracker = BudgetTracker(budget=budget)

    # Advance time past deadline
    frozen_utcnow.set(anchor + timedelta(seconds=35))

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check()

    error = cast("BudgetExceededError", exc_info.value)
    assert error.exceeded_dimension == "deadline"


def test_tracker_thread_safety() -> None:
    """BudgetTracker should be thread-safe for concurrent updates."""
    budget = Budget(max_total_tokens=100000)
    tracker = BudgetTracker(budget=budget)
    num_threads = 10
    updates_per_thread = 100

    def update_tracker(thread_id: int) -> None:
        for i in range(updates_per_thread):
            usage = TokenUsage(input_tokens=1, output_tokens=1)
            tracker.record_cumulative(f"eval-{thread_id}-{i}", usage)

    threads = [
        threading.Thread(target=update_tracker, args=(tid,))
        for tid in range(num_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    consumed = tracker.consumed
    expected_total = num_threads * updates_per_thread * 2
    assert consumed.total_tokens == expected_total


def test_tracker_handles_none_token_counts() -> None:
    """Tracker should handle TokenUsage with None values."""
    budget = Budget(max_total_tokens=1000)
    tracker = BudgetTracker(budget=budget)

    tracker.record_cumulative("eval-1", TokenUsage(input_tokens=None, output_tokens=50))
    tracker.record_cumulative(
        "eval-2", TokenUsage(input_tokens=100, output_tokens=None)
    )

    consumed = tracker.consumed
    assert consumed.input_tokens == 100
    assert consumed.output_tokens == 50


# BudgetExceededError tests


def test_error_has_budget_reference() -> None:
    """Error should contain reference to the budget."""
    budget = Budget(max_total_tokens=100)
    consumed = TokenUsage(input_tokens=80, output_tokens=30)

    error = BudgetExceededError(
        budget=budget,
        consumed=consumed,
        exceeded_dimension="total_tokens",
    )

    assert error.budget is budget
    assert error.consumed is consumed
    assert error.exceeded_dimension == "total_tokens"


def test_error_str_representation() -> None:
    """Error string should indicate which dimension was exceeded."""
    budget = Budget(max_total_tokens=100)
    consumed = TokenUsage(input_tokens=80, output_tokens=30)

    error = BudgetExceededError(
        budget=budget,
        consumed=consumed,
        exceeded_dimension="total_tokens",
    )

    assert "total_tokens" in str(error)


def test_error_is_runtime_error() -> None:
    """BudgetExceededError should inherit from RuntimeError."""
    budget = Budget(max_total_tokens=100)
    consumed = TokenUsage(input_tokens=80, output_tokens=30)

    error = BudgetExceededError(
        budget=budget,
        consumed=consumed,
        exceeded_dimension="total_tokens",
    )

    assert isinstance(error, RuntimeError)


# Integration tests


def test_budget_tracker_reuse_across_conversations() -> None:
    """Same tracker should accumulate usage across multiple conversations."""
    budget = Budget(max_total_tokens=500)
    tracker = BudgetTracker(budget=budget)

    # Simulate first conversation
    tracker.record_cumulative("eval-1", TokenUsage(input_tokens=100, output_tokens=50))
    tracker.check()  # Should pass

    # Simulate second conversation
    tracker.record_cumulative("eval-2", TokenUsage(input_tokens=200, output_tokens=100))
    tracker.check()  # Should pass (total: 450)

    # Third conversation would exceed
    tracker.record_cumulative("eval-3", TokenUsage(input_tokens=100, output_tokens=50))

    with pytest.raises(BudgetExceededError):
        tracker.check()


def test_budget_with_all_limits(frozen_utcnow: FrozenUtcNow) -> None:
    """Test budget with all limits set."""
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(minutes=5))

    budget = Budget(
        deadline=deadline,
        max_total_tokens=1000,
        max_input_tokens=600,
        max_output_tokens=400,
    )
    tracker = BudgetTracker(budget=budget)

    # Within all limits
    tracker.record_cumulative("eval-1", TokenUsage(input_tokens=300, output_tokens=200))
    tracker.check()

    # Exceed input limit
    tracker.record_cumulative("eval-2", TokenUsage(input_tokens=400, output_tokens=50))

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check()

    error = cast("BudgetExceededError", exc_info.value)
    assert error.exceeded_dimension == "input_tokens"
