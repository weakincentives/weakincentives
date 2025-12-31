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

"""Tests for evals session assertions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.evals import (
    SessionAssertion,
    all_session_assertions,
    any_session_assertions,
    session_all,
    session_contains,
    session_has,
    session_latest,
)
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class Plan:
    """Test dataclass for session assertions."""

    status: str
    active: bool = True


@dataclass(slots=True, frozen=True)
class ToolCall:
    """Test dataclass for session assertions."""

    name: str
    success: bool


# =============================================================================
# session_has Tests
# =============================================================================


def test_session_has_exact_count_passes() -> None:
    """session_has passes when slice has exact count."""
    session = Session()
    session[Plan].seed(Plan(status="complete"))

    assertion = session_has(Plan, count=1)
    score = assertion(session)

    assert score.passed is True
    assert score.value == 1.0


def test_session_has_exact_count_fails() -> None:
    """session_has fails when count doesn't match."""
    session = Session()
    # No plans seeded

    assertion = session_has(Plan, count=1)
    score = assertion(session)

    assert score.passed is False
    assert score.value == 0.0
    assert "expected 1" in score.reason
    assert "got 0" in score.reason


def test_session_has_min_count_passes() -> None:
    """session_has passes when slice has at least min_count items."""
    session = Session()
    session[Plan].seed([Plan(status="a"), Plan(status="b")])

    assertion = session_has(Plan, min_count=1)
    score = assertion(session)

    assert score.passed is True


def test_session_has_min_count_fails() -> None:
    """session_has fails when slice has fewer than min_count items."""
    session = Session()

    assertion = session_has(Plan, min_count=1)
    score = assertion(session)

    assert score.passed is False
    assert ">= 1" in score.reason


def test_session_has_max_count_passes() -> None:
    """session_has passes when slice has at most max_count items."""
    session = Session()
    session[Plan].seed(Plan(status="a"))

    assertion = session_has(Plan, max_count=2)
    score = assertion(session)

    assert score.passed is True


def test_session_has_max_count_fails() -> None:
    """session_has fails when slice has more than max_count items."""
    session = Session()
    session[Plan].seed([Plan(status="a"), Plan(status="b"), Plan(status="c")])

    assertion = session_has(Plan, max_count=2)
    score = assertion(session)

    assert score.passed is False
    assert "<= 2" in score.reason


def test_session_has_range_passes() -> None:
    """session_has passes when count is within range."""
    session = Session()
    session[Plan].seed([Plan(status="a"), Plan(status="b")])

    assertion = session_has(Plan, min_count=1, max_count=3)
    score = assertion(session)

    assert score.passed is True


def test_session_has_empty_slice_with_count_zero() -> None:
    """session_has passes for empty slice when count=0."""
    session = Session()

    assertion = session_has(Plan, count=0)
    score = assertion(session)

    assert score.passed is True


def test_session_has_mutual_exclusion() -> None:
    """session_has raises ValueError when count used with min/max."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        session_has(Plan, count=1, min_count=1)


# =============================================================================
# session_latest Tests
# =============================================================================


def test_session_latest_passes() -> None:
    """session_latest passes when latest item matches predicate."""
    session = Session()
    session[Plan].seed([Plan(status="pending"), Plan(status="complete")])

    assertion = session_latest(Plan, lambda p: p.status == "complete")
    score = assertion(session)

    assert score.passed is True
    assert score.value == 1.0


def test_session_latest_fails_no_match() -> None:
    """session_latest fails when latest item doesn't match."""
    session = Session()
    session[Plan].seed([Plan(status="complete"), Plan(status="pending")])

    assertion = session_latest(Plan, lambda p: p.status == "complete")
    score = assertion(session)

    assert score.passed is False
    assert "does not match" in score.reason


def test_session_latest_fails_empty_slice() -> None:
    """session_latest fails when slice is empty."""
    session = Session()

    assertion = session_latest(Plan, lambda p: p.status == "complete")
    score = assertion(session)

    assert score.passed is False
    assert "no items" in score.reason


def test_session_latest_custom_reason() -> None:
    """session_latest uses custom reason on failure."""
    session = Session()

    assertion = session_latest(Plan, lambda p: True, reason="custom failure")
    score = assertion(session)

    assert score.passed is False
    assert score.reason == "custom failure"


# =============================================================================
# session_contains Tests
# =============================================================================


def test_session_contains_passes() -> None:
    """session_contains passes when any item matches predicate."""
    session = Session()
    session[ToolCall].seed(
        [
            ToolCall(name="read", success=True),
            ToolCall(name="search", success=True),
            ToolCall(name="write", success=False),
        ]
    )

    assertion = session_contains(ToolCall, lambda t: t.name == "search")
    score = assertion(session)

    assert score.passed is True
    assert score.value == 1.0


def test_session_contains_fails_no_match() -> None:
    """session_contains fails when no item matches predicate."""
    session = Session()
    session[ToolCall].seed(
        [
            ToolCall(name="read", success=True),
            ToolCall(name="write", success=False),
        ]
    )

    assertion = session_contains(ToolCall, lambda t: t.name == "search")
    score = assertion(session)

    assert score.passed is False
    assert "no matching items" in score.reason


def test_session_contains_empty_slice() -> None:
    """session_contains fails on empty slice."""
    session = Session()

    assertion = session_contains(ToolCall, lambda t: True)
    score = assertion(session)

    assert score.passed is False


def test_session_contains_custom_reason() -> None:
    """session_contains uses custom reason on failure."""
    session = Session()

    assertion = session_contains(Plan, lambda p: True, reason="missing plan")
    score = assertion(session)

    assert score.passed is False
    assert score.reason == "missing plan"


# =============================================================================
# session_all Tests
# =============================================================================


def test_session_all_passes() -> None:
    """session_all passes when all items match predicate."""
    session = Session()
    session[ToolCall].seed(
        [
            ToolCall(name="read", success=True),
            ToolCall(name="write", success=True),
        ]
    )

    assertion = session_all(ToolCall, lambda t: t.success)
    score = assertion(session)

    assert score.passed is True
    assert score.value == 1.0


def test_session_all_fails() -> None:
    """session_all fails when any item doesn't match predicate."""
    session = Session()
    session[ToolCall].seed(
        [
            ToolCall(name="read", success=True),
            ToolCall(name="write", success=False),
        ]
    )

    assertion = session_all(ToolCall, lambda t: t.success)
    score = assertion(session)

    assert score.passed is False
    assert "not all items match" in score.reason


def test_session_all_empty_slice_passes() -> None:
    """session_all passes on empty slice (vacuous truth)."""
    session = Session()

    assertion = session_all(ToolCall, lambda t: t.success)
    score = assertion(session)

    assert score.passed is True
    assert score.value == 1.0


def test_session_all_custom_reason() -> None:
    """session_all uses custom reason on failure."""
    session = Session()
    session[ToolCall].seed(ToolCall(name="fail", success=False))

    assertion = session_all(ToolCall, lambda t: t.success, reason="all must succeed")
    score = assertion(session)

    assert score.passed is False
    assert score.reason == "all must succeed"


# =============================================================================
# all_session_assertions Tests
# =============================================================================


def test_all_session_assertions_passes() -> None:
    """all_session_assertions passes when all assertions pass."""
    session = Session()
    session[Plan].seed(Plan(status="complete"))

    combined = all_session_assertions(
        session_has(Plan, count=1),
        session_latest(Plan, lambda p: p.status == "complete"),
    )
    score = combined(session)

    assert score.passed is True
    assert score.value == 1.0


def test_all_session_assertions_fails_any() -> None:
    """all_session_assertions fails when any assertion fails."""
    session = Session()
    session[Plan].seed(Plan(status="pending"))

    combined = all_session_assertions(
        session_has(Plan, count=1),  # passes
        session_latest(Plan, lambda p: p.status == "complete"),  # fails
    )
    score = combined(session)

    assert score.passed is False
    assert score.value == 0.5  # mean of 1.0 and 0.0


def test_all_session_assertions_mean_score() -> None:
    """all_session_assertions returns mean of all scores."""
    session = Session()
    session[Plan].seed([Plan(status="a"), Plan(status="b")])

    combined = all_session_assertions(
        session_has(Plan, count=2),  # 1.0
        session_has(Plan, count=2),  # 1.0
    )
    score = combined(session)

    assert score.value == 1.0


def test_all_session_assertions_combines_reasons() -> None:
    """all_session_assertions combines failure reasons."""
    session = Session()

    combined = all_session_assertions(
        session_has(Plan, count=1),
        session_contains(ToolCall, lambda t: True),
    )
    score = combined(session)

    assert score.passed is False
    assert "Plan" in score.reason
    assert "ToolCall" in score.reason


def test_all_session_assertions_empty() -> None:
    """all_session_assertions with no assertions passes with value 1.0."""
    session = Session()

    combined = all_session_assertions()
    score = combined(session)

    assert score.passed is True
    assert score.value == 1.0


# =============================================================================
# any_session_assertions Tests
# =============================================================================


def test_any_session_assertions_passes() -> None:
    """any_session_assertions passes when any assertion passes."""
    session = Session()
    session[Plan].seed(Plan(status="pending"))

    combined = any_session_assertions(
        session_latest(Plan, lambda p: p.status == "complete"),  # fails
        session_has(Plan, count=1),  # passes
    )
    score = combined(session)

    assert score.passed is True
    assert score.value == 1.0  # max score


def test_any_session_assertions_fails_all() -> None:
    """any_session_assertions fails when all assertions fail."""
    session = Session()

    combined = any_session_assertions(
        session_has(Plan, count=1),
        session_has(Plan, count=2),
    )
    score = combined(session)

    assert score.passed is False
    assert score.value == 0.0


def test_any_session_assertions_max_score() -> None:
    """any_session_assertions returns max of all scores."""
    session = Session()
    session[Plan].seed(Plan(status="a"))

    combined = any_session_assertions(
        session_has(Plan, count=1),  # 1.0, passes
        session_has(Plan, count=2),  # 0.0, fails
    )
    score = combined(session)

    assert score.value == 1.0


def test_any_session_assertions_empty() -> None:
    """any_session_assertions with no assertions fails with value 0.0."""
    session = Session()

    combined = any_session_assertions()
    score = combined(session)

    assert score.passed is False
    assert score.value == 0.0


# =============================================================================
# Type alias Tests
# =============================================================================


def test_session_assertion_type_alias() -> None:
    """SessionAssertion type alias can be used for typing."""

    def my_assertion(session: Session) -> None:
        # Just verify the type works
        pass

    assertion: SessionAssertion = session_has(Plan, count=1)
    assert callable(assertion)
