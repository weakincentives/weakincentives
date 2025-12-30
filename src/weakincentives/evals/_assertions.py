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

"""Session assertions for the evaluation framework.

State assertions verify session state after MainLoop execution. They follow
the same pure function pattern as evaluators: (session) -> Score.

Session assertions:
- **session_has**: Assert slice has expected item count
- **session_latest**: Assert latest item matches predicate
- **session_contains**: Assert any item matches predicate
- **session_all**: Assert all items match predicate

Combinators:
- **all_session_assertions**: All assertions must pass (mean score)
- **any_session_assertions**: At least one must pass (max score)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..types.dataclass import SupportsDataclass
from ._types import Score

if TYPE_CHECKING:
    from ..runtime.session import Session


SessionAssertion = Callable[["Session"], Score]
"""Assertion against session state. Pure function: (session) -> Score."""


def session_has[T: SupportsDataclass](
    slice_type: type[T],
    *,
    count: int | None = None,
    min_count: int | None = None,
    max_count: int | None = None,
) -> SessionAssertion:
    """Assert slice has expected item count.

    Args:
        slice_type: The slice type to check.
        count: Exact count required (mutually exclusive with min/max).
        min_count: Minimum items required.
        max_count: Maximum items allowed.

    Returns:
        SessionAssertion that scores based on count constraints.

    Raises:
        ValueError: If count is used with min_count or max_count.

    Example:
        >>> assertion = session_has(Plan, count=1)
        >>> score = assertion(session)
        >>> score.passed  # True if exactly 1 Plan in session
    """
    if count is not None and (min_count is not None or max_count is not None):
        msg = "count is mutually exclusive with min_count and max_count"
        raise ValueError(msg)

    def assert_count(session: Session) -> Score:
        items = session[slice_type].all()
        actual = len(items)

        # Exact count check
        if count is not None:
            if actual == count:
                return Score(value=1.0, passed=True)
            return Score(
                value=0.0,
                passed=False,
                reason=f"{slice_type.__name__}: expected {count}, got {actual}",
            )

        # Range check
        if min_count is not None and actual < min_count:
            return Score(
                value=0.0,
                passed=False,
                reason=f"{slice_type.__name__}: expected >= {min_count}, got {actual}",
            )
        if max_count is not None and actual > max_count:
            return Score(
                value=0.0,
                passed=False,
                reason=f"{slice_type.__name__}: expected <= {max_count}, got {actual}",
            )

        return Score(value=1.0, passed=True)

    return assert_count


def session_latest[T: SupportsDataclass](
    slice_type: type[T],
    predicate: Callable[[T], bool],
    *,
    reason: str = "",
) -> SessionAssertion:
    """Assert latest item in slice matches predicate.

    Args:
        slice_type: The slice type to check.
        predicate: Function that returns True if item matches.
        reason: Optional reason included in Score on failure.

    Returns:
        SessionAssertion that passes if latest item matches.

    Example:
        >>> assertion = session_latest(Plan, lambda p: p.status == "complete")
        >>> score = assertion(session)
    """

    def assert_latest(session: Session) -> Score:
        latest = session[slice_type].latest()
        if latest is None:
            return Score(
                value=0.0,
                passed=False,
                reason=reason or f"{slice_type.__name__}: no items in slice",
            )
        if predicate(latest):
            return Score(value=1.0, passed=True)
        return Score(
            value=0.0,
            passed=False,
            reason=reason or f"{slice_type.__name__}: latest item does not match",
        )

    return assert_latest


def session_contains[T: SupportsDataclass](
    slice_type: type[T],
    predicate: Callable[[T], bool],
    *,
    reason: str = "",
) -> SessionAssertion:
    """Assert at least one item in slice matches predicate.

    Args:
        slice_type: The slice type to check.
        predicate: Function that returns True if item matches.
        reason: Optional reason included in Score on failure.

    Returns:
        SessionAssertion that passes if any item matches.

    Example:
        >>> assertion = session_contains(ToolCall, lambda t: t.name == "search")
        >>> score = assertion(session)
    """

    def assert_contains(session: Session) -> Score:
        items = session[slice_type].all()
        if any(predicate(item) for item in items):
            return Score(value=1.0, passed=True)
        return Score(
            value=0.0,
            passed=False,
            reason=reason or f"{slice_type.__name__}: no matching items",
        )

    return assert_contains


def session_all[T: SupportsDataclass](
    slice_type: type[T],
    predicate: Callable[[T], bool],
    *,
    reason: str = "",
) -> SessionAssertion:
    """Assert all items in slice match predicate.

    Args:
        slice_type: The slice type to check.
        predicate: Function that returns True if item matches.
        reason: Optional reason included in Score on failure.

    Returns:
        SessionAssertion that passes if all items match (or slice is empty).

    Example:
        >>> assertion = session_all(ToolCall, lambda t: t.success)
        >>> score = assertion(session)
    """

    def assert_all(session: Session) -> Score:
        items = session[slice_type].all()
        # Empty slice passes (vacuous truth)
        if not items:
            return Score(value=1.0, passed=True)
        if all(predicate(item) for item in items):
            return Score(value=1.0, passed=True)
        return Score(
            value=0.0,
            passed=False,
            reason=reason or f"{slice_type.__name__}: not all items match",
        )

    return assert_all


def all_session_assertions(*assertions: SessionAssertion) -> SessionAssertion:
    """All session assertions must pass. Score is the mean.

    Combines multiple session assertions conjunctively. All must pass for
    the combined score to pass. The score value is the mean of all
    individual scores.

    Args:
        *assertions: Variable number of session assertion functions.

    Returns:
        A combined session assertion function.

    Example:
        >>> combined = all_session_assertions(
        ...     session_has(Plan, count=1),
        ...     session_latest(Plan, lambda p: p.complete),
        ... )
        >>> score = combined(session)
    """

    def evaluate(session: Session) -> Score:
        scores = [a(session) for a in assertions]
        passed = all(s.passed for s in scores)
        value = sum(s.value for s in scores) / len(scores) if scores else 1.0
        reasons = [s.reason for s in scores if s.reason]
        return Score(value=value, passed=passed, reason="; ".join(reasons))

    return evaluate


def any_session_assertions(*assertions: SessionAssertion) -> SessionAssertion:
    """At least one session assertion must pass. Score is the max.

    Combines multiple session assertions disjunctively. At least one must
    pass for the combined score to pass. The score value is the maximum
    of all individual scores.

    Args:
        *assertions: Variable number of session assertion functions.

    Returns:
        A combined session assertion function.

    Example:
        >>> combined = any_session_assertions(
        ...     session_has(Plan, count=1),
        ...     session_has(Plan, count=2),
        ... )
        >>> score = combined(session)
    """

    def evaluate(session: Session) -> Score:
        scores = [a(session) for a in assertions]
        passed = any(s.passed for s in scores)
        value = max((s.value for s in scores), default=0.0)
        reasons = [s.reason for s in scores if s.reason]
        return Score(value=value, passed=passed, reason="; ".join(reasons))

    return evaluate


__all__ = [
    "SessionAssertion",
    "all_session_assertions",
    "any_session_assertions",
    "session_all",
    "session_contains",
    "session_has",
    "session_latest",
]
