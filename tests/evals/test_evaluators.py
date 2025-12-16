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

"""Tests for built-in evaluators."""

from __future__ import annotations

from weakincentives.evals import (
    Score,
    all_of,
    any_of,
    contains,
    exact_match,
    json_subset,
    within_tolerance,
)

# =============================================================================
# exact_match Tests
# =============================================================================


def test_exact_match_pass() -> None:
    """exact_match passes on equal strings."""
    score = exact_match("hello", "hello")
    assert score.passed is True
    assert score.value == 1.0


def test_exact_match_fail() -> None:
    """exact_match fails on different strings."""
    score = exact_match("hello", "world")
    assert score.passed is False
    assert score.value == 0.0


def test_exact_match_with_ints() -> None:
    """exact_match works with integers."""
    assert exact_match(42, 42).passed is True
    assert exact_match(42, 43).passed is False


def test_exact_match_case_sensitive() -> None:
    """exact_match is case sensitive."""
    assert exact_match("Hello", "hello").passed is False


# =============================================================================
# contains Tests
# =============================================================================


def test_contains_pass() -> None:
    """contains passes when expected is substring."""
    score = contains("The answer is 42.", "42")
    assert score.passed is True
    assert score.value == 1.0


def test_contains_fail() -> None:
    """contains fails when expected is not substring."""
    score = contains("The answer is 42.", "24")
    assert score.passed is False
    assert score.value == 0.0


def test_contains_exact() -> None:
    """contains passes when strings are equal."""
    score = contains("hello", "hello")
    assert score.passed is True


def test_contains_empty_expected() -> None:
    """contains passes for empty expected string."""
    score = contains("anything", "")
    assert score.passed is True


# =============================================================================
# all_of Tests
# =============================================================================


def test_all_of_all_pass() -> None:
    """all_of passes when all evaluators pass."""
    evaluator = all_of(exact_match, exact_match)
    score = evaluator("hello", "hello")
    assert score.passed is True
    assert score.value == 1.0


def test_all_of_one_fails() -> None:
    """all_of fails when one evaluator fails."""

    def always_pass(output: str, expected: str) -> Score:
        del output, expected
        return Score(value=1.0, passed=True)

    def always_fail(output: str, expected: str) -> Score:
        del output, expected
        return Score(value=0.0, passed=False, reason="always fails")

    evaluator = all_of(always_pass, always_fail)
    score = evaluator("x", "y")
    assert score.passed is False
    assert score.value == 0.5  # Mean of 1.0 and 0.0
    assert "always fails" in score.reason


def test_all_of_aggregates_scores() -> None:
    """all_of computes mean score."""

    def score_75(output: str, expected: str) -> Score:
        del output, expected
        return Score(value=0.75, passed=True)

    def score_25(output: str, expected: str) -> Score:
        del output, expected
        return Score(value=0.25, passed=False)

    evaluator = all_of(score_75, score_25)
    score = evaluator("x", "y")
    assert score.value == 0.5
    assert score.passed is False


def test_all_of_combines_reasons() -> None:
    """all_of combines all reasons."""

    def reason_a(output: str, expected: str) -> Score:
        del output, expected
        return Score(value=1.0, passed=True, reason="reason A")

    def reason_b(output: str, expected: str) -> Score:
        del output, expected
        return Score(value=1.0, passed=True, reason="reason B")

    evaluator = all_of(reason_a, reason_b)
    score = evaluator("x", "y")
    assert "reason A" in score.reason
    assert "reason B" in score.reason


# =============================================================================
# any_of Tests
# =============================================================================


def test_any_of_one_passes() -> None:
    """any_of passes when at least one evaluator passes."""

    def always_pass(output: str, expected: str) -> Score:
        del output, expected
        return Score(value=1.0, passed=True)

    def always_fail(output: str, expected: str) -> Score:
        del output, expected
        return Score(value=0.0, passed=False)

    evaluator = any_of(always_pass, always_fail)
    score = evaluator("x", "y")
    assert score.passed is True
    assert score.value == 1.0  # Max of 1.0 and 0.0


def test_any_of_all_fail() -> None:
    """any_of fails when all evaluators fail."""

    def always_fail(output: str, expected: str) -> Score:
        del output, expected
        return Score(value=0.0, passed=False)

    evaluator = any_of(always_fail, always_fail)
    score = evaluator("x", "y")
    assert score.passed is False
    assert score.value == 0.0


def test_any_of_takes_max_score() -> None:
    """any_of takes maximum score value."""

    def score_25(output: str, expected: str) -> Score:
        del output, expected
        return Score(value=0.25, passed=False)

    def score_75(output: str, expected: str) -> Score:
        del output, expected
        return Score(value=0.75, passed=True)

    evaluator = any_of(score_25, score_75)
    score = evaluator("x", "y")
    assert score.value == 0.75
    assert score.passed is True


# =============================================================================
# within_tolerance Tests
# =============================================================================


def test_within_tolerance_exact() -> None:
    """within_tolerance passes on exact match."""
    evaluator = within_tolerance(0.01)
    score = evaluator(3.14, 3.14)
    assert score.passed is True
    assert score.value == 1.0


def test_within_tolerance_pass() -> None:
    """within_tolerance passes within tolerance."""
    evaluator = within_tolerance(0.1)
    score = evaluator(3.14, 3.10)
    assert score.passed is True
    assert score.value > 0.5


def test_within_tolerance_fail() -> None:
    """within_tolerance fails outside tolerance."""
    evaluator = within_tolerance(0.01)
    score = evaluator(3.14, 3.00)
    assert score.passed is False
    assert score.value == 0.0  # Diff exceeds tolerance


def test_within_tolerance_reason_contains_diff() -> None:
    """within_tolerance includes diff in reason."""
    evaluator = within_tolerance(1.0)
    score = evaluator(5.0, 5.5)
    assert "diff=" in score.reason


def test_within_tolerance_zero() -> None:
    """within_tolerance with zero tolerance."""
    evaluator = within_tolerance(0.0)
    assert evaluator(1.0, 1.0).passed is True
    assert evaluator(1.0, 1.1).passed is False


# =============================================================================
# json_subset Tests
# =============================================================================


def test_json_subset_pass() -> None:
    """json_subset passes when all expected keys present."""
    output = {"a": 1, "b": 2, "c": 3}
    expected = {"a": 1, "b": 2}
    score = json_subset(output, expected)
    assert score.passed is True
    assert score.value == 1.0


def test_json_subset_fail_missing_key() -> None:
    """json_subset fails when key is missing."""
    output = {"a": 1}
    expected = {"a": 1, "b": 2}
    score = json_subset(output, expected)
    assert score.passed is False
    assert "missing or wrong" in score.reason


def test_json_subset_fail_wrong_value() -> None:
    """json_subset fails when value is wrong."""
    output = {"a": 1, "b": 3}
    expected = {"a": 1, "b": 2}
    score = json_subset(output, expected)
    assert score.passed is False
    assert "missing or wrong" in score.reason


def test_json_subset_empty_expected() -> None:
    """json_subset passes for empty expected."""
    output = {"a": 1}
    expected: dict[str, object] = {}
    score = json_subset(output, expected)
    assert score.passed is True


# =============================================================================
# Combinator Integration Tests
# =============================================================================


def test_nested_combinators() -> None:
    """Combinators can be nested."""
    inner = all_of(exact_match, exact_match)
    outer = any_of(inner, contains)
    score = outer("hello", "hello")
    assert score.passed is True


def test_combinator_with_real_evaluators() -> None:
    """Real-world combinator usage pattern."""
    evaluator = all_of(
        contains,
        lambda output, expected: Score(
            value=1.0 if len(output) > len(expected) else 0.5,
            passed=len(output) > len(expected),
        ),
    )
    score = evaluator("The answer is 42", "42")
    assert score.passed is True
