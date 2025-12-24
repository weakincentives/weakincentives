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

"""Tests for evals evaluators and combinators."""

from __future__ import annotations

from weakincentives.evals import Score, all_of, any_of, contains, exact_match

# =============================================================================
# exact_match Tests
# =============================================================================


def test_exact_match_pass() -> None:
    """exact_match passes on equal values."""
    score = exact_match("hello", "hello")
    assert score.passed is True
    assert score.value == 1.0


def test_exact_match_fail() -> None:
    """exact_match fails on different values."""
    score = exact_match("hello", "world")
    assert score.passed is False
    assert score.value == 0.0


def test_exact_match_with_numbers() -> None:
    """exact_match works with numeric types."""
    assert exact_match(42, 42).passed is True
    assert exact_match(42, 43).passed is False
    assert exact_match(3.14, 3.14).passed is True


def test_exact_match_with_lists() -> None:
    """exact_match works with sequences."""
    assert exact_match([1, 2, 3], [1, 2, 3]).passed is True
    assert exact_match([1, 2, 3], [1, 2]).passed is False


def test_exact_match_case_sensitive() -> None:
    """exact_match is case-sensitive for strings."""
    assert exact_match("Hello", "hello").passed is False


# =============================================================================
# contains Tests
# =============================================================================


def test_contains_pass() -> None:
    """contains passes when expected is in output."""
    score = contains("The answer is 42.", "42")
    assert score.passed is True
    assert score.value == 1.0


def test_contains_fail() -> None:
    """contains fails when expected is not in output."""
    score = contains("The answer is 42.", "43")
    assert score.passed is False
    assert score.value == 0.0


def test_contains_substring() -> None:
    """contains matches substrings."""
    assert contains("Hello, World!", "World").passed is True
    assert contains("Hello, World!", "Goodbye").passed is False


def test_contains_empty_expected() -> None:
    """contains with empty expected always passes."""
    assert contains("anything", "").passed is True


def test_contains_case_sensitive() -> None:
    """contains is case-sensitive."""
    assert contains("Hello", "HELLO").passed is False


# =============================================================================
# all_of Tests
# =============================================================================


def test_all_of_all_pass() -> None:
    """all_of passes when all evaluators pass."""
    evaluator = all_of(exact_match, contains)
    score = evaluator("hello", "hello")
    assert score.passed is True
    assert score.value == 1.0  # Mean of 1.0 and 1.0


def test_all_of_one_fails() -> None:
    """all_of fails when any evaluator fails."""
    evaluator = all_of(exact_match, contains)
    score = evaluator("hello world", "hello")
    # exact_match fails (hello world != hello), contains passes (hello in hello world)
    assert score.passed is False
    assert score.value == 0.5  # Mean of 0.0 and 1.0


def test_all_of_all_fail() -> None:
    """all_of fails when all evaluators fail."""
    evaluator = all_of(exact_match, contains)
    score = evaluator("foo", "bar")
    assert score.passed is False
    assert score.value == 0.0


def test_all_of_collects_reasons() -> None:
    """all_of collects non-empty reasons."""

    def reason_evaluator(output: str, expected: str) -> Score:
        return Score(value=0.5, passed=True, reason="test reason")

    evaluator = all_of(reason_evaluator, reason_evaluator)
    score = evaluator("a", "b")
    assert "test reason" in score.reason


def test_all_of_single_evaluator() -> None:
    """all_of works with a single evaluator."""
    evaluator = all_of(exact_match)
    score = evaluator("hello", "hello")
    assert score.passed is True
    assert score.value == 1.0


# =============================================================================
# any_of Tests
# =============================================================================


def test_any_of_all_pass() -> None:
    """any_of passes when all evaluators pass."""
    evaluator = any_of(exact_match, contains)
    score = evaluator("hello", "hello")
    assert score.passed is True
    assert score.value == 1.0  # Max of 1.0 and 1.0


def test_any_of_one_passes() -> None:
    """any_of passes when at least one evaluator passes."""
    evaluator = any_of(exact_match, contains)
    score = evaluator("hello world", "hello")
    # exact_match fails, contains passes
    assert score.passed is True
    assert score.value == 1.0  # Max of 0.0 and 1.0


def test_any_of_all_fail() -> None:
    """any_of fails when all evaluators fail."""
    evaluator = any_of(exact_match, contains)
    score = evaluator("foo", "bar")
    assert score.passed is False
    assert score.value == 0.0


def test_any_of_collects_reasons() -> None:
    """any_of collects non-empty reasons."""

    def reason_evaluator(output: str, expected: str) -> Score:
        return Score(value=0.5, passed=True, reason="test reason")

    evaluator = any_of(reason_evaluator, reason_evaluator)
    score = evaluator("a", "b")
    assert "test reason" in score.reason


def test_any_of_single_evaluator() -> None:
    """any_of works with a single evaluator."""
    evaluator = any_of(exact_match)
    score = evaluator("hello", "hello")
    assert score.passed is True
    assert score.value == 1.0


# =============================================================================
# Nested Combinator Tests
# =============================================================================


def test_nested_combinators() -> None:
    """Combinators can be nested."""

    def always_pass(output: str, expected: str) -> Score:
        return Score(value=1.0, passed=True)

    def always_fail(output: str, expected: str) -> Score:
        return Score(value=0.0, passed=False)

    # (exact_match AND contains) OR always_pass
    inner = all_of(exact_match, contains)
    outer = any_of(inner, always_pass)

    # Inner fails (foo != bar, bar not in foo), but always_pass passes
    score = outer("foo", "bar")
    assert score.passed is True
    assert score.value == 1.0  # Max of 0.0 and 1.0


def test_combinators_preserve_score_semantics() -> None:
    """Combinators correctly compute mean vs max."""

    def half_score(output: str, expected: str) -> Score:
        return Score(value=0.5, passed=True)

    def quarter_score(output: str, expected: str) -> Score:
        return Score(value=0.25, passed=True)

    # all_of should return mean
    all_eval = all_of(half_score, quarter_score)
    all_score = all_eval("a", "b")
    assert all_score.value == 0.375  # Mean of 0.5 and 0.25

    # any_of should return max
    any_eval = any_of(half_score, quarter_score)
    any_score = any_eval("a", "b")
    assert any_score.value == 0.5  # Max of 0.5 and 0.25
