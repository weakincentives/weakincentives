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

"""Built-in evaluators and combinators for the evaluation framework."""

from __future__ import annotations

from collections.abc import Callable

from ._types import Score


def exact_match[T](output: T, expected: T) -> Score:
    """Exact equality check.

    Compares output and expected using Python equality (``==``).
    Returns score of 1.0 on match, 0.0 otherwise.

    Args:
        output: The actual output to evaluate.
        expected: The expected output.

    Returns:
        Score with value=1.0 if output equals expected, else value=0.0.

    Example:
        >>> score = exact_match("hello", "hello")
        >>> score.passed
        True
        >>> score.value
        1.0
    """
    passed = output == expected
    return Score(value=1.0 if passed else 0.0, passed=passed)


def contains(output: str, expected: str) -> Score:
    """Check if expected appears in output.

    Substring presence check for string comparison.
    Returns score of 1.0 if expected is in output, 0.0 otherwise.

    Args:
        output: The actual output string.
        expected: The expected substring.

    Returns:
        Score with value=1.0 if expected is in output, else value=0.0.

    Example:
        >>> score = contains("The answer is 42.", "42")
        >>> score.passed
        True
    """
    passed = expected in output
    return Score(value=1.0 if passed else 0.0, passed=passed)


def all_of[O, E](*evaluators: Callable[[O, E], Score]) -> Callable[[O, E], Score]:
    """All evaluators must pass. Score is the mean.

    Combines multiple evaluators conjunctively. All must pass for
    the combined score to pass. The score value is the mean of all
    individual scores.

    Args:
        *evaluators: Variable number of evaluator functions.

    Returns:
        A combined evaluator function.

    Example:
        >>> evaluator = all_of(exact_match, contains)
        >>> score = evaluator("hello", "hello")
        >>> score.passed  # Both pass
        True
    """

    def evaluate(output: O, expected: E) -> Score:
        scores = [e(output, expected) for e in evaluators]
        passed = all(s.passed for s in scores)
        value = sum(s.value for s in scores) / len(scores)
        reasons = [s.reason for s in scores if s.reason]
        return Score(value=value, passed=passed, reason="; ".join(reasons))

    return evaluate


def any_of[O, E](*evaluators: Callable[[O, E], Score]) -> Callable[[O, E], Score]:
    """At least one evaluator must pass. Score is the max.

    Combines multiple evaluators disjunctively. At least one must pass
    for the combined score to pass. The score value is the maximum of
    all individual scores.

    Args:
        *evaluators: Variable number of evaluator functions.

    Returns:
        A combined evaluator function.

    Example:
        >>> evaluator = any_of(exact_match, contains)
        >>> score = evaluator("hello world", "hello")
        >>> score.passed  # contains passes
        True
    """

    def evaluate(output: O, expected: E) -> Score:
        scores = [e(output, expected) for e in evaluators]
        passed = any(s.passed for s in scores)
        value = max(s.value for s in scores)
        reasons = [s.reason for s in scores if s.reason]
        return Score(value=value, passed=passed, reason="; ".join(reasons))

    return evaluate


__all__ = [
    "all_of",
    "any_of",
    "contains",
    "exact_match",
]
