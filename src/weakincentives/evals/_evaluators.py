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

"""Built-in evaluators and combinators for the evaluation framework.

This module provides:
- Basic evaluators: ``exact_match``, ``contains``
- Evaluator combinators: ``all_of``, ``any_of``
- Utilities: ``is_session_aware``, ``adapt``

Evaluators are functions with signature ``(output, expected) -> Score``.
Session-aware evaluators add a third ``session`` parameter for behavioral
assertions against session state (tool calls, token usage, etc.).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, get_args, get_type_hints

from ._types import Evaluator, Score, SessionEvaluator

if TYPE_CHECKING:
    from ..runtime.session import SessionProtocol, SessionViewProtocol


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


_SESSION_TYPE_NAMES = frozenset({"SessionProtocol", "SessionViewProtocol"})
"""Names of session protocol types that indicate a session-aware evaluator."""


def _is_session_type(type_hint: object) -> bool:
    """Check if a type hint is or contains a session protocol type.

    Uses type name matching to avoid runtime protocol checks.
    Handles Union types (e.g., SessionProtocol | SessionViewProtocol).
    """
    # Get type name for direct comparison
    type_name = getattr(type_hint, "__name__", None)
    if type_name in _SESSION_TYPE_NAMES:
        return True

    # Handle Union types (including | syntax which becomes UnionType at runtime)
    # Use get_args directly - works for both Union and | syntax
    args = get_args(type_hint)
    if args:
        return any(_is_session_type(arg) for arg in args)

    return False


def _check_string_annotation(hint: str, fn_globals: dict[str, object]) -> bool:
    """Check if a string type annotation refers to a session type.

    Args:
        hint: The string annotation to check.
        fn_globals: The function's __globals__ dict for resolving names.

    Returns:
        True if the string annotation refers to a session type.
    """
    # Direct name match
    if hint in _SESSION_TYPE_NAMES:
        return True
    # Try to resolve the string in the function's global namespace
    # This handles cases like `session: SVP` where SVP is an alias
    if hint in fn_globals:
        resolved = fn_globals[hint]
        return _is_session_type(resolved)
    # String annotation that couldn't be resolved - check for substring match
    # to catch patterns like "SessionProtocol" within longer type strings
    return any(name in hint for name in _SESSION_TYPE_NAMES)


def is_session_aware(fn: Callable[..., Score]) -> bool:
    """Check if evaluator accepts a session parameter.

    Inspects the function's type hints to determine if it expects a session
    parameter. A session-aware evaluator has a third parameter typed as
    SessionProtocol, SessionViewProtocol, or a union containing them.

    Requires explicit type hints - functions without session type annotations
    are not considered session-aware. This allows runtime dispatch between
    standard and session-aware evaluators in ``EvalLoop`` and combinators.

    Args:
        fn: The evaluator function to check.

    Returns:
        True if the evaluator has an explicit session type hint as its
        third parameter.

    Example:
        >>> from weakincentives.evals import exact_match, tool_called
        >>> is_session_aware(exact_match)  # False - standard evaluator
        False
        >>> is_session_aware(tool_called("search"))  # True - session-aware
        True
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    # Must have at least 3 parameters to be session-aware
    if len(params) < 3:  # noqa: PLR2004
        return False

    third_param = params[2].name

    # Try to get type hints for proper detection
    # First try get_type_hints for resolved annotations
    try:
        hints = get_type_hints(fn)
        if third_param in hints:
            return _is_session_type(hints[third_param])
    except Exception:  # nosec B110
        # get_type_hints can fail with nested functions due to unresolved forward refs
        # Fall back to raw annotations
        pass

    # Try raw __annotations__ (may contain string forward refs or unresolved types)
    annotations = getattr(fn, "__annotations__", {})
    if third_param in annotations:
        hint = annotations[third_param]
        if isinstance(hint, str):
            return _check_string_annotation(hint, getattr(fn, "__globals__", {}))
        return _is_session_type(hint)

    # No type hint found - require explicit annotation
    return False


def adapt[O, E](evaluator: Evaluator) -> SessionEvaluator:
    """Adapt a standard evaluator to session-aware signature.

    The session parameter is ignored, allowing standard evaluators
    to compose with session-aware evaluators.

    Args:
        evaluator: A standard (output, expected) -> Score evaluator.

    Returns:
        A session-aware evaluator that ignores the session parameter.

    Example:
        >>> adapted = adapt(exact_match)
        >>> score = adapted(output, expected, session)  # session is ignored
    """

    def evaluate(
        output: O,
        expected: E,
        session: SessionProtocol | SessionViewProtocol,
    ) -> Score:
        _ = session  # Unused - adapts standard evaluator to session-aware signature
        return evaluator(output, expected)

    return evaluate  # type: ignore[return-value]


def all_of[O, E](
    *evaluators: Evaluator | SessionEvaluator,
) -> SessionEvaluator:
    """All evaluators must pass. Score is the mean.

    Combines multiple evaluators conjunctively. All must pass for
    the combined score to pass. The score value is the mean of all
    individual scores.

    Automatically adapts standard evaluators to session-aware signature.

    Args:
        *evaluators: Variable number of evaluator functions (standard or
            session-aware).

    Returns:
        A session-aware combined evaluator function.

    Example:
        >>> from weakincentives.evals import exact_match, tool_called
        >>> evaluator = all_of(exact_match, tool_called("search"))
        >>> score = evaluator("hello", "hello", session)
        >>> score.passed  # Both must pass
    """
    # Adapt standard evaluators to session-aware signature at runtime
    # Type ignore needed: is_session_aware narrows the type at runtime but
    # the static type checker cannot verify this transformation
    adapted: list[SessionEvaluator] = [  # type: ignore[assignment]
        e if is_session_aware(e) else adapt(e)  # type: ignore[arg-type]
        for e in evaluators
    ]

    def evaluate(
        output: O, expected: E, session: SessionProtocol | SessionViewProtocol
    ) -> Score:
        scores: list[Score] = [e(output, expected, session) for e in adapted]
        passed = all(s.passed for s in scores)
        value = sum(s.value for s in scores) / len(scores)
        reasons = [s.reason for s in scores if s.reason]
        return Score(value=value, passed=passed, reason="; ".join(reasons))

    return evaluate  # type: ignore[return-value]


def any_of[O, E](
    *evaluators: Evaluator | SessionEvaluator,
) -> SessionEvaluator:
    """At least one evaluator must pass. Score is the max.

    Combines multiple evaluators disjunctively. At least one must pass
    for the combined score to pass. The score value is the maximum of
    all individual scores.

    Automatically adapts standard evaluators to session-aware signature.

    Args:
        *evaluators: Variable number of evaluator functions (standard or
            session-aware).

    Returns:
        A session-aware combined evaluator function.

    Example:
        >>> from weakincentives.evals import exact_match, tool_called
        >>> evaluator = any_of(exact_match, tool_called("search"))
        >>> score = evaluator("hello world", "hello", session)
        >>> score.passed  # At least one must pass
    """
    # Adapt standard evaluators to session-aware signature at runtime
    # Type ignore needed: is_session_aware narrows the type at runtime but
    # the static type checker cannot verify this transformation
    adapted: list[SessionEvaluator] = [  # type: ignore[assignment]
        e if is_session_aware(e) else adapt(e)  # type: ignore[arg-type]
        for e in evaluators
    ]

    def evaluate(
        output: O, expected: E, session: SessionProtocol | SessionViewProtocol
    ) -> Score:
        scores: list[Score] = [e(output, expected, session) for e in adapted]
        passed = any(s.passed for s in scores)
        value = max(s.value for s in scores)
        reasons = [s.reason for s in scores if s.reason]
        return Score(value=value, passed=passed, reason="; ".join(reasons))

    return evaluate  # type: ignore[return-value]


__all__ = [
    "adapt",
    "all_of",
    "any_of",
    "contains",
    "exact_match",
    "is_session_aware",
]
