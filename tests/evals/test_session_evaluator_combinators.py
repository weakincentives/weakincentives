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

"""Tests for session-aware evaluator combinators and is_session_aware detection."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

from weakincentives.evals import (
    all_of,
    all_tools_succeeded,
    any_of,
    exact_match,
    token_usage_under,
    tool_called,
)
from weakincentives.runtime.events import PromptExecuted, TokenUsage, ToolInvoked
from weakincentives.runtime.session import SessionViewProtocol


def make_tool_invoked(
    name: str,
    *,
    params: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> ToolInvoked:
    """Create a ToolInvoked event for testing."""
    return ToolInvoked(
        prompt_name="test",
        adapter="openai",
        name=name,
        params=params or {},
        result=result or {"success": True},
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )


def make_prompt_executed(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> PromptExecuted:
    """Create a PromptExecuted event for testing."""
    return PromptExecuted(
        prompt_name="test",
        adapter="openai",
        result={"output": "test"},
        session_id=uuid4(),
        created_at=datetime.now(UTC),
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def make_mock_session(
    tool_invocations: list[ToolInvoked] | None = None,
    prompt_executions: list[PromptExecuted] | None = None,
    custom_slices: dict[type, tuple[Any, ...]] | None = None,
) -> SessionViewProtocol:
    """Create a mock session with specified slices.

    Args:
        tool_invocations: ToolInvoked events to populate.
        prompt_executions: PromptExecuted events to populate.
        custom_slices: Additional custom slices as {type: (items,)}.

    Returns:
        Mock session implementing SessionViewProtocol.
    """
    session = Mock(spec=SessionViewProtocol)
    tool_invocations = tool_invocations or []
    prompt_executions = prompt_executions or []
    custom_slices = custom_slices or {}

    def get_slice(slice_type: type) -> Mock:
        slice_mock = Mock()
        if slice_type is ToolInvoked:
            slice_mock.all.return_value = tuple(tool_invocations)
            slice_mock.where.side_effect = lambda pred: iter(
                t for t in tool_invocations if pred(t)
            )
        elif slice_type is PromptExecuted:
            slice_mock.all.return_value = tuple(prompt_executions)
            slice_mock.where.side_effect = lambda pred: iter(
                p for p in prompt_executions if pred(p)
            )
        elif slice_type in custom_slices:
            items = custom_slices[slice_type]
            slice_mock.all.return_value = items
            slice_mock.where.side_effect = lambda pred: iter(
                item for item in items if pred(item)
            )
        else:
            slice_mock.all.return_value = ()
            slice_mock.where.side_effect = lambda pred: iter(())
        return slice_mock

    session.__getitem__ = Mock(side_effect=get_slice)
    return session


# =============================================================================
# Combinator Tests with Session Evaluators
# =============================================================================


def test_all_of_mixed_evaluators() -> None:
    """all_of works with both standard and session-aware evaluators."""
    session = make_mock_session(
        tool_invocations=[make_tool_invoked("search")],
    )

    evaluator = all_of(
        exact_match,  # Standard evaluator
        tool_called("search"),  # Session-aware evaluator
    )

    # Both pass
    score = evaluator("hello", "hello", session)
    assert score.passed is True

    # exact_match fails
    score = evaluator("hello", "world", session)
    assert score.passed is False


def test_any_of_mixed_evaluators() -> None:
    """any_of works with both standard and session-aware evaluators."""
    session = make_mock_session(
        tool_invocations=[make_tool_invoked("search")],
    )

    evaluator = any_of(
        exact_match,  # Standard evaluator
        tool_called("search"),  # Session-aware evaluator
    )

    # Both pass
    score = evaluator("hello", "hello", session)
    assert score.passed is True

    # exact_match fails but tool_called passes
    score = evaluator("hello", "world", session)
    assert score.passed is True


def test_nested_session_evaluators() -> None:
    """Session evaluators can be nested in combinators."""
    session = make_mock_session(
        tool_invocations=[
            make_tool_invoked("search", result={"success": True}),
        ],
        prompt_executions=[
            make_prompt_executed(input_tokens=100, output_tokens=50),
        ],
    )

    # Complex composed evaluator
    evaluator = all_of(
        exact_match,
        all_of(
            tool_called("search"),
            all_tools_succeeded(),
        ),
        token_usage_under(200),
    )

    score = evaluator("hello", "hello", session)
    assert score.passed is True


def test_combinator_collects_session_evaluator_reasons() -> None:
    """Combinators collect reasons from session evaluators."""
    session = make_mock_session(tool_invocations=[])

    evaluator = all_of(
        tool_called("search"),
        tool_called("fetch"),
    )

    score = evaluator(None, None, session)
    assert score.passed is False
    assert "search" in score.reason
    assert "fetch" in score.reason


# =============================================================================
# is_session_aware Tests
# =============================================================================


def test_is_session_aware_detects_session_protocol() -> None:
    """is_session_aware detects SessionProtocol type hint."""
    from weakincentives.evals import Score, is_session_aware
    from weakincentives.runtime.session import SessionProtocol

    def session_evaluator(
        output: object, expected: object, session: SessionProtocol
    ) -> Score:
        return Score(value=1.0, passed=True)

    assert is_session_aware(session_evaluator) is True


def test_is_session_aware_detects_session_view_protocol() -> None:
    """is_session_aware detects SessionViewProtocol type hint."""
    from weakincentives.evals import Score, is_session_aware
    from weakincentives.runtime.session import (  # noqa: F401
        SessionViewProtocol,  # Used in string annotation due to PEP 563
    )

    def session_evaluator(
        output: object, expected: object, session: SessionViewProtocol
    ) -> Score:
        return Score(value=1.0, passed=True)

    assert is_session_aware(session_evaluator) is True


def test_is_session_aware_detects_union_type() -> None:
    """is_session_aware detects union of session protocols."""
    from weakincentives.evals import Score, is_session_aware
    from weakincentives.runtime.session import (  # noqa: F401
        SessionProtocol,  # Used in string annotation due to PEP 563
        SessionViewProtocol,  # Used in string annotation due to PEP 563
    )

    def session_evaluator(
        output: object,
        expected: object,
        session: SessionProtocol | SessionViewProtocol,
    ) -> Score:
        return Score(value=1.0, passed=True)

    assert is_session_aware(session_evaluator) is True


def test_is_session_aware_rejects_standard_evaluator() -> None:
    """is_session_aware returns False for 2-param evaluators."""
    from weakincentives.evals import Score, is_session_aware

    def standard_evaluator(output: object, expected: object) -> Score:
        return Score(value=1.0, passed=True)

    assert is_session_aware(standard_evaluator) is False


def test_is_session_aware_rejects_non_session_third_param() -> None:
    """is_session_aware returns False when third param is not a session type."""
    from weakincentives.evals import Score, is_session_aware

    def not_session_evaluator(output: object, expected: object, context: str) -> Score:
        return Score(value=1.0, passed=True)

    assert is_session_aware(not_session_evaluator) is False


def test_is_session_aware_with_builtin_evaluators() -> None:
    """is_session_aware correctly identifies built-in evaluators."""
    from weakincentives.evals import (
        exact_match,
        is_session_aware,
        tool_called,
    )

    # exact_match is a standard 2-param evaluator
    assert is_session_aware(exact_match) is False

    # tool_called returns a session-aware evaluator
    assert is_session_aware(tool_called("search")) is True


def test_is_session_aware_rejects_unannotated_third_param() -> None:
    """is_session_aware returns False for unannotated 3+ param functions."""
    from collections.abc import Callable

    from weakincentives.evals import is_session_aware

    # Create a function with 3 params but no type annotations dynamically
    # This avoids ruff's ANN rules while testing the behavior
    def make_unannotated() -> Callable[..., None]:
        exec(
            """
def unannotated_evaluator(output, expected, session):
    pass
""",
            {},
            (result := {}),
        )
        return result["unannotated_evaluator"]  # type: ignore[return-value]

    evaluator = make_unannotated()
    # Requires explicit type hints - unannotated functions are not session-aware
    assert is_session_aware(evaluator) is False


def test_is_session_aware_handles_non_session_type_object() -> None:
    """is_session_aware returns False when raw annotation is a non-session type object."""
    from weakincentives.evals._evaluators import _is_session_type

    # Test _is_session_type with a plain type (not a session protocol)
    assert _is_session_type(str) is False
    assert _is_session_type(int) is False
    assert _is_session_type(object) is False


def test_check_string_annotation_resolves_from_globals() -> None:
    """_check_string_annotation resolves aliases from function globals."""
    from weakincentives.evals._evaluators import _check_string_annotation
    from weakincentives.runtime.session import SessionViewProtocol

    # Test with a globals dict containing the alias
    fn_globals: dict[str, object] = {"SVP": SessionViewProtocol}
    assert _check_string_annotation("SVP", fn_globals) is True

    # Test with unresolved alias (not in globals)
    assert _check_string_annotation("UnknownType", {}) is False


def test_is_session_aware_with_raw_type_annotation() -> None:
    """is_session_aware handles raw type annotations when get_type_hints fails."""
    from collections.abc import Callable

    from weakincentives.evals import is_session_aware

    # Create a function where get_type_hints fails due to undefined forward ref
    # but __annotations__ contains actual type objects (not strings)
    def make_evaluator() -> Callable[..., None]:
        # Include an undefined forward reference in return type to make get_type_hints fail
        # But keep the session parameter as an actual type object
        exec(
            """
def raw_type_evaluator(output, expected, session):
    pass
# Mix: string forward ref (undefined) and actual types
# This causes get_type_hints to fail but annotations['session'] is type object
raw_type_evaluator.__annotations__ = {
    'output': object,
    'expected': object,
    'session': str,
    'return': 'UndefinedForwardRef'  # This makes get_type_hints fail
}
""",
            {},
            (result := {}),
        )
        return result["raw_type_evaluator"]  # type: ignore[return-value]

    evaluator = make_evaluator()
    # get_type_hints fails due to undefined return type, falls back to raw annotations
    # session annotation is actual str type (not string), tests line 161
    assert is_session_aware(evaluator) is False
