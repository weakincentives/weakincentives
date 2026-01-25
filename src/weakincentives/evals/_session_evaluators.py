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

"""Session-aware evaluators for behavioral assertions.

Session-aware evaluators enable assertions against session state, checking
not just what the agent produced, but how it got there. This includes
verifying tool usage patterns, token budgets, and custom state invariants.

All evaluators in this module return factory functions that create
SessionEvaluator instances. They access session state via the slice
accessor pattern: ``session[EventType].all()``.

Available evaluators:
- ``tool_called(name)`` - Assert a tool was called at least once
- ``tool_not_called(name)`` - Assert a tool was never called
- ``tool_call_count(name, min_count, max_count)`` - Assert call count bounds
- ``all_tools_succeeded()`` - Assert no tool failures occurred
- ``token_usage_under(max_tokens)`` - Assert total tokens within budget
- ``slice_contains(type, predicate)`` - Assert slice contains matching items
"""

from __future__ import annotations

from collections.abc import Callable

from ..runtime.events import PromptExecuted, ToolInvoked
from ..runtime.session import SessionProtocol, SessionViewProtocol
from ._types import Score, SessionEvaluator


def tool_called(name: str) -> SessionEvaluator:
    """Assert that a tool was called at least once.

    Args:
        name: The tool name to check for.

    Returns:
        SessionEvaluator that passes if the tool was called.

    Example:
        >>> evaluator = tool_called("search")
        >>> score = evaluator(output, expected, session)
        >>> score.passed  # True if "search" was called
    """

    def evaluate(
        output: object,
        expected: object,
        session: SessionProtocol | SessionViewProtocol,
    ) -> Score:
        calls = session[ToolInvoked].all()
        matching = [c for c in calls if c.name == name]
        passed = len(matching) > 0
        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"tool '{name}' called {len(matching)} time(s)",
        )

    return evaluate


def tool_not_called(name: str) -> SessionEvaluator:
    """Assert that a tool was never called.

    Args:
        name: The tool name that should not appear.

    Returns:
        SessionEvaluator that passes if the tool was not called.

    Example:
        >>> evaluator = tool_not_called("dangerous_tool")
        >>> score = evaluator(output, expected, session)
        >>> score.passed  # True if "dangerous_tool" was never called
    """

    def evaluate(
        output: object,
        expected: object,
        session: SessionProtocol | SessionViewProtocol,
    ) -> Score:
        calls = session[ToolInvoked].all()
        matching = [c for c in calls if c.name == name]
        passed = len(matching) == 0
        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"tool '{name}' called {len(matching)} time(s)"
            if not passed
            else "",
        )

    return evaluate


def tool_call_count(
    name: str,
    *,
    min_count: int = 0,
    max_count: int | None = None,
) -> SessionEvaluator:
    """Assert tool call count is within bounds.

    Args:
        name: The tool name to count.
        min_count: Minimum number of calls required (inclusive).
        max_count: Maximum number of calls allowed (inclusive). None = no limit.

    Returns:
        SessionEvaluator that passes if count is within bounds.

    Example:
        >>> evaluator = tool_call_count("search", min_count=1, max_count=3)
        >>> score = evaluator(output, expected, session)
    """

    def evaluate(
        output: object,
        expected: object,
        session: SessionProtocol | SessionViewProtocol,
    ) -> Score:
        calls = session[ToolInvoked].all()
        count = sum(1 for c in calls if c.name == name)
        passed = count >= min_count and (max_count is None or count <= max_count)
        bounds = f">= {min_count}" if max_count is None else f"{min_count}-{max_count}"

        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"tool '{name}' called {count} times (expected {bounds})",
        )

    return evaluate


def all_tools_succeeded() -> SessionEvaluator:
    """Assert all tool invocations succeeded.

    Checks the 'success' field in each ToolInvoked.result dict.
    Tools without a 'success' field are assumed to have succeeded.

    Returns:
        SessionEvaluator that passes if no tool failures occurred.

    Example:
        >>> evaluator = all_tools_succeeded()
        >>> score = evaluator(output, expected, session)
        >>> score.passed  # True if no tools returned success=False
    """

    def evaluate(
        output: object,
        expected: object,
        session: SessionProtocol | SessionViewProtocol,
    ) -> Score:
        calls = session[ToolInvoked].all()
        if not calls:
            return Score(value=1.0, passed=True)

        failures: list[str] = []
        for call in calls:
            result = call.result
            if isinstance(result, dict) and result.get("success") is False:  # pyright: ignore[reportUnknownMemberType]
                failures.append(call.name)

        passed = len(failures) == 0
        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"failed tools: {failures}" if failures else "",
        )

    return evaluate


def token_usage_under(max_tokens: int) -> SessionEvaluator:
    """Assert total token usage is under budget.

    Sums input_tokens + output_tokens across all PromptExecuted events.

    Args:
        max_tokens: Maximum total tokens allowed.

    Returns:
        SessionEvaluator that passes if usage is under budget.

    Example:
        >>> evaluator = token_usage_under(5000)
        >>> score = evaluator(output, expected, session)
        >>> score.passed  # True if total tokens <= 5000
    """

    def evaluate(
        output: object,
        expected: object,
        session: SessionProtocol | SessionViewProtocol,
    ) -> Score:
        executions = session[PromptExecuted].all()
        total = 0
        for ex in executions:
            if ex.usage:
                total += (ex.usage.input_tokens or 0) + (ex.usage.output_tokens or 0)

        passed = total <= max_tokens
        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"used {total} tokens (limit: {max_tokens})",
        )

    return evaluate


def slice_contains[T](
    slice_type: type[T],
    predicate: Callable[[T], bool],
    *,
    min_count: int = 1,
) -> SessionEvaluator:
    """Assert a session slice contains items matching a predicate.

    Queries the session for all items of the specified slice type and
    counts how many match the predicate. Useful for asserting against
    custom slice types beyond the built-in ToolInvoked/PromptExecuted.

    Args:
        slice_type: The slice type to query from the session. Must be a
            type registered as a session slice.
        predicate: Function that receives each item and returns True if
            it matches the assertion criteria.
        min_count: Minimum number of matching items required for the
            evaluator to pass. Defaults to 1.

    Returns:
        SessionEvaluator that passes if at least ``min_count`` items match.

    Example:
        >>> from myapp.slices import PlanStep
        >>> evaluator = slice_contains(
        ...     PlanStep,
        ...     lambda step: step.status == "completed",
        ...     min_count=3,
        ... )
        >>> # Passes if at least 3 PlanStep items have status="completed"
    """

    def evaluate(
        output: object,
        expected: object,
        session: SessionProtocol | SessionViewProtocol,
    ) -> Score:
        # Use the slice accessor's where method which returns an iterator
        # Type ignore needed: slice_type is generic T, session indexing is dynamic
        accessor = session[slice_type]  # type: ignore[index]
        items: tuple[object, ...] = tuple(accessor.where(predicate))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType,reportUnknownVariableType]
        passed = len(items) >= min_count
        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"found {len(items)} matching items (need >= {min_count})",
        )

    return evaluate


__all__ = [
    "all_tools_succeeded",
    "slice_contains",
    "token_usage_under",
    "tool_call_count",
    "tool_called",
    "tool_not_called",
]
