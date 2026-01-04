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
    """Assert slice contains items matching predicate.

    Args:
        slice_type: The slice type to query.
        predicate: Function to test each item.
        min_count: Minimum matching items required.

    Returns:
        SessionEvaluator that passes if enough items match.

    Example:
        >>> evaluator = slice_contains(
        ...     PlanStep,
        ...     lambda step: step.status == "completed",
        ...     min_count=1,
        ... )
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
