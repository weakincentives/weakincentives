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

"""High-level orchestration of prompt evaluation.

:class:`AgentLoop` drives a :class:`weakincentives.core.ProviderAdapter`
through render → call → tool → repeat, enforcing optional deadlines and
budgets. It publishes lifecycle events to the session so transcripts and
evaluators can observe progress without coupling to the loop internals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, final

from weakincentives.core import (
    Budget,
    ContractError,
    Deadline,
    PromptExecuted,
    ProviderAdapter,
    Session,
    Snapshotable,
    ToolPolicy,
    ToolResult,
    Usage,
    execute_tool,
)
from weakincentives.core.prompt import Prompt
from weakincentives.core.protocols import PromptResponse, ToolCall

__all__ = [
    "AgentIterationCompleted",
    "AgentIterationStarted",
    "AgentLoop",
    "AgentLoopFinished",
    "AgentLoopResult",
    "AgentLoopStarted",
    "MaxIterationsExceeded",
]

_DEFAULT_MAX_ITERATIONS = 10


# =============================================================================
# Loop-emitted events
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class AgentLoopStarted:
    """Emitted before the first iteration of an agent loop."""

    ns: str
    key: str


@final
@dataclass(frozen=True, slots=True)
class AgentIterationStarted:
    """Emitted at the start of each iteration."""

    iteration: int


@final
@dataclass(frozen=True, slots=True)
class AgentIterationCompleted:
    """Emitted after each iteration's tool dispatch finishes."""

    iteration: int
    text: str
    tool_calls: int
    finish_reason: str | None


@final
@dataclass(frozen=True, slots=True)
class AgentLoopFinished:
    """Emitted after the final iteration of an agent loop."""

    iterations: int
    finish_reason: str | None


# =============================================================================
# Result + errors
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    """Return value of :meth:`AgentLoop.run`."""

    final_response: PromptResponse
    iterations: int
    tool_results: tuple[ToolResult[Any], ...]


class MaxIterationsExceeded(ContractError):
    """Raised when the loop reaches ``max_iterations`` without terminating."""


# =============================================================================
# AgentLoop
# =============================================================================


@final
class AgentLoop:
    """Iterative driver around a :class:`ProviderAdapter`.

    Each iteration:

    1. Renders the prompt.
    2. Calls ``adapter.evaluate(rendered, session, deadline=...)``.
    3. If the response carries tool calls, dispatches them through
       :func:`weakincentives.core.execute_tool`.
    4. Stops when ``finish_reason`` is set, when there are no tool calls,
       or when ``max_iterations`` is reached.

    Deadlines and budgets are checked between iterations and after every
    provider response.
    """

    def __init__(
        self,
        *,
        adapter: ProviderAdapter,
        session: Session,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        snapshotables: Sequence[Snapshotable[Any]] = (),
        policies: Sequence[ToolPolicy] = (),
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    ) -> None:
        super().__init__()
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._adapter = adapter
        self._session = session
        self._deadline = deadline
        self._budget = budget
        self._snapshotables = tuple(snapshotables)
        self._policies = tuple(policies)
        self._max_iterations = max_iterations

    def run(self, prompt: Prompt) -> AgentLoopResult:
        self._session.publish(AgentLoopStarted(ns=prompt.ns, key=prompt.key))
        tool_results: list[ToolResult[Any]] = []
        last_response = PromptResponse(text="")
        terminated = False
        iterations = 0
        for iteration in range(1, self._max_iterations + 1):
            iterations = iteration
            self._check_limits()
            self._session.publish(AgentIterationStarted(iteration=iteration))
            response = self._adapter.evaluate(
                prompt.render(), self._session, deadline=self._deadline
            )
            last_response = response
            self._record_usage(response.usage)
            self._check_limits()
            results = self._dispatch(prompt, response.tool_calls)
            tool_results.extend(results)
            self._session.publish(
                AgentIterationCompleted(
                    iteration=iteration,
                    text=response.text,
                    tool_calls=len(response.tool_calls),
                    finish_reason=response.finish_reason,
                )
            )
            if response.finish_reason is not None or not response.tool_calls:
                terminated = True
                break
        if not terminated:
            raise MaxIterationsExceeded(
                f"agent loop exceeded {self._max_iterations} iterations"
            )
        self._session.publish(
            AgentLoopFinished(
                iterations=iterations,
                finish_reason=last_response.finish_reason,
            )
        )
        self._session.publish(
            PromptExecuted(ns=prompt.ns, key=prompt.key, tool_calls=len(tool_results))
        )
        return AgentLoopResult(
            final_response=last_response,
            iterations=iterations,
            tool_results=tuple(tool_results),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dispatch(
        self, prompt: Prompt, calls: tuple[ToolCall, ...]
    ) -> tuple[ToolResult[Any], ...]:
        results: list[ToolResult[Any]] = []
        for call in calls:
            result = execute_tool(
                prompt,
                self._session,
                call.name,
                call.arguments,
                snapshotables=self._snapshotables,
                policies=self._policies,
            )
            results.append(result)
        return tuple(results)

    def _record_usage(self, usage: Usage) -> None:
        if self._budget is not None:
            self._budget.record(usage)

    def _check_limits(self) -> None:
        if self._deadline is not None and self._deadline.expired():
            from weakincentives.core import DeadlineExceededError

            raise DeadlineExceededError("agent loop deadline expired")
        if self._budget is not None:
            self._budget.check()
