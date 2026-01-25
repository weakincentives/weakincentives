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

"""Core adapter interfaces shared across provider integrations.

This module defines the abstract contracts and data structures for LLM
provider adapters. All provider-specific implementations (OpenAI, LiteLLM,
Claude Agent SDK) derive from these base types.

Key exports:
    ProviderAdapter: Abstract base class for all adapter implementations.
    PromptResponse: Structured result containing text and parsed output.
    PromptEvaluationError: Exception for evaluation failures with phase info.
    PromptEvaluationPhase: Literal type for categorizing error phases.

Example:
    >>> from weakincentives.adapters.core import ProviderAdapter, PromptResponse
    >>> # Create a concrete adapter (e.g., OpenAIAdapter)
    >>> adapter: ProviderAdapter = get_adapter()
    >>> with my_prompt.resources:
    ...     response: PromptResponse = adapter.evaluate(my_prompt, session=session)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from ..budget import Budget, BudgetTracker
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..errors import WinkError
from ..prompt import Prompt
from ..runtime.session.protocols import SessionProtocol

if TYPE_CHECKING:
    from ..runtime.run_context import RunContext
    from ..runtime.watchdog import Heartbeat


@FrozenDataclass()
class PromptResponse[OutputT]:
    """Structured result emitted by an adapter evaluation.

    Contains both raw text and parsed output from a prompt evaluation.
    The output is typed according to the prompt's output schema.

    Attributes:
        prompt_name: Fully-qualified name of the evaluated prompt
            (format: ``namespace.key``).
        text: Raw text content from the model response. May be ``None``
            if the response contained only tool calls or structured output.
        output: Parsed and validated output matching the prompt's output type.
            ``None`` if parsing failed or the prompt has no output schema.

    Example:
        >>> response = adapter.evaluate(my_prompt, session=session)
        >>> if response.output is not None:
        ...     process_result(response.output)
        >>> if response.text:
        ...     log_raw_response(response.text)
    """

    prompt_name: str
    text: str | None
    output: OutputT | None


class ProviderAdapter(ABC):
    """Abstract base class for LLM provider integrations.

    A ProviderAdapter bridges between the weakincentives prompt system and
    a specific LLM provider (OpenAI, Anthropic, LiteLLM, etc.). It handles:

    - Rendering prompts into provider-specific request formats
    - Executing model API calls with timeout and budget enforcement
    - Processing responses and parsing structured outputs
    - Orchestrating multi-turn tool call loops

    Implementers must override the ``evaluate`` method to provide
    provider-specific logic. Built-in adapters include:

    - ``OpenAIAdapter``: Direct OpenAI API integration
    - ``LiteLLMAdapter``: Multi-provider support via LiteLLM
    - ``ClaudeAgentSDKAdapter``: Anthropic Claude Agent SDK integration

    Example:
        >>> from weakincentives.adapters import OpenAIAdapter
        >>> adapter = OpenAIAdapter(model="gpt-4")
        >>> with my_prompt.resources:
        ...     response = adapter.evaluate(my_prompt, session=session)
        >>> print(response.output)

    Note:
        This class is generic over the output type but uses a custom
        ``__class_getitem__`` to support runtime subscripting without
        creating new types at runtime.
    """

    @classmethod
    def __class_getitem__(cls, _: object) -> type[ProviderAdapter[Any]]:
        return cls

    @abstractmethod
    def evaluate[OutputT](
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
    ) -> PromptResponse[OutputT]:
        """Evaluate a prompt against this provider and return a structured response.

        Executes the full prompt lifecycle: rendering, API call, response parsing,
        and tool call orchestration (if tools are declared on the prompt).

        Args:
            prompt: The prompt to evaluate. Must be within an active
                ``with prompt.resources:`` context for resource injection.
            session: Session for state management, event dispatch, and
                visibility overrides. Use ``session[VisibilityOverrides]``
                to configure section visibility before calling.
            deadline: Optional timeout for the entire evaluation. Raises
                ``DeadlineExceeded`` if the deadline expires during execution.
            budget: Optional token/cost budget for this evaluation. Creates
                a new ``BudgetTracker`` if ``budget_tracker`` is not provided.
            budget_tracker: Optional shared tracker for multi-prompt budget
                enforcement. Takes precedence over ``budget`` if both provided.
            heartbeat: Optional liveness callback. The adapter beats at key
                execution points (LLM calls, tool boundaries). Tool handlers
                can access via ``ToolContext.beat()`` for long operations.
            run_context: Optional tracing context threaded through telemetry
                events (``PromptRendered``, ``PromptExecuted``, ``ToolInvoked``)
                for distributed tracing integration.

        Returns:
            A ``PromptResponse`` containing the raw text and parsed output.

        Raises:
            PromptEvaluationError: If evaluation fails at any phase (request,
                response parsing, tool execution, or budget exceeded).
            DeadlineExceeded: If the deadline expires during evaluation.

        Example:
            >>> with prompt.resources:
            ...     response = adapter.evaluate(
            ...         prompt,
            ...         session=session,
            ...         deadline=Deadline.from_timeout(30.0),
            ...         budget=Budget(max_tokens=1000),
            ...     )
        """

        ...


class PromptEvaluationError(WinkError, RuntimeError):
    """Raised when prompt evaluation against a provider fails.

    This exception captures detailed context about evaluation failures,
    including the phase where the failure occurred and any provider-specific
    diagnostic information.

    Attributes:
        message: Human-readable description of the failure.
        prompt_name: Fully-qualified name of the prompt that failed
            (format: ``namespace.key``).
        phase: The evaluation phase where the error occurred. One of:
            ``"request"`` (API call failed), ``"response"`` (parsing failed),
            ``"tool"`` (tool execution failed), or ``"budget"`` (limits exceeded).
        provider_payload: Optional dict containing provider-specific error
            details (e.g., API error codes, rate limit info, validation errors).

    Example:
        >>> try:
        ...     response = adapter.evaluate(prompt, session=session)
        ... except PromptEvaluationError as e:
        ...     if e.phase == "budget":
        ...         handle_budget_exceeded(e)
        ...     elif e.phase == "tool":
        ...         log_tool_failure(e.prompt_name, e.provider_payload)
        ...     else:
        ...         raise
    """

    def __init__(
        self,
        message: str,
        *,
        prompt_name: str,
        phase: PromptEvaluationPhase,
        provider_payload: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a PromptEvaluationError.

        Args:
            message: Human-readable description of what went wrong.
            prompt_name: The fully-qualified name of the prompt that failed.
            phase: The evaluation phase where the error occurred.
            provider_payload: Optional provider-specific diagnostic data.
        """
        super().__init__(message)
        self.message = message
        self.prompt_name = prompt_name
        self.phase: PromptEvaluationPhase = phase
        self.provider_payload = provider_payload


PromptEvaluationPhase = Literal["request", "response", "tool", "budget"]
"""Phases where a prompt evaluation error can occur.

Used to categorize ``PromptEvaluationError`` by failure point:

- ``"request"``: Error while building or sending the API request
  (network failures, authentication errors, malformed prompts).
- ``"response"``: Error while processing the provider response
  (JSON parsing, schema validation, unexpected response format).
- ``"tool"``: Error during tool call execution
  (tool handler exceptions, validation failures, timeout).
- ``"budget"``: Budget limits exceeded during evaluation
  (token count, cost limit, or request count exceeded).

Use the ``PROMPT_EVALUATION_PHASE_*`` constants for type-safe comparisons.
"""

PROMPT_EVALUATION_PHASE_REQUEST: PromptEvaluationPhase = "request"
"""Prompt evaluation failed while issuing the provider request."""

PROMPT_EVALUATION_PHASE_RESPONSE: PromptEvaluationPhase = "response"
"""Prompt evaluation failed while handling the provider response."""

PROMPT_EVALUATION_PHASE_TOOL: PromptEvaluationPhase = "tool"
"""Prompt evaluation failed while handling a tool invocation."""

PROMPT_EVALUATION_PHASE_BUDGET: PromptEvaluationPhase = "budget"
"""Prompt evaluation failed due to budget limits being exceeded."""


__all__ = [
    "PROMPT_EVALUATION_PHASE_BUDGET",
    "PROMPT_EVALUATION_PHASE_REQUEST",
    "PROMPT_EVALUATION_PHASE_RESPONSE",
    "PROMPT_EVALUATION_PHASE_TOOL",
    "Budget",
    "BudgetTracker",
    "PromptEvaluationError",
    "PromptEvaluationPhase",
    "PromptResponse",
    "ProviderAdapter",
]
