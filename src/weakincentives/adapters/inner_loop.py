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

"""Inner loop orchestration for provider adapters.

This module provides the InnerLoop class that coordinates the conversation
lifecycle with a provider:
1. Prepare initial messages and tool specifications
2. Call the provider repeatedly until a final response is produced
3. Execute tools as requested by the provider
4. Parse and return structured output when configured
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, NoReturn, cast
from uuid import uuid4

from ..budget import BudgetExceededError, BudgetTracker
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..prompt.prompt import Prompt, RenderedPrompt
from ..protocols.dispatcher import HandlerFailureProtocol
from ..runtime.events import PromptExecuted, PromptRendered
from ..runtime.execution_state import ExecutionState
from ..runtime.logging import StructuredLogger, get_logger
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from ._names import AdapterName
from ._provider_protocols import ProviderChoice, ProviderMessage, ProviderToolCall
from .core import (
    PROMPT_EVALUATION_PHASE_BUDGET,
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PromptEvaluationError,
    PromptEvaluationPhase,
    PromptResponse,
    SessionProtocol,
)
from .response_parser import ResponseParser
from .throttle import (
    ThrottleError,
    ThrottlePolicy,
    details_from_error,
    jittered_backoff,
    new_throttle_policy,
    sleep_for,
)
from .tool_executor import (
    ToolExecutor,
    ToolMessageSerializer,
)
from .utilities import (
    ToolArgumentsParser,
    ToolChoice,
    deadline_provider_payload,
    extract_payload,
    format_dispatch_failures,
    parse_tool_arguments,
    serialize_tool_call,
    token_usage_from_payload,
    tool_to_spec,
)

if TYPE_CHECKING:
    from ..prompt.tool import Tool
    from .core import ProviderAdapter


logger: StructuredLogger = get_logger(
    __name__, context={"component": "adapters.inner_loop"}
)


ProviderCall = Callable[
    [
        list[dict[str, Any]],
        Sequence[Mapping[str, Any]],
        ToolChoice | None,
        Mapping[str, Any] | None,
    ],
    object,
]
"""Callable responsible for invoking the provider with assembled payloads."""


ChoiceSelector = Callable[[object], ProviderChoice]
"""Callable that extracts the relevant choice from a provider response."""


@FrozenDataclass()
class InnerLoopInputs[OutputT]:
    """Inputs required to start a conversation with a provider."""

    adapter_name: AdapterName
    adapter: ProviderAdapter[OutputT]
    prompt: Prompt[OutputT]
    prompt_name: str
    rendered: RenderedPrompt[OutputT]
    render_inputs: tuple[SupportsDataclass, ...]
    initial_messages: list[dict[str, Any]]


@FrozenDataclass()
class InnerLoopConfig:
    """Configuration and collaborators required to run the inner loop.

    The execution_state provides unified access to session and resources.
    Session is accessed via execution_state.session.
    """

    execution_state: ExecutionState
    tool_choice: ToolChoice
    response_format: Mapping[str, Any] | None
    require_structured_output_text: bool
    call_provider: ProviderCall
    select_choice: ChoiceSelector
    serialize_tool_message_fn: ToolMessageSerializer
    format_dispatch_failures: Callable[[Sequence[HandlerFailureProtocol]], str] = (
        format_dispatch_failures
    )
    parse_arguments: ToolArgumentsParser = parse_tool_arguments
    logger_override: StructuredLogger | None = None
    deadline: Deadline | None = None
    throttle_policy: ThrottlePolicy = field(default_factory=new_throttle_policy)
    budget_tracker: BudgetTracker | None = None

    @property
    def session(self) -> SessionProtocol:
        """Get session from execution state."""
        return self.execution_state.session

    def with_defaults(self, rendered: RenderedPrompt[object]) -> InnerLoopConfig:
        """Fill in optional settings using rendered prompt metadata."""

        return replace(self, deadline=self.deadline or rendered.deadline)


@dataclass(slots=True)
class InnerLoop[OutputT]:
    """Coordinate the inner loop of a conversational exchange with a provider.

    This class orchestrates the conversation lifecycle:
    1. Prepare initial messages and tool specifications
    2. Call the provider repeatedly until a final response is produced
    3. Execute tools as requested by the provider
    4. Parse and return structured output when configured

    The loop handles throttling, deadline enforcement, and budget tracking.
    """

    inputs: InnerLoopInputs[OutputT]
    config: InnerLoopConfig
    _evaluation_id: str = field(init=False)
    _log: StructuredLogger = field(init=False)
    _messages: list[dict[str, Any]] = field(init=False)
    _tool_specs: list[dict[str, Any]] = field(init=False)
    _provider_payload: dict[str, Any] | None = field(init=False, default=None)
    _next_tool_choice: ToolChoice = field(init=False)
    _tool_executor: ToolExecutor = field(init=False)
    _response_parser: ResponseParser[OutputT] = field(init=False)
    _rendered: RenderedPrompt[OutputT] = field(init=False)
    _deadline: Deadline | None = field(init=False)

    def __post_init__(self) -> None:
        normalized_config = self.config.with_defaults(self.inputs.rendered)
        self._deadline = normalized_config.deadline
        if self._deadline is not None and (
            self.inputs.rendered.deadline is not self._deadline
        ):
            self._rendered = replace(self.inputs.rendered, deadline=self._deadline)
        else:
            self._rendered = self.inputs.rendered

    def _raise_deadline_error(
        self, message: str, *, phase: PromptEvaluationPhase
    ) -> NoReturn:
        raise PromptEvaluationError(
            message,
            prompt_name=self.inputs.prompt_name,
            phase=phase,
            provider_payload=deadline_provider_payload(self._deadline),
        )

    def _ensure_deadline_remaining(
        self, message: str, *, phase: PromptEvaluationPhase
    ) -> None:
        if self._deadline is None:
            return
        if self._deadline.remaining() <= timedelta(0):
            self._raise_deadline_error(message, phase=phase)

    def _record_and_check_budget(self) -> None:
        """Record cumulative token usage and check budget limits."""
        if self.config.budget_tracker is None:
            return

        usage = token_usage_from_payload(self._provider_payload)
        if usage is not None:
            self.config.budget_tracker.record_cumulative(self._evaluation_id, usage)

        self._check_budget()

    def _check_budget(self) -> None:
        """Check budget limits and raise if exceeded."""
        if self.config.budget_tracker is None:
            return

        try:
            self.config.budget_tracker.check()
        except BudgetExceededError as error:
            raise PromptEvaluationError(
                str(error),
                prompt_name=self.inputs.prompt_name,
                phase=PROMPT_EVALUATION_PHASE_BUDGET,
                provider_payload=self._provider_payload,
            ) from error

    def run(self) -> PromptResponse[OutputT]:
        """Execute the inner loop and return the final response."""

        self._prepare()

        while True:
            response = self._issue_provider_request()

            self._provider_payload = extract_payload(response)
            self._record_and_check_budget()

            choice = self.config.select_choice(response)
            message = choice.message
            if message is None:
                raise PromptEvaluationError(
                    "Provider response did not include a message payload.",
                    prompt_name=self.inputs.prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_RESPONSE,
                    provider_payload=self._provider_payload,
                )

            tool_calls = list(message.tool_calls or [])

            if not tool_calls:
                return self._finalize_response(message)

            self._handle_tool_calls(message, tool_calls)

    def _issue_provider_request(self) -> object:
        attempts = 0
        total_delay = timedelta(0)
        throttle_policy = self.config.throttle_policy

        while True:
            attempts += 1
            self._ensure_deadline_remaining(
                "Deadline expired before provider request.",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            )

            try:
                return self.config.call_provider(
                    self._messages,
                    self._tool_specs,
                    self._next_tool_choice if self._tool_specs else None,
                    self.config.response_format,
                )
            except ThrottleError as error:
                attempts = max(error.attempts, attempts)
                if not error.retry_safe:
                    raise

                if attempts >= throttle_policy.max_attempts:
                    raise ThrottleError(
                        "Throttle retry budget exhausted.",
                        prompt_name=self.inputs.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        details=details_from_error(
                            error, attempts=attempts, retry_safe=False
                        ),
                    ) from error

                delay = jittered_backoff(
                    policy=throttle_policy,
                    attempt=attempts,
                    retry_after=error.retry_after,
                )

                if self._deadline is not None and self._deadline.remaining() <= delay:
                    raise ThrottleError(
                        "Deadline expired before retrying after throttling.",
                        prompt_name=self.inputs.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        details=details_from_error(
                            error, attempts=attempts, retry_safe=False
                        ),
                    ) from error

                total_delay += delay
                if total_delay > throttle_policy.max_total_delay:
                    raise ThrottleError(
                        "Throttle retry window exceeded configured budget.",
                        prompt_name=self.inputs.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        details=details_from_error(
                            error, attempts=attempts, retry_safe=False
                        ),
                    ) from error

                self._log.warning(
                    "Provider throttled request.",
                    event="prompt_throttled",
                    context={
                        "attempt": attempts,
                        "retry_after_seconds": error.retry_after.total_seconds()
                        if error.retry_after
                        else None,
                        "kind": error.kind,
                        "delay_seconds": delay.total_seconds(),
                    },
                )
                sleep_for(delay)

    def _prepare(self) -> None:
        """Initialize execution state prior to the provider loop."""

        self._evaluation_id = str(uuid4())
        self._messages = list(self.inputs.initial_messages)
        self._log = (self.config.logger_override or logger).bind(
            adapter=self.inputs.adapter_name,
            prompt=self.inputs.prompt_name,
            evaluation_id=self._evaluation_id,
        )
        self._log.info(
            "Prompt execution started.",
            event="prompt_execution_started",
            context={
                "tool_count": len(self._rendered.tools),
            },
        )

        tools = list(self._rendered.tools)
        self._tool_specs = [tool_to_spec(tool) for tool in tools]
        tool_registry: Mapping[
            str, Tool[SupportsDataclassOrNone, SupportsToolResult]
        ] = {tool.name: tool for tool in tools}
        self._provider_payload = None
        self._next_tool_choice = self.config.tool_choice

        self._tool_executor = ToolExecutor(
            adapter_name=self.inputs.adapter_name,
            adapter=self.inputs.adapter,
            prompt=self.inputs.prompt,
            prompt_name=self.inputs.prompt_name,
            rendered=self._rendered,
            execution_state=self.config.execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=self.config.serialize_tool_message_fn,
            format_dispatch_failures=self.config.format_dispatch_failures,
            parse_arguments=self.config.parse_arguments,
            logger_override=self.config.logger_override,
            deadline=self._deadline,
            budget_tracker=self.config.budget_tracker,
        )
        self._response_parser = ResponseParser[OutputT](
            prompt_name=self.inputs.prompt_name,
            rendered=self._rendered,
            require_structured_output_text=self.config.require_structured_output_text,
        )

        self._dispatch_rendered_event()

    def _dispatch_rendered_event(self) -> None:
        """Dispatch the PromptRendered event."""

        dispatch_result = self.config.session.dispatcher.dispatch(
            PromptRendered(
                prompt_ns=self.inputs.prompt.ns,
                prompt_key=self.inputs.prompt.key,
                prompt_name=self.inputs.prompt.name,
                adapter=self.inputs.adapter_name,
                session_id=getattr(self.config.session, "session_id", None),
                render_inputs=self.inputs.render_inputs,
                rendered_prompt=self._rendered.text,
                descriptor=self._rendered.descriptor,
                created_at=datetime.now(UTC),
                event_id=uuid4(),
            )
        )
        if not dispatch_result.ok:
            failure_handlers = [
                getattr(failure.handler, "__qualname__", repr(failure.handler))
                for failure in dispatch_result.errors
            ]
            self._log.error(
                "Prompt rendered dispatch failed.",
                event="prompt_rendered_dispatch_failed",
                context={
                    "failure_count": len(dispatch_result.errors),
                    "failed_handlers": failure_handlers,
                },
            )
        else:
            self._log.debug(
                "Prompt rendered event dispatched.",
                event="prompt_rendered_dispatched",
                context={"handler_count": dispatch_result.handled_count},
            )

    def _handle_tool_calls(
        self,
        message: ProviderMessage,
        tool_calls: Sequence[ProviderToolCall],
    ) -> None:
        """Execute provider tool calls and record emitted messages."""

        assistant_tool_calls = [serialize_tool_call(call) for call in tool_calls]
        self._messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": assistant_tool_calls,
            }
        )

        tool_messages, next_choice = self._tool_executor.execute(
            tool_calls, self._provider_payload
        )
        self._messages.extend(tool_messages)

        self._check_budget()

        if isinstance(self._next_tool_choice, Mapping):
            # Mapping tool_choice always specifies a function call - reset to auto
            self._next_tool_choice = next_choice

    def _finalize_response(self, message: ProviderMessage) -> PromptResponse[OutputT]:
        """Assemble and dispatch the final prompt response."""

        self._ensure_deadline_remaining(
            "Deadline expired while finalizing provider response.",
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
        )

        self._check_budget()

        output, text_value = self._response_parser.parse(
            message, self._provider_payload
        )
        tool_message_records = self._tool_executor.tool_message_records

        if (
            output is not None
            and tool_message_records
            and tool_message_records[-1][0].success
        ):
            last_result, last_message = tool_message_records[-1]
            serialized = self.config.serialize_tool_message_fn(
                last_result, payload=output
            )
            if "output" in last_message:
                last_message["output"] = serialized
            else:
                last_message["content"] = serialized

        response_payload = PromptResponse(
            prompt_name=self.inputs.prompt_name,
            text=text_value,
            output=output,
        )

        self._dispatch_executed_event(response_payload)

        return response_payload

    def _dispatch_executed_event(
        self, response_payload: PromptResponse[OutputT]
    ) -> None:
        """Dispatch the PromptExecuted event."""

        usage = token_usage_from_payload(self._provider_payload)

        dispatch_result = self.config.session.dispatcher.dispatch(
            PromptExecuted(
                prompt_name=self.inputs.prompt_name,
                adapter=self.inputs.adapter_name,
                result=cast(PromptResponse[object], response_payload),
                session_id=getattr(self.config.session, "session_id", None),
                created_at=datetime.now(UTC),
                usage=usage,
                event_id=uuid4(),
            )
        )
        if not dispatch_result.ok:
            failure_handlers = [
                getattr(failure.handler, "__qualname__", repr(failure.handler))
                for failure in dispatch_result.errors
            ]
            self._log.error(
                "Prompt execution dispatch failed.",
                event="prompt_execution_dispatch_failed",
                context={
                    "failure_count": len(dispatch_result.errors),
                    "failed_handlers": failure_handlers,
                },
            )
            dispatch_result.raise_if_errors()
        self._log.info(
            "Prompt execution completed.",
            event="prompt_execution_succeeded",
            context={
                "tool_count": len(self._tool_executor.tool_message_records),
                "has_output": response_payload.output is not None,
                "text_length": len(response_payload.text or "")
                if response_payload.text
                else 0,
                "structured_output": self._response_parser.should_parse_structured_output,
                "handler_count": dispatch_result.handled_count,
            },
        )


def run_inner_loop[
    OutputT,
](
    *,
    inputs: InnerLoopInputs[OutputT],
    config: InnerLoopConfig,
) -> PromptResponse[OutputT]:
    """Execute the inner loop of a conversation with a provider.

    This is the primary entry point for running a conversation. It creates
    an InnerLoop instance and executes it.
    """

    loop = InnerLoop[OutputT](inputs=inputs, config=config)
    return loop.run()


__all__ = [
    "ChoiceSelector",
    "InnerLoop",
    "InnerLoopConfig",
    "InnerLoopInputs",
    "ProviderCall",
    "run_inner_loop",
]
