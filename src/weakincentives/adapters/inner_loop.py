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
lifecycle with a provider. The loop is organized into explicit phases:

1. **Prepare**: Initialize messages, tool specs, and phase components
2. **Call**: Invoke the provider via ProviderCaller (handles throttle/retry)
3. **Execute**: Run tools via ToolExecutor when provider requests them
4. **Parse**: Extract structured output via ResponseParser

Each phase is handled by a dedicated component:
- ProviderCaller: Provider invocation with throttle retry logic
- ToolExecutor: Tool execution with transactional rollback
- ResponseParser: Response parsing and structured output extraction
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
from ..runtime.events import HandlerFailure, PromptExecuted, PromptRendered
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.run_context import RunContext
from ..types import AdapterName
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from ._provider_protocols import ProviderChoice, ProviderMessage, ProviderToolCall
from .core import (
    PROMPT_EVALUATION_PHASE_BUDGET,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PromptEvaluationError,
    PromptEvaluationPhase,
    PromptResponse,
    SessionProtocol,
)
from .provider_caller import ProviderCall, ProviderCaller
from .response_parser import ResponseParser
from .throttle import ThrottlePolicy, new_throttle_policy
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
    from ..runtime.watchdog import Heartbeat
    from .core import ProviderAdapter


logger: StructuredLogger = get_logger(
    __name__, context={"component": "adapters.inner_loop"}
)


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

    Provides access to session for transactional tool execution. Prompt
    resources are accessed via the prompt in InnerLoopInputs.

    When ``heartbeat`` is provided, the inner loop beats at key execution
    points (each iteration, before/after provider calls) to prove liveness.
    Tool handlers receive the heartbeat via ToolContext.beat().
    """

    session: SessionProtocol
    tool_choice: ToolChoice
    response_format: Mapping[str, Any] | None
    require_structured_output_text: bool
    call_provider: ProviderCall
    select_choice: ChoiceSelector
    serialize_tool_message_fn: ToolMessageSerializer
    format_dispatch_failures: Callable[[Sequence[HandlerFailure]], str] = (
        format_dispatch_failures
    )
    parse_arguments: ToolArgumentsParser = parse_tool_arguments
    logger_override: StructuredLogger | None = None
    deadline: Deadline | None = None
    throttle_policy: ThrottlePolicy = field(default_factory=new_throttle_policy)
    budget_tracker: BudgetTracker | None = None
    heartbeat: Heartbeat | None = None
    run_context: RunContext | None = None

    def with_defaults(self, rendered: RenderedPrompt[object]) -> InnerLoopConfig:
        """Fill in optional settings using rendered prompt metadata."""

        return replace(self, deadline=self.deadline or rendered.deadline)


@dataclass(slots=True)
class InnerLoop[OutputT]:
    """Coordinate the inner loop of a conversational exchange with a provider.

    This class orchestrates the conversation lifecycle through explicit phases:

    1. **Prepare**: Initialize messages, tool specs, and phase components
    2. **Call**: Invoke the provider via ProviderCaller (handles throttle/retry)
    3. **Execute**: Run tools via ToolExecutor when provider requests them
    4. **Parse**: Extract structured output via ResponseParser

    Each phase is handled by a dedicated component:
    - ProviderCaller: Provider invocation with throttle retry logic
    - ToolExecutor: Tool execution with transactional rollback
    - ResponseParser: Response parsing and structured output extraction

    The loop also handles deadline enforcement and budget tracking at each stage.
    """

    inputs: InnerLoopInputs[OutputT]
    config: InnerLoopConfig
    _evaluation_id: str = field(init=False)
    _log: StructuredLogger = field(init=False)
    _messages: list[dict[str, Any]] = field(init=False)
    _tool_specs: list[dict[str, Any]] = field(init=False)
    _provider_payload: dict[str, Any] | None = field(init=False, default=None)
    _next_tool_choice: ToolChoice = field(init=False)
    # Phase components
    _provider_caller: ProviderCaller = field(init=False)
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

    def _beat(self) -> None:
        """Record a heartbeat if available."""
        if self.config.heartbeat is not None:
            self.config.heartbeat.beat()

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
        """Execute the inner loop and return the final response.

        The main loop follows this pattern:
        1. Call provider via ProviderCaller (handles throttle/retry)
        2. Extract and validate response
        3. If tool calls present, execute via ToolExecutor and loop back
        4. Otherwise, parse final response via ResponseParser
        """
        self._prepare()

        while True:
            # Beat at start of each iteration
            self._beat()

            # Phase: Call provider (with throttle/retry handling)
            response = self._provider_caller.call(
                self._messages,
                self._tool_specs,
                self._next_tool_choice,
                self.config.response_format,
            )

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
                # Phase: Parse final response
                return self._finalize_response(message)

            # Phase: Execute tools
            self._handle_tool_calls(message, tool_calls)

    def _prepare(self) -> None:
        """Initialize execution state and phase components prior to the loop.

        Creates the three phase components:
        - ProviderCaller: Handles provider invocation with throttle retry
        - ToolExecutor: Handles tool execution with transactional rollback
        - ResponseParser: Handles response parsing and structured output
        """
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

        # Phase component: ProviderCaller (handles throttle/retry)
        self._provider_caller = ProviderCaller(
            call_provider=self.config.call_provider,
            prompt_name=self.inputs.prompt_name,
            throttle_policy=self.config.throttle_policy,
            deadline=self._deadline,
            log=self._log,
        )

        # Phase component: ToolExecutor (handles tool execution)
        self._tool_executor = ToolExecutor(
            adapter_name=self.inputs.adapter_name,
            adapter=self.inputs.adapter,
            prompt=self.inputs.prompt,
            prompt_name=self.inputs.prompt_name,
            rendered=self._rendered,
            session=self.config.session,
            tool_registry=tool_registry,
            serialize_tool_message_fn=self.config.serialize_tool_message_fn,
            format_dispatch_failures=self.config.format_dispatch_failures,
            parse_arguments=self.config.parse_arguments,
            logger_override=self.config.logger_override,
            deadline=self._deadline,
            budget_tracker=self.config.budget_tracker,
            heartbeat=self.config.heartbeat,
            run_context=self.config.run_context,
        )

        # Phase component: ResponseParser (handles structured output)
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
                run_context=self.config.run_context,
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
                run_context=self.config.run_context,
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
    "ProviderCaller",
    "run_inner_loop",
]
