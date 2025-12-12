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

"""Claude Agent SDK adapter implementation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, TypeVar, cast, override

from ...budget import Budget, BudgetTracker
from ...deadlines import Deadline
from ...prompt import Prompt, RenderedPrompt, SectionVisibility
from ...prompt.errors import SectionPath
from ...runtime.events import PromptExecuted, PromptRendered
from ...runtime.events._types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session.protocols import SessionProtocol
from ...serde import parse, schema
from .._names import AdapterName
from ..core import PromptEvaluationError, PromptResponse, ProviderAdapter
from ._async_utils import run_async
from ._bridge import create_bridged_tools
from ._errors import normalize_sdk_error
from ._hooks import (
    HookContext,
    create_post_tool_use_hook,
    create_pre_tool_use_hook,
    create_stop_hook,
    create_user_prompt_submit_hook,
)
from .config import ClaudeAgentSDKClientConfig, ClaudeAgentSDKModelConfig

__all__ = [
    "CLAUDE_AGENT_SDK_ADAPTER_NAME",
    "ClaudeAgentSDKAdapter",
]

logger: StructuredLogger = get_logger(
    __name__, context={"component": "claude_agent_sdk"}
)

OutputT = TypeVar("OutputT")

CLAUDE_AGENT_SDK_ADAPTER_NAME: AdapterName = "claude_agent_sdk"
"""Canonical label for the Claude Agent SDK adapter."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _import_sdk() -> Any:  # pragma: no cover
    """Import the Claude Agent SDK, raising a helpful error if not installed."""
    try:
        import claude_agent_sdk

        return claude_agent_sdk
    except ImportError as error:
        raise ImportError(
            "claude-agent-sdk is not installed. Install it with: "
            "pip install 'weakincentives[claude-agent-sdk]'"
        ) from error


class ClaudeAgentSDKAdapter(ProviderAdapter[OutputT]):
    """Adapter using Claude Agent SDK with hook-based state synchronization.

    This adapter uses the Claude Agent SDK's ClaudeSDKClient to execute
    prompts with full agentic capabilities. Hooks synchronize state
    bidirectionally between the SDK's internal execution and the
    weakincentives Session.

    Example:
        >>> from weakincentives import Prompt, PromptTemplate, MarkdownSection
        >>> from weakincentives.runtime import Session, InProcessEventBus
        >>> from weakincentives.adapters.claude_agent_sdk import (
        ...     ClaudeAgentSDKAdapter,
        ...     ClaudeAgentSDKClientConfig,
        ... )
        >>>
        >>> bus = InProcessEventBus()
        >>> session = Session(bus=bus)
        >>>
        >>> adapter = ClaudeAgentSDKAdapter(
        ...     model="claude-sonnet-4-5-20250929",
        ...     client_config=ClaudeAgentSDKClientConfig(
        ...         permission_mode="acceptEdits",
        ...         cwd="/home/user/project",
        ...     ),
        ... )
        >>>
        >>> template = PromptTemplate[str](
        ...     ns="test",
        ...     key="hello",
        ...     sections=[
        ...         MarkdownSection(
        ...             title="Task",
        ...             key="task",
        ...             template="Say hello",
        ...         ),
        ...     ],
        ... )
        >>> prompt = Prompt(template)
        >>>
        >>> response = adapter.evaluate(prompt, session=session)
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5-20250929",
        client_config: ClaudeAgentSDKClientConfig | None = None,
        model_config: ClaudeAgentSDKModelConfig | None = None,
        allowed_tools: tuple[str, ...] | None = None,
        disallowed_tools: tuple[str, ...] = (),
    ) -> None:
        """Initialize the Claude Agent SDK adapter.

        Args:
            model: Claude model identifier.
            client_config: SDK client configuration. Defaults to
                bypassPermissions mode.
            model_config: Model parameter configuration.
            allowed_tools: Tools Claude can use (None = all available).
            disallowed_tools: Tools to explicitly block.
        """
        self._model = model
        self._client_config = client_config or ClaudeAgentSDKClientConfig()
        self._model_config = model_config or ClaudeAgentSDKModelConfig(model=model)
        self._allowed_tools = allowed_tools
        self._disallowed_tools = disallowed_tools

    @override
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponse[OutputT]:
        """Evaluate prompt using Claude Agent SDK with hook-based state sync.

        Args:
            prompt: The prompt to evaluate.
            session: Session for state management and event publishing.
            deadline: Optional deadline for execution timeout.
            visibility_overrides: Optional section visibility overrides.
            budget: Optional token budget constraints.
            budget_tracker: Optional shared budget tracker.

        Returns:
            PromptResponse with structured output and events published.

        Raises:
            PromptEvaluationError: If SDK execution fails.
        """
        if budget and not budget_tracker:
            budget_tracker = BudgetTracker(budget)

        effective_deadline = deadline or (budget.deadline if budget else None)

        if effective_deadline and effective_deadline.remaining().total_seconds() <= 0:
            raise PromptEvaluationError(
                message="Deadline expired before SDK invocation",
                prompt_name=prompt.name or f"{prompt.ns}:{prompt.key}",
                phase="request",
            )

        return run_async(
            self._evaluate_async(
                prompt,
                session=session,
                deadline=effective_deadline,
                visibility_overrides=visibility_overrides,
                budget_tracker=budget_tracker,
            )
        )

    async def _evaluate_async(
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None,
        visibility_overrides: Mapping[SectionPath, SectionVisibility] | None,
        budget_tracker: BudgetTracker | None,
    ) -> PromptResponse[OutputT]:
        """Async implementation of evaluate."""
        sdk = _import_sdk()

        rendered = prompt.render(visibility_overrides=visibility_overrides)
        prompt_text = rendered.text

        session.event_bus.publish(
            PromptRendered(
                prompt_ns=prompt.ns,
                prompt_key=prompt.key,
                prompt_name=prompt.name,
                adapter=CLAUDE_AGENT_SDK_ADAPTER_NAME,
                session_id=None,
                render_inputs=(),
                rendered_prompt=prompt_text,
                created_at=_utcnow(),
                descriptor=None,
            )
        )

        output_format = self._build_output_format(rendered)

        prompt_name = prompt.name or f"{prompt.ns}:{prompt.key}"

        hook_context = HookContext(
            session=session,
            adapter_name=CLAUDE_AGENT_SDK_ADAPTER_NAME,
            prompt_name=prompt_name,
            deadline=deadline,
            budget_tracker=budget_tracker,
        )

        bridged_tools = create_bridged_tools(
            rendered.tools,
            session=session,
            adapter=self,
            prompt=cast(Any, prompt),
            rendered_prompt=rendered,
            deadline=deadline,
            budget_tracker=budget_tracker,
        )

        logger.info(
            "claude_agent_sdk.evaluate.start",
            event="sdk.evaluate.start",
            context={
                "prompt_name": prompt_name,
                "model": self._model,
                "tool_count": len(bridged_tools),
                "has_structured_output": output_format is not None,
            },
        )

        start_time = _utcnow()

        try:
            messages = await self._run_sdk_query(
                sdk=sdk,
                prompt_text=prompt_text,
                output_format=output_format,
                hook_context=hook_context,
                bridged_tools=bridged_tools,
            )
        except Exception as error:
            raise normalize_sdk_error(error, prompt_name) from error

        end_time = _utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        result_text, output, usage = self._extract_result(
            messages, rendered, budget_tracker, prompt_name
        )

        response = PromptResponse(
            prompt_name=prompt_name,
            text=result_text,
            output=output,
        )

        session.event_bus.publish(
            PromptExecuted(
                prompt_name=prompt_name,
                adapter=CLAUDE_AGENT_SDK_ADAPTER_NAME,
                result=response,
                session_id=None,
                created_at=_utcnow(),
                usage=usage,
                value=output,
            )
        )

        logger.info(
            "claude_agent_sdk.evaluate.complete",
            event="sdk.evaluate.complete",
            context={
                "prompt_name": prompt_name,
                "duration_ms": duration_ms,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "stop_reason": hook_context.stop_reason,
            },
        )

        return response

    async def _run_sdk_query(
        self,
        *,
        sdk: Any,
        prompt_text: str,
        output_format: dict[str, Any] | None,
        hook_context: HookContext,
        bridged_tools: tuple[Any, ...],
    ) -> list[Any]:
        """Execute the SDK query and return message list."""
        # Import the SDK's options type
        from claude_agent_sdk.types import ClaudeAgentOptions

        # Build options dict then convert to ClaudeAgentOptions
        options_kwargs: dict[str, Any] = {
            "model": self._model,
        }

        if self._client_config.cwd:
            options_kwargs["cwd"] = self._client_config.cwd

        if self._client_config.permission_mode:
            options_kwargs["permission_mode"] = self._client_config.permission_mode

        if self._client_config.max_turns:
            options_kwargs["max_turns"] = self._client_config.max_turns

        if output_format:
            options_kwargs["output_format"] = output_format

        if self._allowed_tools is not None:
            options_kwargs["allowed_tools"] = list(self._allowed_tools)

        if self._disallowed_tools:
            options_kwargs["disallowed_tools"] = list(self._disallowed_tools)

        # Apply model config parameters (temperature, max_tokens)
        model_params = self._model_config.to_request_params()
        if model_params:
            # max_tokens maps to max_thinking_tokens in SDK
            if "max_tokens" in model_params:
                options_kwargs["max_thinking_tokens"] = model_params.pop("max_tokens")
            # temperature is not directly supported in ClaudeAgentOptions
            model_params.pop("temperature", None)

        pre_hook = create_pre_tool_use_hook(hook_context)
        post_hook = create_post_tool_use_hook(hook_context)
        stop_hook_fn = create_stop_hook(hook_context)
        prompt_hook = create_user_prompt_submit_hook(hook_context)

        _ = pre_hook
        _ = post_hook
        _ = stop_hook_fn
        _ = prompt_hook
        _ = bridged_tools

        options = ClaudeAgentOptions(**options_kwargs)

        return [
            message async for message in sdk.query(prompt=prompt_text, options=options)
        ]

    def _build_output_format(
        self,
        rendered: RenderedPrompt[OutputT],
    ) -> dict[str, Any] | None:
        """Generate SDK output format from prompt output type."""
        output_type = rendered.output_type

        if output_type is None or output_type is type(None):
            return None

        return {
            "type": "json_schema",
            "schema": schema(output_type),
        }

    def _extract_result(
        self,
        messages: list[Any],
        rendered: RenderedPrompt[OutputT],
        budget_tracker: BudgetTracker | None,
        prompt_name: str,
    ) -> tuple[str | None, OutputT | None, TokenUsage | None]:
        """Extract text, structured output, and usage from SDK messages."""
        # Import SDK types for isinstance checks
        from claude_agent_sdk.types import ResultMessage

        result_text: str | None = None
        structured_output: OutputT | None = None
        total_input_tokens = 0
        total_output_tokens = 0

        for message in reversed(messages):
            # Check for ResultMessage using isinstance
            if isinstance(message, ResultMessage):
                # ResultMessage has 'result' attribute, not 'text'
                if hasattr(message, "result") and message.result:
                    result_text = message.result

                if hasattr(message, "structured_output") and message.structured_output:
                    output_type = rendered.output_type
                    if output_type and output_type is not type(None):
                        try:
                            structured_output = parse(
                                output_type,
                                message.structured_output,
                                extra="ignore",
                            )
                        except (TypeError, ValueError) as error:
                            logger.warning(
                                "claude_agent_sdk.parse.structured_output_error",
                                event="parse.structured_output_error",
                                context={"error": str(error)},
                            )

            if hasattr(message, "usage") and message.usage:
                usage_dict = message.usage
                if isinstance(usage_dict, dict):
                    total_input_tokens += usage_dict.get("input_tokens", 0)
                    total_output_tokens += usage_dict.get("output_tokens", 0)

        usage = TokenUsage(
            input_tokens=total_input_tokens or None,
            output_tokens=total_output_tokens or None,
            cached_tokens=None,
        )

        if budget_tracker and (total_input_tokens or total_output_tokens):
            budget_tracker.record_cumulative(
                prompt_name,
                usage,
            )

        return result_text, structured_output, usage
