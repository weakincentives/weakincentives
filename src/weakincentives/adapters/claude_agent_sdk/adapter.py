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
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, TypeVar, cast, override
from uuid import UUID

from ...budget import Budget, BudgetTracker
from ...dataclasses import FrozenDataclass
from ...deadlines import Deadline
from ...prompt import Prompt, SectionVisibility
from ...prompt.errors import SectionPath
from ...runtime.events import PromptExecuted, PromptRendered, TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ...serde import parse, schema
from .._names import CLAUDE_AGENT_SDK_ADAPTER_NAME
from ..core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)
from ._async_utils import run_sync
from ._errors import normalize_sdk_error
from .config import ClaudeAgentSDKClientConfig, ClaudeAgentSDKModelConfig

if TYPE_CHECKING:
    from ...prompt.rendering import RenderedPrompt

__all__ = [
    "CLAUDE_AGENT_SDK_ADAPTER_NAME",
    "ClaudeAgentSDKAdapter",
]


logger: StructuredLogger = get_logger(
    __name__, context={"component": "claude_agent_sdk.adapter"}
)

OutputT = TypeVar("OutputT")

_DEFAULT_MODEL: Final[str] = "claude-sonnet-4-5-20250929"


@FrozenDataclass()
class _EvaluationContext[OutputT]:
    """Internal context for prompt evaluation."""

    prompt_name: str
    render_inputs: tuple[Any, ...]
    rendered: RenderedPrompt[OutputT]
    output_format: dict[str, Any] | None


class ClaudeAgentSDKAdapter(ProviderAdapter[Any]):
    """Adapter using Claude Agent SDK with hook-based state synchronization.

    This adapter delegates tool execution to Claude Code via the official
    claude-agent-sdk Python package. It uses the SDK's Hook system to
    synchronize state bidirectionally between SDK execution and the
    weakincentives Session.

    Args:
        model: Claude model identifier.
        client_config: SDK client configuration.
        model_config: Model parameter configuration.
        allowed_tools: Tools Claude can use (None = all available).
        disallowed_tools: Tools to explicitly block.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        client_config: ClaudeAgentSDKClientConfig | None = None,
        model_config: ClaudeAgentSDKModelConfig | None = None,
        allowed_tools: tuple[str, ...] | None = None,
        disallowed_tools: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._model = model
        self._client_config = client_config or ClaudeAgentSDKClientConfig()
        self._model_config = model_config
        self._allowed_tools = allowed_tools
        self._disallowed_tools = disallowed_tools

        # Evaluation state (set during evaluate)
        self._stop_reason: str | None = None

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
            deadline: Optional execution deadline.
            visibility_overrides: Optional section visibility overrides.
            budget: Optional budget limits.
            budget_tracker: Optional shared budget tracker.

        Returns:
            PromptResponse with text and structured output.

        Raises:
            PromptEvaluationError: If evaluation fails.
        """
        # Setup evaluation context
        context = self._setup_evaluation(
            prompt,
            deadline=deadline,
            visibility_overrides=visibility_overrides,
        )

        # Create tracker if budget provided but tracker not supplied
        effective_tracker = budget_tracker
        if effective_tracker is None and budget is not None:
            effective_tracker = BudgetTracker(budget=budget)

        # Run async evaluation synchronously
        return run_sync(
            self._evaluate_async(
                prompt=prompt,
                session=session,
                context=context,
                deadline=deadline,
                budget_tracker=effective_tracker,
            )
        )

    async def _evaluate_async(
        self,
        *,
        prompt: Prompt[OutputT],
        session: SessionProtocol,
        context: _EvaluationContext[OutputT],
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
    ) -> PromptResponse[OutputT]:
        """Async implementation of prompt evaluation."""
        prompt_name = context.prompt_name
        rendered = context.rendered

        # Publish PromptRendered event
        session_id = cast(UUID | None, getattr(session, "session_id", None))
        _ = session.event_bus.publish(
            PromptRendered(
                prompt_ns=prompt.ns,
                prompt_key=prompt.key,
                prompt_name=prompt_name,
                adapter=CLAUDE_AGENT_SDK_ADAPTER_NAME,
                session_id=session_id,
                render_inputs=context.render_inputs,
                rendered_prompt=rendered.text,
                created_at=datetime.now(UTC),
            )
        )

        logger.info(
            "claude_agent_sdk.evaluate.start",
            event="sdk.evaluate.start",
            context={
                "prompt_name": prompt_name,
                "model": self._model,
                "tool_count": len(rendered.tools),
                "has_structured_output": context.output_format is not None,
            },
        )

        start_time = datetime.now(UTC)

        try:
            # Import SDK lazily to avoid import errors when not installed
            from claude_agent_sdk import ClaudeSDKClient
            from claude_agent_sdk.types import ClaudeAgentOptions

            # Build hooks for state synchronization
            hooks = self._build_hooks(
                session=session,
                deadline=deadline,
                budget_tracker=budget_tracker,
                prompt_name=prompt_name,
            )

            # Build client config kwargs
            client_kwargs = self._client_config.to_client_kwargs()

            # Construct options with all settings
            options = ClaudeAgentOptions(
                model=self._model,
                hooks=hooks,  # type: ignore[arg-type]
                permission_mode=client_kwargs.get("permission_mode"),
                cwd=client_kwargs.get("cwd"),
                add_dirs=client_kwargs.get("add_dirs", []),
                env=client_kwargs.get("env", {}),
                max_turns=client_kwargs.get("max_turns"),
                include_partial_messages=client_kwargs.get(
                    "include_partial_messages", False
                ),
                allowed_tools=list(self._allowed_tools)
                if self._allowed_tools is not None
                else [],
                disallowed_tools=list(self._disallowed_tools)
                if self._disallowed_tools
                else [],
                output_format=context.output_format,
            )

            # Create client and run query
            result: object | None = None
            async with ClaudeSDKClient(options) as client:
                # Execute query with rendered prompt
                result = await client.query(rendered.text)

            end_time = datetime.now(UTC)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            # Extract result message and structured output
            text_result: str | None = None
            output: OutputT | None = None
            usage: TokenUsage | None = None

            if result is not None:
                # Extract text from result
                text_result = self._extract_text_result(result)

                # Extract usage
                result_usage = getattr(result, "usage", None)
                if result_usage is not None:
                    usage = TokenUsage(
                        input_tokens=result_usage.get("input_tokens"),
                        output_tokens=result_usage.get("output_tokens"),
                    )

                # Record usage to budget tracker
                if budget_tracker is not None and usage is not None:
                    evaluation_id = f"{prompt_name}:{start_time.isoformat()}"
                    budget_tracker.record_cumulative(evaluation_id, usage)

                # Extract structured output if expected
                if context.output_format is not None:
                    output = self._extract_structured_output(
                        result,
                        prompt.template.output_type
                        if hasattr(prompt.template, "output_type")
                        else None,
                    )

            logger.info(
                "claude_agent_sdk.evaluate.complete",
                event="sdk.evaluate.complete",
                context={
                    "prompt_name": prompt_name,
                    "duration_ms": duration_ms,
                    "input_tokens": usage.input_tokens if usage else None,
                    "output_tokens": usage.output_tokens if usage else None,
                    "stop_reason": self._stop_reason,
                },
            )

            # Publish PromptExecuted event
            _ = session.event_bus.publish(
                PromptExecuted(
                    prompt_name=prompt_name,
                    adapter=CLAUDE_AGENT_SDK_ADAPTER_NAME,
                    result=text_result,
                    session_id=session_id,
                    created_at=datetime.now(UTC),
                    usage=usage,
                    value=output,
                )
            )

            return PromptResponse(
                prompt_name=prompt_name,
                text=text_result,
                output=output,
            )

        except ImportError as exc:
            raise PromptEvaluationError(
                message=(
                    "Claude Agent SDK not installed. "
                    "Install: pip install 'weakincentives[claude-agent-sdk]'"
                ),
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ) from exc
        except Exception as exc:
            # Check if it's an SDK-specific error
            error_type = type(exc).__name__
            if error_type in {
                "CLINotFoundError",
                "CLIConnectionError",
                "ProcessError",
                "CLIJSONDecodeError",
            }:
                raise normalize_sdk_error(exc, prompt_name=prompt_name) from exc

            # Re-raise other exceptions
            raise PromptEvaluationError(
                message=str(exc),
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ) from exc

    def _setup_evaluation(
        self,
        prompt: Prompt[OutputT],
        *,
        deadline: Deadline | None,
        visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
    ) -> _EvaluationContext[OutputT]:
        """Setup evaluation context from prompt."""
        prompt_name = prompt.name or prompt.template.__class__.__name__

        # Check deadline not already expired
        self._ensure_deadline_not_expired(deadline, prompt_name)

        # Render prompt
        rendered = prompt.render(visibility_overrides=visibility_overrides)

        # Build output format for structured output
        output_format = self._build_output_format(prompt)

        return _EvaluationContext(
            prompt_name=prompt_name,
            render_inputs=prompt.params,
            rendered=rendered,
            output_format=output_format,
        )

    @staticmethod
    def _ensure_deadline_not_expired(
        deadline: Deadline | None,
        prompt_name: str,
    ) -> None:
        """Raise if deadline is already expired."""
        if deadline is not None and deadline.remaining() <= timedelta(0):
            raise PromptEvaluationError(
                "Deadline expired before evaluation started.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
                provider_payload={
                    "deadline_at": deadline.expires_at.isoformat()
                    if deadline.expires_at
                    else None,
                },
            )

    @staticmethod
    def _build_output_format(
        prompt: Prompt[OutputT],
    ) -> dict[str, Any] | None:
        """Generate SDK output format from prompt output type."""
        # Get output type from prompt template's structured output config
        structured_output = getattr(prompt.template, "_structured_output", None)
        if structured_output is None:
            return None

        output_type = getattr(structured_output, "output_type", None)
        if output_type is None or output_type is type(None):
            return None

        return {
            "type": "json_schema",
            "schema": schema(output_type),
        }

    def _build_hooks(
        self,
        *,
        session: SessionProtocol,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        prompt_name: str,
    ) -> dict[str, list[Any]]:
        """Construct hook configuration for state synchronization."""
        from claude_agent_sdk.types import HookMatcher

        from ._hooks import (
            build_post_tool_use_hook,
            build_pre_tool_use_hook,
            build_stop_hook,
            build_user_prompt_submit_hook,
        )

        def on_stop(reason: str) -> None:
            self._stop_reason = reason

        pre_tool_use = build_pre_tool_use_hook(
            session=session,
            deadline=deadline,
            budget_tracker=budget_tracker,
            adapter_name=CLAUDE_AGENT_SDK_ADAPTER_NAME,
            prompt_name=prompt_name,
        )

        post_tool_use = build_post_tool_use_hook(
            session=session,
            adapter_name=CLAUDE_AGENT_SDK_ADAPTER_NAME,
            prompt_name=prompt_name,
        )

        user_prompt_submit = build_user_prompt_submit_hook(
            session=session,
            adapter_name=CLAUDE_AGENT_SDK_ADAPTER_NAME,
            prompt_name=prompt_name,
        )

        stop_hook = build_stop_hook(
            session=session,
            adapter_name=CLAUDE_AGENT_SDK_ADAPTER_NAME,
            prompt_name=prompt_name,
            on_stop=on_stop,
        )

        # Return hook matchers in SDK format
        return {
            "PreToolUse": [HookMatcher(hooks=[pre_tool_use], timeout=30.0)],
            "PostToolUse": [HookMatcher(hooks=[post_tool_use], timeout=30.0)],
            "UserPromptSubmit": [HookMatcher(hooks=[user_prompt_submit], timeout=10.0)],
            "Stop": [HookMatcher(hooks=[stop_hook], timeout=10.0)],
        }

    @staticmethod
    def _extract_text_result(result: object) -> str | None:
        """Extract text from SDK result."""
        # Try common attributes
        if hasattr(result, "text"):
            text = getattr(result, "text")
            if isinstance(text, str):
                return text

        if hasattr(result, "content"):
            content = getattr(result, "content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Concatenate text blocks
                texts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        item_dict = cast(dict[str, object], item)
                        if item_dict.get("type") == "text":
                            text_val = item_dict.get("text", "")
                            if isinstance(text_val, str):
                                texts.append(text_val)
                    elif hasattr(item, "text"):
                        texts.append(str(getattr(item, "text")))
                return "".join(texts) if texts else None

        return None

    @staticmethod
    def _extract_structured_output(
        result: object,
        output_type: type[OutputT] | None,
    ) -> OutputT | None:
        """Parse structured output from SDK result."""
        if output_type is None:
            return None

        # Check for structured_output attribute
        structured = getattr(result, "structured_output", None)
        if structured is not None:
            try:
                return cast(OutputT, parse(output_type, structured, extra="ignore"))
            except Exception:
                logger.warning(
                    "claude_agent_sdk.structured_output.parse_error",
                    event="sdk.structured_output.parse_error",
                    context={"output_type": output_type.__name__},
                )

        return None
