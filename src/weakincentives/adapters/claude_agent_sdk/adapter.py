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

from datetime import UTC, datetime
from typing import Any, TypeVar, cast, override

from ...budget import Budget, BudgetTracker
from ...deadlines import Deadline
from ...prompt import Prompt, RenderedPrompt
from ...prompt.errors import VisibilityExpansionRequired
from ...runtime.events import PromptExecuted, PromptRendered
from ...runtime.events._types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session import append
from ...runtime.session.protocols import SessionProtocol
from ...serde import parse, schema
from .._names import AdapterName
from ..core import PromptEvaluationError, PromptResponse, ProviderAdapter
from ._async_utils import run_async
from ._bridge import create_bridged_tools, create_mcp_server
from ._client import ClientConfig, ClientSession, QueryResult
from ._errors import normalize_sdk_error
from ._hooks import (
    HookContext,
    create_notification_hook,
    create_post_tool_use_hook,
    create_pre_compact_hook,
    create_pre_tool_use_hook,
    create_stop_hook,
    create_subagent_start_hook,
    create_subagent_stop_hook,
    create_user_prompt_submit_hook,
)
from ._notifications import Notification
from .config import ClaudeAgentSDKClientConfig, ClaudeAgentSDKModelConfig
from .isolation import EphemeralHome

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


def _import_sdk_types() -> Any:  # pragma: no cover
    """Import SDK types for hook registration."""
    try:
        from claude_agent_sdk.types import HookMatcher

        return HookMatcher
    except ImportError as error:
        raise ImportError(
            "claude-agent-sdk is not installed. Install it with: "
            "pip install 'weakincentives[claude-agent-sdk]'"
        ) from error


class ClaudeAgentSDKAdapter(ProviderAdapter[OutputT]):
    """Adapter using Claude Agent SDK with hook-based state synchronization.

    This adapter uses the Claude Agent SDK's ClaudeSDKClient to execute
    prompts with full agentic capabilities. The client-based approach
    enables:

    - Real cancellation via interrupt()
    - Streaming progress through receive_response()
    - Clean lifecycle management with connect/disconnect
    - Session continuity across multiple queries

    Hooks synchronize state bidirectionally between the SDK's internal
    execution and the weakincentives Session.

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
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass(frozen=True)
        ... class TaskResult:
        ...     message: str
        ...
        >>> template = PromptTemplate[TaskResult](
        ...     ns="test",
        ...     key="hello",
        ...     sections=(
        ...         MarkdownSection(
        ...             title="Task",
        ...             key="task",
        ...             template="Say hello",
        ...         ),
        ...     ),
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
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponse[OutputT]:
        """Evaluate prompt using Claude Agent SDK with hook-based state sync.

        Visibility overrides are managed exclusively via Session state using the
        VisibilityOverrides state slice. Use session.mutate(VisibilityOverrides)
        to set visibility overrides before calling evaluate().

        Args:
            prompt: The prompt to evaluate.
            session: Session for state management and event publishing.
            deadline: Optional deadline for execution timeout.
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
                budget_tracker=budget_tracker,
            )
        )

    async def _evaluate_async(
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
    ) -> PromptResponse[OutputT]:
        """Async implementation of evaluate using ClaudeSDKClient."""
        rendered = prompt.render(
            session=session,
        )
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

        # Register Notification reducer if not already registered
        session.mutate(Notification).register(Notification, append)

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
            adapter_name=CLAUDE_AGENT_SDK_ADAPTER_NAME,
            prompt_name=prompt_name,
        )

        logger.info(
            "claude_agent_sdk.evaluate.start",
            event="sdk.evaluate.start",
            context={
                "prompt_name": prompt_name,
                "model": self._model,
                "tool_count": len(bridged_tools),
                "has_structured_output": output_format is not None,
                "isolated": self._client_config.isolation is not None,
            },
        )

        start_time = _utcnow()

        # Create ephemeral home for isolation if configured
        ephemeral_home: EphemeralHome | None = None
        if self._client_config.isolation:
            ephemeral_home = EphemeralHome(
                self._client_config.isolation,
                workspace_path=self._client_config.cwd,
            )

        try:
            query_result = await self._run_client_query(
                prompt_text=prompt_text,
                output_format=output_format,
                hook_context=hook_context,
                bridged_tools=bridged_tools,
                ephemeral_home=ephemeral_home,
                deadline=deadline,
                budget_tracker=budget_tracker,
            )
        except VisibilityExpansionRequired:
            # Progressive disclosure: let this propagate to the caller
            raise
        except Exception as error:
            raise normalize_sdk_error(error, prompt_name) from error
        finally:
            # Always clean up ephemeral home
            if ephemeral_home:
                ephemeral_home.cleanup()

        end_time = _utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Update hook context stop reason from query result
        if query_result.stop_reason:
            hook_context.stop_reason = query_result.stop_reason

        result_text, output, usage = self._extract_result(
            query_result, rendered, budget_tracker, prompt_name
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
                "interrupted": query_result.interrupted,
            },
        )

        return response

    async def _run_client_query(
        self,
        *,
        prompt_text: str,
        output_format: dict[str, Any] | None,
        hook_context: HookContext,
        bridged_tools: tuple[Any, ...],
        ephemeral_home: EphemeralHome | None = None,
        deadline: Deadline | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> QueryResult:
        """Execute query using ClaudeSDKClient and return result."""
        # Build client configuration
        client_config = self._build_client_config(
            output_format=output_format,
            hook_context=hook_context,
            bridged_tools=bridged_tools,
            ephemeral_home=ephemeral_home,
        )

        # Create and use client session
        async with ClientSession(
            client_config,
            hook_context,
            deadline=deadline,
            budget_tracker=budget_tracker,
        ) as client:
            return await client.query(
                prompt_text,
                session_id=hook_context.prompt_name,
            )

    def _build_client_config(
        self,
        *,
        output_format: dict[str, Any] | None,
        hook_context: HookContext,
        bridged_tools: tuple[Any, ...],
        ephemeral_home: EphemeralHome | None = None,
    ) -> ClientConfig:
        """Build ClientConfig from adapter configuration."""
        HookMatcher = _import_sdk_types()

        # Build hooks
        pre_hook = create_pre_tool_use_hook(hook_context)
        post_hook = create_post_tool_use_hook(
            hook_context,
            stop_on_structured_output=self._client_config.stop_on_structured_output,
        )
        stop_hook_fn = create_stop_hook(hook_context)
        prompt_hook = create_user_prompt_submit_hook(hook_context)
        subagent_start_hook = create_subagent_start_hook(hook_context)
        subagent_stop_hook = create_subagent_stop_hook(hook_context)
        pre_compact_hook = create_pre_compact_hook(hook_context)
        notification_hook = create_notification_hook(hook_context)

        hooks: dict[str, list[Any]] = {
            "PreToolUse": [HookMatcher(matcher=None, hooks=[pre_hook])],
            "PostToolUse": [HookMatcher(matcher=None, hooks=[post_hook])],
            "Stop": [HookMatcher(matcher=None, hooks=[stop_hook_fn])],
            "UserPromptSubmit": [HookMatcher(matcher=None, hooks=[prompt_hook])],
            "SubagentStart": [HookMatcher(matcher=None, hooks=[subagent_start_hook])],
            "SubagentStop": [HookMatcher(matcher=None, hooks=[subagent_stop_hook])],
            "PreCompact": [HookMatcher(matcher=None, hooks=[pre_compact_hook])],
            "Notification": [HookMatcher(matcher=None, hooks=[notification_hook])],
        }

        # Build MCP servers config
        mcp_servers: dict[str, Any] | None = None
        if bridged_tools:
            mcp_server_config = create_mcp_server(bridged_tools)
            mcp_servers = {"wink": mcp_server_config}

        # Build env and setting_sources from ephemeral home
        env: dict[str, str] | None = None
        setting_sources: list[str] | None = None
        if ephemeral_home:
            env = ephemeral_home.get_env()
            setting_sources = ephemeral_home.get_setting_sources()

        return ClientConfig(
            model=self._model,
            cwd=self._client_config.cwd,
            permission_mode=self._client_config.permission_mode,
            max_turns=self._client_config.max_turns,
            output_format=output_format,
            allowed_tools=self._allowed_tools,
            disallowed_tools=self._disallowed_tools,
            suppress_stderr=self._client_config.suppress_stderr,
            env=env,
            setting_sources=setting_sources,
            mcp_servers=mcp_servers,
            hooks=hooks,
        )

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
        query_result: QueryResult,
        rendered: RenderedPrompt[OutputT],
        budget_tracker: BudgetTracker | None,
        prompt_name: str,
    ) -> tuple[str | None, OutputT | None, TokenUsage | None]:
        """Extract text, structured output, and usage from QueryResult."""
        result_text = query_result.result_text
        structured_output: OutputT | None = None

        # Parse structured output if present
        if query_result.structured_output:
            output_type = rendered.output_type
            if output_type and output_type is not type(None):
                try:
                    structured_output = parse(
                        output_type,
                        query_result.structured_output,
                        extra="ignore",
                    )
                except (TypeError, ValueError) as error:
                    logger.warning(
                        "claude_agent_sdk.parse.structured_output_error",
                        event="parse.structured_output_error",
                        context={"error": str(error)},
                    )

        usage = TokenUsage(
            input_tokens=query_result.input_tokens or None,
            output_tokens=query_result.output_tokens or None,
            cached_tokens=None,
        )

        if budget_tracker and (query_result.input_tokens or query_result.output_tokens):
            budget_tracker.record_cumulative(
                prompt_name,
                usage,
            )

        return result_text, structured_output, usage
