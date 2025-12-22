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
from pathlib import Path
from typing import Any, TypeVar, cast, override

from ...budget import Budget, BudgetTracker
from ...contrib.tools.filesystem import Filesystem
from ...contrib.tools.filesystem_host import HostFilesystem
from ...deadlines import Deadline
from ...prompt import Prompt, RenderedPrompt
from ...prompt.errors import VisibilityExpansionRequired
from ...prompt.tool import ResourceRegistry
from ...runtime.events import PromptExecuted, PromptRendered
from ...runtime.events._types import TokenUsage
from ...runtime.execution_state import ExecutionState
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session import append_all
from ...runtime.session.protocols import SessionProtocol
from ...serde import parse, schema
from .._names import AdapterName
from ..core import PromptEvaluationError, PromptResponse, ProviderAdapter
from ._async_utils import run_async
from ._bridge import create_bridged_tools, create_mcp_server
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
        resources: ResourceRegistry | None = None,
    ) -> PromptResponse[OutputT]:
        """Evaluate prompt using Claude Agent SDK with hook-based state sync.

        Visibility overrides are managed exclusively via Session state using the
        VisibilityOverrides state slice. Use session[VisibilityOverrides]
        to set visibility overrides before calling evaluate().

        Args:
            prompt: The prompt to evaluate.
            session: Session for state management and event publishing.
            deadline: Optional deadline for execution timeout.
            budget: Optional token budget constraints.
            budget_tracker: Optional shared budget tracker.
            resources: Optional resources to inject (merged with workspace resources,
                user-provided resources take precedence).

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
                resources=resources,
            )
        )

    async def _evaluate_async(  # noqa: PLR0914
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        resources: ResourceRegistry | None,
    ) -> PromptResponse[OutputT]:
        """Async implementation of evaluate."""
        sdk = _import_sdk()

        rendered = prompt.render(
            session=session,
        )
        prompt_text = rendered.text

        session.dispatcher.dispatch(
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
        session[Notification].register(Notification, append_all)

        # Get filesystem from workspace section if present, otherwise create one
        # from the working directory. This ensures MCP-bridged tools operate on
        # the same filesystem as the SDK's native tools.
        filesystem = prompt.filesystem()
        if filesystem is None:
            workspace_root = self._client_config.cwd or str(Path.cwd())
            filesystem = HostFilesystem(_root=workspace_root)

        # Create ExecutionState for transactional tool execution
        # Build workspace resources from prompt, then merge with user-provided resources
        # Both bridged MCP tools and native SDK tools will use this for rollback
        workspace_resources = ResourceRegistry.build({Filesystem: filesystem})
        effective_resources = (
            workspace_resources.merge(resources)
            if resources is not None
            else workspace_resources
        )
        execution_state = ExecutionState(session=session, resources=effective_resources)

        # Add execution_state to hook context for native tool transactions
        hook_context = HookContext(
            execution_state=execution_state,
            adapter_name=CLAUDE_AGENT_SDK_ADAPTER_NAME,
            prompt_name=prompt_name,
            deadline=deadline,
            budget_tracker=budget_tracker,
        )

        bridged_tools = create_bridged_tools(
            rendered.tools,
            execution_state=execution_state,
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
            messages = await self._run_sdk_query(
                sdk=sdk,
                prompt_text=prompt_text,
                output_format=output_format,
                hook_context=hook_context,
                bridged_tools=bridged_tools,
                ephemeral_home=ephemeral_home,
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

        result_text, output, usage = self._extract_result(
            messages, rendered, budget_tracker, prompt_name
        )

        response = PromptResponse(
            prompt_name=prompt_name,
            text=result_text,
            output=output,
        )

        session.dispatcher.dispatch(
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
            },
        )

        return response

    async def _run_sdk_query(  # noqa: C901
        self,
        *,
        sdk: Any,
        prompt_text: str,
        output_format: dict[str, Any] | None,
        hook_context: HookContext,
        bridged_tools: tuple[Any, ...],
        ephemeral_home: EphemeralHome | None = None,
    ) -> list[Any]:
        """Execute the SDK query and return message list."""
        # Import the SDK's types
        from claude_agent_sdk.types import ClaudeAgentOptions, HookMatcher

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

        if self._client_config.max_budget_usd is not None:
            options_kwargs["max_budget_usd"] = self._client_config.max_budget_usd

        if self._client_config.betas:
            options_kwargs["betas"] = list(self._client_config.betas)

        if output_format:
            options_kwargs["output_format"] = output_format

        if self._allowed_tools is not None:
            options_kwargs["allowed_tools"] = list(self._allowed_tools)

        if self._disallowed_tools:
            options_kwargs["disallowed_tools"] = list(self._disallowed_tools)

        # Apply isolation configuration if ephemeral home is provided
        if ephemeral_home:
            # Set environment variables including redirected HOME
            options_kwargs["env"] = ephemeral_home.get_env()
            # Prevent loading any external settings
            options_kwargs["setting_sources"] = ephemeral_home.get_setting_sources()

        # Apply model config parameters
        # Note: The Claude Agent SDK does not expose max_tokens or temperature
        # parameters directly. It manages token budgets internally.
        if self._model_config.max_thinking_tokens is not None:
            options_kwargs["max_thinking_tokens"] = (
                self._model_config.max_thinking_tokens
            )

        # Register custom tools via MCP server if any are provided
        if bridged_tools:
            # create_mcp_server returns an McpSdkServerConfig directly
            mcp_server_config = create_mcp_server(bridged_tools)
            options_kwargs["mcp_servers"] = {
                "wink": mcp_server_config,
            }

        # Suppress stderr if configured (hides bun errors and CLI noise)
        if self._client_config.suppress_stderr:
            options_kwargs["stderr"] = lambda _: None

        # Create async hook callbacks
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

        # Build hooks dict with HookMatcher wrappers
        # matcher=None matches all tools
        options_kwargs["hooks"] = {
            "PreToolUse": [HookMatcher(matcher=None, hooks=[pre_hook])],
            "PostToolUse": [HookMatcher(matcher=None, hooks=[post_hook])],
            "Stop": [HookMatcher(matcher=None, hooks=[stop_hook_fn])],
            "UserPromptSubmit": [HookMatcher(matcher=None, hooks=[prompt_hook])],
            "SubagentStart": [HookMatcher(matcher=None, hooks=[subagent_start_hook])],
            "SubagentStop": [HookMatcher(matcher=None, hooks=[subagent_stop_hook])],
            "PreCompact": [HookMatcher(matcher=None, hooks=[pre_compact_hook])],
            "Notification": [HookMatcher(matcher=None, hooks=[notification_hook])],
        }

        options = ClaudeAgentOptions(**options_kwargs)

        # Use streaming mode (AsyncIterable) to enable hook support.
        # The SDK's query() function only initializes hooks when
        # is_streaming_mode=True, which requires an AsyncIterable prompt.
        async def stream_prompt() -> Any:  # noqa: RUF029
            """Yield a single user message in streaming format."""
            yield {
                "type": "user",
                "message": {"role": "user", "content": prompt_text},
                "parent_tool_use_id": None,
                "session_id": hook_context.prompt_name,
            }

        return [
            message
            async for message in sdk.query(prompt=stream_prompt(), options=options)
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

    def _try_parse_structured_output(
        self,
        message: Any,
        rendered: RenderedPrompt[OutputT],
    ) -> OutputT | None:
        """Attempt to parse structured output from a message."""
        if not (hasattr(message, "structured_output") and message.structured_output):
            return None
        output_type = rendered.output_type
        if not output_type or output_type is type(None):
            return None  # pragma: no cover - defensive for prompts without output type
        try:
            return parse(output_type, message.structured_output, extra="ignore")
        except (TypeError, ValueError) as error:
            logger.warning(
                "claude_agent_sdk.parse.structured_output_error",
                event="parse.structured_output_error",
                context={"error": str(error)},
            )
            return None

    def _extract_result(
        self,
        messages: list[Any],
        rendered: RenderedPrompt[OutputT],
        budget_tracker: BudgetTracker | None,
        prompt_name: str,
    ) -> tuple[str | None, OutputT | None, TokenUsage | None]:
        """Extract text, structured output, and usage from SDK messages."""
        from claude_agent_sdk.types import ResultMessage

        result_text: str | None = None
        structured_output: OutputT | None = None
        total_input_tokens = 0
        total_output_tokens = 0

        for message in reversed(messages):
            if isinstance(message, ResultMessage):
                if hasattr(message, "result") and message.result:
                    result_text = message.result
                structured_output = self._try_parse_structured_output(message, rendered)

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
            budget_tracker.record_cumulative(prompt_name, usage)

        return result_text, structured_output, usage
