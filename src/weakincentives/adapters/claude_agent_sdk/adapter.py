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

import contextlib
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast, override

from ...budget import Budget, BudgetTracker
from ...deadlines import Deadline
from ...filesystem import Filesystem, HostFilesystem
from ...prompt import Prompt, RenderedPrompt
from ...prompt.errors import VisibilityExpansionRequired
from ...prompt.protocols import PromptProtocol
from ...runtime.events import PromptExecuted, PromptRendered
from ...runtime.events.types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from ...runtime.watchdog import Heartbeat
from ...serde import parse, schema
from ...types import AdapterName
from ..core import PromptEvaluationError, PromptResponse, ProviderAdapter
from ._async_utils import run_async
from ._bridge import create_bridged_tools, create_mcp_server
from ._errors import normalize_sdk_error
from ._hook_registry import HookRegistry
from ._hooks import HookContext
from ._log_aggregator import ClaudeLogAggregator
from ._query_builder import SdkQueryBuilder
from ._task_completion import TaskCompletionContext
from ._visibility_signal import VisibilityExpansionSignal
from .config import ClaudeAgentSDKClientConfig, ClaudeAgentSDKModelConfig
from .isolation import EphemeralHome, IsolationConfig

__all__ = [
    "CLAUDE_AGENT_SDK_ADAPTER_NAME",
    "ClaudeAgentSDKAdapter",
]

logger: StructuredLogger = get_logger(
    __name__, context={"component": "claude_agent_sdk"}
)

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


def _extract_content_block(block: dict[str, Any]) -> dict[str, Any]:
    """Extract full content from a single content block for logging."""
    block_type = block.get("type")
    result: dict[str, Any] = {"type": block_type}

    if block_type == "tool_use":
        result["name"] = block.get("name", "unknown")
        result["id"] = block.get("id", "")
        # Include full input for complete tracing
        if "input" in block:
            result["input"] = block["input"]
    elif block_type == "text":
        result["text"] = block.get("text", "")
    elif block_type == "tool_result":
        result["tool_use_id"] = block.get("tool_use_id", "")
        result["content"] = block.get("content", "")
        if "is_error" in block:
            result["is_error"] = block["is_error"]
    else:
        # For other types, include the whole block
        result.update(block)

    return result


def _extract_list_content(content: list[Any]) -> list[dict[str, Any]]:
    """Extract full content from content block list."""
    return [
        _extract_content_block(block) for block in content if isinstance(block, dict)
    ]


def _extract_inner_message_content(inner_msg: dict[str, Any]) -> dict[str, Any]:
    """Extract full content from the inner message dict."""
    result: dict[str, Any] = {}
    role = inner_msg.get("role")
    if role:
        result["role"] = role
    content = inner_msg.get("content")
    if isinstance(content, str):
        result["content"] = content
    elif isinstance(content, list):
        result["content_blocks"] = _extract_list_content(content)
    return result


def _extract_message_content(message: Any) -> dict[str, Any]:
    """Extract full content from an SDK message for debug logging."""
    result: dict[str, Any] = {}

    # Try to get the inner message dict (common pattern in SDK messages)
    inner_msg = getattr(message, "message", None)
    if isinstance(inner_msg, dict):
        result.update(_extract_inner_message_content(inner_msg))

    # ResultMessage specific: extract the full result field
    sdk_result = getattr(message, "result", None)
    if sdk_result and isinstance(sdk_result, str):
        result["result"] = sdk_result

    # Structured output - include full content
    structured_output = getattr(message, "structured_output", None)
    if structured_output:
        result["structured_output"] = structured_output

    # Include usage if present
    usage = getattr(message, "usage", None)
    if usage:
        result["usage"] = usage if isinstance(usage, dict) else str(usage)
        # Extract thinking tokens for extended thinking mode
        if isinstance(usage, dict):
            result["input_tokens"] = usage.get("input_tokens")
            result["output_tokens"] = usage.get("output_tokens")
            # Check for cache-related fields
            result["cache_read_input_tokens"] = usage.get("cache_read_input_tokens")
            result["cache_creation_input_tokens"] = usage.get(
                "cache_creation_input_tokens"
            )

    # Extract thinking content if present (extended thinking mode)
    thinking = getattr(message, "thinking", None)
    if thinking:
        result["has_thinking"] = True
        if isinstance(thinking, str):  # pragma: no cover
            result["thinking_preview"] = thinking[:200]
            result["thinking_length"] = len(thinking)

    return result


class ClaudeAgentSDKAdapter[OutputT](ProviderAdapter[OutputT]):
    """Adapter using Claude Agent SDK with hook-based state synchronization.

    This adapter uses the Claude Agent SDK's ClaudeSDKClient to execute
    prompts with full agentic capabilities. Hooks synchronize state
    bidirectionally between the SDK's internal execution and the
    weakincentives Session.

    Example:
        >>> from weakincentives import Prompt, PromptTemplate, MarkdownSection
        >>> from weakincentives.runtime import Session, InProcessDispatcher
        >>> from weakincentives.adapters.claude_agent_sdk import (
        ...     ClaudeAgentSDKAdapter,
        ...     ClaudeAgentSDKClientConfig,
        ... )
        >>>
        >>> bus = InProcessDispatcher()
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
        # Buffer for capturing stderr output for debug logging
        self._stderr_buffer: list[str] = []

        logger.debug(
            "claude_agent_sdk.adapter.init",
            event="adapter.init",
            context={
                "model": model,
                "permission_mode": self._client_config.permission_mode,
                "cwd": self._client_config.cwd,
                "max_turns": self._client_config.max_turns,
                "max_budget_usd": self._client_config.max_budget_usd,
                "suppress_stderr": self._client_config.suppress_stderr,
                "stop_on_structured_output": self._client_config.stop_on_structured_output,
                "has_isolation_config": self._client_config.isolation is not None,
                "allowed_tools": allowed_tools,
                "disallowed_tools": disallowed_tools,
                "max_thinking_tokens": self._model_config.max_thinking_tokens,
            },
        )

    @override
    def evaluate(
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
        """Evaluate prompt using Claude Agent SDK with hook-based state sync.

        Visibility overrides are managed exclusively via Session state using the
        VisibilityOverrides state slice. Use session[VisibilityOverrides]
        to set visibility overrides before calling evaluate().

        Resources should be bound to the prompt via ``prompt.bind(resources=...)``
        before calling evaluate(). The adapter will merge workspace resources
        (filesystem) with any pre-bound resources.

        When ``heartbeat`` is provided, the adapter beats at key execution
        points (tool calls via hooks) to prove liveness. Bridged tool handlers
        receive the heartbeat via ToolContext.beat().

        Args:
            prompt: The prompt to evaluate.
            session: Session for state management and event dispatching.
            deadline: Optional deadline for execution timeout.
            budget: Optional token budget constraints.
            budget_tracker: Optional shared budget tracker.
            heartbeat: Optional heartbeat for lease extension.

        Returns:
            PromptResponse with structured output and events dispatched.

        Raises:
            PromptEvaluationError: If SDK execution fails.
        """
        if budget and not budget_tracker:
            budget_tracker = BudgetTracker(budget)

        effective_deadline = deadline or (budget.deadline if budget else None)

        prompt_name = prompt.name or f"{prompt.ns}:{prompt.key}"

        logger.debug(
            "claude_agent_sdk.evaluate.entry",
            event="evaluate.entry",
            context={
                "prompt_name": prompt_name,
                "prompt_ns": prompt.ns,
                "prompt_key": prompt.key,
                "has_deadline": effective_deadline is not None,
                "deadline_remaining_seconds": (
                    effective_deadline.remaining().total_seconds()
                    if effective_deadline
                    else None
                ),
                "has_budget": budget is not None,
                "has_budget_tracker": budget_tracker is not None,
                "has_heartbeat": heartbeat is not None,
            },
        )

        if effective_deadline and effective_deadline.remaining().total_seconds() <= 0:
            logger.debug(
                "claude_agent_sdk.evaluate.deadline_expired",
                event="evaluate.deadline_expired",
                context={"prompt_name": prompt_name},
            )
            raise PromptEvaluationError(
                message="Deadline expired before SDK invocation",
                prompt_name=prompt_name,
                phase="request",
            )

        return run_async(
            self._evaluate_async(
                prompt,
                session=session,
                deadline=effective_deadline,
                budget_tracker=budget_tracker,
                heartbeat=heartbeat,
                run_context=run_context,
            )
        )

    async def _evaluate_async(
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        heartbeat: Heartbeat | None,
        run_context: RunContext | None,
    ) -> PromptResponse[OutputT]:
        """Async implementation of evaluate."""
        sdk = _import_sdk()

        # Clear stderr buffer for this evaluation
        self._stderr_buffer.clear()

        rendered = prompt.render(
            session=session,
        )
        prompt_text = rendered.text

        logger.debug(
            "claude_agent_sdk.evaluate.rendered",
            event="evaluate.rendered",
            context={
                "prompt_text_length": len(prompt_text),
                "tool_count": len(rendered.tools),
                "tool_names": [t.name for t in rendered.tools],
                "has_output_type": rendered.output_type is not None,
                "output_type": (
                    rendered.output_type.__name__
                    if rendered.output_type
                    and hasattr(rendered.output_type, "__name__")
                    else str(rendered.output_type)
                ),
            },
        )

        session.dispatcher.dispatch(
            PromptRendered(
                prompt_ns=prompt.ns,
                prompt_key=prompt.key,
                prompt_name=prompt.name,
                adapter=CLAUDE_AGENT_SDK_ADAPTER_NAME,
                session_id=getattr(session, "session_id", None),
                render_inputs=(),
                rendered_prompt=prompt_text,
                created_at=_utcnow(),
                descriptor=None,
                run_context=run_context,
            )
        )

        output_format = self._build_output_format(rendered)

        prompt_name = prompt.name or f"{prompt.ns}:{prompt.key}"

        # Determine effective cwd: explicit config, or create a temp folder
        # when no workspace section is present. This ensures agents start in
        # an empty directory rather than inheriting the host's cwd.
        temp_workspace_dir: str | None = None
        effective_cwd: str | None = self._client_config.cwd

        if prompt.filesystem() is None:
            if effective_cwd is None:
                # Create an empty temp folder as the default workspace
                temp_workspace_dir = tempfile.mkdtemp(prefix="wink-sdk-")
                effective_cwd = temp_workspace_dir
                logger.debug(
                    "claude_agent_sdk.evaluate.temp_workspace_created",
                    event="evaluate.temp_workspace_created",
                    context={"temp_workspace_dir": temp_workspace_dir},
                )
            filesystem = HostFilesystem(_root=effective_cwd)
            prompt = prompt.bind(resources={Filesystem: filesystem})
            logger.debug(
                "claude_agent_sdk.evaluate.filesystem_bound",
                event="evaluate.filesystem_bound",
                context={"effective_cwd": effective_cwd},
            )

        try:
            # Enter resource context for lifecycle management
            with prompt.resources:
                return await self._run_with_prompt_context(
                    sdk=sdk,
                    prompt=prompt,
                    prompt_name=prompt_name,
                    prompt_text=prompt_text,
                    rendered=rendered,
                    session=session,
                    output_format=output_format,
                    deadline=deadline,
                    budget_tracker=budget_tracker,
                    effective_cwd=effective_cwd,
                    heartbeat=heartbeat,
                    run_context=run_context,
                )
        finally:
            # Clean up temp workspace if we created one
            if temp_workspace_dir:
                shutil.rmtree(temp_workspace_dir, ignore_errors=True)

    async def _run_with_prompt_context[OutputT](
        self,
        *,
        sdk: Any,
        prompt: Prompt[OutputT],
        prompt_name: str,
        prompt_text: str,
        rendered: RenderedPrompt[OutputT],
        session: SessionProtocol,
        output_format: dict[str, Any] | None,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        effective_cwd: str | None,
        heartbeat: Heartbeat | None,
        run_context: RunContext | None,
    ) -> PromptResponse[OutputT]:
        """Run SDK query within prompt context."""
        logger.debug(
            "claude_agent_sdk.run_context.entry",
            event="run_context.entry",
            context={
                "prompt_name": prompt_name,
                "effective_cwd": effective_cwd,
                "has_output_format": output_format is not None,
            },
        )

        # Create hook context for native tool transactions
        hook_context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name=CLAUDE_AGENT_SDK_ADAPTER_NAME,
            prompt_name=prompt_name,
            deadline=deadline,
            budget_tracker=budget_tracker,
            heartbeat=heartbeat,
            run_context=run_context,
        )

        # Create visibility signal for progressive disclosure support.
        # When a tool raises VisibilityExpansionRequired, the bridge stores
        # the exception in this signal. We check it after SDK query completes.
        visibility_signal = VisibilityExpansionSignal()

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
            heartbeat=heartbeat,
            run_context=run_context,
            visibility_signal=visibility_signal,
        )

        logger.debug(
            "claude_agent_sdk.run_context.bridged_tools",
            event="run_context.bridged_tools",
            context={
                "bridged_tool_count": len(bridged_tools),
                "bridged_tool_names": [
                    getattr(t, "name", str(t)) for t in bridged_tools
                ],
            },
        )

        logger.info(
            "claude_agent_sdk.evaluate.start",
            event="sdk.evaluate.start",
            context={
                "prompt_name": prompt_name,
                "model": self._model,
                "tool_count": len(bridged_tools),
                "has_structured_output": output_format is not None,
                "isolated": True,  # Always isolated by default
            },
        )

        start_time = _utcnow()

        # Always create ephemeral home for hermetic isolation.
        # This prevents the SDK from reading the host's ~/.claude configuration,
        # which may have alternative providers configured (e.g., AWS Bedrock).
        # Use provided isolation config or a secure default.
        isolation = self._client_config.isolation or IsolationConfig()
        ephemeral_home = EphemeralHome(
            isolation,
            workspace_path=effective_cwd,
        )

        # Build network policy representation for logging
        network_policy_repr: str | None = None
        if isolation.network_policy:
            network_policy_repr = str(isolation.network_policy)

        # Count skills if provided (SkillConfig | None)
        skill_count = 1 if isolation.skills else 0

        logger.debug(
            "claude_agent_sdk.run_context.isolation",
            event="run_context.isolation",
            context={
                "ephemeral_home_path": str(ephemeral_home.home_path),
                "workspace_path": effective_cwd,
                "network_policy": network_policy_repr,
                "sandbox_enabled": (
                    isolation.sandbox.enabled if isolation.sandbox else True
                ),
                "has_api_key_override": isolation.api_key is not None,
                "include_host_env": isolation.include_host_env,
                "skill_count": skill_count,
            },
        )

        # Create log aggregator to monitor .claude directory for log files
        log_aggregator = ClaudeLogAggregator(
            claude_dir=ephemeral_home.claude_dir,
            prompt_name=prompt_name,
        )

        try:
            async with log_aggregator.run():
                messages = await self._run_sdk_query(
                    sdk=sdk,
                    prompt_text=prompt_text,
                    output_format=output_format,
                    hook_context=hook_context,
                    bridged_tools=bridged_tools,
                    ephemeral_home=ephemeral_home,
                    effective_cwd=effective_cwd,
                    visibility_signal=visibility_signal,
                )
        except VisibilityExpansionRequired:
            # Progressive disclosure: let this propagate to the caller
            logger.debug(
                "claude_agent_sdk.run_context.visibility_expansion_required",
                event="run_context.visibility_expansion_required",
                context={"prompt_name": prompt_name},
            )
            raise
        except Exception as error:
            # Capture stderr for debugging when SDK fails
            captured_stderr = (
                "\n".join(self._stderr_buffer) if self._stderr_buffer else None
            )
            logger.debug(
                "claude_agent_sdk.run_context.sdk_error",
                event="run_context.sdk_error",
                context={
                    "prompt_name": prompt_name,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "stderr_output": captured_stderr,
                    "exit_code": getattr(error, "exit_code", None),
                },
            )
            raise normalize_sdk_error(
                error, prompt_name, stderr_output=captured_stderr
            ) from error
        finally:
            # Always clean up ephemeral home
            ephemeral_home.cleanup()

        end_time = _utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        result_text, output, usage = self._extract_result(
            messages, rendered, budget_tracker, prompt_name
        )

        # Final verification: if we got structured output but tasks are incomplete,
        # reject the output. This catches cases where the SDK captured output before
        # our hooks could prevent it.
        self._verify_task_completion(
            output,
            session,
            hook_context.stop_reason,
            prompt_name,
            deadline=deadline,
            budget_tracker=budget_tracker,
            prompt=cast("PromptProtocol[Any]", prompt),
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
                session_id=getattr(session, "session_id", None),
                created_at=_utcnow(),
                usage=usage,
                run_context=run_context,
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

    def _create_stderr_handler(self) -> Callable[[str], None]:
        """Create a stderr handler that buffers output for debug logging.

        Even when stderr is suppressed from display, we capture it for
        debugging purposes when errors occur.
        """

        def stderr_handler(line: str) -> None:
            # Always buffer stderr for debug logging on errors
            self._stderr_buffer.append(line)
            # Log individual stderr lines at DEBUG level
            logger.debug(
                "claude_agent_sdk.sdk_query.stderr",
                event="sdk_query.stderr",
                context={"line": line.rstrip()},
            )

        return stderr_handler

    async def _run_sdk_query(
        self,
        *,
        sdk: Any,
        prompt_text: str,
        output_format: dict[str, Any] | None,
        hook_context: HookContext,
        bridged_tools: tuple[Any, ...],
        ephemeral_home: EphemeralHome,
        effective_cwd: str | None = None,
        visibility_signal: VisibilityExpansionSignal,
    ) -> list[Any]:
        """Execute the SDK query and return message list."""
        from claude_agent_sdk.types import ClaudeAgentOptions

        logger.debug(
            "claude_agent_sdk.sdk_query.entry",
            event="sdk_query.entry",
            context={
                "prompt_text_preview": prompt_text[:500] if prompt_text else "",
                "has_output_format": output_format is not None,
                "bridged_tool_count": len(bridged_tools),
            },
        )

        # Build SDK options using the builder pattern
        query_options = self._build_sdk_query_options(
            ephemeral_home=ephemeral_home,
            effective_cwd=effective_cwd,
            output_format=output_format,
            bridged_tools=bridged_tools,
            hook_context=hook_context,
        )

        # Log SDK options (excluding sensitive data)
        self._log_sdk_options(query_options)

        options = ClaudeAgentOptions(**query_options.to_kwargs())

        # Execute the query and collect messages
        messages = await self._execute_sdk_query(
            sdk=sdk,
            prompt_text=prompt_text,
            options=options,
            hook_context=hook_context,
        )

        logger.debug(
            "claude_agent_sdk.sdk_query.complete",
            event="sdk_query.complete",
            context={
                "message_count": len(messages),
                "stderr_line_count": len(self._stderr_buffer),
                "stats_tool_count": hook_context.stats.tool_count,
                "stats_turn_count": hook_context.stats.turn_count,
                "stats_subagent_count": hook_context.stats.subagent_count,
                "stats_compact_count": hook_context.stats.compact_count,
                "stats_input_tokens": hook_context.stats.total_input_tokens,
                "stats_output_tokens": hook_context.stats.total_output_tokens,
                "stats_hook_errors": hook_context.stats.hook_errors,
            },
        )

        # Check for visibility expansion signal from progressive disclosure.
        # If a tool raised VisibilityExpansionRequired, the bridge stored it
        # in the signal. We re-raise it here so the caller can handle it.
        stored_exc = visibility_signal.get_and_clear()
        if stored_exc is not None:
            logger.debug(
                "claude_agent_sdk.sdk_query.visibility_expansion_detected",
                event="sdk_query.visibility_expansion_detected",
                context={
                    "section_keys": stored_exc.section_keys,
                    "reason": stored_exc.reason,
                },
            )
            raise stored_exc

        return messages

    def _build_sdk_query_options(
        self,
        *,
        ephemeral_home: EphemeralHome,
        effective_cwd: str | None,
        output_format: dict[str, Any] | None,
        bridged_tools: tuple[Any, ...],
        hook_context: HookContext,
    ) -> Any:
        """Build SDK query options using the builder pattern.

        Args:
            ephemeral_home: Ephemeral home with isolation config.
            effective_cwd: Working directory for SDK operations.
            output_format: Structured output format specification.
            bridged_tools: Tuple of bridged tools to register.
            hook_context: Hook context for creating hooks.

        Returns:
            SdkQueryOptions with all configured values.
        """
        builder = SdkQueryBuilder(self._model)

        # Apply configurations
        builder.with_cwd(effective_cwd)
        builder.with_client_config(self._client_config)
        builder.with_model_config(self._model_config)
        builder.with_ephemeral_home(ephemeral_home)
        builder.with_output_format(output_format)
        builder.with_tool_constraints(
            allowed_tools=self._allowed_tools,
            disallowed_tools=self._disallowed_tools,
        )
        builder.with_stderr_handler(self._create_stderr_handler())

        # Log environment configuration
        env_vars = ephemeral_home.get_env()
        logger.debug(
            "claude_agent_sdk.sdk_query.env_configured",
            event="sdk_query.env_configured",
            context={
                "home_override": env_vars.get("HOME"),
                "has_api_key": "ANTHROPIC_API_KEY" in env_vars,
                "env_var_count": len(env_vars),
                "env_keys": [k for k in env_vars if "KEY" not in k.upper()],
            },
        )

        # Register MCP server for bridged tools
        if bridged_tools:
            mcp_server_config = create_mcp_server(bridged_tools)
            builder.with_mcp_server("wink", mcp_server_config)
            logger.debug(
                "claude_agent_sdk.sdk_query.mcp_server_configured",
                event="sdk_query.mcp_server_configured",
                context={"mcp_server_name": "wink"},
            )

        # Create and register hooks using the registry
        hook_registry = HookRegistry(hook_context)
        checker = self._client_config.task_completion_checker
        hook_set = hook_registry.create_hook_set(
            stop_on_structured_output=self._client_config.stop_on_structured_output,
            task_completion_checker=checker,
        )
        hooks_dict = HookRegistry.to_sdk_hooks(hook_set)
        builder.with_hooks(hooks_dict)

        logger.debug(
            "claude_agent_sdk.sdk_query.hooks_registered",
            event="sdk_query.hooks_registered",
            context={"hook_types": HookRegistry.get_hook_type_names()},
        )

        return builder.build()

    @staticmethod
    def _log_sdk_options(query_options: Any) -> None:
        """Log SDK options (excluding sensitive data).

        Args:
            query_options: SdkQueryOptions to log.
        """
        logger.debug(
            "claude_agent_sdk.sdk_query.options",
            event="sdk_query.options",
            context={
                "model": query_options.model,
                "cwd": query_options.cwd,
                "permission_mode": query_options.permission_mode,
                "max_turns": query_options.max_turns,
                "max_budget_usd": query_options.max_budget_usd,
                "max_thinking_tokens": query_options.max_thinking_tokens,
                "has_output_format": query_options.output_format is not None,
                "allowed_tools": (
                    list(query_options.allowed_tools)
                    if query_options.allowed_tools is not None
                    else None
                ),
                "disallowed_tools": list(query_options.disallowed_tools),
                "has_mcp_servers": bool(query_options.mcp_servers),
                "betas": list(query_options.betas),
            },
        )

    async def _execute_sdk_query(
        self,
        *,
        sdk: Any,
        prompt_text: str,
        options: Any,
        hook_context: HookContext,
    ) -> list[Any]:
        """Execute the SDK query and collect messages.

        Uses streaming mode (AsyncIterable) to enable hook support.
        The SDK's query() function only initializes hooks when
        is_streaming_mode=True, which requires an AsyncIterable prompt.

        Args:
            sdk: The Claude Agent SDK instance.
            prompt_text: The prompt text to send.
            options: ClaudeAgentOptions for the query.
            hook_context: Hook context for tracking stats.

        Returns:
            List of messages from the SDK query.
        """

        async def stream_prompt() -> Any:
            """Yield a single user message in streaming format."""
            yield {
                "type": "user",
                "message": {"role": "user", "content": prompt_text},
                "parent_tool_use_id": None,
                "session_id": hook_context.prompt_name,
            }

        logger.debug(
            "claude_agent_sdk.sdk_query.executing",
            event="sdk_query.executing",
            context={"prompt_name": hook_context.prompt_name},
        )

        messages: list[Any] = []
        async for message in sdk.query(prompt=stream_prompt(), options=options):
            messages.append(message)

            # Extract message content for logging
            content = _extract_message_content(message)

            # Update cumulative token stats in hook context
            if content.get("input_tokens"):
                hook_context.stats.total_input_tokens += content["input_tokens"]
            if content.get("output_tokens"):
                hook_context.stats.total_output_tokens += content["output_tokens"]

            # Log each message at DEBUG level for troubleshooting
            logger.debug(
                "claude_agent_sdk.sdk_query.message_received",
                event="sdk_query.message_received",
                context={
                    "message_type": type(message).__name__,
                    "message_index": len(messages) - 1,
                    "cumulative_input_tokens": hook_context.stats.total_input_tokens,
                    "cumulative_output_tokens": hook_context.stats.total_output_tokens,
                    **content,
                },
            )

        return messages

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

    def _verify_task_completion(
        self,
        output: Any,
        session: SessionProtocol,
        stop_reason: str | None,
        prompt_name: str,
        *,
        deadline: Deadline | None = None,
        budget_tracker: BudgetTracker | None = None,
        prompt: PromptProtocol[Any] | None = None,
    ) -> None:
        """Verify task completion if checker is configured.

        Raises PromptEvaluationError if structured output was produced but
        tasks are incomplete. Skips verification if deadline or budget is
        exhausted (partial output is acceptable when resources run out).

        Args:
            output: The structured output to verify.
            session: The session containing state.
            stop_reason: Why the agent stopped.
            prompt_name: Name of the prompt for error reporting.
            deadline: Optional deadline to check for exhaustion.
            budget_tracker: Optional budget tracker to check for exhaustion.
            prompt: Optional prompt for filesystem access.
        """
        checker = self._client_config.task_completion_checker
        if output is None or checker is None:
            return

        # Skip verification if deadline exceeded - can't do more work
        if deadline is not None and deadline.remaining().total_seconds() <= 0:
            logger.debug(
                "claude_agent_sdk.verify.deadline_exceeded",
                event="sdk.verify.deadline_exceeded",
                context={"prompt_name": prompt_name, "stop_reason": stop_reason},
            )
            return

        # Skip verification if budget exhausted - can't do more work
        if budget_tracker is not None:
            budget = budget_tracker.budget
            consumed = budget_tracker.consumed
            consumed_total = (consumed.input_tokens or 0) + (
                consumed.output_tokens or 0
            )
            if (
                budget.max_total_tokens is not None
                and consumed_total >= budget.max_total_tokens
            ):
                logger.debug(
                    "claude_agent_sdk.verify.budget_exhausted",
                    event="sdk.verify.budget_exhausted",
                    context={"prompt_name": prompt_name, "stop_reason": stop_reason},
                )
                return

        # Get filesystem from prompt resources if available
        filesystem: Filesystem | None = None
        if prompt is not None:
            with contextlib.suppress(LookupError, AttributeError):
                filesystem = prompt.resources.get(Filesystem)

        context = TaskCompletionContext(
            session=session,
            tentative_output=output,
            stop_reason=stop_reason or "structured_output",
            filesystem=filesystem,
            adapter=self,
        )
        completion = checker.check(context)
        if not completion.complete:
            logger.warning(
                "claude_agent_sdk.evaluate.incomplete_tasks",
                event="sdk.evaluate.incomplete_tasks",
                context={
                    "prompt_name": prompt_name,
                    "feedback": completion.feedback,
                    "stop_reason": stop_reason,
                },
            )
            raise PromptEvaluationError(
                message=f"Tasks incomplete: {completion.feedback}",
                prompt_name=prompt_name,
                phase="response",
            )
