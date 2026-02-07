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

import asyncio
import contextlib
import inspect
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast, override
from uuid import uuid4

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
from ...runtime.session.rendered_tools import RenderedTools, ToolSchema
from ...runtime.watchdog import Heartbeat
from ...serde import parse, schema
from ...types import AdapterName
from ..core import PromptEvaluationError, PromptResponse, ProviderAdapter
from ..tool_spec import tool_to_spec
from ._async_utils import run_async
from ._bridge import MCPToolExecutionState, create_bridged_tools, create_mcp_server
from ._errors import normalize_sdk_error
from ._hooks import (
    HookConstraints,
    HookContext,
    create_post_tool_use_hook,
    create_pre_compact_hook,
    create_pre_tool_use_hook,
    create_stop_hook,
    create_subagent_stop_hook,
    create_task_completion_stop_hook,
    create_user_prompt_submit_hook,
)
from ._task_completion import TaskCompletionContext
from ._transcript_collector import TranscriptCollector
from ._visibility_signal import VisibilityExpansionSignal
from .config import ClaudeAgentSDKClientConfig, ClaudeAgentSDKModelConfig
from .isolation import EphemeralHome, IsolationConfig, get_default_model

if TYPE_CHECKING:
    from ...prompt.tool import Tool
    from ...types.dataclass import SupportsDataclassOrNone, SupportsToolResult

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


def _collapse_nullable_any_of(any_of: object) -> dict[str, Any] | None:
    """Collapse ``anyOf`` nullable unions into ``type=[..., "null"]``."""
    nullable_arity = 2
    if not isinstance(any_of, list) or len(any_of) != nullable_arity:
        return None

    if not all(isinstance(entry, dict) for entry in any_of):
        return None

    entries = cast(list[dict[str, Any]], any_of)
    null_index = next(
        (
            index
            for index, entry in enumerate(entries)
            if entry.get("type") == "null" and len(entry) == 1
        ),
        None,
    )
    if null_index is None:
        return None

    non_null_entry = dict(entries[1 - null_index])
    schema_type = non_null_entry.get("type")
    if isinstance(schema_type, str):
        non_null_entry["type"] = [schema_type, "null"]
        return non_null_entry
    if isinstance(schema_type, list) and all(
        isinstance(value, str) for value in schema_type
    ):
        typed_values = cast(list[str], schema_type)
        non_null_entry["type"] = (
            typed_values if "null" in typed_values else [*typed_values, "null"]
        )
        return non_null_entry
    return None


def _normalize_claude_output_schema(raw_schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize serde JSON schema for Claude structured output compatibility."""
    normalized = dict(raw_schema)

    if normalized.get("type") == "object":
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["properties"] = {
                key: _normalize_claude_output_schema(cast(dict[str, Any], value))
                if isinstance(value, dict)
                else value
                for key, value in properties.items()
            }

    if "items" in normalized and isinstance(normalized["items"], dict):
        normalized["items"] = _normalize_claude_output_schema(
            cast(dict[str, Any], normalized["items"])
        )

    for combinator in ("anyOf", "oneOf", "allOf"):
        combinator_items = normalized.get(combinator)
        if isinstance(combinator_items, list):
            normalized[combinator] = [
                _normalize_claude_output_schema(cast(dict[str, Any], entry))
                if isinstance(entry, dict)
                else entry
                for entry in combinator_items
            ]

    for defs_key in ("$defs", "definitions"):
        defs = normalized.get(defs_key)
        if isinstance(defs, dict):
            normalized[defs_key] = {
                key: _normalize_claude_output_schema(cast(dict[str, Any], value))
                if isinstance(value, dict)
                else value
                for key, value in defs.items()
            }

    collapsed_nullable = _collapse_nullable_any_of(normalized.get("anyOf"))
    return collapsed_nullable if collapsed_nullable is not None else normalized


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
        ...     model="claude-opus-4-6",
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
        model: str | None = None,
        client_config: ClaudeAgentSDKClientConfig | None = None,
        model_config: ClaudeAgentSDKModelConfig | None = None,
        allowed_tools: tuple[str, ...] | None = None,
        disallowed_tools: tuple[str, ...] = (),
    ) -> None:
        """Initialize the Claude Agent SDK adapter.

        Args:
            model: Claude model identifier. Defaults to the environment-aware
                default: Bedrock model ID when Bedrock is configured,
                Anthropic API model name otherwise.
            client_config: SDK client configuration. Defaults to
                bypassPermissions mode.
            model_config: Model parameter configuration.
            allowed_tools: Tools Claude can use (None = all available).
            disallowed_tools: Tools to explicitly block.
        """
        resolved_model = model if model is not None else get_default_model()
        self._model = resolved_model
        self._client_config = client_config or ClaudeAgentSDKClientConfig()
        self._model_config = model_config or ClaudeAgentSDKModelConfig(
            model=resolved_model
        )
        self._allowed_tools = allowed_tools
        self._disallowed_tools = disallowed_tools
        # Buffer for capturing stderr output for debug logging
        self._stderr_buffer: list[str] = []

        logger.debug(
            "claude_agent_sdk.adapter.init",
            event="adapter.init",
            context={
                "model": resolved_model,
                "permission_mode": self._client_config.permission_mode,
                "cwd": self._client_config.cwd,
                "max_turns": self._client_config.max_turns,
                "max_budget_usd": self._client_config.max_budget_usd,
                "suppress_stderr": self._client_config.suppress_stderr,
                "stop_on_structured_output": self._client_config.stop_on_structured_output,
                "has_isolation_config": self._client_config.isolation is not None,
                "allowed_tools": allowed_tools,
                "disallowed_tools": disallowed_tools,
                "reasoning": self._model_config.reasoning,
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

        # Generate shared event correlation data
        render_event_id = uuid4()
        session_id = getattr(session, "session_id", None)
        created_at = _utcnow()

        session.dispatcher.dispatch(
            PromptRendered(
                prompt_ns=prompt.ns,
                prompt_key=prompt.key,
                prompt_name=prompt.name,
                adapter=CLAUDE_AGENT_SDK_ADAPTER_NAME,
                session_id=session_id,
                render_inputs=(),
                rendered_prompt=prompt_text,
                created_at=created_at,
                descriptor=None,
                run_context=run_context,
                event_id=render_event_id,
            )
        )

        # Dispatch RenderedTools with tool schemas
        def _extract_tool_schema(
            tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
        ) -> ToolSchema:
            spec = tool_to_spec(tool)
            fn = spec["function"]
            return ToolSchema(
                name=fn["name"],
                description=fn["description"],
                parameters=fn["parameters"],
            )

        tool_schemas = tuple(_extract_tool_schema(tool) for tool in rendered.tools)
        tools_dispatch_result = session.dispatcher.dispatch(
            RenderedTools(
                prompt_ns=prompt.ns,
                prompt_key=prompt.key,
                tools=tool_schemas,
                render_event_id=render_event_id,
                session_id=session_id,
                created_at=created_at,
            )
        )
        if not tools_dispatch_result.ok:
            logger.error(
                "claude_agent_sdk.evaluate.rendered_tools_dispatch_failed",
                event="rendered_tools_dispatch_failed",
                context={
                    "failure_count": len(tools_dispatch_result.errors),
                    "tool_count": len(tool_schemas),
                },
            )
        else:
            logger.debug(
                "claude_agent_sdk.evaluate.rendered_tools_dispatched",
                event="rendered_tools_dispatched",
                context={
                    "tool_count": len(tool_schemas),
                    "handler_count": tools_dispatch_result.handled_count,
                },
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
        elif effective_cwd is None:
            # Workspace section provides a filesystem - derive cwd from its root
            # so the SDK subprocess starts in the workspace directory.
            fs = prompt.filesystem()
            if isinstance(fs, HostFilesystem):
                effective_cwd = fs.root
                logger.debug(
                    "claude_agent_sdk.evaluate.cwd_from_workspace",
                    event="evaluate.cwd_from_workspace",
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

        # Create shared state for MCP tool_use_id tracking between hooks and bridge
        mcp_tool_state = MCPToolExecutionState()

        # Create hook context for native tool transactions
        constraints = HookConstraints(
            deadline=deadline,
            budget_tracker=budget_tracker,
            heartbeat=heartbeat,
            run_context=run_context,
            mcp_tool_state=mcp_tool_state,
        )
        hook_context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name=CLAUDE_AGENT_SDK_ADAPTER_NAME,
            prompt_name=prompt_name,
            constraints=constraints,
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
            mcp_tool_state=mcp_tool_state,
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

        # Mount skills from rendered prompt into ephemeral home
        skills = rendered.skills
        if skills:
            ephemeral_home.mount_skills(skills)

        # Build network policy representation for logging
        network_policy_repr: str | None = None
        if isolation.network_policy:
            network_policy_repr = str(isolation.network_policy)

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
                "skill_count": len(skills),
            },
        )

        # Create transcript collector to monitor SDK transcripts (if enabled)
        transcript_config = self._client_config.transcript_collection
        collector: TranscriptCollector | None = None
        if transcript_config is not None:
            collector = TranscriptCollector(
                prompt_name=prompt_name,
                config=transcript_config,
            )

        try:
            async with collector.run() if collector else contextlib.nullcontext():
                messages = await self._run_sdk_query(
                    sdk=sdk,
                    prompt_text=prompt_text,
                    output_format=output_format,
                    hook_context=hook_context,
                    bridged_tools=bridged_tools,
                    ephemeral_home=ephemeral_home,
                    effective_cwd=effective_cwd,
                    visibility_signal=visibility_signal,
                    collector=collector,
                )
        except VisibilityExpansionRequired:
            # Progressive disclosure: let this propagate to the caller
            logger.debug(
                "claude_agent_sdk.run_context.visibility_expansion_required",
                event="run_context.visibility_expansion_required",
                context={"prompt_name": prompt_name},
            )
            raise
        except PromptEvaluationError:
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
        self._raise_if_missing_required_structured_output(
            rendered=rendered,
            prompt_name=prompt_name,
            messages=messages,
            result_text=result_text,
            output=output,
            stop_reason=hook_context.stop_reason,
        )

        # Final verification: log a warning if tasks are incomplete.
        # The task completion checker provides feedback during execution via hooks.
        # We don't reject the output here - we return whatever the agent produced,
        # allowing partial progress even if all tasks weren't completed.
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

    def _supported_option_names(
        self,
        options_type: type[Any],
    ) -> set[str] | None:
        """Return supported option names for ClaudeAgentOptions.

        Returns None when the options type accepts arbitrary keyword arguments.
        """
        dataclass_fields = getattr(options_type, "__dataclass_fields__", None)
        if isinstance(dataclass_fields, dict):
            return set(dataclass_fields)

        try:
            signature = inspect.signature(options_type)
        except (TypeError, ValueError):
            return None

        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        ):
            return None

        return {
            name
            for name, param in signature.parameters.items()
            if param.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }

    def _filter_unsupported_options(
        self,
        options_kwargs: dict[str, Any],
        *,
        options_type: type[Any],
    ) -> dict[str, Any]:
        """Drop SDK option kwargs unsupported by the installed SDK version."""
        supported_names = self._supported_option_names(options_type)
        if supported_names is None:
            return options_kwargs

        unsupported = sorted(
            key for key in options_kwargs if key not in supported_names
        )
        if not unsupported:
            return options_kwargs

        for key in unsupported:
            options_kwargs.pop(key, None)

        logger.info(
            "claude_agent_sdk.sdk_query.options_filtered",
            event="sdk_query.options_filtered",
            context={
                "unsupported_option_names": unsupported,
            },
        )
        return options_kwargs

    def _add_client_config_options(
        self,
        options_kwargs: dict[str, Any],
        effective_cwd: str | None,
        output_format: dict[str, Any] | None,
    ) -> None:
        """Add client config options to the options dictionary."""
        # Add non-None optional values using dict comprehension
        optional_values = {
            "cwd": effective_cwd,
            "permission_mode": self._client_config.permission_mode,
            "max_turns": self._client_config.max_turns,
            "output_format": output_format,
            "reasoning": self._model_config.reasoning,
        }
        options_kwargs.update(
            {k: v for k, v in optional_values.items() if v is not None}
        )

        # Handle special cases that need explicit None checks or transformations
        if self._client_config.max_budget_usd is not None:
            options_kwargs["max_budget_usd"] = self._client_config.max_budget_usd
        if self._client_config.betas:
            options_kwargs["betas"] = list(self._client_config.betas)
        if self._allowed_tools is not None:
            options_kwargs["allowed_tools"] = list(self._allowed_tools)
        if self._disallowed_tools:
            options_kwargs["disallowed_tools"] = list(self._disallowed_tools)

    def _build_sdk_options_kwargs(
        self,
        *,
        output_format: dict[str, Any] | None,
        bridged_tools: tuple[Any, ...],
        ephemeral_home: EphemeralHome,
        effective_cwd: str | None,
    ) -> dict[str, Any]:
        """Build the SDK options dictionary for ClaudeAgentOptions."""
        options_kwargs: dict[str, Any] = {"model": self._model}

        # Add client config options
        self._add_client_config_options(options_kwargs, effective_cwd, output_format)

        # Apply isolation configuration from ephemeral home
        env_vars = ephemeral_home.get_env()
        options_kwargs["env"] = env_vars
        options_kwargs["setting_sources"] = ephemeral_home.get_setting_sources()

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

        # Register custom tools via MCP server if any are provided
        if bridged_tools:
            mcp_server_config = create_mcp_server(bridged_tools)
            options_kwargs["mcp_servers"] = {"wink": mcp_server_config}
            logger.debug(
                "claude_agent_sdk.sdk_query.mcp_server_configured",
                event="sdk_query.mcp_server_configured",
                context={"mcp_server_name": "wink"},
            )

        options_kwargs["stderr"] = self._create_stderr_handler()
        return options_kwargs

    def _build_hooks_config(
        self,
        *,
        hook_context: HookContext,
        collector: TranscriptCollector | None,
    ) -> dict[str, list[Any]]:
        """Build the hooks configuration for the SDK client."""
        from claude_agent_sdk.types import HookMatcher

        checker = self._client_config.task_completion_checker
        pre_hook = create_pre_tool_use_hook(hook_context)
        post_hook = create_post_tool_use_hook(
            hook_context,
            stop_on_structured_output=self._client_config.stop_on_structured_output,
            task_completion_checker=checker,
        )

        # Use task completion stop hook if checker is configured
        if checker is not None:  # pragma: no cover - tested via hook tests
            stop_hook_fn = create_task_completion_stop_hook(
                hook_context,
                checker=checker,
            )
        else:
            stop_hook_fn = create_stop_hook(hook_context)

        prompt_hook = create_user_prompt_submit_hook(hook_context)
        subagent_stop_hook = create_subagent_stop_hook(hook_context)
        pre_compact_hook = create_pre_compact_hook(hook_context)

        # Get collector hooks (empty if collector is disabled)
        collector_hooks = collector.hooks_config() if collector else {}

        hook_types = [
            "PreToolUse",
            "PostToolUse",
            "Stop",
            "UserPromptSubmit",
            "SubagentStop",
            "PreCompact",
        ]

        logger.debug(
            "claude_agent_sdk.sdk_query.hooks_registered",
            event="sdk_query.hooks_registered",
            context={"hook_types": hook_types},
        )

        return {
            "PreToolUse": [
                HookMatcher(matcher=None, hooks=[pre_hook]),
                *collector_hooks.get("PreToolUse", []),
            ],
            "PostToolUse": [
                HookMatcher(matcher=None, hooks=[post_hook]),
                *collector_hooks.get("PostToolUse", []),
            ],
            "Stop": [
                HookMatcher(matcher=None, hooks=[stop_hook_fn]),
                *collector_hooks.get("Stop", []),
            ],
            "UserPromptSubmit": [
                HookMatcher(matcher=None, hooks=[prompt_hook]),
                *collector_hooks.get("UserPromptSubmit", []),
            ],
            "SubagentStop": [
                HookMatcher(matcher=None, hooks=[subagent_stop_hook]),
                *collector_hooks.get("SubagentStop", []),
            ],
            "PreCompact": [
                HookMatcher(matcher=None, hooks=[pre_compact_hook]),
                *collector_hooks.get("PreCompact", []),
            ],
        }

    def _check_continuation_constraints(
        self,
        hook_context: HookContext,
        continuation_round: int,
    ) -> bool:
        """Check deadline and budget constraints before a round.

        Returns True if constraints are satisfied and we should continue.
        Returns False if constraints are exceeded and we should stop.
        """
        # Check deadline before each round
        if (
            hook_context.deadline
            and hook_context.deadline.remaining().total_seconds() <= 0
        ):
            logger.info(
                "claude_agent_sdk.sdk_query.deadline_exceeded",
                event="sdk_query.deadline_exceeded",
                context={
                    "continuation_round": continuation_round,
                    "prompt_name": hook_context.prompt_name,
                },
            )
            return False

        # Check token budget before each round
        if hook_context.budget_tracker:
            try:
                hook_context.budget_tracker.check()
            except Exception as budget_error:
                logger.info(
                    "claude_agent_sdk.sdk_query.token_budget_exceeded",
                    event="sdk_query.token_budget_exceeded",
                    context={
                        "continuation_round": continuation_round,
                        "total_input_tokens": hook_context.stats.total_input_tokens,
                        "total_output_tokens": hook_context.stats.total_output_tokens,
                        "error": str(budget_error),
                    },
                )
                return False

        return True

    def _update_token_stats(
        self,
        hook_context: HookContext,
        content: dict[str, Any],
    ) -> None:
        """Update token statistics from message content."""
        if content.get("input_tokens"):
            hook_context.stats.total_input_tokens += content["input_tokens"]
        if content.get("output_tokens"):
            hook_context.stats.total_output_tokens += content["output_tokens"]

        # Update budget tracker if available with cumulative totals
        if hook_context.budget_tracker:
            from ...runtime.events import TokenUsage

            hook_context.budget_tracker.record_cumulative(
                hook_context.prompt_name,
                TokenUsage(
                    input_tokens=hook_context.stats.total_input_tokens,
                    output_tokens=hook_context.stats.total_output_tokens,
                ),
            )

    def _check_task_completion(
        self,
        checker: Any,
        round_messages: list[Any],
        hook_context: HookContext,
    ) -> tuple[bool, str | None]:
        """Check if task is complete and return (should_continue, feedback).

        Returns:
            Tuple of (should_continue, feedback). If should_continue is True,
            feedback contains the continuation message to send.
        """
        if not round_messages:
            return (False, None)

        last_message = round_messages[-1]
        tentative_output = getattr(last_message, "structured_output", None)
        if tentative_output is None:
            tentative_output = getattr(last_message, "result", None)

        completion_context = TaskCompletionContext(
            session=hook_context.session,
            tentative_output=tentative_output,
            stop_reason="message_stream_complete",
            filesystem=None,
        )

        result = checker.check(completion_context)

        if not result.complete and result.feedback:
            return (True, result.feedback)

        if result.complete:
            logger.debug(
                "claude_agent_sdk.sdk_query.task_complete",
                event="sdk_query.task_complete",
                context={"feedback": result.feedback},
            )

        return (False, None)

    def _log_message_received(
        self,
        message: Any,
        messages: list[Any],
        continuation_round: int,
        hook_context: HookContext,
        content: dict[str, Any],
    ) -> None:
        """Log message receipt at DEBUG level."""
        logger.debug(
            "claude_agent_sdk.sdk_query.message_received",
            event="sdk_query.message_received",
            context={
                "message_type": type(message).__name__,
                "message_index": len(messages) - 1,
                "continuation_round": continuation_round,
                "cumulative_input_tokens": hook_context.stats.total_input_tokens,
                "cumulative_output_tokens": hook_context.stats.total_output_tokens,
                **content,
            },
        )

    def _should_continue_loop(
        self,
        continuation_round: int,
        max_continuation_rounds: int | None,
    ) -> bool:
        """Check if continuation loop should continue."""
        return (
            max_continuation_rounds is None
            or continuation_round < max_continuation_rounds
        )

    def _check_and_raise_visibility_signal(
        self,
        visibility_signal: VisibilityExpansionSignal,
    ) -> None:
        """Check for visibility expansion signal and raise if present."""
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

    def _resolve_response_wait_timeout(
        self,
        *,
        hook_context: HookContext,
        continuation_round: int,
        message_count: int,
    ) -> tuple[float | None, bool]:
        """Resolve wait timeout for the next response message.

        Returns:
            Tuple of (timeout_seconds, should_stop_stream_reading).
        """
        if hook_context.deadline is None:
            return (None, False)

        wait_timeout = hook_context.deadline.remaining().total_seconds()
        if wait_timeout <= 0:
            logger.info(
                "claude_agent_sdk.sdk_query.deadline_exceeded_during_stream_wait",
                event="sdk_query.deadline_exceeded_during_stream_wait",
                context={
                    "prompt_name": hook_context.prompt_name,
                    "continuation_round": continuation_round,
                    "message_count": message_count,
                    "deadline_remaining_seconds": wait_timeout,
                },
            )
            return (None, True)

        return (wait_timeout, False)

    async def _next_response_message(
        self,
        *,
        response_stream: Any,
        hook_context: HookContext,
        continuation_round: int,
        message_count: int,
    ) -> Any | None:
        """Read the next message from the SDK response stream."""
        wait_timeout, should_stop = self._resolve_response_wait_timeout(
            hook_context=hook_context,
            continuation_round=continuation_round,
            message_count=message_count,
        )
        if should_stop:
            return None

        try:
            if wait_timeout is None:
                return await anext(response_stream)
            return await asyncio.wait_for(
                anext(response_stream),
                timeout=wait_timeout,
            )
        except StopAsyncIteration:
            return None
        except TimeoutError as error:
            raise PromptEvaluationError(
                message="Deadline exceeded while waiting for Claude SDK response stream.",
                prompt_name=hook_context.prompt_name,
                phase="response",
                provider_payload={
                    "continuation_round": continuation_round,
                    "message_count": message_count,
                    "deadline_remaining_seconds": (
                        hook_context.deadline.remaining().total_seconds()
                        if hook_context.deadline is not None
                        else None
                    ),
                },
            ) from error

    async def _collect_round_messages(
        self,
        *,
        client: Any,
        messages: list[Any],
        continuation_round: int,
        hook_context: HookContext,
    ) -> list[Any]:
        """Collect a single response round from the SDK."""
        round_messages: list[Any] = []
        response_stream = client.receive_response()

        while True:
            message = await self._next_response_message(
                response_stream=response_stream,
                hook_context=hook_context,
                continuation_round=continuation_round,
                message_count=len(messages),
            )
            if message is None:
                break

            messages.append(message)
            round_messages.append(message)

            # Extract message content, update token stats, and log
            content = _extract_message_content(message)
            self._update_token_stats(hook_context, content)
            self._log_message_received(
                message, messages, continuation_round, hook_context, content
            )

        return round_messages

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
        collector: TranscriptCollector | None,
    ) -> list[Any]:
        """Execute the SDK query and return message list."""
        from claude_agent_sdk import ClaudeSDKClient
        from claude_agent_sdk.types import ClaudeAgentOptions

        logger.debug(
            "claude_agent_sdk.sdk_query.entry",
            event="sdk_query.entry",
            context={
                "prompt_text_preview": (prompt_text or "")[:500],
                "has_output_format": output_format is not None,
                "bridged_tool_count": len(bridged_tools),
            },
        )

        # Build options using helper methods
        options_kwargs = self._build_sdk_options_kwargs(
            output_format=output_format,
            bridged_tools=bridged_tools,
            ephemeral_home=ephemeral_home,
            effective_cwd=effective_cwd,
        )
        options_kwargs["hooks"] = self._build_hooks_config(
            hook_context=hook_context,
            collector=collector,
        )
        options_kwargs = self._filter_unsupported_options(
            options_kwargs,
            options_type=ClaudeAgentOptions,
        )

        # Log SDK options (excluding sensitive data)
        logger.debug(
            "claude_agent_sdk.sdk_query.options",
            event="sdk_query.options",
            context={
                "model": options_kwargs.get("model"),
                "cwd": options_kwargs.get("cwd"),
                "permission_mode": options_kwargs.get("permission_mode"),
                "max_turns": options_kwargs.get("max_turns"),
                "max_budget_usd": options_kwargs.get("max_budget_usd"),
                "reasoning": options_kwargs.get("reasoning"),
                "has_output_format": "output_format" in options_kwargs,
                "allowed_tools": options_kwargs.get("allowed_tools"),
                "disallowed_tools": options_kwargs.get("disallowed_tools"),
                "has_mcp_servers": "mcp_servers" in options_kwargs,
                "betas": options_kwargs.get("betas"),
            },
        )

        options = ClaudeAgentOptions(**options_kwargs)

        # Create ClaudeSDKClient for direct control
        client = ClaudeSDKClient(options=options)

        logger.debug(
            "claude_agent_sdk.sdk_query.connecting",
            event="sdk_query.connecting",
            context={"prompt_name": hook_context.prompt_name},
        )

        # Connect WITHOUT a prompt stream. This prevents the SDK from starting
        # stream_input() which would close stdin after the generator finishes.
        # Instead, we use client.query() for the initial message and all
        # continuations, then manually close stdin when done.
        #
        # This approach allows multi-turn conversations where continuation
        # messages are sent based on task completion checking.
        await client.connect(prompt=None)

        # Send the initial prompt via query() - this writes directly to transport
        await client.query(prompt=prompt_text, session_id=hook_context.prompt_name)

        logger.debug(
            "claude_agent_sdk.sdk_query.executing",
            event="sdk_query.executing",
            context={"prompt_name": hook_context.prompt_name},
        )

        messages: list[Any] = []
        # Only use max rounds as fallback if no deadline or budget configured
        has_constraints = bool(hook_context.deadline or hook_context.budget_tracker)
        max_continuation_rounds = None if has_constraints else 100
        continuation_round = 0

        logger.debug(
            "claude_agent_sdk.sdk_query.loop_config",
            event="sdk_query.loop_config",
            context={
                "has_constraints": has_constraints,
                "has_deadline": hook_context.deadline is not None,
                "has_budget": hook_context.budget_tracker is not None,
                "max_rounds": max_continuation_rounds,
                "prompt_name": hook_context.prompt_name,
            },
        )

        checker = self._client_config.task_completion_checker

        try:
            while self._should_continue_loop(  # pragma: no branch
                continuation_round, max_continuation_rounds
            ):
                # Check deadline and budget constraints
                if not self._check_continuation_constraints(
                    hook_context, continuation_round
                ):
                    break

                # Receive messages until ResultMessage using receive_response().
                # This exits after ResultMessage without waiting for subprocess to exit,
                # allowing us to send continuation messages if needed.
                round_messages = await self._collect_round_messages(
                    client=client,
                    messages=messages,
                    continuation_round=continuation_round,
                    hook_context=hook_context,
                )

                # Handle empty message stream (e.g., after continuation)
                if not round_messages:
                    logger.warning(
                        "claude_agent_sdk.sdk_query.empty_message_stream",
                        event="sdk_query.empty_message_stream",
                        context={
                            "continuation_round": continuation_round,
                            "prompt_name": hook_context.prompt_name,
                        },
                    )
                    break  # Exit if no messages received

                # Check if we should continue based on task completion
                if checker is not None:
                    should_continue, feedback = self._check_task_completion(
                        checker, round_messages, hook_context
                    )
                    if should_continue and feedback:
                        logger.info(
                            "claude_agent_sdk.sdk_query.continuation_required",
                            event="sdk_query.continuation_required",
                            context={
                                "feedback": feedback[:200],
                                "continuation_round": continuation_round + 1,
                            },
                        )
                        continuation_round += 1
                        await client.query(
                            prompt=feedback,
                            session_id=hook_context.prompt_name,
                        )
                        continue

                # Exit loop if no checker, no messages, or completion check passed
                break

        finally:
            # Close stdin to signal EOF to the subprocess, then disconnect.
            # Since we didn't use stream_input(), we need to manually close stdin.
            if client._transport is not None:
                await client._transport.end_input()
            logger.debug(
                "claude_agent_sdk.sdk_query.disconnecting",
                event="sdk_query.disconnecting",
                context={
                    "prompt_name": hook_context.prompt_name,
                    "continuation_rounds": continuation_round,
                },
            )
            await client.disconnect()

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

        # Check for visibility expansion signal and re-raise if present
        self._check_and_raise_visibility_signal(visibility_signal)

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
            "schema": _normalize_claude_output_schema(schema(output_type)),
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

    def _raise_if_missing_required_structured_output(
        self,
        *,
        rendered: RenderedPrompt[OutputT],
        prompt_name: str,
        messages: list[Any],
        result_text: str | None,
        output: OutputT | None,
        stop_reason: str | None,
    ) -> None:
        """Raise when structured output is required but no response was produced."""
        output_type = rendered.output_type
        if output_type is None or output_type is type(None):
            return
        if output is not None or result_text is not None:
            return

        message_type_counts: dict[str, int] = {}
        for message in messages:
            message_type = type(message).__name__
            message_type_counts[message_type] = (
                message_type_counts.get(message_type, 0) + 1
            )
        stderr_tail = [line.rstrip() for line in self._stderr_buffer[-20:]]

        logger.warning(
            "claude_agent_sdk.evaluate.missing_structured_output",
            event="sdk.evaluate.missing_structured_output",
            context={
                "prompt_name": prompt_name,
                "message_count": len(messages),
                "message_type_counts": message_type_counts,
                "stop_reason": stop_reason,
                "stderr_tail": stderr_tail or None,
            },
        )

        raise PromptEvaluationError(
            message="Structured output prompt returned no text and no structured output.",
            prompt_name=prompt_name,
            phase="response",
            provider_payload={
                "output_type": (
                    output_type.__name__
                    if hasattr(output_type, "__name__")
                    else str(output_type)
                ),
                "message_count": len(messages),
                "message_type_counts": message_type_counts,
                "stop_reason": stop_reason,
                "stderr_tail": stderr_tail or None,
            },
        )

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
            # Log warning but don't fail - the task completion checker provides
            # feedback during execution via hooks. At the end, we should return
            # whatever output the agent produced, even if tasks are incomplete.
            # This allows the agent to make progress even if it doesn't complete
            # all planned tasks within the available turns/budget.
            logger.warning(
                "claude_agent_sdk.evaluate.incomplete_tasks",
                event="sdk.evaluate.incomplete_tasks",
                context={
                    "prompt_name": prompt_name,
                    "feedback": completion.feedback,
                    "stop_reason": stop_reason,
                    "has_output": output is not None,
                },
            )
            # Don't raise an error - let the response be returned with whatever
            # output the agent managed to produce
