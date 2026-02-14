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
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from ...runtime.session.rendered_tools import RenderedTools, ToolSchema
from ...runtime.watchdog import Heartbeat
from ...types import AdapterName
from ..core import PromptEvaluationError, PromptResponse, ProviderAdapter
from ..tool_spec import tool_to_spec
from ._async_utils import run_async
from ._bridge import MCPToolExecutionState, create_bridged_tools
from ._ephemeral_home import EphemeralHome
from ._errors import normalize_sdk_error
from ._hooks import HookConstraints, HookContext
from ._result_extraction import (
    build_output_format,
    extract_result,
    raise_if_missing_required_structured_output,
    verify_task_completion,
)
from ._sdk_execution import run_sdk_query
from ._sdk_options import (
    build_hooks_config,
    build_sdk_options_kwargs,
    filter_unsupported_options,
)
from ._transcript_collector import TranscriptCollector
from ._visibility_signal import VisibilityExpansionSignal
from .config import ClaudeAgentSDKClientConfig, ClaudeAgentSDKModelConfig
from .isolation import IsolationConfig, get_default_model

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


def _dispatch_render_events(
    prompt: Prompt[Any],
    rendered: RenderedPrompt[Any],
    session: SessionProtocol,
    run_context: RunContext | None,
) -> None:
    """Dispatch PromptRendered and RenderedTools events."""
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
            rendered_prompt=rendered.text,
            created_at=created_at,
            descriptor=None,
            run_context=run_context,
            event_id=render_event_id,
        )
    )

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
    tools_result = session.dispatcher.dispatch(
        RenderedTools(
            prompt_ns=prompt.ns,
            prompt_key=prompt.key,
            tools=tool_schemas,
            render_event_id=render_event_id,
            session_id=session_id,
            created_at=created_at,
        )
    )
    if not tools_result.ok:
        logger.error(
            "claude_agent_sdk.evaluate.rendered_tools_dispatch_failed",
            event="rendered_tools_dispatch_failed",
            context={
                "failure_count": len(tools_result.errors),
                "tool_count": len(tool_schemas),
            },
        )


def _resolve_workspace(
    prompt: Prompt[Any],
    client_config: ClaudeAgentSDKClientConfig,
) -> tuple[Prompt[Any], str | None, str | None]:
    """Resolve workspace cwd and filesystem binding."""
    temp_workspace_dir: str | None = None
    effective_cwd: str | None = client_config.cwd

    if prompt.filesystem() is None:
        if effective_cwd is None:
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
        fs = prompt.filesystem()
        if isinstance(fs, HostFilesystem):
            effective_cwd = fs.root
            logger.debug(
                "claude_agent_sdk.evaluate.cwd_from_workspace",
                event="evaluate.cwd_from_workspace",
                context={"effective_cwd": effective_cwd},
            )

    return prompt, effective_cwd, temp_workspace_dir


def _setup_ephemeral_home(
    client_config: ClaudeAgentSDKClientConfig,
    rendered: RenderedPrompt[Any],
    effective_cwd: str | None,
) -> tuple[IsolationConfig, EphemeralHome]:
    """Create and configure ephemeral home for hermetic isolation."""
    isolation = client_config.isolation or IsolationConfig()
    ephemeral_home = EphemeralHome(isolation, workspace_path=effective_cwd)

    skills = rendered.skills
    if skills:
        ephemeral_home.mount_skills(skills)

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
    return isolation, ephemeral_home


def _build_and_dispatch_response[OutputT](
    *,
    messages: list[Any],
    rendered: RenderedPrompt[OutputT],
    budget_tracker: BudgetTracker | None,
    prompt_name: str,
    hook_context: HookContext,
    stderr_buffer: list[str],
    session: SessionProtocol,
    client_config: ClaudeAgentSDKClientConfig,
    adapter: Any,
    deadline: Deadline | None,
    prompt: Prompt[OutputT],
    run_context: RunContext | None,
    duration_ms: int,
) -> PromptResponse[OutputT]:
    """Extract result, validate, dispatch events, and return response."""
    result_text, output, usage = extract_result(
        messages, rendered, budget_tracker, prompt_name
    )
    raise_if_missing_required_structured_output(
        rendered=rendered,
        prompt_name=prompt_name,
        messages=messages,
        result_text=result_text,
        output=output,
        stop_reason=hook_context.stop_reason,
        stderr_buffer=stderr_buffer,
    )

    verify_task_completion(
        output,
        session,
        hook_context.stop_reason,
        prompt_name,
        deadline=deadline,
        budget_tracker=budget_tracker,
        prompt=cast("PromptProtocol[Any]", prompt),
        client_config=client_config,
        adapter=adapter,
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


class ClaudeAgentSDKAdapter[OutputT](ProviderAdapter[OutputT]):
    """Adapter using Claude Agent SDK with hook-based state synchronization.

    Uses the Claude Agent SDK's ClaudeSDKClient to execute prompts with
    full agentic capabilities. Hooks synchronize state bidirectionally
    between the SDK's internal execution and the weakincentives Session.
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
        """Evaluate prompt using Claude Agent SDK with hook-based state sync."""
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
        self._stderr_buffer.clear()

        rendered = prompt.render(session=session)
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

        _dispatch_render_events(prompt, rendered, session, run_context)

        output_format = build_output_format(rendered)
        prompt_name = prompt.name or f"{prompt.ns}:{prompt.key}"
        prompt, effective_cwd, temp_workspace_dir = _resolve_workspace(
            prompt, self._client_config
        )

        try:
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
            if temp_workspace_dir:
                shutil.rmtree(temp_workspace_dir, ignore_errors=True)

    def _create_hook_context_and_tools[OutputT](
        self,
        *,
        prompt: Prompt[OutputT],
        prompt_name: str,
        rendered: RenderedPrompt[OutputT],
        session: SessionProtocol,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        heartbeat: Heartbeat | None,
        run_context: RunContext | None,
    ) -> tuple[HookContext, VisibilityExpansionSignal, tuple[Any, ...]]:
        """Create hook context, visibility signal, and bridged tools."""
        mcp_tool_state = MCPToolExecutionState()
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
        return hook_context, visibility_signal, bridged_tools

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
        hook_context, visibility_signal, bridged_tools = (
            self._create_hook_context_and_tools(
                prompt=prompt,
                prompt_name=prompt_name,
                rendered=rendered,
                session=session,
                deadline=deadline,
                budget_tracker=budget_tracker,
                heartbeat=heartbeat,
                run_context=run_context,
            )
        )

        logger.info(
            "claude_agent_sdk.evaluate.start",
            event="sdk.evaluate.start",
            context={
                "prompt_name": prompt_name,
                "model": self._model,
                "tool_count": len(bridged_tools),
                "has_structured_output": output_format is not None,
                "isolated": True,
            },
        )

        start_time = _utcnow()
        _isolation, ephemeral_home = _setup_ephemeral_home(
            self._client_config, rendered, effective_cwd
        )

        transcript_config = self._client_config.transcript_collection
        collector: TranscriptCollector | None = None
        if transcript_config is not None:
            session_id = getattr(session, "session_id", None)
            collector = TranscriptCollector(
                prompt_name=prompt_name,
                config=transcript_config,
                session_id=str(session_id) if session_id else None,
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
            logger.debug(
                "claude_agent_sdk.run_context.visibility_expansion_required",
                event="run_context.visibility_expansion_required",
                context={"prompt_name": prompt_name},
            )
            raise
        except PromptEvaluationError:
            raise
        except Exception as error:
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
            ephemeral_home.cleanup()

        duration_ms = int((_utcnow() - start_time).total_seconds() * 1000)
        return _build_and_dispatch_response(
            messages=messages,
            rendered=rendered,
            budget_tracker=budget_tracker,
            prompt_name=prompt_name,
            hook_context=hook_context,
            stderr_buffer=self._stderr_buffer,
            session=session,
            client_config=self._client_config,
            adapter=self,
            deadline=deadline,
            prompt=prompt,
            run_context=run_context,
            duration_ms=duration_ms,
        )

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
        from claude_agent_sdk.types import ClaudeAgentOptions

        options_kwargs = build_sdk_options_kwargs(
            model=self._model,
            output_format=output_format,
            bridged_tools=bridged_tools,
            ephemeral_home=ephemeral_home,
            effective_cwd=effective_cwd,
            client_config=self._client_config,
            model_config=self._model_config,
            allowed_tools=self._allowed_tools,
            disallowed_tools=self._disallowed_tools,
            stderr_buffer=self._stderr_buffer,
        )
        options_kwargs["hooks"] = build_hooks_config(
            hook_context=hook_context,
            client_config=self._client_config,
            collector=collector,
        )
        options_kwargs = filter_unsupported_options(
            options_kwargs,
            options_type=ClaudeAgentOptions,
        )

        return await run_sdk_query(
            options_kwargs=options_kwargs,
            prompt_text=prompt_text,
            hook_context=hook_context,
            visibility_signal=visibility_signal,
            stderr_buffer=self._stderr_buffer,
            task_completion_checker=self._client_config.task_completion_checker,
        )
