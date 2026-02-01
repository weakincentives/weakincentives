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

"""OpenCode ACP adapter implementation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast, override

from ...budget import Budget, BudgetTracker
from ...dataclasses import FrozenDataclass
from ...deadlines import Deadline
from ...prompt import Prompt, RenderedPrompt
from ...prompt.errors import VisibilityExpansionRequired
from ...prompt.protocols import PromptProtocol, RenderedPromptProtocol
from ...prompt.tool import ToolContext, ToolResult
from ...runtime.events import PromptExecuted, PromptRendered
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from ...runtime.watchdog import Heartbeat
from ...types import OPENCODE_ACP_ADAPTER_NAME
from ..claude_agent_sdk._bridge import create_bridged_tools, create_mcp_server
from ..claude_agent_sdk._visibility_signal import VisibilityExpansionSignal
from ..core import PromptEvaluationError, PromptResponse, ProviderAdapter
from ..response_parser import build_json_schema_response_format
from ._async import run_async
from ._events import is_wink_tool_call, map_tool_call_to_event
from ._state import OpenCodeACPSessionState
from ._structured_output import (
    StructuredOutputParams,
    StructuredOutputSignal,
    create_structured_output_tool_spec,
)
from .client import OpenCodeACPClient
from .config import OpenCodeACPAdapterConfig, OpenCodeACPClientConfig
from .workspace import OpenCodeWorkspaceSection

__all__ = [
    "OpenCodeACPAdapter",
]

logger: StructuredLogger = get_logger(__name__, context={"component": "opencode_acp"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


@FrozenDataclass()
class _StructuredOutputToolSpec:
    """Internal spec for the structured_output bridged tool."""

    name: str = "structured_output"
    description: str = "Submit your final structured output."
    input_schema: dict[str, Any] | None = None


class OpenCodeACPAdapter[OutputT](ProviderAdapter[OutputT]):
    """Adapter using OpenCode ACP for agentic execution.

    This adapter delegates prompt execution to OpenCode via the Agent Client
    Protocol (ACP). WINK handles prompt composition, resource binding, and
    session telemetry while OpenCode handles the agentic execution including
    planning, reasoning, tool calls, and file edits.

    Example:
        >>> from weakincentives import Prompt, PromptTemplate, MarkdownSection
        >>> from weakincentives.runtime import Session, InProcessDispatcher
        >>> from weakincentives.adapters.opencode_acp import (
        ...     OpenCodeACPAdapter,
        ...     OpenCodeACPClientConfig,
        ... )
        >>>
        >>> bus = InProcessDispatcher()
        >>> session = Session(dispatcher=bus)
        >>>
        >>> adapter = OpenCodeACPAdapter(
        ...     client_config=OpenCodeACPClientConfig(
        ...         cwd="/absolute/path/to/workspace",
        ...         permission_mode="auto",
        ...         allow_file_reads=True,
        ...     ),
        ... )
        >>>
        >>> template = PromptTemplate(
        ...     ns="demo",
        ...     key="opencode",
        ...     sections=(
        ...         MarkdownSection(
        ...             title="Task",
        ...             key="task",
        ...             template="List the files in the repo and summarize.",
        ...         ),
        ...     ),
        ... )
        >>> prompt = Prompt(template)
        >>>
        >>> with prompt.resources:
        ...     resp = adapter.evaluate(prompt, session=session)
        >>>
        >>> print(resp.text)
    """

    def __init__(
        self,
        *,
        client_config: OpenCodeACPClientConfig | None = None,
        adapter_config: OpenCodeACPAdapterConfig | None = None,
    ) -> None:
        """Initialize the OpenCode ACP adapter.

        Args:
            client_config: Client-level configuration for OpenCode ACP.
            adapter_config: Adapter-level configuration.
        """
        self._client_config = client_config or OpenCodeACPClientConfig()
        self._adapter_config = adapter_config or OpenCodeACPAdapterConfig()

        logger.debug(
            "opencode_acp.adapter.init",
            event="adapter.init",
            context={
                "opencode_bin": self._client_config.opencode_bin,
                "opencode_args": self._client_config.opencode_args,
                "cwd": self._client_config.cwd,
                "permission_mode": self._client_config.permission_mode,
                "allow_file_reads": self._client_config.allow_file_reads,
                "allow_file_writes": self._client_config.allow_file_writes,
                "mode_id": self._adapter_config.mode_id,
                "model_id": self._adapter_config.model_id,
                "reuse_session": self._client_config.reuse_session,
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
        """Evaluate prompt using OpenCode ACP.

        Args:
            prompt: The prompt to evaluate.
            session: Session for state management and event dispatching.
            deadline: Optional deadline for execution timeout.
            budget: Optional token budget constraints.
            budget_tracker: Optional shared budget tracker.
            heartbeat: Optional heartbeat for lease extension.
            run_context: Optional run context for tracing.

        Returns:
            PromptResponse with text and structured output.

        Raises:
            PromptEvaluationError: If execution fails.
        """
        if budget and not budget_tracker:
            budget_tracker = BudgetTracker(budget)

        effective_deadline = deadline or (budget.deadline if budget else None)

        prompt_name = prompt.name or f"{prompt.ns}:{prompt.key}"

        logger.debug(
            "opencode_acp.evaluate.entry",
            event="evaluate.entry",
            context={
                "prompt_name": prompt_name,
                "prompt_ns": prompt.ns,
                "prompt_key": prompt.key,
                "has_deadline": effective_deadline is not None,
                "has_budget": budget is not None,
            },
        )

        if effective_deadline and effective_deadline.remaining().total_seconds() <= 0:
            logger.debug(
                "opencode_acp.evaluate.deadline_expired",
                event="evaluate.deadline_expired",
                context={"prompt_name": prompt_name},
            )
            raise PromptEvaluationError(
                message="Deadline expired before ACP invocation",
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
        prompt_name = prompt.name or f"{prompt.ns}:{prompt.key}"

        # Render the prompt
        rendered = prompt.render(session=session)
        prompt_text = rendered.text

        logger.debug(
            "opencode_acp.evaluate.rendered",
            event="evaluate.rendered",
            context={
                "prompt_text_length": len(prompt_text),
                "tool_count": len(rendered.tools),
                "tool_names": [t.name for t in rendered.tools],
                "has_output_type": rendered.output_type is not None,
            },
        )

        # Dispatch PromptRendered event
        session.dispatcher.dispatch(
            PromptRendered(
                prompt_ns=prompt.ns,
                prompt_key=prompt.key,
                prompt_name=prompt.name,
                adapter=OPENCODE_ACP_ADAPTER_NAME,
                session_id=getattr(session, "session_id", None),
                render_inputs=(),
                rendered_prompt=prompt_text,
                created_at=_utcnow(),
                descriptor=None,
                run_context=run_context,
            )
        )

        # Find workspace section if present
        workspace = self._find_workspace_section(cast(PromptProtocol[Any], prompt))

        # Enter resource context for lifecycle management
        with prompt.resources:
            return await self._run_with_prompt_context(
                prompt=prompt,
                prompt_name=prompt_name,
                prompt_text=prompt_text,
                rendered=rendered,
                session=session,
                deadline=deadline,
                budget_tracker=budget_tracker,
                heartbeat=heartbeat,
                run_context=run_context,
                workspace=workspace,
            )

    def _find_workspace_section(
        self, prompt: PromptProtocol[Any]
    ) -> OpenCodeWorkspaceSection | None:
        """Find an OpenCodeWorkspaceSection in the prompt if present."""
        for section in prompt.template.sections:
            if isinstance(section, OpenCodeWorkspaceSection):
                return section
        return None

    async def _run_with_prompt_context(
        self,
        *,
        prompt: Prompt[OutputT],
        prompt_name: str,
        prompt_text: str,
        rendered: RenderedPrompt[OutputT],
        session: SessionProtocol,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        heartbeat: Heartbeat | None,
        run_context: RunContext | None,
        workspace: OpenCodeWorkspaceSection | None,
    ) -> PromptResponse[OutputT]:
        """Run ACP within prompt context."""
        start_time = _utcnow()

        # Create visibility signal for progressive disclosure
        visibility_signal = VisibilityExpansionSignal()

        # Create structured output signal
        structured_output_signal = StructuredOutputSignal()

        # Create bridged tools for WINK tools
        bridged_tools = create_bridged_tools(
            rendered.tools,
            session=session,
            adapter=self,
            prompt=cast(Any, prompt),
            rendered_prompt=rendered,
            deadline=deadline,
            budget_tracker=budget_tracker,
            adapter_name=OPENCODE_ACP_ADAPTER_NAME,
            prompt_name=prompt_name,
            heartbeat=heartbeat,
            run_context=run_context,
            visibility_signal=visibility_signal,
        )

        # Create structured_output tool if output type declared
        structured_output_tool = None
        if rendered.output_type is not None and rendered.output_type is not type(None):
            structured_output_tool = self._create_structured_output_tool(
                rendered=rendered,
                prompt_name=prompt_name,
                signal=structured_output_signal,
            )

        # Combine tools
        all_bridged_tools = bridged_tools
        if structured_output_tool is not None:
            all_bridged_tools = (*bridged_tools, structured_output_tool)

        # Create MCP server with all bridged tools
        mcp_server_config = None
        if all_bridged_tools:
            mcp_server_config = create_mcp_server(
                all_bridged_tools,
                server_name="wink-tools",
            )

        logger.debug(
            "opencode_acp.run_context.tools_bridged",
            event="run_context.tools_bridged",
            context={
                "bridged_tool_count": len(all_bridged_tools),
                "has_structured_output_tool": structured_output_tool is not None,
            },
        )

        # Determine cwd
        cwd = self._resolve_cwd(workspace)

        # Create client
        client = OpenCodeACPClient(
            self._client_config,
            workspace=workspace,
        )

        try:
            # Connect and initialize
            await client.connect()
            await client.initialize()

            # Create or load session
            await self._establish_session(
                client=client,
                session=session,
                cwd=cwd,
                mcp_server_config=mcp_server_config,
                workspace=workspace,
            )

            # Set mode/model (best-effort)
            if self._adapter_config.mode_id:
                await client.set_mode(self._adapter_config.mode_id)
            if self._adapter_config.model_id:
                await client.set_model(self._adapter_config.model_id)

            # Send prompt
            await client.prompt(prompt_text)

            # Wait for updates to drain
            await self._drain_updates(client, deadline)

            # Check for visibility expansion signal
            stored_exc = visibility_signal.get_and_clear()
            if stored_exc is not None:
                logger.debug(
                    "opencode_acp.run_context.visibility_expansion_detected",
                    event="run_context.visibility_expansion_detected",
                    context={
                        "section_keys": stored_exc.section_keys,
                        "reason": stored_exc.reason,
                    },
                )
                raise stored_exc

            # Extract results
            result_text, output = self._extract_results(
                client=client,
                rendered=rendered,
                structured_output_signal=structured_output_signal,
                prompt_name=prompt_name,
                session=session,
                run_context=run_context,
            )

            end_time = _utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            response = PromptResponse(
                prompt_name=prompt_name,
                text=result_text,
                output=output,
            )

            # Dispatch PromptExecuted event
            session.dispatcher.dispatch(
                PromptExecuted(
                    prompt_name=prompt_name,
                    adapter=OPENCODE_ACP_ADAPTER_NAME,
                    result=response,
                    session_id=getattr(session, "session_id", None),
                    created_at=_utcnow(),
                    usage=None,  # ACP doesn't reliably expose token usage
                    run_context=run_context,
                )
            )

            logger.info(
                "opencode_acp.evaluate.complete",
                event="evaluate.complete",
                context={
                    "prompt_name": prompt_name,
                    "duration_ms": duration_ms,
                    "has_output": output is not None,
                },
            )

            return response

        except VisibilityExpansionRequired:
            # Progressive disclosure - let this propagate
            raise

        except Exception as error:
            logger.warning(
                "opencode_acp.evaluate.error",
                event="evaluate.error",
                context={
                    "prompt_name": prompt_name,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
            raise PromptEvaluationError(
                message=str(error),
                prompt_name=prompt_name,
                phase="request",
            ) from error

        finally:
            await client.disconnect()

    def _resolve_cwd(self, workspace: OpenCodeWorkspaceSection | None) -> str:
        """Resolve the working directory for the session."""
        if workspace is not None:
            return str(workspace.temp_dir)

        if self._client_config.cwd is not None:
            return self._client_config.cwd

        return str(Path.cwd().resolve())

    def _create_structured_output_tool(
        self,
        rendered: RenderedPromptProtocol[Any],
        prompt_name: str,
        signal: StructuredOutputSignal,
    ) -> Any:
        """Create the structured_output bridged tool.

        Returns a tool-like object compatible with create_mcp_server.
        """

        # Build JSON schema
        schema_format = build_json_schema_response_format(
            cast(RenderedPrompt[Any], rendered), prompt_name
        )
        json_schema: dict[str, Any] = {}
        if schema_format is not None:
            json_schema_wrapper = schema_format.get("json_schema")
            if isinstance(json_schema_wrapper, dict):
                wrapper_dict = cast(dict[str, Any], json_schema_wrapper)
                schema_value = wrapper_dict.get("schema")
                if isinstance(schema_value, dict):
                    json_schema = cast(dict[str, Any], schema_value)

        spec = create_structured_output_tool_spec(json_schema)

        # Create a wrapper that captures signal
        def handler(
            params: StructuredOutputParams, *, context: ToolContext
        ) -> ToolResult[None]:
            from ._structured_output import structured_output_handler

            return structured_output_handler(
                params,
                context=context,
                signal=signal,
                rendered=rendered,
            )

        # Create a Tool-like object for BridgedTool
        from ...prompt.tool import Tool

        return Tool[StructuredOutputParams, None](
            name="structured_output",
            description=spec["description"],
            handler=handler,
        )

    async def _establish_session(
        self,
        *,
        client: OpenCodeACPClient,
        session: SessionProtocol,
        cwd: str,
        mcp_server_config: Any,
        workspace: OpenCodeWorkspaceSection | None,
    ) -> str:
        """Establish a new or load existing OpenCode session."""
        mcp_servers: list[dict[str, Any]] = []
        if mcp_server_config is not None:
            mcp_servers.append({"name": "wink", "config": mcp_server_config})

        # Add user-provided MCP servers
        mcp_servers.extend(
            {
                "name": server_cfg.name,
                "command": server_cfg.command,
                "args": list(server_cfg.args),
                "env": server_cfg.env,
            }
            for server_cfg in self._client_config.mcp_servers
        )

        # Compute workspace fingerprint
        workspace_fingerprint = (
            workspace.workspace_fingerprint if workspace is not None else None
        )

        # Try to reuse session if configured
        if self._client_config.reuse_session:
            state = session[OpenCodeACPSessionState].latest()
            can_reuse = (
                state is not None
                and state.cwd == cwd
                and state.workspace_fingerprint == workspace_fingerprint
            )
            if can_reuse and state is not None:
                if await client.load_session(
                    session_id=state.session_id,
                    cwd=cwd,
                    mcp_servers=mcp_servers,
                ):
                    logger.debug(
                        "opencode_acp.session.reused",
                        event="session.reused",
                        context={"session_id": state.session_id},
                    )
                    return state.session_id

        # Create new session
        session_id = await client.new_session(cwd=cwd, mcp_servers=mcp_servers)

        # Store session state for potential reuse
        session[OpenCodeACPSessionState].seed(
            OpenCodeACPSessionState(
                session_id=session_id,
                cwd=cwd,
                workspace_fingerprint=workspace_fingerprint,
            )
        )

        return session_id

    async def _drain_updates(
        self,
        client: OpenCodeACPClient,
        deadline: Deadline | None,
    ) -> None:
        """Wait for updates to drain after prompt.

        Uses quiet_period_ms to wait until no updates for a period,
        with total wait capped by deadline.
        """
        quiet_period = self._adapter_config.quiet_period_ms / 1000.0

        # Simple wait - in a real implementation we'd monitor for updates
        # and reset the timer on each one
        if deadline is not None:
            remaining = deadline.remaining().total_seconds()
            wait_time = min(quiet_period, max(0, remaining))
        else:
            wait_time = quiet_period

        if wait_time > 0:
            await asyncio.sleep(wait_time)

    def _extract_results(
        self,
        *,
        client: OpenCodeACPClient,
        rendered: RenderedPrompt[OutputT],
        structured_output_signal: StructuredOutputSignal,
        prompt_name: str,
        session: SessionProtocol,
        run_context: RunContext | None,
    ) -> tuple[str | None, OutputT | None]:
        """Extract text and structured output from session.

        Also dispatches ToolInvoked events for non-WINK tools.
        """
        accumulator = client.accumulator

        # Get text result
        if self._adapter_config.emit_thought_chunks:
            result_text = accumulator.final_text_with_thoughts
        else:
            result_text = accumulator.final_text

        # Dispatch ToolInvoked events for non-WINK tools
        for call in accumulator.completed_tool_calls():
            if is_wink_tool_call(call.tool_name, call.mcp_server_name):
                # Skip - BridgedTool already emitted the event
                continue

            event = map_tool_call_to_event(
                tool_name=call.tool_name,
                tool_use_id=call.tool_use_id,
                params=call.params,
                result=call.result,
                success=call.status == "completed",
                prompt_name=prompt_name,
                adapter_name=OPENCODE_ACP_ADAPTER_NAME,
                session_id=getattr(session, "session_id", None),
                run_context=run_context,
            )
            session.dispatcher.dispatch(event)

        # Get structured output
        output: OutputT | None = None
        if rendered.output_type is not None and rendered.output_type is not type(None):
            data, error = structured_output_signal.get()
            if data is not None:
                output = cast(OutputT, data)
            elif error is not None:
                logger.warning(
                    "opencode_acp.extract.structured_output_error",
                    event="extract.structured_output_error",
                    context={"error": error},
                )
                raise PromptEvaluationError(
                    message=f"Structured output validation failed: {error}",
                    prompt_name=prompt_name,
                    phase="response",
                )
            elif not structured_output_signal.is_set():
                logger.warning(
                    "opencode_acp.extract.structured_output_missing",
                    event="extract.structured_output_missing",
                )
                raise PromptEvaluationError(
                    message="Model did not call structured_output tool",
                    prompt_name=prompt_name,
                    phase="response",
                )

        return result_text or None, output
