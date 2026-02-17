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

"""Generic ACP adapter implementation."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast, override
from uuid import uuid4

from ...budget import Budget, BudgetTracker
from ...clock import SYSTEM_CLOCK, AsyncSleeper, MonotonicClock
from ...deadlines import Deadline
from ...filesystem import Filesystem, HostFilesystem
from ...prompt import Prompt, RenderedPrompt
from ...prompt.errors import VisibilityExpansionRequired
from ...prompt.structured_output import OutputParseError, parse_structured_output
from ...runtime.events import PromptExecuted, PromptRendered
from ...runtime.events.types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from ...runtime.session.rendered_tools import RenderedTools
from ...runtime.transcript import TranscriptEmitter
from ...runtime.watchdog import Heartbeat
from ...types import ACP_ADAPTER_NAME, AdapterName
from .._shared._bridge import BridgedTool, create_bridged_tools
from .._shared._visibility_signal import VisibilityExpansionSignal
from ..core import PromptEvaluationError, PromptResponse, ProviderAdapter
from ..tool_spec import extract_tool_schema
from ._async import run_async
from ._events import dispatch_tool_invoked, extract_token_usage
from ._guardrails import (
    accumulate_usage,
    append_feedback as _append_feedback,
    check_task_completion,
)
from ._mcp_http import MCPHttpServer, create_mcp_tool_server
from ._structured_output import create_structured_output_tool
from ._transcript import ACPTranscriptBridge
from .client import ACPClient
from .config import ACPAdapterConfig, ACPClientConfig

__all__ = ["ACPAdapter"]

logger: StructuredLogger = get_logger(__name__, context={"component": "acp_adapter"})


def _noop() -> None:
    pass


def _utcnow() -> datetime:
    return SYSTEM_CLOCK.utcnow()


def _extract_chunk_text(chunk: Any) -> str:
    """Extract text content from an ACP update chunk.

    The ``content`` attribute may be a plain string, a ``TextContentBlock``
    with a ``.text`` attribute, or a list of content blocks.
    """
    raw = getattr(chunk, "content", "")
    if isinstance(raw, str):
        return raw
    # TextContentBlock or similar pydantic model
    text = getattr(raw, "text", None)
    if isinstance(text, str):
        return text
    # List of content blocks
    if isinstance(raw, list):
        return "".join(getattr(b, "text", str(b)) for b in raw if b)
    return str(raw) if raw else ""


class ACPAdapter(ProviderAdapter[Any]):
    """Generic ACP adapter for agentic prompt evaluation.

    Spawns an ACP-compatible agent binary as a subprocess, communicates over
    the Agent Client Protocol (JSON-RPC 2.0 on stdio), and bridges WINK tools
    via an in-process MCP HTTP server.

    Subclass hooks allow OpenCode-specific (or other agent-specific) behavior.
    """

    def __init__(
        self,
        *,
        adapter_config: ACPAdapterConfig | None = None,
        client_config: ACPClientConfig | None = None,
        async_sleeper: AsyncSleeper = SYSTEM_CLOCK,
        clock: MonotonicClock = SYSTEM_CLOCK,
    ) -> None:
        super().__init__()
        self._adapter_config = adapter_config or ACPAdapterConfig()
        self._client_config = client_config or ACPClientConfig()
        self._async_sleeper = async_sleeper
        self._clock = clock

        logger.debug(
            "acp.adapter.init",
            event="adapter.init",
            context={
                "agent_bin": self._client_config.agent_bin,
                "cwd": self._client_config.cwd,
                "permission_mode": self._client_config.permission_mode,
            },
        )

    @property
    @override
    def adapter_name(self) -> str:
        """Return the canonical adapter name. Override in subclasses."""
        return self._adapter_name()

    def _adapter_name(self) -> AdapterName:
        """Return the canonical adapter name. Override in subclasses."""
        return ACP_ADAPTER_NAME

    @override
    def evaluate[OutputT](
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
        """Evaluate prompt using an ACP agent."""
        if budget and not budget_tracker:
            budget_tracker = BudgetTracker(budget)

        effective_deadline = deadline or (budget.deadline if budget else None)
        prompt_name = prompt.name or f"{prompt.ns}:{prompt.key}"

        if effective_deadline and effective_deadline.remaining().total_seconds() <= 0:
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

    async def _evaluate_async[OutputT](
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
        rendered = prompt.render(session=session)
        prompt_text = rendered.text
        prompt_name = prompt.name or f"{prompt.ns}:{prompt.key}"
        adapter_name = self._adapter_name()

        session_id = getattr(session, "session_id", None)
        render_event_id = uuid4()
        created_at = _utcnow()

        # Dispatch PromptRendered
        _ = session.dispatcher.dispatch(
            PromptRendered(
                prompt_ns=prompt.ns,
                prompt_key=prompt.key,
                prompt_name=prompt.name,
                adapter=adapter_name,
                session_id=session_id,
                render_inputs=(),
                rendered_prompt=prompt_text,
                created_at=created_at,
                descriptor=None,
                run_context=run_context,
                event_id=render_event_id,
            )
        )

        # Dispatch RenderedTools
        tool_schemas = tuple(extract_tool_schema(tool) for tool in rendered.tools)
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
                "acp.evaluate.rendered_tools_dispatch_failed",
                event="rendered_tools_dispatch_failed",
                context={
                    "failure_count": len(tools_result.errors),
                    "tool_count": len(tool_schemas),
                },
            )

        # Determine CWD
        effective_cwd, temp_workspace_dir, prompt = self._resolve_cwd(prompt)

        try:
            with prompt.resources:
                return await self._run_acp(
                    prompt=prompt,
                    prompt_name=prompt_name,
                    prompt_text=prompt_text,
                    rendered=rendered,
                    session=session,
                    deadline=deadline,
                    budget_tracker=budget_tracker,
                    heartbeat=heartbeat,
                    run_context=run_context,
                    effective_cwd=effective_cwd,
                )
        finally:
            if temp_workspace_dir:
                shutil.rmtree(temp_workspace_dir, ignore_errors=True)

    def _resolve_cwd[OutputT](
        self, prompt: Prompt[OutputT]
    ) -> tuple[str, str | None, Prompt[OutputT]]:
        """Determine the effective cwd and optionally bind a filesystem."""
        temp_workspace_dir: str | None = None
        effective_cwd: str | None = self._client_config.cwd

        if prompt.filesystem() is None:
            if effective_cwd is None:
                temp_workspace_dir = tempfile.mkdtemp(prefix="wink-acp-")
                effective_cwd = temp_workspace_dir
            filesystem = HostFilesystem(_root=effective_cwd)
            prompt = prompt.bind(resources={Filesystem: filesystem})
        elif effective_cwd is None:
            fs = prompt.filesystem()
            if isinstance(fs, HostFilesystem):
                effective_cwd = fs.root

        if effective_cwd is None:
            effective_cwd = str(Path.cwd().resolve())

        return effective_cwd, temp_workspace_dir, prompt

    async def _run_acp[OutputT](
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
        effective_cwd: str,
    ) -> PromptResponse[OutputT]:
        """Run the full ACP protocol flow."""
        adapter_name = self._adapter_name()
        visibility_signal = VisibilityExpansionSignal()

        all_tools, structured_capture = self._prepare_tools(
            rendered=rendered,
            session=session,
            prompt=prompt,
            deadline=deadline,
            budget_tracker=budget_tracker,
            adapter_name=adapter_name,
            prompt_name=prompt_name,
            heartbeat=heartbeat,
            run_context=run_context,
            visibility_signal=visibility_signal,
        )

        def _feedback_hook(
            name: str, args: dict[str, Any], result: dict[str, Any]
        ) -> None:
            is_error = result.get("isError", False)
            content = result.get("content", [])
            _append_feedback(
                content,
                is_error=is_error,
                prompt=prompt,
                session=session,
                deadline=deadline,
            )

        mcp_server = create_mcp_tool_server(
            tuple(all_tools), post_call_hook=_feedback_hook
        )

        start_time = _utcnow()
        client = ACPClient(
            self._client_config, workspace_root=effective_cwd, clock=self._clock
        )

        # Create transcript bridge for debug bundle visibility.
        session_id = getattr(session, "session_id", None)
        emitter = TranscriptEmitter(
            prompt_name=prompt_name,
            adapter=str(adapter_name),
            session_id=str(session_id) if session_id else None,
        )
        bridge = ACPTranscriptBridge(emitter)
        client.set_transcript_bridge(bridge)
        emitter.start()

        try:
            bridge.on_user_message(prompt_text)
            result = await self._execute_protocol(
                client=client,
                mcp_server=mcp_server,
                session=session,
                prompt_name=prompt_name,
                prompt_text=prompt_text,
                rendered=rendered,
                effective_cwd=effective_cwd,
                deadline=deadline,
                budget_tracker=budget_tracker,
                prompt=prompt,
                run_context=run_context,
                visibility_signal=visibility_signal,
                structured_capture=structured_capture,
            )
        except VisibilityExpansionRequired:
            raise
        except PromptEvaluationError:
            raise
        except Exception as error:
            raise PromptEvaluationError(
                message=f"ACP execution failed: {error}",
                prompt_name=prompt_name,
                phase="request",
            ) from error
        finally:
            bridge.flush()
            emitter.stop()

        return self._finalize_response(
            result=result,
            rendered=rendered,
            prompt_name=prompt_name,
            adapter_name=adapter_name,
            session=session,
            budget_tracker=budget_tracker,
            run_context=run_context,
            start_time=start_time,
            structured_capture=structured_capture,
        )

    def _prepare_tools[OutputT](
        self,
        *,
        rendered: RenderedPrompt[OutputT],
        session: SessionProtocol,
        prompt: Prompt[OutputT],
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        adapter_name: str,
        prompt_name: str,
        heartbeat: Heartbeat | None,
        run_context: RunContext | None,
        visibility_signal: VisibilityExpansionSignal,
    ) -> tuple[list[BridgedTool | Any], Any]:
        """Build bridged tools and optional structured output tool."""
        bridged_tools = create_bridged_tools(
            rendered.tools,
            session=session,
            adapter=self,
            prompt=cast(Any, prompt),
            rendered_prompt=rendered,
            deadline=deadline,
            budget_tracker=budget_tracker,
            adapter_name=adapter_name,
            prompt_name=prompt_name,
            heartbeat=heartbeat,
            run_context=run_context,
            visibility_signal=visibility_signal,
        )

        all_tools: list[BridgedTool | Any] = list(bridged_tools)
        structured_capture = None

        if rendered.output_type is not None:
            so_tool, structured_capture = create_structured_output_tool(
                rendered.output_type,
                container=rendered.container or "object",
            )
            all_tools.append(so_tool)

        return all_tools, structured_capture

    def _finalize_response[OutputT](
        self,
        *,
        result: tuple[str | None, TokenUsage | None],
        rendered: RenderedPrompt[OutputT],
        prompt_name: str,
        adapter_name: str,
        session: SessionProtocol,
        budget_tracker: BudgetTracker | None,
        run_context: RunContext | None,
        start_time: datetime,
        structured_capture: Any,
    ) -> PromptResponse[OutputT]:
        """Build PromptResponse and dispatch PromptExecuted."""
        accumulated_text, usage = result
        duration_ms = int((_utcnow() - start_time).total_seconds() * 1000)

        output: OutputT | None = None
        if rendered.output_type is not None:
            output = self._resolve_structured_output(
                accumulated_text, rendered, prompt_name, structured_capture
            )

        if budget_tracker and usage:
            budget_tracker.record_cumulative(prompt_name, usage)

        response = PromptResponse(
            prompt_name=prompt_name,
            text=accumulated_text,
            output=output,
        )

        _ = session.dispatcher.dispatch(
            PromptExecuted(
                prompt_name=prompt_name,
                adapter=adapter_name,
                result=response,
                session_id=getattr(session, "session_id", None),
                created_at=_utcnow(),
                usage=usage,
                run_context=run_context,
            )
        )

        logger.info(
            "acp.evaluate.complete",
            event="evaluate.complete",
            context={
                "prompt_name": prompt_name,
                "duration_ms": duration_ms,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
            },
        )

        return response

    async def _execute_protocol[OutputT](
        self,
        *,
        client: ACPClient,
        mcp_server: Any,
        session: SessionProtocol,
        prompt_name: str,
        prompt_text: str,
        rendered: RenderedPrompt[OutputT],
        effective_cwd: str,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None = None,
        prompt: Prompt[OutputT] | None = None,
        run_context: RunContext | None,
        visibility_signal: VisibilityExpansionSignal,
        structured_capture: Any,
    ) -> tuple[str | None, TokenUsage | None]:
        """Execute the ACP protocol flow."""
        try:
            from acp import spawn_agent_process
        except ImportError as e:
            msg = (
                "agent-client-protocol is required for the ACP adapter. "
                "Install with: pip install 'weakincentives[acp]'"
            )
            raise ImportError(msg) from e

        adapter_name = self._adapter_name()

        # Start MCP HTTP server
        mcp_http = MCPHttpServer(mcp_server, server_name="wink-tools")
        await mcp_http.start()

        env_cleanup: Callable[[], None] = _noop
        try:
            wink_mcp = mcp_http.to_http_mcp_server()
            mcp_servers = [wink_mcp, *self._client_config.mcp_servers]
            merged_env, env_cleanup = self._prepare_execution_env(
                rendered=rendered,
                effective_cwd=effective_cwd,
            )

            # Use 10 MB readline limit (asyncio default is 64 KB) to
            # handle large JSON-RPC messages on stdio.
            stdio_limit = 10 * 1024 * 1024

            async with spawn_agent_process(
                client,
                self._client_config.agent_bin,
                *self._client_config.agent_args,
                cwd=effective_cwd,
                env=merged_env,
                transport_kwargs={"limit": stdio_limit},
            ) as (conn, _proc):
                try:
                    acp_session_id = await asyncio.wait_for(
                        self._handshake(conn, effective_cwd, mcp_servers),
                        timeout=self._client_config.startup_timeout_s,
                    )
                except TimeoutError:
                    raise PromptEvaluationError(
                        message=(
                            "ACP handshake timed out after "
                            f"{self._client_config.startup_timeout_s}s"
                        ),
                        prompt_name=prompt_name,
                        phase="request",
                    ) from None
                await self._configure_session(conn, acp_session_id)

                return await self._run_prompt_loop(
                    conn=conn,
                    acp_session_id=acp_session_id,
                    client=client,
                    session=session,
                    adapter_name=adapter_name,
                    prompt_name=prompt_name,
                    prompt_text=prompt_text,
                    deadline=deadline,
                    budget_tracker=budget_tracker,
                    prompt=prompt,
                    run_context=run_context,
                    visibility_signal=visibility_signal,
                    structured_capture=structured_capture,
                )
        finally:
            await mcp_http.stop()
            env_cleanup()

    async def _run_prompt_loop[OutputT](
        self,
        *,
        conn: Any,
        acp_session_id: str,
        client: ACPClient,
        session: SessionProtocol,
        adapter_name: str,
        prompt_name: str,
        prompt_text: str,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        prompt: Prompt[OutputT] | None,
        run_context: RunContext | None,
        visibility_signal: VisibilityExpansionSignal,
        structured_capture: Any,
    ) -> tuple[str | None, TokenUsage | None]:
        """Run the prompt turn + task completion continuation loop."""
        from acp.schema import TextContentBlock

        max_continuation_rounds = 10
        continuation_round = 0
        current_prompt_text = prompt_text
        accumulated_text: str | None = None
        usage: TokenUsage | None = None

        while True:
            prompt_resp = await self._send_prompt(
                conn=conn,
                acp_session_id=acp_session_id,
                text=current_prompt_text,
                prompt_name=prompt_name,
                deadline=deadline,
                text_content_block_cls=TextContentBlock,
            )
            await self._drain_quiet_period(client, deadline)

            # Skip empty response check on first round only.
            need_empty_check = continuation_round == 0 and not (
                structured_capture is not None and structured_capture.called
            )
            if need_empty_check:
                self._detect_empty_response(client, prompt_resp)

            stored_exc = visibility_signal.get_and_clear()
            if stored_exc is not None:
                raise stored_exc

            for tc_id, tc_data in client.tool_call_tracker.items():
                dispatch_tool_invoked(
                    session=session,
                    adapter_name=adapter_name,
                    prompt_name=prompt_name,
                    run_context=run_context,
                    tool_call_id=tc_id,
                    title=tc_data.get("title", ""),
                    status=tc_data.get("status", "completed"),
                    rendered_output=tc_data.get("output", ""),
                )

            accumulated_text = self._extract_text(client)
            turn_usage = extract_token_usage(prompt_resp.usage if prompt_resp else None)
            if turn_usage is not None:
                usage = accumulate_usage(usage, turn_usage)

            should_continue, feedback = check_task_completion(
                prompt=prompt,
                session=session,
                accumulated_text=accumulated_text,
                deadline=deadline,
                budget_tracker=budget_tracker,
            )
            if should_continue and continuation_round < max_continuation_rounds:
                current_prompt_text = feedback  # type: ignore[assignment]
                continuation_round += 1
                client.reset_tracking()
                continue
            break

        return accumulated_text, usage

    async def _send_prompt(
        self,
        *,
        conn: Any,
        acp_session_id: str,
        text: str,
        prompt_name: str,
        deadline: Deadline | None,
        text_content_block_cls: Any,
    ) -> Any:
        """Send a single prompt and return the response."""
        prompt_coro = conn.prompt(
            [text_content_block_cls(type="text", text=text)],
            session_id=acp_session_id,
        )
        timeout_s = deadline.remaining().total_seconds() if deadline else None
        try:
            return await asyncio.wait_for(prompt_coro, timeout=timeout_s)
        except TimeoutError:
            raise PromptEvaluationError(
                message="ACP prompt timed out (deadline expired)",
                prompt_name=prompt_name,
                phase="request",
            ) from None

    def _build_env(self) -> dict[str, str] | None:
        """Build merged environment variables.

        When ``config.env`` is set, the full ``os.environ`` is forwarded with
        config entries taking precedence.  This mirrors stdlib
        ``subprocess.Popen`` behaviour where ``env=None`` inherits the parent
        environment.  Returning ``None`` (no config env) lets the subprocess
        inherit the parent env via the stdlib default.
        """
        if not self._client_config.env:
            return None
        import os

        return {**os.environ, **self._client_config.env}

    async def _handshake(
        self,
        conn: Any,
        effective_cwd: str,
        mcp_servers: list[Any],
    ) -> str:
        """Initialize and create session. Returns session ID."""
        from acp import PROTOCOL_VERSION
        from acp.schema import (
            ClientCapabilities,
            FileSystemCapability,
            Implementation,
        )

        _ = await conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(
                    read_text_file=self._client_config.allow_file_reads,
                    write_text_file=self._client_config.allow_file_writes,
                ),
                terminal=False,
            ),
            client_info=Implementation(
                name="wink",
                title="WINK",
                version="0.1.0",
            ),
        )

        new_session_resp = await conn.new_session(
            cwd=effective_cwd,
            mcp_servers=mcp_servers,
        )
        acp_session_id: str = new_session_resp.session_id

        # Validate model against available models
        available_models = (
            new_session_resp.models.available_models if new_session_resp.models else []
        )
        if self._adapter_config.model_id:
            self._validate_model(self._adapter_config.model_id, available_models)

        return acp_session_id

    async def _configure_session(self, conn: Any, session_id: str) -> None:
        """Configure model and mode on the session (best-effort)."""
        if self._adapter_config.model_id:
            try:
                await conn.set_session_model(
                    session_id=session_id,
                    model_id=self._adapter_config.model_id,
                )
            except Exception as err:
                logger.warning(
                    "acp.set_model.failed",
                    event="set_model.failed",
                    context={"error": str(err)},
                )

        if self._adapter_config.mode_id:
            try:
                await conn.set_session_mode(
                    session_id=session_id,
                    mode_id=self._adapter_config.mode_id,
                )
            except Exception as err:
                self._handle_mode_error(err)

    _MAX_DRAIN_S: float = 30.0

    async def _drain_quiet_period(
        self,
        client: ACPClient,
        deadline: Deadline | None,
    ) -> None:
        """Wait until no new updates arrive for quiet_period_ms.

        If no updates have been received (``client.last_update_time is None``),
        the drain exits immediately.  A hard cap of ``_MAX_DRAIN_S`` prevents
        unbounded waiting when no deadline is set.
        """
        if client.last_update_time is None:
            return

        quiet_s = self._adapter_config.quiet_period_ms / 1000.0
        now = self._clock.monotonic()
        hard_cap = now + self._MAX_DRAIN_S
        deadline_time = (
            self._clock.monotonic() + deadline.remaining().total_seconds()
            if deadline
            else None
        )
        if deadline_time is not None:
            effective_deadline = min(deadline_time, hard_cap)
        else:
            effective_deadline = hard_cap

        while True:
            now = self._clock.monotonic()
            if now >= effective_deadline:
                break

            snapshot = client.last_update_time
            if snapshot is None:
                break

            elapsed = now - snapshot
            if elapsed >= quiet_s:
                break

            wait_s = quiet_s - elapsed
            wait_s = min(wait_s, effective_deadline - now)
            await self._async_sleeper.async_sleep(wait_s)

    def _extract_text(self, client: ACPClient) -> str | None:
        """Extract accumulated text from client message chunks."""
        if not client.message_chunks:
            return None

        parts: list[str] = []

        if self._adapter_config.emit_thought_chunks and client.thought_chunks:
            for chunk in client.thought_chunks:
                text = _extract_chunk_text(chunk)
                if text:
                    parts.append(text)

        for chunk in client.message_chunks:
            text = _extract_chunk_text(chunk)
            if text:
                parts.append(text)

        return "".join(parts) if parts else None

    def _resolve_structured_output[OutputT](
        self,
        accumulated_text: str | None,
        rendered: RenderedPrompt[OutputT],
        prompt_name: str,
        structured_capture: Any,
    ) -> OutputT | None:
        """Resolve structured output from capture or text."""
        if structured_capture is not None and structured_capture.called:
            try:
                return cast(
                    OutputT,
                    parse_structured_output(
                        json.dumps(structured_capture.data), rendered
                    ),
                )
            except (OutputParseError, TypeError, ValueError) as error:
                raise PromptEvaluationError(
                    message=f"Failed to parse structured output: {error}",
                    prompt_name=prompt_name,
                    phase="response",
                ) from error

        if accumulated_text:
            try:
                return cast(
                    OutputT, parse_structured_output(accumulated_text, rendered)
                )
            except (OutputParseError, TypeError, ValueError) as error:
                raise PromptEvaluationError(
                    message=f"Failed to parse structured output: {error}",
                    prompt_name=prompt_name,
                    phase="response",
                    provider_payload={"raw_text": accumulated_text[:2000]},
                ) from error

        raise PromptEvaluationError(
            message="Structured output required but model did not produce output",
            prompt_name=prompt_name,
            phase="response",
        )

    # -- Subclass hooks --

    def _validate_model(self, model_id: str, available_models: list[Any]) -> None:
        """Validate model_id against available models. Override in subclasses."""

    def _handle_mode_error(self, error: Exception) -> None:
        """Handle set_session_mode failure. Override in subclasses."""
        logger.warning(
            "acp.set_mode.failed",
            event="set_mode.failed",
            context={"error": str(error)},
        )

    def _detect_empty_response(self, client: ACPClient, prompt_resp: Any) -> None:
        """Detect empty or invalid response. Override in subclasses."""

    def _prepare_execution_env(
        self,
        *,
        rendered: RenderedPrompt[Any],
        effective_cwd: str,
    ) -> tuple[dict[str, str] | None, Callable[[], None]]:
        """Prepare subprocess environment. Override for isolation/ephemeral home.

        Returns:
            Tuple of (env dict or None to inherit parent, cleanup callable).
        """
        return self._build_env(), _noop
