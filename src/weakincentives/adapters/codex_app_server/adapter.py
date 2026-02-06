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

"""Codex App Server adapter implementation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, cast, override
from uuid import uuid4

from ...budget import Budget, BudgetTracker
from ...deadlines import Deadline
from ...filesystem import Filesystem, HostFilesystem
from ...prompt import Prompt, RenderedPrompt
from ...prompt.errors import VisibilityExpansionRequired
from ...runtime.events import PromptExecuted, PromptRendered
from ...runtime.events.types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from ...runtime.watchdog import Heartbeat
from ...serde import parse, schema
from ...types import AdapterName
from .._shared._bridge import BridgedTool, create_bridged_tools
from .._shared._visibility_signal import VisibilityExpansionSignal
from ..core import PromptEvaluationError, PromptResponse, ProviderAdapter
from ._async import run_async
from ._events import (
    dispatch_item_tool_invoked,
    extract_token_usage,
    map_codex_error_phase,
)
from .client import CodexAppServerClient, CodexClientError
from .config import (
    ApiKeyAuth,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
)

__all__ = [
    "CODEX_APP_SERVER_ADAPTER_NAME",
    "CodexAppServerAdapter",
]

logger: StructuredLogger = get_logger(
    __name__, context={"component": "codex_app_server"}
)

CODEX_APP_SERVER_ADAPTER_NAME: AdapterName = "codex_app_server"
"""Canonical label for the Codex App Server adapter."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _bridged_tools_to_dynamic_specs(
    tools: tuple[BridgedTool, ...],
) -> list[dict[str, Any]]:
    """Convert BridgedTool list to Codex DynamicToolSpec format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
        }
        for tool in tools
    ]


def _openai_strict_schema(s: dict[str, Any]) -> dict[str, Any]:
    """Adapt a WINK serde schema for OpenAI/Codex structured output.

    OpenAI's structured output requires:
    - ``additionalProperties: false`` on all object types
    - All properties listed in ``required`` when additionalProperties is false

    WINK's ``serde.schema()`` emits ``additionalProperties: true`` by
    default and only marks fields without defaults as required.
    """
    out = dict(s)
    if out.get("type") == "object":
        out["additionalProperties"] = False
        props: dict[str, Any] | None = out.get("properties")
        if isinstance(props, dict):
            out["properties"] = {
                k: _openai_strict_schema(cast(dict[str, Any], v))
                if isinstance(v, dict)
                else v
                for k, v in props.items()
            }
            # OpenAI requires all properties in required when
            # additionalProperties is false.
            out["required"] = list(props.keys())

    # Recurse into array items
    if "items" in out and isinstance(out["items"], dict):
        out["items"] = _openai_strict_schema(cast(dict[str, Any], out["items"]))

    # Recurse into combinators
    for combinator in ("anyOf", "oneOf", "allOf"):
        if combinator in out and isinstance(out[combinator], list):
            out[combinator] = [
                _openai_strict_schema(cast(dict[str, Any], entry))
                if isinstance(entry, dict)
                else entry
                for entry in out[combinator]
            ]

    # Recurse into schema definitions
    for defs_key in ("$defs", "definitions"):
        if defs_key in out and isinstance(out[defs_key], dict):
            out[defs_key] = {
                k: _openai_strict_schema(cast(dict[str, Any], v))
                if isinstance(v, dict)
                else v
                for k, v in out[defs_key].items()
            }

    return out


class _ThreadState(NamedTuple):
    """Adapter-internal state for Codex thread reuse."""

    thread_id: str
    cwd: str
    dynamic_tool_names: tuple[str, ...]


class CodexAppServerAdapter(ProviderAdapter[Any]):
    """Adapter using the Codex App Server for agentic prompt evaluation.

    Spawns ``codex app-server`` as a subprocess and communicates over
    NDJSON stdio using the JSON-RPC protocol (without ``"jsonrpc"`` header).
    WINK tools are bridged as Codex dynamic tools.
    """

    def __init__(
        self,
        *,
        model_config: CodexAppServerModelConfig | None = None,
        client_config: CodexAppServerClientConfig | None = None,
    ) -> None:
        super().__init__()
        self._model_config = model_config or CodexAppServerModelConfig()
        self._client_config = client_config or CodexAppServerClientConfig()
        self._last_thread: _ThreadState | None = None

        logger.debug(
            "codex_app_server.adapter.init",
            event="adapter.init",
            context={
                "model": self._model_config.model,
                "codex_bin": self._client_config.codex_bin,
                "cwd": self._client_config.cwd,
                "approval_policy": self._client_config.approval_policy,
                "sandbox_mode": self._client_config.sandbox_mode,
                "ephemeral": self._client_config.ephemeral,
            },
        )

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
        """Evaluate prompt using the Codex App Server."""
        if budget and not budget_tracker:
            budget_tracker = BudgetTracker(budget)

        effective_deadline = deadline or (budget.deadline if budget else None)
        prompt_name = prompt.name or f"{prompt.ns}:{prompt.key}"

        if effective_deadline and effective_deadline.remaining().total_seconds() <= 0:
            raise PromptEvaluationError(
                message="Deadline expired before Codex invocation",
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

        session_id = getattr(session, "session_id", None)

        # Dispatch PromptRendered
        _ = session.dispatcher.dispatch(
            PromptRendered(
                prompt_ns=prompt.ns,
                prompt_key=prompt.key,
                prompt_name=prompt.name,
                adapter=CODEX_APP_SERVER_ADAPTER_NAME,
                session_id=session_id,
                render_inputs=(),
                rendered_prompt=prompt_text,
                created_at=_utcnow(),
                descriptor=None,
                run_context=run_context,
                event_id=uuid4(),
            )
        )

        # Determine CWD
        effective_cwd, temp_workspace_dir, prompt = self._resolve_cwd(prompt)

        try:
            with prompt.resources:
                return await self._run_codex(
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
        """Determine the effective cwd and optionally bind a filesystem.

        Returns (effective_cwd, temp_workspace_dir_or_none, maybe_rebound_prompt).
        """
        temp_workspace_dir: str | None = None
        effective_cwd: str | None = self._client_config.cwd

        if prompt.filesystem() is None:
            if effective_cwd is None:
                temp_workspace_dir = tempfile.mkdtemp(prefix="wink-codex-")
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

    async def _run_codex[OutputT](
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
        """Run the full Codex protocol flow."""
        visibility_signal = VisibilityExpansionSignal()

        bridged_tools = create_bridged_tools(
            rendered.tools,
            session=session,
            adapter=self,
            prompt=cast(Any, prompt),
            rendered_prompt=rendered,
            deadline=deadline,
            budget_tracker=budget_tracker,
            adapter_name=CODEX_APP_SERVER_ADAPTER_NAME,
            prompt_name=prompt_name,
            heartbeat=heartbeat,
            run_context=run_context,
            visibility_signal=visibility_signal,
        )

        dynamic_tool_specs = _bridged_tools_to_dynamic_specs(bridged_tools)
        tool_lookup: dict[str, BridgedTool] = {t.name: t for t in bridged_tools}

        output_schema: dict[str, Any] | None = None
        if rendered.output_type is not None:
            output_schema = _openai_strict_schema(schema(rendered.output_type))

        client = CodexAppServerClient(
            codex_bin=self._client_config.codex_bin,
            env=self._client_config.env,
            suppress_stderr=self._client_config.suppress_stderr,
        )

        start_time = _utcnow()
        try:
            result = await self._execute_protocol(
                client=client,
                session=session,
                prompt_name=prompt_name,
                prompt_text=prompt_text,
                effective_cwd=effective_cwd,
                dynamic_tool_specs=dynamic_tool_specs,
                tool_lookup=tool_lookup,
                output_schema=output_schema,
                deadline=deadline,
                budget_tracker=budget_tracker,
                run_context=run_context,
                visibility_signal=visibility_signal,
            )
        except VisibilityExpansionRequired:
            raise
        except CodexClientError as error:
            raise PromptEvaluationError(
                message=str(error),
                prompt_name=prompt_name,
                phase="request",
                provider_payload={"stderr": client.stderr_output[-8192:]},
            ) from error
        except PromptEvaluationError:
            raise
        except Exception as error:
            raise PromptEvaluationError(
                message=f"Codex execution failed: {error}",
                prompt_name=prompt_name,
                phase="request",
                provider_payload={"stderr": client.stderr_output[-8192:]},
            ) from error
        finally:
            await client.stop()

        accumulated_text, usage = result
        end_time = _utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        output: OutputT | None = None
        if output_schema is not None and accumulated_text:
            output = self._parse_structured_output(
                accumulated_text, rendered, prompt_name
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
                adapter=CODEX_APP_SERVER_ADAPTER_NAME,
                result=response,
                session_id=getattr(session, "session_id", None),
                created_at=_utcnow(),
                usage=usage,
                run_context=run_context,
            )
        )

        logger.info(
            "codex_app_server.evaluate.complete",
            event="evaluate.complete",
            context={
                "prompt_name": prompt_name,
                "duration_ms": duration_ms,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
            },
        )

        return response

    async def _execute_protocol(
        self,
        *,
        client: CodexAppServerClient,
        session: SessionProtocol,
        prompt_name: str,
        prompt_text: str,
        effective_cwd: str,
        dynamic_tool_specs: list[dict[str, Any]],
        tool_lookup: dict[str, BridgedTool],
        output_schema: dict[str, Any] | None,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        run_context: RunContext | None,
        visibility_signal: VisibilityExpansionSignal,
    ) -> tuple[str | None, TokenUsage | None]:
        """Execute the Codex protocol (init → thread → turn → stream).

        Returns (accumulated_text, usage).
        """
        await client.start()

        # 1. Initialize
        _ = await client.send_request(
            "initialize",
            {
                "clientInfo": {
                    "name": self._client_config.client_name,
                    "title": "WINK Agent",
                    "version": self._client_config.client_version,
                },
                "capabilities": {"experimentalApi": True},
            },
            timeout=self._client_config.startup_timeout_s,
        )
        await client.send_notification("initialized")

        # 2. Authenticate
        await self._authenticate(client)

        # 3. Thread
        thread_id = await self._start_thread(client, effective_cwd, dynamic_tool_specs)

        # 4. Turn
        turn_result = await self._start_turn(
            client, thread_id, prompt_text, output_schema
        )
        turn_id: int = turn_result["turn"]["id"]

        # 5. Stream
        accumulated_text, usage = await self._stream_turn(
            client=client,
            session=session,
            prompt_name=prompt_name,
            thread_id=thread_id,
            turn_id=turn_id,
            tool_lookup=tool_lookup,
            deadline=deadline,
            run_context=run_context,
        )

        # 6. Visibility signal
        stored_exc = visibility_signal.get_and_clear()
        if stored_exc is not None:
            raise stored_exc

        return accumulated_text, usage

    async def _authenticate(self, client: CodexAppServerClient) -> None:
        """Perform authentication if auth_mode is configured."""
        auth = self._client_config.auth_mode
        if auth is None:
            return

        if isinstance(auth, ApiKeyAuth):
            _ = await client.send_request(
                "account/login/start",
                {"type": "apiKey", "apiKey": auth.api_key},
            )
        else:
            # ExternalTokenAuth
            _ = await client.send_request(
                "account/login/start",
                {
                    "type": "chatgptAuthTokens",
                    "idToken": auth.id_token,
                    "accessToken": auth.access_token,
                },
            )

    async def _start_thread(
        self,
        client: CodexAppServerClient,
        effective_cwd: str,
        dynamic_tool_specs: list[dict[str, Any]],
    ) -> str:
        """Start or resume a Codex thread. Returns the thread ID."""
        if self._client_config.reuse_thread:
            thread_id = await self._try_resume_thread(
                client, effective_cwd, dynamic_tool_specs
            )
            if thread_id is not None:
                return thread_id

        return await self._create_thread(client, effective_cwd, dynamic_tool_specs)

    async def _try_resume_thread(
        self,
        client: CodexAppServerClient,
        effective_cwd: str,
        dynamic_tool_specs: list[dict[str, Any]],
    ) -> str | None:
        """Try to resume an existing thread. Returns thread ID or None."""
        state = self._last_thread
        if state is None or state.cwd != effective_cwd:
            return None
        current_names = tuple(sorted(spec["name"] for spec in dynamic_tool_specs))
        if state.dynamic_tool_names != current_names:
            return None
        try:
            result = await client.send_request(
                "thread/resume", {"threadId": state.thread_id}
            )
            return result["thread"]["id"]
        except CodexClientError:
            logger.debug(
                "codex_app_server.thread_resume_failed",
                event="thread_resume_failed",
                context={"thread_id": state.thread_id},
            )
            return None

    async def _create_thread(
        self,
        client: CodexAppServerClient,
        effective_cwd: str,
        dynamic_tool_specs: list[dict[str, Any]],
    ) -> str:
        """Create a new thread and store state for reuse."""
        thread_params: dict[str, Any] = {
            "model": self._model_config.model,
            "cwd": effective_cwd,
            "approvalPolicy": self._client_config.approval_policy,
            "ephemeral": self._client_config.ephemeral,
        }
        if self._client_config.sandbox_mode is not None:
            thread_params["sandbox"] = self._client_config.sandbox_mode
        if dynamic_tool_specs:
            thread_params["dynamicTools"] = dynamic_tool_specs
        if self._client_config.mcp_servers:
            thread_params["config"] = {"mcp_servers": self._client_config.mcp_servers}

        result = await client.send_request("thread/start", thread_params)
        thread_id: str = result["thread"]["id"]

        dynamic_tool_names = tuple(sorted(spec["name"] for spec in dynamic_tool_specs))
        self._last_thread = _ThreadState(
            thread_id=thread_id,
            cwd=effective_cwd,
            dynamic_tool_names=dynamic_tool_names,
        )
        return thread_id

    async def _start_turn(
        self,
        client: CodexAppServerClient,
        thread_id: str,
        prompt_text: str,
        output_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Start a turn and return the response."""
        turn_params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": prompt_text}],
        }
        if self._model_config.effort is not None:
            turn_params["effort"] = self._model_config.effort
        if self._model_config.summary is not None:
            turn_params["summary"] = self._model_config.summary
        if self._model_config.personality is not None:
            turn_params["personality"] = self._model_config.personality
        if output_schema is not None:
            turn_params["outputSchema"] = output_schema

        return await client.send_request("turn/start", turn_params)

    async def _stream_turn(
        self,
        *,
        client: CodexAppServerClient,
        session: SessionProtocol,
        prompt_name: str,
        thread_id: str,
        turn_id: int,
        tool_lookup: dict[str, BridgedTool],
        deadline: Deadline | None,
        run_context: RunContext | None,
    ) -> tuple[str | None, TokenUsage | None]:
        """Stream turn notifications until turn/completed.

        Returns (accumulated_text, token_usage).
        """
        accumulated_text = ""
        usage: TokenUsage | None = None

        watchdog_task = self._create_deadline_watchdog(
            client, thread_id, turn_id, deadline
        )

        try:
            accumulated_text, usage = await self._consume_messages(
                client=client,
                session=session,
                prompt_name=prompt_name,
                tool_lookup=tool_lookup,
                run_context=run_context,
                accumulated_text=accumulated_text,
                usage=usage,
            )
        finally:
            if watchdog_task is not None:
                _ = watchdog_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watchdog_task

        return accumulated_text or None, usage

    def _raise_for_terminal_notification(
        self,
        kind: str,
        value: str,
        prompt_name: str,
        message: dict[str, Any],
    ) -> None:
        """Raise PromptEvaluationError for terminal notification kinds."""
        if kind == "error":
            params: dict[str, Any] = message.get("params", {})
            turn: dict[str, Any] = params.get("turn", {})
            phase = map_codex_error_phase(turn.get("codexErrorInfo"))
            raise PromptEvaluationError(
                message=value,
                prompt_name=prompt_name,
                phase=phase,
            )
        if kind == "interrupted":
            raise PromptEvaluationError(
                message="Turn was interrupted (deadline or user)",
                prompt_name=prompt_name,
                phase="request",
            )

    async def _consume_messages(
        self,
        *,
        client: CodexAppServerClient,
        session: SessionProtocol,
        prompt_name: str,
        tool_lookup: dict[str, BridgedTool],
        run_context: RunContext | None,
        accumulated_text: str,
        usage: TokenUsage | None,
    ) -> tuple[str, TokenUsage | None]:
        """Consume messages from the client until turn/completed."""
        turn_completed = False
        async for message in client.read_messages():
            if "id" in message and "method" in message:
                await self._handle_server_request(client, message, tool_lookup)
                continue

            result = self._process_notification(
                message, session, prompt_name, run_context
            )
            if result is None:
                continue

            accumulated_text, usage, done = self._apply_notification(
                result, message, accumulated_text, usage, prompt_name
            )
            if done:
                turn_completed = True
                break
        if not turn_completed:
            raise PromptEvaluationError(
                message="Codex stream ended before turn completion",
                prompt_name=prompt_name,
                phase="response",
            )
        return accumulated_text, usage

    def _apply_notification(
        self,
        result: tuple[str, str],
        message: dict[str, Any],
        accumulated_text: str,
        usage: TokenUsage | None,
        prompt_name: str,
    ) -> tuple[str, TokenUsage | None, bool]:
        """Apply a single notification result. Returns (text, usage, done)."""
        kind, value = result
        if kind == "text":
            return value, usage, False
        if kind == "delta":
            return accumulated_text + value, usage, False
        if kind == "usage":
            return (
                accumulated_text,
                extract_token_usage(message.get("params", {})),
                False,
            )
        if kind == "done":
            return accumulated_text, usage, True
        self._raise_for_terminal_notification(kind, value, prompt_name, message)
        return accumulated_text, usage, False  # pragma: no cover

    def _create_deadline_watchdog(
        self,
        client: CodexAppServerClient,
        thread_id: str,
        turn_id: int,
        deadline: Deadline | None,
    ) -> asyncio.Task[None] | None:
        """Create a watchdog task if a deadline is set."""
        if deadline is None:
            return None
        remaining = deadline.remaining().total_seconds()
        if remaining <= 0:
            return None
        return asyncio.create_task(
            self._deadline_watchdog(client, thread_id, turn_id, remaining)
        )

    def _process_notification(
        self,
        message: dict[str, Any],
        session: SessionProtocol,
        prompt_name: str,
        run_context: RunContext | None,
    ) -> tuple[str, str] | None:
        """Process a notification message.

        Returns (kind, value) where kind is one of:
        "text", "delta", "usage", "error", "interrupted", "done".
        Returns None for unhandled notification types.
        """
        method = message.get("method", "")
        params: dict[str, Any] = message.get("params", {})

        if method == "item/agentMessage/delta":
            return ("delta", params.get("delta", ""))

        if method == "item/completed":
            return self._handle_item_completed(
                params, session, prompt_name, run_context
            )

        if method == "thread/tokenUsage/updated":
            return ("usage", "")

        if method == "turn/completed":
            return self._handle_turn_completed(params)

        return None

    def _handle_item_completed(
        self,
        params: dict[str, Any],
        session: SessionProtocol,
        prompt_name: str,
        run_context: RunContext | None,
    ) -> tuple[str, str] | None:
        """Handle item/completed notification."""
        item: dict[str, Any] = params.get("item", {})
        item_type = item.get("type", "")

        if item_type == "agentMessage":
            final_text = item.get("text")
            if final_text is not None:
                return ("text", final_text)
            return None

        if item_type in {
            "commandExecution",
            "fileChange",
            "mcpToolCall",
            "webSearch",
        }:
            dispatch_item_tool_invoked(
                item=item,
                session=session,
                adapter_name=CODEX_APP_SERVER_ADAPTER_NAME,
                prompt_name=prompt_name,
                run_context=run_context,
            )
        return None

    def _handle_turn_completed(self, params: dict[str, Any]) -> tuple[str, str]:
        """Handle turn/completed notification."""
        turn: dict[str, Any] = params.get("turn", {})
        status = turn.get("status", "")

        if status == "failed":
            error_info = turn.get("codexErrorInfo")
            additional = turn.get("additionalDetails", "")
            return ("error", f"Turn failed: {error_info or 'unknown'} — {additional}")

        if status == "interrupted":
            return ("interrupted", "")

        return ("done", "")

    async def _handle_server_request(
        self,
        client: CodexAppServerClient,
        message: dict[str, Any],
        tool_lookup: dict[str, BridgedTool],
    ) -> None:
        """Handle a server-initiated request (tool call or approval)."""
        method: str = message["method"]
        params: dict[str, Any] = message.get("params", {})
        request_id: int = message["id"]

        if method == "item/tool/call":
            await self._handle_tool_call(client, request_id, params, tool_lookup)
        elif method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            # In non-interactive execution, we deterministically accept only
            # policies intended to proceed without human gating when requested.
            decision = (
                "accept"
                if self._client_config.approval_policy in {"never", "on-failure"}
                else "decline"
            )
            await client.send_response(request_id, {"decision": decision})
        else:
            await client.send_response(request_id, {})

    async def _handle_tool_call(
        self,
        client: CodexAppServerClient,
        request_id: int,
        params: dict[str, Any],
        tool_lookup: dict[str, BridgedTool],
    ) -> None:
        """Handle an item/tool/call server request."""
        tool_name: str = params.get("tool", "")
        arguments = params.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        bridged_tool = tool_lookup.get(tool_name)
        if bridged_tool is None:
            await client.send_response(
                request_id,
                {
                    "success": False,
                    "contentItems": [
                        {"type": "inputText", "text": f"Unknown tool: {tool_name}"}
                    ],
                },
            )
            return

        mcp_result: dict[str, Any] = await asyncio.to_thread(
            bridged_tool, cast(dict[str, Any], arguments)
        )
        is_error: bool = mcp_result.get("isError", False)
        mcp_content: list[dict[str, Any]] = mcp_result.get("content", [])
        content_items: list[dict[str, str]] = [
            {"type": "inputText", "text": str(c.get("text", ""))}
            for c in mcp_content
            if c.get("type") == "text"
        ]
        await client.send_response(
            request_id,
            {"success": not is_error, "contentItems": content_items},
        )

    async def _deadline_watchdog(
        self,
        client: CodexAppServerClient,
        thread_id: str,
        turn_id: int,
        remaining_seconds: float,
    ) -> None:
        """Sleep until deadline, then interrupt the turn."""
        await asyncio.sleep(remaining_seconds)
        with contextlib.suppress(CodexClientError):
            _ = await client.send_request(
                "turn/interrupt",
                {"threadId": thread_id, "turnId": turn_id},
                timeout=5.0,
            )

    def _parse_structured_output[OutputT](
        self,
        text: str,
        rendered: RenderedPrompt[OutputT],
        prompt_name: str,
    ) -> OutputT | None:
        """Parse JSON text into the expected output type."""
        if rendered.output_type is None:
            return None  # pragma: no cover

        try:
            raw = json.loads(text)
            return cast(OutputT, parse(rendered.output_type, raw))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise PromptEvaluationError(
                message=f"Failed to parse structured output: {error}",
                prompt_name=prompt_name,
                phase="response",
                provider_payload={"raw_text": text[:2000]},
            ) from error
