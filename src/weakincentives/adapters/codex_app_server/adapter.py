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

"""Codex App Server adapter implementation.

Extends the generic ``JsonRpcAdapter`` with Codex-specific protocol hooks:
initialization, authentication, thread/turn management, notification
processing, approval handling, and structured output.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, override

from ...clock import SYSTEM_CLOCK, AsyncSleeper
from ...deadlines import Deadline
from ...prompt import RenderedPrompt
from ...runtime.events.types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session.protocols import SessionProtocol
from ...runtime.transcript import TranscriptEmitter
from ...types import AdapterName
from .._shared._bridge import BridgedTool
from ..core import PromptEvaluationError
from ..jsonrpc import (
    JsonRpcAdapter,
    JsonRpcClient,
    JsonRpcClientConfig,
    JsonRpcClientError,
    JsonRpcMessage,
    NotificationHandler,
    ProtocolContext,
    ServerRequestHandler,
    deadline_remaining_s,
)
from ._ephemeral_home import CodexEphemeralHome
from ._events import (
    extract_token_usage,
    map_codex_error_phase,
)
from ._protocol import (
    authenticate,
    create_thread,
    handle_delta_notification,
    handle_item_completed_notification,
    handle_token_usage_notification,
    handle_tool_call_request,
    handle_turn_completed_notification,
    make_approval_handler,
    start_turn,
)
from ._schema import bridged_tools_to_dynamic_specs, build_output_schema
from ._transcript import CodexTranscriptBridge
from .config import (
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

# Module-level alias used by _create_client and patchable in tests.
CodexAppServerClient = JsonRpcClient


class CodexAppServerAdapter(JsonRpcAdapter[Any]):
    """Adapter using the Codex App Server for agentic prompt evaluation.

    Extends ``JsonRpcAdapter`` with Codex-specific protocol hooks for
    the ``codex app-server`` JSON-RPC protocol (without ``"jsonrpc"``
    header) over stdio or WebSocket.
    """

    def __init__(
        self,
        *,
        model_config: CodexAppServerModelConfig | None = None,
        client_config: CodexAppServerClientConfig | None = None,
        async_sleeper: AsyncSleeper = SYSTEM_CLOCK,
    ) -> None:
        self._model_config = model_config or CodexAppServerModelConfig()
        self._codex_client_config = client_config or CodexAppServerClientConfig()

        # Build the base JsonRpcClientConfig from Codex config.
        base_config = JsonRpcClientConfig(
            transport=self._codex_client_config.transport,
            bin_path=self._codex_client_config.codex_bin,
            bin_args=("app-server",),
            bin_ws_args=("app-server", "--listen"),
            remote_url=self._codex_client_config.remote_url,
            ws_auth_token=self._codex_client_config.ws_auth_token,
            cwd=self._codex_client_config.cwd,
            env=self._codex_client_config.env,
            suppress_stderr=self._codex_client_config.suppress_stderr,
            startup_timeout_s=self._codex_client_config.startup_timeout_s,
            transcript=self._codex_client_config.transcript,
            transcript_emit_raw=self._codex_client_config.transcript_emit_raw,
        )

        super().__init__(
            client_config=base_config,
            async_sleeper=async_sleeper,
        )

        logger.debug(
            "codex_app_server.adapter.init",
            event="adapter.init",
            context={
                "model": self._model_config.model,
                "transport": self._codex_client_config.transport,
                "codex_bin": self._codex_client_config.codex_bin,
                "remote_url": self._codex_client_config.remote_url,
                "cwd": self._codex_client_config.cwd,
                "approval_policy": self._codex_client_config.approval_policy,
                "sandbox_mode": self._codex_client_config.sandbox_mode,
                "ephemeral": self._codex_client_config.ephemeral,
            },
        )

    # ------------------------------------------------------------------
    # Required hooks
    # ------------------------------------------------------------------

    @override
    def _adapter_name(self) -> str:
        return CODEX_APP_SERVER_ADAPTER_NAME

    @override
    def _create_client(self, env: dict[str, str] | None) -> JsonRpcClient:
        return CodexAppServerClient(
            bin_path=self._codex_client_config.codex_bin,
            bin_args=("app-server",),
            bin_ws_args=("app-server", "--listen"),
            env=env,
            suppress_stderr=self._codex_client_config.suppress_stderr,
            transport=self._codex_client_config.transport,
            remote_url=self._codex_client_config.remote_url,
            ws_auth_token=self._codex_client_config.ws_auth_token,
        )

    @override
    async def _initialize_session(
        self,
        client: JsonRpcClient,
        *,
        deadline: Deadline | None,
        prompt_name: str,
        protocol_context: ProtocolContext,
    ) -> str:
        """Run initialize → authenticate → create_thread. Returns thread_id."""
        _ = await client.send_request(
            "initialize",
            {
                "clientInfo": {
                    "name": self._codex_client_config.client_name,
                    "title": "WINK Agent",
                    "version": self._codex_client_config.client_version,
                },
                "capabilities": {"experimentalApi": True},
            },
            timeout=self._codex_client_config.startup_timeout_s,
        )
        await client.send_notification("initialized")

        try:
            timeout = deadline_remaining_s(deadline, prompt_name)
            await authenticate(
                client,
                self._codex_client_config.auth_mode,
                timeout=timeout,
            )

            timeout = deadline_remaining_s(deadline, prompt_name)
            return await create_thread(
                client,
                protocol_context.effective_cwd,
                protocol_context.dynamic_tool_specs,
                client_config=self._codex_client_config,
                model_config=self._model_config,
                timeout=timeout,
            )
        except JsonRpcClientError as error:
            raise PromptEvaluationError(
                message=str(error),
                prompt_name=prompt_name,
                phase="request",
            ) from error

    @override
    async def _start_turn(
        self,
        client: JsonRpcClient,
        session_state: Any,
        prompt_text: str,
        *,
        deadline: Deadline | None,
        prompt_name: str,
        timeout: float | None,
        protocol_context: ProtocolContext,
    ) -> Any:
        """Start a Codex turn. Returns turn result dict."""
        return await start_turn(
            client,
            session_state,  # thread_id
            prompt_text,
            protocol_context.output_schema,
            model_config=self._model_config,
            timeout=timeout,
        )

    @override
    async def _send_interrupt(
        self,
        client: JsonRpcClient,
        session_state: Any,
        turn_state: Any,
    ) -> None:
        """Send turn/interrupt to Codex."""
        turn_id: int = turn_state["turn"]["id"]
        _ = await client.send_request(
            "turn/interrupt",
            {"threadId": session_state, "turnId": turn_id},
            timeout=5.0,
        )

    @override
    def _notification_handlers(self) -> dict[str, NotificationHandler]:
        """Codex notification handlers.

        Adding a new notification is a single dict entry.
        """
        return {
            "item/agentMessage/delta": handle_delta_notification,
            "item/completed": handle_item_completed_notification,
            "thread/tokenUsage/updated": handle_token_usage_notification,
            "turn/completed": handle_turn_completed_notification,
        }

    @override
    def _server_request_handlers(self) -> dict[str, ServerRequestHandler]:
        """Codex server-request handlers.

        Adding a new server request type is a single dict entry.
        """
        policy = self._codex_client_config.approval_policy
        approval = make_approval_handler(policy)
        return {
            "item/tool/call": handle_tool_call_request,
            "item/commandExecution/requestApproval": approval,
            "item/fileChange/requestApproval": approval,
        }

    @override
    def _build_tool_specs(
        self, bridged_tools: tuple[BridgedTool, ...]
    ) -> list[dict[str, object]]:
        return bridged_tools_to_dynamic_specs(bridged_tools)

    @override
    def _build_output_schema(
        self, rendered: RenderedPrompt[Any]
    ) -> dict[str, Any] | None:
        return build_output_schema(rendered)

    @override
    def _extract_token_usage(self, params: dict[str, object]) -> TokenUsage | None:
        return extract_token_usage(params)

    @override
    def _map_error_phase(self, message: JsonRpcMessage) -> str:
        """Map Codex error info to WINK error phase."""
        params: dict[str, object] = message.get("params", {})
        turn_raw = params.get("turn", {})
        turn: dict[str, object] = turn_raw if isinstance(turn_raw, dict) else {}  # ty: ignore[invalid-assignment]  # pyright: ignore[reportUnknownVariableType]
        return map_codex_error_phase(turn.get("codexErrorInfo"))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    @override
    def _setup_environment(
        self,
        rendered: RenderedPrompt[Any],
        config_env: Any,
    ) -> tuple[CodexEphemeralHome | None, dict[str, str] | None]:
        """Create ephemeral home for Codex skill discovery if needed."""
        return _setup_skill_env(rendered, config_env)

    @override
    def _cleanup_environment(self, cleanup_state: Any) -> None:
        if cleanup_state is not None:
            cleanup_state.cleanup()

    @override
    def _create_transcript_bridge(
        self,
        session: SessionProtocol,
        prompt_name: str,
    ) -> CodexTranscriptBridge | None:
        if not self._codex_client_config.transcript:
            return None
        session_id = getattr(session, "session_id", None)
        emitter = TranscriptEmitter(
            prompt_name=prompt_name,
            adapter="codex_app_server",
            session_id=str(session_id) if session_id else None,
            emit_raw=self._codex_client_config.transcript_emit_raw,
        )
        bridge = CodexTranscriptBridge(emitter)
        emitter.start()
        return bridge


def _setup_skill_env(
    rendered: RenderedPrompt[object],
    config_env: Mapping[str, str] | None,
) -> tuple[CodexEphemeralHome | None, dict[str, str] | None]:
    """Create an ephemeral home for skill discovery if needed."""
    ephemeral_home: CodexEphemeralHome | None = None
    if rendered.skills:
        ephemeral_home = CodexEphemeralHome()
        try:
            ephemeral_home.mount_skills(rendered.skills)
        except BaseException:
            ephemeral_home.cleanup()
            raise

    client_env = dict(config_env or {})
    if ephemeral_home is not None:
        client_env.update(ephemeral_home.get_env())

    return ephemeral_home, client_env or None
