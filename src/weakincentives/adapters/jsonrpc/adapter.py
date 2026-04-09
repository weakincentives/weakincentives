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

"""Base JSON-RPC provider adapter.

Provides the generic evaluate → render → bridge → protocol lifecycle
for any agent that speaks a turn-based JSON-RPC protocol over stdio or
WebSocket.  Subclasses implement provider-specific hooks for
initialization, turn management, notification processing, and error
handling.
"""

from __future__ import annotations

import shutil
import tempfile
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override
from uuid import uuid4

from ...budget import Budget, BudgetTracker
from ...clock import SYSTEM_CLOCK, AsyncSleeper
from ...deadlines import Deadline
from ...filesystem import Filesystem, HostFilesystem
from ...prompt import Prompt, RenderedPrompt
from ...prompt.errors import VisibilityExpansionRequired
from ...runtime.events import PromptRendered
from ...runtime.events.types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from ...runtime.session.rendered_tools import RenderedTools
from ...runtime.watchdog import Heartbeat
from .._shared._bridge import BridgedTool, create_bridged_tools
from .._shared._guardrails import resolve_filesystem
from .._shared._visibility_signal import VisibilityExpansionSignal
from ..core import PromptEvaluationError, PromptResponse, ProviderAdapter
from ..tool_spec import extract_tool_schema
from ._async import run_async
from ._protocol import execute_protocol
from ._response import build_response
from ._types import JsonRpcMessage
from .client import JsonRpcClient, JsonRpcClientError
from .config import JsonRpcClientConfig

if TYPE_CHECKING:
    from ...prompt.protocols import PromptProtocol

__all__ = [
    "JsonRpcAdapter",
]

logger: StructuredLogger = get_logger(
    __name__, context={"component": "jsonrpc_adapter"}
)


def _utcnow() -> datetime:
    return SYSTEM_CLOCK.utcnow()


class JsonRpcAdapter[OutputT_co](ProviderAdapter[OutputT_co]):
    """Abstract base class for JSON-RPC provider adapters.

    Provides the full evaluate → render → bridge → protocol lifecycle.
    Subclasses implement provider-specific protocol hooks.
    """

    def __init__(
        self,
        *,
        client_config: JsonRpcClientConfig | None = None,
        async_sleeper: AsyncSleeper = SYSTEM_CLOCK,
    ) -> None:
        super().__init__()
        self._client_config = client_config or JsonRpcClientConfig()
        self._async_sleeper = async_sleeper

    # ------------------------------------------------------------------
    # ProviderAdapter interface
    # ------------------------------------------------------------------

    @property
    @override
    def adapter_name(self) -> str:
        return self._adapter_name()

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
        """Evaluate prompt using the JSON-RPC agent."""
        if budget and not budget_tracker:
            budget_tracker = BudgetTracker(budget)

        effective_deadline = deadline or (budget.deadline if budget else None)
        prompt_name = prompt.name or f"{prompt.ns}:{prompt.key}"

        if effective_deadline and effective_deadline.remaining().total_seconds() <= 0:
            raise PromptEvaluationError(
                message="Deadline expired before invocation",
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

    # ------------------------------------------------------------------
    # Required hooks (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    def _adapter_name(self) -> str:
        """Return the canonical adapter identifier."""
        ...

    @abstractmethod
    def _create_client(self, env: dict[str, str] | None) -> JsonRpcClient:
        """Construct a configured :class:`JsonRpcClient`."""
        ...

    @abstractmethod
    async def _initialize_session(
        self,
        client: JsonRpcClient,
        *,
        deadline: Deadline | None,
        prompt_name: str,
    ) -> object:
        """Run provider-specific handshake and session creation.

        Returns provider-specific session state (e.g. thread_id).
        """
        ...

    @abstractmethod
    async def _start_turn(  # noqa: PLR0913
        self,
        client: JsonRpcClient,
        session_state: object,
        prompt_text: str,
        *,
        deadline: Deadline | None,
        prompt_name: str,
        timeout: float | None,
    ) -> object:
        """Start a turn.  Returns provider-specific turn state."""
        ...

    @abstractmethod
    async def _send_interrupt(
        self,
        client: JsonRpcClient,
        session_state: object,
        turn_state: object,
    ) -> None:
        """Send an interrupt request (deadline enforcement)."""
        ...

    @abstractmethod
    def _process_notification(
        self,
        message: JsonRpcMessage,
        session: SessionProtocol,
        adapter_name: str,
        prompt_name: str,
        run_context: RunContext | None,
    ) -> tuple[str, str] | None:
        """Process a notification message.

        Returns ``(kind, value)`` where kind is one of ``"text"``,
        ``"delta"``, ``"usage"``, ``"done"``, ``"error"``,
        ``"interrupted"``.  Returns ``None`` for unhandled methods.
        """
        ...

    @abstractmethod
    async def _handle_server_request(  # noqa: PLR0913
        self,
        client: JsonRpcClient,
        message: JsonRpcMessage,
        tool_lookup: dict[str, BridgedTool],
        *,
        bridge: object | None = None,
        prompt: PromptProtocol[Any] | None = None,
        session: SessionProtocol | None = None,
        deadline: Deadline | None = None,
    ) -> None:
        """Handle a server-initiated request (tool call, approval, etc.)."""
        ...

    @abstractmethod
    def _build_tool_specs(
        self, bridged_tools: tuple[BridgedTool, ...]
    ) -> list[dict[str, object]]:
        """Convert bridged tools to provider-specific format."""
        ...

    @abstractmethod
    def _build_output_schema(
        self, rendered: RenderedPrompt[Any]
    ) -> dict[str, Any] | None:
        """Build provider-specific output schema."""
        ...

    @abstractmethod
    def _extract_token_usage(self, params: dict[str, object]) -> TokenUsage | None:
        """Extract TokenUsage from a usage notification's params."""
        ...

    @abstractmethod
    def _map_error_phase(self, message: JsonRpcMessage) -> str:
        """Map a provider error notification to a WINK error phase."""
        ...

    # ------------------------------------------------------------------
    # Optional hooks (defaults)
    # ------------------------------------------------------------------

    def _setup_environment(  # noqa: PLR6301
        self,
        rendered: RenderedPrompt[Any],
        config_env: Any,  # noqa: ANN401
    ) -> tuple[Any, dict[str, str] | None]:
        """Setup provider-specific environment (e.g. ephemeral home).

        Returns ``(cleanup_state, env_dict)``.  The base implementation
        returns ``(None, config_env)`` with no special setup.
        """
        return None, dict(config_env) if config_env else None

    def _cleanup_environment(self, cleanup_state: Any) -> None:  # noqa: ANN401
        """Cleanup provider-specific environment after evaluation."""

    def _create_transcript_bridge(  # noqa: PLR6301
        self,
        session: SessionProtocol,
        prompt_name: str,
    ) -> Any | None:  # noqa: ANN401
        """Create a provider-specific transcript bridge.

        Returns ``None`` by default (no transcription).
        """
        return None

    def _stop_transcript_bridge(self, bridge: Any | None) -> None:  # noqa: ANN401, PLR6301
        """Stop the transcript bridge.  Override if needed."""
        if bridge is not None and hasattr(bridge, "emitter"):
            bridge.emitter.stop()

    def _on_user_message_for_transcript(self, bridge: Any | None, text: str) -> None:  # noqa: ANN401, PLR6301
        """Forward a user message to the transcript bridge."""
        if bridge is not None and hasattr(bridge, "on_user_message"):
            bridge.on_user_message(text)

    def _on_notification_for_transcript(  # noqa: PLR6301
        self,
        bridge: Any | None,  # noqa: ANN401
        message: JsonRpcMessage,
    ) -> None:
        """Forward a notification to the transcript bridge."""
        if bridge is not None and hasattr(bridge, "on_notification"):
            method = message.get("method", "")
            params: dict[str, object] = message.get("params", {})
            bridge.on_notification(method, params)

    def _check_task_completion(  # noqa: PLR0911, PLR6301
        self,
        *,
        prompt: PromptProtocol[Any] | None,
        session: SessionProtocol,
        accumulated_text: str | None,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
    ) -> tuple[bool, str | None]:
        """Check if the task is complete.

        Returns ``(should_continue, feedback)``.  Default delegates to
        the prompt's task completion checker.
        """
        if prompt is None:
            return False, None

        checker = prompt.task_completion_checker
        if checker is None:
            return False, None

        if deadline is not None and deadline.remaining().total_seconds() <= 0:
            return False, None

        if budget_tracker is not None:
            from ...budget import BudgetExceededError

            try:
                budget_tracker.check()
            except BudgetExceededError:
                return False, None

        from ...prompt.task_completion import TaskCompletionContext

        filesystem = resolve_filesystem(prompt)
        context = TaskCompletionContext(
            session=session,
            tentative_output=accumulated_text,
            filesystem=filesystem,
        )
        result = checker.check(context)

        if result.complete:
            return False, None
        if not result.feedback:
            return False, None

        return True, result.feedback

    # ------------------------------------------------------------------
    # Internal lifecycle
    # ------------------------------------------------------------------

    async def _evaluate_async[OutputT](  # noqa: PLR0913
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
                "jsonrpc.evaluate.rendered_tools_dispatch_failed",
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
                return await self._run_protocol(
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
                temp_workspace_dir = tempfile.mkdtemp(prefix="wink-jsonrpc-")
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

    async def _run_protocol[OutputT](  # noqa: PLR0913
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
        """Run the full protocol flow."""
        adapter_name = self._adapter_name()
        visibility_signal = VisibilityExpansionSignal()

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

        self._dynamic_tool_specs = self._build_tool_specs(bridged_tools)
        tool_lookup = {t.name: t for t in bridged_tools}
        self._output_schema = self._build_output_schema(rendered)
        self._effective_cwd = effective_cwd
        output_schema = self._output_schema

        env_state, client_env = self._setup_environment(
            rendered, self._client_config.env
        )

        client = self._create_client(client_env)

        start_time = _utcnow()
        try:
            result = await execute_protocol(
                adapter=self,
                client=client,
                session=session,
                adapter_name=adapter_name,
                prompt_name=prompt_name,
                prompt_text=prompt_text,
                tool_lookup=tool_lookup,
                deadline=deadline,
                budget_tracker=budget_tracker,
                run_context=run_context,
                visibility_signal=visibility_signal,
                async_sleeper=self._async_sleeper,
                prompt=cast(Any, prompt),
            )
        except VisibilityExpansionRequired:
            raise
        except JsonRpcClientError as error:
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
                message=f"Execution failed: {error}",
                prompt_name=prompt_name,
                phase="request",
                provider_payload={"stderr": client.stderr_output[-8192:]},
            ) from error
        finally:
            await client.stop()
            self._cleanup_environment(env_state)

        accumulated_text, usage = result
        return build_response(
            accumulated_text=accumulated_text,
            usage=usage,
            output_schema=output_schema,
            rendered=rendered,
            prompt_name=prompt_name,
            adapter_name=adapter_name,
            session=session,
            budget_tracker=budget_tracker,
            run_context=run_context,
            start_time=start_time,
            utcnow=_utcnow(),
        )
