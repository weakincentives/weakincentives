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

import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast, override
from uuid import uuid4

from ...budget import Budget, BudgetTracker
from ...clock import SYSTEM_CLOCK, AsyncSleeper
from ...deadlines import Deadline
from ...filesystem import Filesystem, HostFilesystem
from ...prompt import Prompt, RenderedPrompt
from ...prompt.errors import VisibilityExpansionRequired
from ...runtime.events import PromptRendered
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from ...runtime.session.rendered_tools import RenderedTools
from ...runtime.watchdog import Heartbeat
from ...types import AdapterName
from .._shared._bridge import create_bridged_tools
from .._shared._visibility_signal import VisibilityExpansionSignal
from ..core import PromptEvaluationError, PromptResponse, ProviderAdapter
from ..tool_spec import extract_tool_schema
from ._async import run_async
from ._ephemeral_home import CodexEphemeralHome
from ._protocol import execute_protocol
from ._response import build_response
from ._schema import bridged_tools_to_dynamic_specs, build_output_schema
from .client import CodexAppServerClient, CodexClientError
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


def _utcnow() -> datetime:
    return SYSTEM_CLOCK.utcnow()


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
        async_sleeper: AsyncSleeper = SYSTEM_CLOCK,
    ) -> None:
        super().__init__()
        self._model_config = model_config or CodexAppServerModelConfig()
        self._client_config = client_config or CodexAppServerClientConfig()
        self._async_sleeper = async_sleeper

        logger.debug(
            "codex_app_server.adapter.init",
            event="adapter.init",
            context={
                "model": self._model_config.model,
                "codex_bin": self._client_config.codex_bin,
                "cwd": self._client_config.cwd,
                "approval_policy": self._client_config.approval_policy,
                "sandbox_policy": type(self._client_config.sandbox_policy).__name__,
                "ephemeral": self._client_config.ephemeral,
            },
        )

    @property
    @override
    def adapter_name(self) -> str:
        """Return the canonical Codex App Server adapter name."""
        return CODEX_APP_SERVER_ADAPTER_NAME

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
        render_event_id = uuid4()
        created_at = _utcnow()

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
                "codex_app_server.evaluate.rendered_tools_dispatch_failed",
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

    @staticmethod
    def _setup_skill_env(
        rendered: RenderedPrompt[object],
        config_env: Mapping[str, str] | None,
    ) -> tuple[CodexEphemeralHome | None, dict[str, str] | None]:
        """Create an ephemeral home for skill discovery if needed.

        Returns (ephemeral_home_or_none, merged_env_or_none).
        """
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

        dynamic_tool_specs = bridged_tools_to_dynamic_specs(bridged_tools)
        tool_lookup = {t.name: t for t in bridged_tools}

        output_schema = build_output_schema(rendered)

        ephemeral_home, client_env = self._setup_skill_env(
            rendered, self._client_config.env
        )

        client = CodexAppServerClient(
            codex_bin=self._client_config.codex_bin,
            env=client_env,
            suppress_stderr=self._client_config.suppress_stderr,
        )

        start_time = _utcnow()
        try:
            result = await execute_protocol(
                client_config=self._client_config,
                model_config=self._model_config,
                client=client,
                session=session,
                adapter_name=CODEX_APP_SERVER_ADAPTER_NAME,
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
                async_sleeper=self._async_sleeper,
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
            if ephemeral_home is not None:
                ephemeral_home.cleanup()

        accumulated_text, usage = result
        return build_response(
            accumulated_text=accumulated_text,
            usage=usage,
            output_schema=output_schema,
            rendered=rendered,
            prompt_name=prompt_name,
            adapter_name=CODEX_APP_SERVER_ADAPTER_NAME,
            session=session,
            budget_tracker=budget_tracker,
            run_context=run_context,
            start_time=start_time,
            utcnow=_utcnow(),
        )
