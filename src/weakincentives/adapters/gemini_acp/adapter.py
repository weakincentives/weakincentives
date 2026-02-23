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

"""Gemini CLI ACP adapter — thin wrapper over the generic ACPAdapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, override

from ...prompt import RenderedPrompt
from ...types import GEMINI_ACP_ADAPTER_NAME, AdapterName
from ..acp import build_env
from ..acp.adapter import ACPAdapter
from ..acp.client import ACPClient
from ..core import PromptEvaluationError
from .config import GeminiACPAdapterConfig, GeminiACPClientConfig

__all__ = ["GeminiACPAdapter"]


def _noop() -> None:
    pass


class GeminiACPAdapter(ACPAdapter):
    """Gemini CLI-specific ACP adapter.

    Adds CLI flag injection for model/approval-mode/sandbox, empty response
    detection, and skips unsupported ``session/setModel`` /
    ``session/setMode`` calls.  All protocol logic lives in the parent
    :class:`ACPAdapter`.
    """

    def __init__(
        self,
        *,
        adapter_config: GeminiACPAdapterConfig | None = None,
        client_config: GeminiACPClientConfig | None = None,
    ) -> None:
        super().__init__(
            adapter_config=adapter_config or GeminiACPAdapterConfig(),
            client_config=client_config or GeminiACPClientConfig(),
        )

    @override
    def _adapter_name(self) -> AdapterName:
        return GEMINI_ACP_ADAPTER_NAME

    @override
    def _validate_model(self, model_id: str, available_models: list[Any]) -> None:
        """No-op — Gemini does not return available model lists."""

    @override
    def _detect_empty_response(self, client: ACPClient, prompt_resp: Any) -> None:
        """Raise if Gemini returned zero agent message chunks."""
        if not client.message_chunks:
            raise PromptEvaluationError(
                message=(
                    "Gemini returned an empty response (zero "
                    "AgentMessageChunks). This may indicate an invalid model "
                    "or configuration."
                ),
                prompt_name="",
                phase="response",
            )

    @override
    def _agent_spawn_args(self) -> tuple[str, ...]:
        """Append --model, --approval-mode, and --sandbox flags."""
        args = list(self._client_config.agent_args)
        if self._adapter_config.model_id:
            args.extend(("--model", self._adapter_config.model_id))
        adapter_cfg = self._adapter_config
        if isinstance(adapter_cfg, GeminiACPAdapterConfig):
            if adapter_cfg.approval_mode:
                args.extend(("--approval-mode", adapter_cfg.approval_mode))
            if adapter_cfg.sandbox:
                args.append("--sandbox")
        return tuple(args)

    @override
    def _prepare_execution_env(
        self,
        *,
        rendered: RenderedPrompt[Any],
        effective_cwd: str,
    ) -> tuple[dict[str, str] | None, Callable[[], None]]:
        """Prepare subprocess environment with optional seatbelt profile.

        When ``sandbox_profile`` is set on the adapter config, injects the
        ``SEATBELT_PROFILE`` environment variable so that Gemini's
        ``--sandbox`` flag uses the specified macOS seatbelt profile.
        """
        env_config = dict(self._client_config.env) if self._client_config.env else None
        adapter_cfg = self._adapter_config
        if (
            isinstance(adapter_cfg, GeminiACPAdapterConfig)
            and adapter_cfg.sandbox_profile
        ):
            env = build_env(env_config)
            if env is None:
                import os

                env = dict(os.environ)
            env["SEATBELT_PROFILE"] = adapter_cfg.sandbox_profile
            return env, _noop
        return build_env(env_config), _noop

    @override
    async def _configure_session(self, conn: Any, session_id: str) -> None:
        """No-op — Gemini does not support session/setModel or session/setMode."""
