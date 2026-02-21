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

"""OpenCode ACP adapter — thin wrapper over the generic ACPAdapter."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, override

from ...prompt import RenderedPrompt
from ...types import OPENCODE_ACP_ADAPTER_NAME, AdapterName
from ..acp import ACPAdapter, ACPClient, build_env
from ..core import PromptEvaluationError
from ._ephemeral_home import OpenCodeEphemeralHome
from .config import OpenCodeACPAdapterConfig, OpenCodeACPClientConfig

__all__ = ["OpenCodeACPAdapter"]


class OpenCodeACPAdapter(ACPAdapter):
    """OpenCode-specific ACP adapter.

    Adds model validation against available_models, empty response detection,
    and error-tolerant mode setting. All protocol logic lives in the parent
    :class:`ACPAdapter`.
    """

    def __init__(
        self,
        *,
        adapter_config: OpenCodeACPAdapterConfig | None = None,
        client_config: OpenCodeACPClientConfig | None = None,
    ) -> None:
        super().__init__(
            adapter_config=adapter_config or OpenCodeACPAdapterConfig(),
            client_config=client_config or OpenCodeACPClientConfig(),
        )

    @override
    def _adapter_name(self) -> AdapterName:
        return OPENCODE_ACP_ADAPTER_NAME

    @override
    def _validate_model(self, model_id: str, available_models: list[Any]) -> None:
        """Validate model_id against available models.

        OpenCode silently returns empty for invalid models — we must fail fast.
        """
        if not available_models:
            return  # No model list provided; skip validation

        model_ids = [getattr(m, "model_id", None) for m in available_models]
        if model_id not in model_ids:
            raise PromptEvaluationError(
                message=(
                    f"Model '{model_id}' not found in available models: {model_ids}"
                ),
                prompt_name="",
                phase="request",
            )

    @override
    def _detect_empty_response(self, client: ACPClient, prompt_resp: Any) -> None:
        """Raise if OpenCode returned zero agent message chunks."""
        if not client.message_chunks:
            raise PromptEvaluationError(
                message=(
                    "OpenCode returned an empty response (zero "
                    "AgentMessageChunks). This may indicate an invalid model "
                    "or configuration."
                ),
                prompt_name="",
                phase="response",
            )

    @override
    def _prepare_execution_env(
        self,
        *,
        rendered: RenderedPrompt[Any],
        effective_cwd: str,
    ) -> tuple[dict[str, str] | None, Callable[[], None]]:
        """Prepare subprocess environment with optional ephemeral home.

        When skills are present in the rendered prompt, creates an ephemeral
        home directory, mounts skills, and overrides HOME in the env so
        OpenCode discovers them at ``$HOME/.claude/skills/<name>/SKILL.md``.

        When no skills are present, delegates to the parent implementation.
        """
        if not rendered.skills:
            return super()._prepare_execution_env(
                rendered=rendered,
                effective_cwd=effective_cwd,
            )

        home = OpenCodeEphemeralHome(workspace_path=effective_cwd)
        try:
            home.mount_skills(rendered.skills)
        except Exception:
            home.cleanup()
            raise

        base_env = build_env(self._client_config)
        if base_env is None:
            base_env = dict(os.environ)
        base_env["HOME"] = home.home_path

        return base_env, home.cleanup
