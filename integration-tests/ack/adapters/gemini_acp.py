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

"""ACK fixture for the Gemini CLI ACP adapter."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.runtime.session import Session

from ._protocol import AdapterCapabilities

_MODEL_ENV_VAR = "GEMINI_ACP_TEST_MODEL"


class GeminiACPFixture:
    """Adapter fixture for Gemini CLI ACP integration tests."""

    @property
    def adapter_name(self) -> str:
        return "gemini_acp"

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            text_response=True,
            tool_invocation=True,
            structured_output=True,
            event_emission=True,
            transcript=True,
            rendered_tools_event=True,
            progressive_disclosure=True,
            transactional_tools=True,
            deadline_enforcement=True,
            budget_enforcement=False,
            native_tools=False,
            # --sandbox is incompatible with --experimental-acp (v0.29.5):
            # sandbox-exec breaks ACP's stdio pipe protocol.
            workspace_isolation=False,
            custom_env_forwarding=True,
            network_policy=False,
            sandbox_policy=False,
            skill_installation=False,
            tool_policies=True,
            feedback_providers=True,
            task_completion=True,
        )

    def is_available(self) -> bool:
        return shutil.which("gemini") is not None

    def create_adapter(self, tmp_path: Path) -> ProviderAdapter[object]:
        from weakincentives.adapters.gemini_acp import (
            GeminiACPAdapter,
            GeminiACPAdapterConfig,
            GeminiACPClientConfig,
        )

        model_id = self.get_model() or None
        return GeminiACPAdapter(
            adapter_config=GeminiACPAdapterConfig(
                model_id=model_id,
                approval_mode="yolo",
            ),
            client_config=GeminiACPClientConfig(
                cwd=str(tmp_path),
                allow_file_reads=True,
                allow_file_writes=True,
                permission_mode="auto",
            ),
        )

    def create_adapter_with_sandbox(
        self,
        tmp_path: Path,
        *,
        sandbox_mode: str,
    ) -> ProviderAdapter[object]:
        # --sandbox is incompatible with --experimental-acp in Gemini
        # v0.29.5: the sandbox re-launches gemini via sandbox-exec, which
        # breaks ACP's stdio pipe protocol.  See GEMINI_ACP_ADAPTER.md.
        raise NotImplementedError(
            "--sandbox is incompatible with --experimental-acp (Gemini v0.29.5)"
        )

    def create_adapter_with_env(
        self,
        tmp_path: Path,
        *,
        env: Mapping[str, str],
    ) -> ProviderAdapter[object]:
        from weakincentives.adapters.gemini_acp import (
            GeminiACPAdapter,
            GeminiACPAdapterConfig,
            GeminiACPClientConfig,
        )

        model_id = self.get_model() or None
        return GeminiACPAdapter(
            adapter_config=GeminiACPAdapterConfig(
                model_id=model_id,
                approval_mode="yolo",
            ),
            client_config=GeminiACPClientConfig(
                cwd=str(tmp_path),
                allow_file_reads=True,
                allow_file_writes=True,
                permission_mode="auto",
                env=dict(env),
            ),
        )

    def create_session(self) -> Session:
        return Session(tags={"suite": "ack", "adapter": self.adapter_name})

    def get_model(self) -> str:
        return os.environ.get(_MODEL_ENV_VAR, "")
