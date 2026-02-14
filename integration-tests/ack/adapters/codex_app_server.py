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

"""ACK fixture for the Codex App Server adapter."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.runtime.session import Session

from ._protocol import AdapterCapabilities

_MODEL_ENV_VAR = "CODEX_APP_SERVER_TEST_MODEL"
_DEFAULT_MODEL = "gpt-5.3-codex"


class CodexAppServerFixture:
    """Adapter fixture for Codex App Server integration tests."""

    @property
    def adapter_name(self) -> str:
        return "codex_app_server"

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
            native_tools=True,
            workspace_isolation=True,
            custom_env_forwarding=True,
            network_policy=True,
            sandbox_policy=True,
        )

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    def create_adapter(self, tmp_path: Path) -> ProviderAdapter[object]:
        from weakincentives.adapters.codex_app_server import (
            CodexAppServerAdapter,
            CodexAppServerClientConfig,
            CodexAppServerModelConfig,
        )

        return CodexAppServerAdapter(
            model_config=CodexAppServerModelConfig(model=self.get_model()),
            client_config=CodexAppServerClientConfig(
                cwd=str(tmp_path),
                approval_policy="never",
            ),
        )

    def create_adapter_with_sandbox(
        self,
        tmp_path: Path,
        *,
        sandbox_mode: str,
    ) -> ProviderAdapter[object]:
        from weakincentives.adapters.codex_app_server import (
            CodexAppServerAdapter,
            CodexAppServerClientConfig,
            CodexAppServerModelConfig,
        )

        return CodexAppServerAdapter(
            model_config=CodexAppServerModelConfig(model=self.get_model()),
            client_config=CodexAppServerClientConfig(
                cwd=str(tmp_path),
                approval_policy="never",
                sandbox_mode=sandbox_mode,
            ),
        )

    def create_adapter_with_env(
        self,
        tmp_path: Path,
        *,
        env: Mapping[str, str],
    ) -> ProviderAdapter[object]:
        from weakincentives.adapters.codex_app_server import (
            CodexAppServerAdapter,
            CodexAppServerClientConfig,
            CodexAppServerModelConfig,
        )

        return CodexAppServerAdapter(
            model_config=CodexAppServerModelConfig(model=self.get_model()),
            client_config=CodexAppServerClientConfig(
                cwd=str(tmp_path),
                approval_policy="never",
                env=dict(env),
            ),
        )

    def create_session(self) -> Session:
        return Session(tags={"suite": "ack", "adapter": self.adapter_name})

    def get_model(self) -> str:
        return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)
