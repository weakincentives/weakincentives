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

"""ACK fixture for the OpenCode ACP adapter."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.runtime.session import Session

from ._protocol import AdapterCapabilities

_MODEL_ENV_VAR = "OPENCODE_ACP_TEST_MODEL"


class OpenCodeACPFixture:
    """Adapter fixture for OpenCode ACP integration tests."""

    @property
    def adapter_name(self) -> str:
        return "opencode_acp"

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            text_response=True,
            tool_invocation=True,
            structured_output=True,
            event_emission=True,
            transcript=True,
            rendered_tools_event=False,
            progressive_disclosure=True,
            transactional_tools=True,
            deadline_enforcement=True,
            budget_enforcement=False,
            native_tools=False,
            workspace_isolation=False,
            custom_env_forwarding=False,
            network_policy=False,
            sandbox_policy=False,
            skill_installation=True,
        )

    def is_available(self) -> bool:
        return shutil.which("opencode") is not None

    def create_adapter(self, tmp_path: Path) -> ProviderAdapter[object]:
        from weakincentives.adapters.opencode_acp import (
            OpenCodeACPAdapter,
            OpenCodeACPAdapterConfig,
            OpenCodeACPClientConfig,
        )

        model_id = self.get_model() or None
        return OpenCodeACPAdapter(
            adapter_config=OpenCodeACPAdapterConfig(model_id=model_id),
            client_config=OpenCodeACPClientConfig(
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
    ) -> NoReturn:
        raise NotImplementedError("OpenCode ACP does not support sandbox_policy")

    def create_adapter_with_env(
        self,
        tmp_path: Path,
        *,
        env: Mapping[str, str],
    ) -> NoReturn:
        raise NotImplementedError("OpenCode ACP does not support custom_env_forwarding")

    def create_session(self) -> Session:
        return Session(tags={"suite": "ack", "adapter": self.adapter_name})

    def get_model(self) -> str:
        return os.environ.get(_MODEL_ENV_VAR, "")
