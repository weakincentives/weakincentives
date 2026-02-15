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

"""ACK fixture for the Claude Agent SDK adapter."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import assert_never

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.runtime.session import Session

from ._protocol import AdapterCapabilities, FileSystemMode, NetworkMode, SandboxSpec

_MODEL_ENV_VAR = "CLAUDE_AGENT_SDK_TEST_MODEL"


def _is_bedrock_mode() -> bool:
    return os.getenv("CLAUDE_CODE_USE_BEDROCK") == "1" and "AWS_REGION" in os.environ


class ClaudeAgentSDKFixture:
    """Adapter fixture for Claude Agent SDK integration tests."""

    @property
    def adapter_name(self) -> str:
        return "claude_agent_sdk"

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
            skill_installation=True,
        )

    def is_available(self) -> bool:
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError:
            return False
        return _is_bedrock_mode() or "ANTHROPIC_API_KEY" in os.environ

    def create_adapter(self, tmp_path: Path) -> ProviderAdapter[object]:
        from weakincentives.adapters.claude_agent_sdk import (
            ClaudeAgentSDKAdapter,
            ClaudeAgentSDKClientConfig,
            IsolationConfig,
        )

        return ClaudeAgentSDKAdapter(
            model=self.get_model(),
            client_config=ClaudeAgentSDKClientConfig(
                permission_mode="bypassPermissions",
                cwd=str(tmp_path),
                isolation=IsolationConfig.inherit_host_auth(),
            ),
        )

    def create_adapter_with_sandbox(
        self,
        tmp_path: Path,
        *,
        sandbox: SandboxSpec,
    ) -> ProviderAdapter[object]:
        from weakincentives.adapters.claude_agent_sdk import (
            ClaudeAgentSDKAdapter,
            ClaudeAgentSDKClientConfig,
            IsolationConfig,
            NetworkPolicy,
            SandboxConfig,
        )

        if sandbox.filesystem is FileSystemMode.READ_ONLY:
            sdk_sandbox = SandboxConfig(enabled=True, writable_paths=())
            # acceptEdits respects sandbox boundaries; bypassPermissions
            # skips all permission checks and overrides the sandbox.
            permission_mode = "acceptEdits"
        elif sandbox.filesystem is FileSystemMode.WORKSPACE_WRITE:
            sdk_sandbox = SandboxConfig(
                enabled=True,
                writable_paths=(str(tmp_path),),
            )
            permission_mode = "bypassPermissions"
        else:
            assert_never(sandbox.filesystem)

        network_policy: NetworkPolicy | None
        if sandbox.network is NetworkMode.BLOCKED:
            network_policy = NetworkPolicy.no_network()
        elif sandbox.network is NetworkMode.ENABLED:
            network_policy = None
        else:
            assert_never(sandbox.network)

        return ClaudeAgentSDKAdapter(
            model=self.get_model(),
            client_config=ClaudeAgentSDKClientConfig(
                permission_mode=permission_mode,
                cwd=str(tmp_path),
                isolation=IsolationConfig.inherit_host_auth(
                    sandbox=sdk_sandbox,
                    network_policy=network_policy,
                ),
            ),
        )

    def create_adapter_with_env(
        self,
        tmp_path: Path,
        *,
        env: Mapping[str, str],
    ) -> ProviderAdapter[object]:
        from weakincentives.adapters.claude_agent_sdk import (
            ClaudeAgentSDKAdapter,
            ClaudeAgentSDKClientConfig,
            IsolationConfig,
        )

        return ClaudeAgentSDKAdapter(
            model=self.get_model(),
            client_config=ClaudeAgentSDKClientConfig(
                permission_mode="bypassPermissions",
                cwd=str(tmp_path),
                isolation=IsolationConfig.inherit_host_auth(env=dict(env)),
            ),
        )

    def create_session(self) -> Session:
        return Session(tags={"suite": "ack", "adapter": self.adapter_name})

    def get_model(self) -> str:
        from weakincentives.adapters.claude_agent_sdk import get_default_model

        return os.environ.get(_MODEL_ENV_VAR, get_default_model())
