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

"""Configuration dataclasses for the Claude Agent SDK adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from ...dataclasses import FrozenDataclass
from ..config import LLMConfig

__all__ = [
    "ClaudeAgentSDKClientConfig",
    "ClaudeAgentSDKModelConfig",
    "PermissionMode",
    "SandboxNetworkConfig",
    "SandboxSettings",
    "SettingSource",
]


PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
"""Tool permission handling mode for the Claude Agent SDK."""

SettingSource = Literal["user", "project", "local"]
"""Configuration file sources for the SDK."""


@FrozenDataclass()
class SandboxNetworkConfig:
    """Network-specific sandbox configuration.

    Attributes:
        allow_local_binding: Whether to allow binding to local addresses.
        allow_unix_sockets: Unix socket paths that are permitted.
    """

    allow_local_binding: bool = False
    allow_unix_sockets: tuple[str, ...] = ()


@FrozenDataclass()
class SandboxSettings:
    """Sandboxing configuration for SDK execution.

    Attributes:
        enabled: Whether sandboxing is enabled.
        auto_allow_bash_if_sandboxed: Automatically allow bash commands when sandboxed.
        excluded_commands: Commands to exclude from sandbox restrictions.
        allow_unsandboxed_commands: Whether to permit commands outside the sandbox.
        network: Network-specific sandbox configuration.
    """

    enabled: bool = False
    auto_allow_bash_if_sandboxed: bool = False
    excluded_commands: tuple[str, ...] = ()
    allow_unsandboxed_commands: bool = False
    network: SandboxNetworkConfig | None = None


@FrozenDataclass()
class ClaudeAgentSDKClientConfig:
    """Client-level configuration for Claude Agent SDK.

    These parameters control the SDK client behavior and execution environment.

    Attributes:
        permission_mode: Tool permission handling mode.
            - "default": Standard permission checks.
            - "acceptEdits": Auto-accept file edits.
            - "plan": Planning mode only.
            - "bypassPermissions": Skip all permission checks.
        cwd: Working directory for SDK operations. None uses current directory.
        add_dirs: Additional directories the SDK can access.
        env: Environment variables passed to SDK subprocess.
        setting_sources: Config file sources to load. Empty tuple means isolated.
        sandbox: Sandboxing configuration for execution.
        max_turns: Maximum conversation turns. None means unlimited.
        include_partial_messages: Whether to include streaming partial messages.
    """

    permission_mode: PermissionMode = "bypassPermissions"
    cwd: str | None = None
    add_dirs: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    setting_sources: tuple[SettingSource, ...] = ()
    sandbox: SandboxSettings | None = None
    max_turns: int | None = None
    include_partial_messages: bool = False

    def to_client_kwargs(self) -> dict[str, Any]:
        """Convert non-None fields to SDK client constructor kwargs."""
        kwargs: dict[str, Any] = {
            "permission_mode": self.permission_mode,
        }
        if self.cwd is not None:
            kwargs["cwd"] = self.cwd
        if self.add_dirs:
            kwargs["add_dirs"] = list(self.add_dirs)
        if self.env is not None:
            kwargs["env"] = dict(self.env)
        if self.setting_sources:
            kwargs["setting_sources"] = list(self.setting_sources)
        if self.sandbox is not None:
            kwargs["sandbox"] = _sandbox_to_dict(self.sandbox)
        if self.max_turns is not None:
            kwargs["max_turns"] = self.max_turns
        if self.include_partial_messages:
            kwargs["include_partial_messages"] = self.include_partial_messages
        return kwargs


@FrozenDataclass()
class ClaudeAgentSDKModelConfig(LLMConfig):
    """Model-level configuration for Claude Agent SDK.

    Extends LLMConfig with parameters specific to the Claude Agent SDK.

    Attributes:
        model: Claude model identifier. Defaults to claude-sonnet-4-5.
    """

    model: str = "claude-sonnet-4-5-20250929"


def _sandbox_to_dict(sandbox: SandboxSettings) -> dict[str, Any]:
    """Convert SandboxSettings to SDK-compatible dict format."""
    result: dict[str, Any] = {
        "enabled": sandbox.enabled,
    }
    if sandbox.auto_allow_bash_if_sandboxed:
        result["auto_allow_bash_if_sandboxed"] = sandbox.auto_allow_bash_if_sandboxed
    if sandbox.excluded_commands:
        result["excluded_commands"] = list(sandbox.excluded_commands)
    if sandbox.allow_unsandboxed_commands:
        result["allow_unsandboxed_commands"] = sandbox.allow_unsandboxed_commands
    if sandbox.network is not None:
        result["network"] = _network_config_to_dict(sandbox.network)
    return result


def _network_config_to_dict(network: SandboxNetworkConfig) -> dict[str, Any]:
    """Convert SandboxNetworkConfig to SDK-compatible dict format."""
    result: dict[str, Any] = {}
    if network.allow_local_binding:
        result["allow_local_binding"] = network.allow_local_binding
    if network.allow_unix_sockets:
        result["allow_unix_sockets"] = list(network.allow_unix_sockets)
    return result
