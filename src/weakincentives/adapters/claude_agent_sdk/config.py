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
from typing import Literal

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
"""Permission handling mode for Claude Agent SDK tool execution."""

SettingSource = Literal["user", "project", "local"]
"""Configuration file source for SDK settings."""


@FrozenDataclass()
class SandboxNetworkConfig:
    """Network-specific sandbox configuration."""

    allow_local_binding: bool = False
    allow_unix_sockets: tuple[str, ...] = ()


@FrozenDataclass()
class SandboxSettings:
    """Sandboxing configuration for SDK execution."""

    enabled: bool = False
    auto_allow_bash_if_sandboxed: bool = False
    excluded_commands: tuple[str, ...] = ()
    allow_unsandboxed_commands: bool = False
    network: SandboxNetworkConfig | None = None


@FrozenDataclass()
class ClaudeAgentSDKClientConfig:
    """Client-level configuration for Claude Agent SDK.

    Attributes:
        permission_mode: Tool permission handling mode. Defaults to
            ``"bypassPermissions"`` for programmatic access.
        cwd: Working directory for SDK operations. None uses the current
            working directory.
        add_dirs: Additional directories accessible to the SDK.
        env: Environment variables passed to the SDK process.
        setting_sources: Configuration file sources (empty = isolated from
            user/project settings).
        sandbox: Sandboxing configuration for SDK execution.
        max_turns: Maximum number of conversation turns. None means unlimited.
        include_partial_messages: Whether to include streaming partial messages
            in the result.
    """

    permission_mode: PermissionMode = "bypassPermissions"
    cwd: str | None = None
    add_dirs: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    setting_sources: tuple[SettingSource, ...] = ()
    sandbox: SandboxSettings | None = None
    max_turns: int | None = None
    include_partial_messages: bool = False


@FrozenDataclass()
class ClaudeAgentSDKModelConfig(LLMConfig):
    """Model-level configuration for Claude Agent SDK.

    Extends LLMConfig with parameters specific to Claude models via the
    Claude Agent SDK.

    Attributes:
        model: Claude model identifier. Defaults to the latest Sonnet model.

    Notes:
        The Claude Agent SDK does not support ``seed``, ``stop``,
        ``presence_penalty``, or ``frequency_penalty``. If any of these fields
        are provided, ``ClaudeAgentSDKModelConfig`` raises ``ValueError``.
    """

    model: str = "claude-sonnet-4-5-20250929"

    def __post_init__(self) -> None:
        unsupported: dict[str, object | None] = {
            "seed": self.seed,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        set_unsupported = [
            key for key, value in unsupported.items() if value is not None
        ]
        if set_unsupported:
            raise ValueError(
                "Unsupported Claude Agent SDK parameters: "
                + ", ".join(sorted(set_unsupported))
                + ". Remove them from ClaudeAgentSDKModelConfig."
            )
