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

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
SettingSource = Literal["user", "project", "local"]


@FrozenDataclass()
class SandboxNetworkConfig:
    """Network configuration used when enabling the SDK sandbox."""

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
    """Client-level configuration for the Claude Agent SDK."""

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
    """Model-level configuration for Claude Agent SDK."""

    model: str = "claude-sonnet-4-5-20250929"
