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

"""Claude Agent SDK adapter for weakincentives.

This module provides integration with the Claude Agent SDK (claude-agent-sdk),
enabling weakincentives prompts to leverage Claude's full agentic capabilities
through the official SDK.

The adapter uses the SDK's Hook system to synchronize state bidirectionally
between the SDK's internal execution and the weakincentives Session, preserving
the event-driven architecture while delegating tool execution to Claude Code.

Example:
    >>> from weakincentives import Prompt, MarkdownSection
    >>> from weakincentives.runtime import Session, InProcessEventBus
    >>> from weakincentives.adapters.claude_agent_sdk import (
    ...     ClaudeAgentSDKAdapter,
    ...     ClaudeAgentSDKClientConfig,
    ... )
    >>>
    >>> bus = InProcessEventBus()
    >>> session = Session(bus=bus)
    >>>
    >>> adapter = ClaudeAgentSDKAdapter(
    ...     model="claude-sonnet-4-5-20250929",
    ...     client_config=ClaudeAgentSDKClientConfig(
    ...         permission_mode="acceptEdits",
    ...         cwd="/home/user/project",
    ...     ),
    ... )
    >>>
    >>> response = adapter.evaluate(prompt, session=session)
"""

from .._names import CLAUDE_AGENT_SDK_ADAPTER_NAME
from .adapter import ClaudeAgentSDKAdapter
from .config import (
    ClaudeAgentSDKClientConfig,
    ClaudeAgentSDKModelConfig,
    PermissionMode,
    SandboxNetworkConfig,
    SandboxSettings,
    SettingSource,
)
from .workspace import (
    ClaudeAgentWorkspace,
    HostMount,
    HostMountPreview,
    WorkspaceBudgetExceededError,
    WorkspaceSecurityError,
    cleanup_workspace,
    create_workspace,
)

__all__ = [
    "CLAUDE_AGENT_SDK_ADAPTER_NAME",
    "ClaudeAgentSDKAdapter",
    "ClaudeAgentSDKClientConfig",
    "ClaudeAgentSDKModelConfig",
    "ClaudeAgentWorkspace",
    "HostMount",
    "HostMountPreview",
    "PermissionMode",
    "SandboxNetworkConfig",
    "SandboxSettings",
    "SettingSource",
    "WorkspaceBudgetExceededError",
    "WorkspaceSecurityError",
    "cleanup_workspace",
    "create_workspace",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
