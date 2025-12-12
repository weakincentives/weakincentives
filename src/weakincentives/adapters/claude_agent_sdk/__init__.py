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

This adapter enables weakincentives prompts to leverage Claude's full agentic
capabilities through the official claude-code-sdk Python package. It uses the
SDK's hook system to synchronize state bidirectionally between SDK execution
and the weakincentives Session.

Example:
    >>> from weakincentives import Prompt, MarkdownSection
    >>> from weakincentives.runtime import Session, InProcessEventBus
    >>> from weakincentives.adapters.claude_agent_sdk import (
    ...     ClaudeAgentSDKAdapter,
    ...     ClaudeAgentSDKClientConfig,
    ...     ClaudeAgentWorkspaceSection,
    ...     HostMount,
    ... )
    >>>
    >>> bus = InProcessEventBus()
    >>> session = Session(bus=bus)
    >>>
    >>> # Create workspace section with host files
    >>> workspace = ClaudeAgentWorkspaceSection(
    ...     session=session,
    ...     mounts=[HostMount(host_path="src")],
    ...     allowed_host_roots=["/home/user/project"],
    ... )
    >>>
    >>> adapter = ClaudeAgentSDKAdapter(
    ...     model="claude-sonnet-4-5-20250929",
    ...     client_config=ClaudeAgentSDKClientConfig(
    ...         permission_mode="acceptEdits",
    ...         cwd=str(workspace.temp_dir),
    ...     ),
    ... )
    >>>
    >>> prompt = Prompt[str](
    ...     ns="test",
    ...     key="hello",
    ...     sections=[
    ...         MarkdownSection(
    ...             title="Task",
    ...             key="task",
    ...             template="Review the code in src/",
    ...         ),
    ...         workspace,
    ...     ],
    ... )
    >>>
    >>> response = adapter.evaluate(prompt, session=session)
    >>> workspace.cleanup()
"""

from __future__ import annotations

from .adapter import CLAUDE_AGENT_SDK_ADAPTER_NAME, ClaudeAgentSDKAdapter
from .config import (
    ClaudeAgentSDKClientConfig,
    ClaudeAgentSDKModelConfig,
    PermissionMode,
    SandboxNetworkConfig,
    SandboxSettings,
    SettingSource,
)
from .workspace import (
    ClaudeAgentWorkspaceSection,
    HostMount,
    HostMountPreview,
    WorkspaceBudgetExceededError,
    WorkspaceSecurityError,
)

__all__ = [
    "CLAUDE_AGENT_SDK_ADAPTER_NAME",
    "ClaudeAgentSDKAdapter",
    "ClaudeAgentSDKClientConfig",
    "ClaudeAgentSDKModelConfig",
    "ClaudeAgentWorkspaceSection",
    "HostMount",
    "HostMountPreview",
    "PermissionMode",
    "SandboxNetworkConfig",
    "SandboxSettings",
    "SettingSource",
    "WorkspaceBudgetExceededError",
    "WorkspaceSecurityError",
]
