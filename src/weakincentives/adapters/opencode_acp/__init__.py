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

"""OpenCode ACP adapter for evaluating prompts via OpenCode's ACP server.

This package provides the OpenCodeACPAdapter which delegates prompt execution
to OpenCode via the Agent Client Protocol (ACP). The adapter handles:

- Prompt composition and rendering
- MCP tool bridging for WINK tools
- Session state management and reuse
- Workspace isolation with host mount support
- Structured output via dedicated MCP tool
- Event dispatch (PromptRendered, ToolInvoked, PromptExecuted)

Requirements
------------
- OpenCode CLI installed and available on PATH as ``opencode``
- ACP Python SDK: ``agent-client-protocol>=0.7.1``
- Claude Agent SDK: ``claude-agent-sdk>=0.1.15`` (for MCP server infrastructure)

Install with::

    pip install 'weakincentives[acp]'

Example
-------
Basic usage::

    from weakincentives import Prompt, PromptTemplate, MarkdownSection
    from weakincentives.runtime import Session, InProcessDispatcher
    from weakincentives.adapters.opencode_acp import (
        OpenCodeACPAdapter,
        OpenCodeACPClientConfig,
    )

    bus = InProcessDispatcher()
    session = Session(dispatcher=bus)

    template = PromptTemplate(
        ns="demo",
        key="opencode",
        sections=(
            MarkdownSection(
                title="Task",
                key="task",
                template="List the files in the repo and summarize.",
            ),
        ),
    )
    prompt = Prompt(template)

    adapter = OpenCodeACPAdapter(
        client_config=OpenCodeACPClientConfig(
            cwd="/absolute/path/to/workspace",
            permission_mode="auto",
            allow_file_reads=True,
            allow_file_writes=False,
        )
    )

    with prompt.resources:
        resp = adapter.evaluate(prompt, session=session)

    print(resp.text)

Configuration
-------------
The adapter uses two configuration classes:

- :class:`OpenCodeACPClientConfig`: Client-level settings for the ACP connection
  including executable path, working directory, permission mode, and file access.

- :class:`OpenCodeACPAdapterConfig`: Adapter-level settings including mode/model
  selection and output handling.

Workspace Management
--------------------
Use :class:`OpenCodeWorkspaceSection` to set up an isolated workspace with
host file mounts::

    from weakincentives.adapters.opencode_acp import (
        OpenCodeWorkspaceSection,
        HostMount,
    )

    workspace = OpenCodeWorkspaceSection(
        session=session,
        mounts=[
            HostMount(
                host_path="/path/to/project",
                mount_path="project",
                exclude_glob=("*.pyc", "__pycache__/*"),
                max_bytes=10_000_000,
            ),
        ],
        allowed_host_roots=[Path("/path/to")],
    )

See Also
--------
- :mod:`weakincentives.adapters.claude_agent_sdk`: Claude Agent SDK adapter
- ``specs/OPENCODE_ACP_ADAPTER.md``: Full specification
"""

from __future__ import annotations

from ...types import OPENCODE_ACP_ADAPTER_NAME
from .adapter import OpenCodeACPAdapter
from .config import OpenCodeACPAdapterConfig, OpenCodeACPClientConfig, PermissionMode
from .workspace import (
    HostMount,
    HostMountPreview,
    McpServerConfig,
    OpenCodeWorkspaceSection,
    WorkspaceBudgetExceededError,
    WorkspaceSecurityError,
)

__all__ = [
    "OPENCODE_ACP_ADAPTER_NAME",
    "HostMount",
    "HostMountPreview",
    "McpServerConfig",
    "OpenCodeACPAdapter",
    "OpenCodeACPAdapterConfig",
    "OpenCodeACPClientConfig",
    "OpenCodeWorkspaceSection",
    "PermissionMode",
    "WorkspaceBudgetExceededError",
    "WorkspaceSecurityError",
]


def __dir__() -> list[str]:
    return sorted(__all__)
