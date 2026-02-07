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

"""Shared utilities used by multiple adapter implementations.

This package contains code that is needed by more than one adapter
(e.g. both the Claude Agent SDK adapter and the Codex App Server adapter).
Individual adapters should import from this package rather than from
each other's private modules.
"""

from __future__ import annotations

from ._async_utils import run_async
from ._bridge import (
    BridgedTool,
    MCPToolExecutionState,
    create_bridged_tools,
    create_mcp_server,
)
from ._visibility_signal import VisibilityExpansionSignal

__all__ = [
    "BridgedTool",
    "MCPToolExecutionState",
    "VisibilityExpansionSignal",
    "create_bridged_tools",
    "create_mcp_server",
    "run_async",
]
