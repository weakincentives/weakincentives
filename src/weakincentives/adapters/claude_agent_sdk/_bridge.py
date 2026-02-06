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

"""Tool bridge — re-exports from shared module."""

from __future__ import annotations

from .._shared._bridge import (
    BridgedTool,
    MCPToolExecutionState,
    create_bridged_tools,
    create_mcp_server,
    make_async_handler,
)

__all__ = [
    "BridgedTool",
    "MCPToolExecutionState",
    "create_bridged_tools",
    "create_mcp_server",
    "make_async_handler",
]
