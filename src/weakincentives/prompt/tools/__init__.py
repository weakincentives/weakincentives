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

"""Tool registration and result helpers used by prompts.

The :mod:`weakincentives.prompt.tools` layer holds the abstractions that wire
up tool handlers to prompts and normalize their outputs for downstream
rendering and telemetry.
"""

from __future__ import annotations

from .tool import Tool, ToolContext, ToolHandler
from .tool_result import ToolResult, render_tool_payload

__all__ = ["Tool", "ToolContext", "ToolHandler", "ToolResult", "render_tool_payload"]
