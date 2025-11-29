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

"""Prompt namespace."""

from __future__ import annotations

from .markdown import MarkdownSection
from .prompt import Prompt
from .structured_output import OutputParseError, StructuredOutputConfig, parse_structured_output
from .tool import Tool, ToolContext, ToolExample, ToolHandler
from .tool_result import ToolResult

__all__ = [
    "MarkdownSection",
    "OutputParseError",
    "Prompt",
    "StructuredOutputConfig",
    "Tool",
    "ToolContext",
    "ToolExample",
    "ToolHandler",
    "ToolResult",
    "parse_structured_output",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
