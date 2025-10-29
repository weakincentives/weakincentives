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

"""Prompt module scaffolding."""

from __future__ import annotations

from ._types import SupportsDataclass
from .errors import (
    PromptError,
    PromptRenderError,
    PromptValidationError,
    SectionPath,
)
from .prompt import Prompt, PromptSectionNode, RenderedPrompt
from .section import Section
from .structured import OutputParseError, parse_output
from .text import TextSection
from .tool import Tool, ToolResult

__all__ = [
    "Prompt",
    "RenderedPrompt",
    "PromptSectionNode",
    "PromptError",
    "PromptRenderError",
    "PromptValidationError",
    "Section",
    "SectionPath",
    "SupportsDataclass",
    "TextSection",
    "Tool",
    "ToolResult",
    "OutputParseError",
    "parse_output",
]
