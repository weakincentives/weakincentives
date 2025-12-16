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

"""Prompt authoring primitives exposed by :mod:`weakincentives.prompt`.

This module exports essential types for prompt authoring. Advanced types
are available in submodules:

- :mod:`weakincentives.prompt.overrides` — Prompt override infrastructure
- :mod:`weakincentives.prompt.protocols` — Protocol types
- :mod:`weakincentives.prompt.rendering` — RenderedPrompt and related types
- :mod:`weakincentives.prompt.structured_output` — Structured output parsing
- :mod:`weakincentives.prompt.task_examples` — TaskExample, TaskExamplesSection
- :mod:`weakincentives.prompt.visibility_overrides` — Visibility override events
- :mod:`weakincentives.prompt.progressive_disclosure` — OpenSectionsParams
"""

from __future__ import annotations

from .errors import (
    PromptError,
    PromptRenderError,
    PromptValidationError,
)
from .markdown import MarkdownSection
from .prompt import Prompt, PromptTemplate
from .section import Section, SectionVisibility
from .tool import Tool, ToolContext, ToolExample, ToolHandler
from .tool_result import ToolResult

__all__ = [
    "MarkdownSection",
    "Prompt",
    "PromptError",
    "PromptRenderError",
    "PromptTemplate",
    "PromptValidationError",
    "Section",
    "SectionVisibility",
    "Tool",
    "ToolContext",
    "ToolExample",
    "ToolHandler",
    "ToolResult",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
