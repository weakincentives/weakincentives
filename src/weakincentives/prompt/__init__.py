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

"""Prompt authoring primitives exposed by :mod:`weakincentives.prompt`."""

from __future__ import annotations

from ._types import SupportsDataclass
from .composition import (
    DelegationPrompt,
    DelegationSummaryParams,
    DelegationSummarySection,
    ParentPromptParams,
    ParentPromptSection,
    RecapParams,
    RecapSection,
)
from .errors import (
    PromptError,
    PromptRenderError,
    PromptValidationError,
    SectionPath,
)
from .markdown import MarkdownSection
from .prompt import Prompt
from .registry import (
    PromptRegistry,
    clear_registry,
    get_prompt,
    iter_prompts,
    register_prompt,
    unregister_prompt,
)
from .section import Section
from .structured_output import OutputParseError, parse_structured_output
from .tool import Tool, ToolContext, ToolHandler
from .tool_result import ToolResult
from .versioning import (
    PromptDescriptor,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionDescriptor,
    SectionOverride,
    ToolDescriptor,
    ToolOverride,
)

__all__ = [
    "DelegationPrompt",
    "DelegationSummaryParams",
    "DelegationSummarySection",
    "MarkdownSection",
    "OutputParseError",
    "ParentPromptParams",
    "ParentPromptSection",
    "Prompt",
    "PromptDescriptor",
    "PromptError",
    "PromptOverride",
    "PromptOverridesError",
    "PromptOverridesStore",
    "PromptRegistry",
    "PromptRenderError",
    "PromptValidationError",
    "RecapParams",
    "RecapSection",
    "Section",
    "SectionDescriptor",
    "SectionOverride",
    "SectionPath",
    "SupportsDataclass",
    "Tool",
    "ToolContext",
    "ToolDescriptor",
    "ToolHandler",
    "ToolOverride",
    "ToolResult",
    "clear_registry",
    "get_prompt",
    "iter_prompts",
    "parse_structured_output",
    "register_prompt",
    "unregister_prompt",
]
