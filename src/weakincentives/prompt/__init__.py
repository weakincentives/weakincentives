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

from ._types import SupportsDataclass, SupportsToolResult
from .chapter import Chapter, ChaptersExpansionPolicy
from .composition import (
    DelegationParams,
    DelegationPrompt,
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
from .overrides import (
    LocalPromptOverridesStore,
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionDescriptor,
    SectionOverride,
    ToolDescriptor,
    ToolOverride,
    hash_json,
    hash_text,
)
from .prompt import Prompt
from .protocols import (
    PromptProtocol,
    ProviderAdapterProtocol,
    RenderedPromptProtocol,
)
from .section import Section
from .structured_output import (
    OutputParseError,
    StructuredOutputConfig,
    parse_structured_output,
)
from .tool import Tool, ToolContext, ToolHandler
from .tool_result import ToolResult

__all__ = [
    "Chapter",
    "ChaptersExpansionPolicy",
    "DelegationParams",
    "DelegationPrompt",
    "DelegationSummarySection",
    "LocalPromptOverridesStore",
    "MarkdownSection",
    "OutputParseError",
    "ParentPromptParams",
    "ParentPromptSection",
    "Prompt",
    "PromptDescriptor",
    "PromptError",
    "PromptLike",
    "PromptOverride",
    "PromptOverridesError",
    "PromptOverridesStore",
    "PromptProtocol",
    "PromptRenderError",
    "PromptValidationError",
    "ProviderAdapterProtocol",
    "RecapParams",
    "RecapSection",
    "RenderedPromptProtocol",
    "Section",
    "SectionDescriptor",
    "SectionOverride",
    "SectionPath",
    "StructuredOutputConfig",
    "SupportsDataclass",
    "SupportsToolResult",
    "Tool",
    "ToolContext",
    "ToolDescriptor",
    "ToolHandler",
    "ToolOverride",
    "ToolResult",
    "hash_json",
    "hash_text",
    "parse_structured_output",
]
