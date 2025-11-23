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

"""Stable prompt API grouped into layered subpackages.

The top-level package surfaces the public interfaces documented in
:mod:`weakincentives.prompt.api` while keeping implementation details organized
under :mod:`weakincentives.prompt.sections`, :mod:`weakincentives.prompt.tools`,
:mod:`weakincentives.prompt.composition`, and :mod:`weakincentives.prompt.overrides`.
"""

from __future__ import annotations

from . import api, composition, overrides, sections, tools
from .api import (
    Chapter,
    ChapterDescriptor,
    ChaptersExpansionPolicy,
    DelegationParams,
    DelegationPrompt,
    DelegationSummarySection,
    LocalPromptOverridesStore,
    MarkdownSection,
    OutputParseError,
    ParentPromptParams,
    ParentPromptSection,
    Prompt,
    PromptDescriptor,
    PromptError,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    PromptProtocol,
    PromptRenderError,
    PromptValidationError,
    ProviderAdapterProtocol,
    RecapParams,
    RecapSection,
    RenderedPromptProtocol,
    Section,
    SectionDescriptor,
    SectionOverride,
    SectionPath,
    StructuredOutputConfig,
    SupportsDataclass,
    SupportsToolResult,
    Tool,
    ToolContext,
    ToolDescriptor,
    ToolHandler,
    ToolOverride,
    ToolRenderableResult,
    ToolResult,
    hash_json,
    hash_text,
    parse_structured_output,
)

__all__ = [
    "Chapter",
    "ChapterDescriptor",
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
    "ToolRenderableResult",
    "ToolResult",
    "api",
    "composition",
    "hash_json",
    "hash_text",
    "overrides",
    "parse_structured_output",
    "sections",
    "tools",
]
