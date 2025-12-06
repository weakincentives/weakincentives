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

from ._types import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
    ToolRenderableResult,
)
from .errors import (
    PromptError,
    PromptRenderError,
    PromptValidationError,
    SectionPath,
    VisibilityExpansionRequired,
)
from .hosted_tool import HostedTool
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
from .progressive_disclosure import OpenSectionsParams
from .prompt import Prompt, PromptTemplate, SectionNode
from .protocols import (
    PromptProtocol,
    PromptTemplateProtocol,
    ProviderAdapterProtocol,
    RenderedPromptProtocol,
)
from .rendering import RenderedPrompt
from .section import Section, SectionVisibility
from .structured_output import (
    OutputParseError,
    StructuredOutputConfig,
    parse_structured_output,
)
from .task_examples import TaskExample, TaskExamplesSection, TaskStep
from .tool import Tool, ToolContext, ToolExample, ToolHandler
from .tool_result import ToolResult

__all__ = [
    "DelegationParams",
    "DelegationPrompt",
    "DelegationSummarySection",
    "HostedTool",
    "LocalPromptOverridesStore",
    "MarkdownSection",
    "OpenSectionsParams",
    "OutputParseError",
    "Prompt",
    "PromptDescriptor",
    "PromptError",
    "PromptLike",
    "PromptOverride",
    "PromptOverridesError",
    "PromptOverridesStore",
    "PromptProtocol",
    "PromptRenderError",
    "PromptTemplate",
    "PromptTemplateProtocol",
    "PromptValidationError",
    "ProviderAdapterProtocol",
    "RenderedPrompt",
    "RenderedPromptProtocol",
    "Section",
    "SectionDescriptor",
    "SectionNode",
    "SectionOverride",
    "SectionPath",
    "SectionVisibility",
    "StructuredOutputConfig",
    "SupportsDataclass",
    "SupportsDataclassOrNone",
    "SupportsToolResult",
    "TaskExample",
    "TaskExamplesSection",
    "TaskStep",
    "Tool",
    "ToolContext",
    "ToolDescriptor",
    "ToolExample",
    "ToolHandler",
    "ToolOverride",
    "ToolRenderableResult",
    "ToolResult",
    "VisibilityExpansionRequired",
    "hash_json",
    "hash_text",
    "parse_structured_output",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
