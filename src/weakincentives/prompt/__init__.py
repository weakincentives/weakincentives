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

"""Prompt authoring primitives."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import *  # noqa: F403

__all__ = [
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
    "ToolExample",
    "ToolHandler",
    "ToolOverride",
    "ToolRenderableResult",
    "ToolResult",
    "api",
    "hash_json",
    "hash_text",
    "parse_structured_output",
]

api: object | None = None


def __getattr__(name: str) -> object:
    module = globals().get("api")
    if module is None:
        module = import_module(f"{__name__}.api")
        globals()["api"] = module
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
