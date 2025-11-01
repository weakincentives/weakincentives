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

"""Public package surface for weakincentives."""

from __future__ import annotations

from .prompt import (
    MarkdownSection,
    OutputParseError,
    Prompt,
    PromptError,
    PromptRenderError,
    PromptValidationError,
    RenderedPrompt,
    Section,
    SectionNode,
    SectionPath,
    SupportsDataclass,
    Tool,
    ToolResult,
    parse_structured_output,
)
from .session import (
    DataEvent,
    PromptData,
    Session,
    ToolData,
    TypedReducer,
    append,
    replace_latest,
    select_all,
    select_latest,
    select_where,
    upsert_by,
)

parse_structured = parse_structured_output

__all__ = [
    "Prompt",
    "RenderedPrompt",
    "Section",
    "SectionNode",
    "SectionPath",
    "MarkdownSection",
    "Tool",
    "ToolResult",
    "PromptError",
    "PromptRenderError",
    "PromptValidationError",
    "OutputParseError",
    "SupportsDataclass",
    "parse_structured_output",
    "parse_structured",
    "Session",
    "DataEvent",
    "ToolData",
    "PromptData",
    "TypedReducer",
    "append",
    "replace_latest",
    "upsert_by",
    "select_all",
    "select_latest",
    "select_where",
]


def __dir__() -> list[str]:  # pragma: no cover - convenience for REPL
    return sorted({*globals().keys(), *__all__})
