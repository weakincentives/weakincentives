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

"""Curated public surface for :mod:`weakincentives`."""

from __future__ import annotations

from .adapters import PromptResponse
from .dbc import (
    dbc_active,
    dbc_enabled,
    ensure,
    invariant,
    pure,
    require,
    skip_invariant,
)
from .deadlines import Deadline
from .prompt import (
    MarkdownSection,
    Prompt,
    SupportsDataclass,
    Tool,
    ToolContext,
    ToolHandler,
    ToolResult,
    parse_structured_output,
)
from .runtime import StructuredLogger, configure_logging, get_logger

__all__ = [
    "Deadline",
    "MarkdownSection",
    "Prompt",
    "PromptResponse",
    "StructuredLogger",
    "SupportsDataclass",
    "Tool",
    "ToolContext",
    "ToolHandler",
    "ToolResult",
    "configure_logging",
    "dbc_active",
    "dbc_enabled",
    "ensure",
    "get_logger",
    "invariant",
    "parse_structured_output",
    "pure",
    "require",
    "skip_invariant",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
