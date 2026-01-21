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

"""Session-stored tool schema records captured at prompt render time.

This module provides the RenderedTools dataclass for storing the complete
list of tools available at render time, including their JSON Schema definitions.
One entry is appended per prompt render as a LOG slice.
"""

from __future__ import annotations

from dataclasses import field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from ...dataclasses import FrozenDataclass

if TYPE_CHECKING:
    from collections.abc import Mapping


@FrozenDataclass()
class ToolSchema:
    """Schema definition for a single tool at render time.

    Captures the tool's name, description, and JSON Schema for parameters.
    """

    name: str = field(metadata={"description": "Tool name (1-64 lowercase ASCII)."})
    description: str = field(
        metadata={"description": "Tool description (1-200 ASCII chars)."}
    )
    parameters: Mapping[str, Any] = field(
        metadata={"description": "JSON Schema for tool parameters."}
    )


@FrozenDataclass()
class RenderedTools:
    """Record of tools available at prompt render time.

    This dataclass is stored as a LOG slice in the session with one entry
    appended per prompt render. It captures the complete tool schema information
    for audit, debugging, and analysis purposes.

    The render_event_id correlates this record with the corresponding
    PromptRendered event.

    Usage::

        from weakincentives.runtime.session import RenderedTools

        # Query rendered tools from session
        session[RenderedTools].all()  # All render records
        session[RenderedTools].latest()  # Most recent render

        # Find tools for a specific render
        session[RenderedTools].where(
            lambda r: r.render_event_id == some_event_id
        )

    """

    prompt_ns: str = field(metadata={"description": "Prompt namespace."})
    prompt_key: str = field(metadata={"description": "Prompt key within namespace."})
    tools: tuple[ToolSchema, ...] = field(
        metadata={"description": "Tool schemas available at render time."}
    )
    render_event_id: UUID = field(
        metadata={"description": "ID of the corresponding PromptRendered event."}
    )
    session_id: UUID | None = field(
        default=None, metadata={"description": "Session ID if available."}
    )
    created_at: datetime = field(
        default_factory=lambda: datetime.now(__import__("datetime").UTC),
        metadata={"description": "Timestamp when tools were rendered."},
    )
    event_id: UUID = field(
        default_factory=uuid4,
        metadata={"description": "Unique identifier for this record."},
    )

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return the names of all tools in this render."""
        return tuple(tool.name for tool in self.tools)

    @property
    def tool_count(self) -> int:
        """Return the number of tools available at render time."""
        return len(self.tools)

    def get_tool(self, name: str) -> ToolSchema | None:
        """Return the schema for a tool by name, or None if not found."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


__all__ = [
    "RenderedTools",
    "ToolSchema",
]
