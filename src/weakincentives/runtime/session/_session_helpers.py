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

# pyright: reportImportCycles=false

"""Session utility functions and constants.

Free functions and type aliases extracted from session.py for
session ID validation, tag normalization, and tree traversal.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast
from uuid import UUID

from ...types.dataclass import SupportsDataclass
from ..events import PromptExecuted, PromptRendered, ToolInvoked
from .rendered_tools import RenderedTools

if TYPE_CHECKING:
    from .session import Session

__all__ = [
    "PROMPT_EXECUTED_TYPE",
    "PROMPT_RENDERED_TYPE",
    "RENDERED_TOOLS_TYPE",
    "SESSION_ID_BYTE_LENGTH",
    "TOOL_INVOKED_TYPE",
    "created_at_has_tz",
    "created_at_is_utc",
    "iter_sessions_bottom_up",
    "normalize_tags",
    "session_id_is_well_formed",
]

type DataEvent = PromptExecuted | PromptRendered | ToolInvoked

SESSION_ID_BYTE_LENGTH: Final[int] = 16

# Type casts for telemetry event types used in slice policy initialization
PROMPT_RENDERED_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], PromptRendered
)
TOOL_INVOKED_TYPE: type[SupportsDataclass] = cast(type[SupportsDataclass], ToolInvoked)
PROMPT_EXECUTED_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], PromptExecuted
)
RENDERED_TOOLS_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], RenderedTools
)


def session_id_is_well_formed(session: Session) -> bool:
    return len(session.session_id.bytes) == SESSION_ID_BYTE_LENGTH


def created_at_has_tz(session: Session) -> bool:
    return session.created_at.tzinfo is not None


def created_at_is_utc(session: Session) -> bool:
    return session.created_at.tzinfo == UTC


def normalize_tags(
    tags: Mapping[object, object] | None,
    *,
    session_id: UUID,
    parent: Session | None,
) -> Mapping[str, str]:
    normalized: dict[str, str] = {}

    if tags is not None:
        for key, value in tags.items():
            if not isinstance(key, str) or not isinstance(value, str):
                msg = "Session tags must be string key/value pairs."
                raise TypeError(msg)
            normalized[key] = value

    normalized["session_id"] = str(session_id)
    if parent is not None:
        _ = normalized.setdefault("parent_session_id", str(parent.session_id))

    return MappingProxyType(normalized)


def iter_sessions_bottom_up(root: Session) -> Iterator[Session]:
    """Yield sessions from the leaves up to the provided root session."""

    visited: set[Session] = set()

    def _walk(node: Session) -> Iterator[Session]:
        if node in visited:
            return
        visited.add(node)
        children = node.children
        if children:
            for child in children:
                yield from _walk(child)
        yield node

    yield from _walk(root)
