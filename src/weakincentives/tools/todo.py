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

"""Session-scoped todo list tools."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, cast, override

from ..prompt import SupportsDataclass, SupportsToolResult
from ..prompt.errors import PromptRenderError
from ..prompt.markdown import MarkdownSection
from ..prompt.tool import Tool, ToolContext, ToolExample
from ..prompt.tool_result import ToolResult
from ..runtime.session import Session, append, select_latest
from ._context import ensure_context_uses_session
from .errors import ToolValidationError


@dataclass(slots=True, frozen=True)
class TodoList(SupportsDataclass):
    """Ordered todo items tracked within the active session."""

    items: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized_items = tuple(self.items)
        object.__setattr__(self, "items", normalized_items)


@dataclass(slots=True, frozen=True)
class TodoReadParams(SupportsDataclass):
    """Placeholder params for `todo_read` (no fields required)."""

    pass


_TODO_TEMPLATE = textwrap.dedent(
    """
    Use `todo_read` before writing so you do not overwrite someone else's list.
    - `todo_read` returns the latest todo list (empty when nothing is stored yet).
    - `todo_write` replaces the list with the ordered items you supply. Items must
      be non-empty strings.
    """
).strip()


class TodoToolsSection(MarkdownSection[SupportsDataclass]):
    """Prompt section exposing the todo tools."""

    _params_type: ClassVar[type[SupportsDataclass] | None] = TodoReadParams

    def __init__(
        self,
        *,
        session: Session,
        accepts_overrides: bool = False,
    ) -> None:
        self._session = session
        tools = _build_tools(section=self, accepts_overrides=accepts_overrides)
        super().__init__(
            title="Todo Tools",
            key="todo.tools",
            template=_TODO_TEMPLATE,
            default_params=TodoReadParams(),
            tools=tools,
            accepts_overrides=accepts_overrides,
        )
        self._initialize_session(session)

    @property
    def session(self) -> Session:
        return self._session

    @staticmethod
    def _initialize_session(session: Session) -> None:
        session.register_reducer(TodoList, append)

    @override
    def clone(self, **kwargs: object) -> TodoToolsSection:
        session = kwargs.get("session")
        if not isinstance(session, Session):
            msg = "session is required to clone TodoToolsSection."
            raise TypeError(msg)
        return TodoToolsSection(
            session=session, accepts_overrides=self.accepts_overrides
        )

    @override
    def render(self, params: SupportsDataclass | None, depth: int, number: str) -> str:
        if params is not None and not isinstance(params, TodoReadParams):
            raise PromptRenderError(
                "Todo tools section does not accept parameters.",
                dataclass_type=TodoReadParams,
            )
        return self.render_with_template(_TODO_TEMPLATE, None, depth, number)

    @override
    def original_body_template(self) -> str:
        return _TODO_TEMPLATE


def _build_tools(
    *,
    section: TodoToolsSection,
    accepts_overrides: bool,
) -> tuple[Tool[SupportsDataclass, SupportsToolResult], ...]:
    suite = _TodoToolSuite(section=section)
    return cast(
        tuple[Tool[SupportsDataclass, SupportsToolResult], ...],
        (
            Tool[TodoList, TodoList](
                name="todo_write",
                description="Replace the todo list with the items you provide.",
                handler=suite.write,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample[
                        TodoList,
                        TodoList,
                    ](
                        description="Record an ordered list of follow-ups.",
                        input=TodoList(
                            items=("trace podman logs", "triage lint failures")
                        ),
                        output=TodoList(
                            items=("trace podman logs", "triage lint failures")
                        ),
                    ),
                ),
            ),
            Tool[TodoReadParams, TodoList](
                name="todo_read",
                description="Return the latest stored todo list or an empty list when missing.",
                handler=suite.read,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample[
                        TodoReadParams,
                        TodoList,
                    ](
                        description="Fetch the current todo snapshot.",
                        input=TodoReadParams(),
                        output=TodoList(),
                    ),
                ),
            ),
        ),
    )


class _TodoToolSuite:
    """Handlers for the todo tool suite bound to a section instance."""

    def __init__(self, *, section: TodoToolsSection) -> None:
        super().__init__()
        self._section = section

    def write(self, params: TodoList, *, context: ToolContext) -> ToolResult[TodoList]:
        ensure_context_uses_session(context=context, session=self._section.session)
        normalized_items = _normalize_items(params.items)
        snapshot = TodoList(items=normalized_items)
        message = f"Stored {len(snapshot.items)} todo item(s)."
        return ToolResult(message=message, value=snapshot)

    def read(
        self, params: TodoReadParams, *, context: ToolContext
    ) -> ToolResult[TodoList]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del params
        latest = select_latest(self._section.session, TodoList)
        if latest is not None:
            return ToolResult(message="Returning latest todo list.", value=latest)
        return ToolResult(
            message="No todo list found; returning an empty list.", value=TodoList()
        )


def _normalize_items(items: Sequence[object]) -> tuple[str, ...]:
    normalized: list[str] = []
    for index, raw in enumerate(items):
        if not isinstance(raw, str):
            message = f"Todo item {index + 1} must be a string."
            raise ToolValidationError(message)
        candidate = raw.strip()
        if not candidate:
            message = "Todo items cannot be empty or whitespace."
            raise ToolValidationError(message)
        normalized.append(candidate)
    return tuple(normalized)


__all__ = ["TodoList", "TodoReadParams", "TodoToolsSection"]
