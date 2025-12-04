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

"""Regression tests for the todo tool suite."""

from __future__ import annotations

from typing import cast

import pytest

from tests.tools.helpers import build_tool_context, find_tool, invoke_tool
from weakincentives.prompt.errors import PromptRenderError
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session, select_latest
from weakincentives.tools import TodoList, TodoReadParams, TodoToolsSection
from weakincentives.tools.errors import ToolValidationError


def test_todo_write_appends_normalized_snapshot() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = TodoToolsSection(session=session)
    write_tool = find_tool(section, "todo_write")

    result = invoke_tool(
        bus,
        write_tool,
        TodoList(items=("  trace podman logs  ", "triage lint failures")),
        session=session,
    )

    assert result.value == TodoList(items=("trace podman logs", "triage lint failures"))
    latest = select_latest(session, TodoList)
    assert latest == TodoList(items=("trace podman logs", "triage lint failures"))


def test_todo_read_returns_latest_snapshot() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = TodoToolsSection(session=session)
    write_tool = find_tool(section, "todo_write")
    read_tool = find_tool(section, "todo_read")

    invoke_tool(bus, write_tool, TodoList(items=("triage",)), session=session)
    read_result = invoke_tool(bus, read_tool, TodoReadParams(), session=session)

    assert read_result.value == TodoList(items=("triage",))


def test_todo_read_returns_empty_when_missing() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = TodoToolsSection(session=session)
    read_tool = find_tool(section, "todo_read")

    result = invoke_tool(bus, read_tool, TodoReadParams(), session=session)

    assert result.value == TodoList()
    latest = select_latest(session, TodoList)
    assert latest == TodoList()


def test_todo_write_rejects_invalid_items() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = TodoToolsSection(session=session)
    write_tool = find_tool(section, "todo_write")
    handler = write_tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError, match="cannot be empty"):
        handler(TodoList(items=("   ",)), context=build_tool_context(bus, session))
    with pytest.raises(ToolValidationError, match="must be a string"):
        handler(
            TodoList(items=cast(tuple[str, ...], ("ship", 1))),
            context=build_tool_context(bus, session),
        )


def test_todo_section_clone_requires_session() -> None:
    session = Session()
    section = TodoToolsSection(session=session)

    with pytest.raises(TypeError, match="session is required"):
        section.clone(session=None)


def test_todo_section_clone_uses_provided_session() -> None:
    session = Session()
    section = TodoToolsSection(session=session)
    other_session = Session()

    cloned = section.clone(session=other_session)

    assert isinstance(cloned, TodoToolsSection)
    assert cloned.session is other_session
    assert cloned.accepts_overrides is section.accepts_overrides


def test_todo_render_rejects_params() -> None:
    session = Session()
    section = TodoToolsSection(session=session)

    with pytest.raises(PromptRenderError):
        section.render(TodoList(items=("noop",)), depth=0, number="1.")


def test_todo_original_body_template_matches_prompt_copy() -> None:
    session = Session()
    section = TodoToolsSection(session=session)

    assert "todo_read" in section.original_body_template()


def test_todo_list_normalizes_items() -> None:
    todo_list = TodoList(items=(" first", "second "))

    assert todo_list.items == (" first", "second ")


def test_todo_tools_reject_mismatched_context_session() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = TodoToolsSection(session=session)
    write_tool = find_tool(section, "todo_write")
    handler = write_tool.handler
    assert handler is not None

    mismatched_session = Session(bus=bus)
    with pytest.raises(RuntimeError, match="session does not match"):
        handler(
            TodoList(items=("ship",)),
            context=build_tool_context(bus, mismatched_session),
        )
