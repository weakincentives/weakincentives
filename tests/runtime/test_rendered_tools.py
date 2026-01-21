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

"""Tests for session-stored rendered tools."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import (
    RenderedTools,
    Session,
    SlicePolicy,
    ToolSchema,
)

if TYPE_CHECKING:
    from tests.conftest import SessionFactory

pytestmark = pytest.mark.core


def make_tool_schema(name: str, description: str = "A test tool.") -> ToolSchema:
    """Create a test ToolSchema."""
    return ToolSchema(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "arg": {"type": "string"},
            },
            "additionalProperties": False,
        },
    )


def make_rendered_tools(
    *,
    prompt_ns: str = "test",
    prompt_key: str = "prompt",
    tools: tuple[ToolSchema, ...] = (),
    render_event_id: None = None,
    session_id: None = None,
) -> RenderedTools:
    """Create a test RenderedTools event."""
    return RenderedTools(
        prompt_ns=prompt_ns,
        prompt_key=prompt_key,
        tools=tools,
        render_event_id=render_event_id if render_event_id else uuid4(),
        session_id=session_id,
        created_at=datetime.now(UTC),
    )


class TestToolSchema:
    """Tests for ToolSchema dataclass."""

    def test_tool_schema_creation(self) -> None:
        """ToolSchema stores name, description, and parameters."""
        params = {"type": "object", "properties": {"x": {"type": "integer"}}}
        schema = ToolSchema(
            name="my_tool",
            description="Does something useful.",
            parameters=params,
        )

        assert schema.name == "my_tool"
        assert schema.description == "Does something useful."
        assert schema.parameters == params

    def test_tool_schema_is_frozen(self) -> None:
        """ToolSchema is immutable."""
        schema = make_tool_schema("test")
        with pytest.raises(AttributeError):
            schema.name = "other"  # type: ignore[misc]


class TestRenderedTools:
    """Tests for RenderedTools dataclass."""

    def test_rendered_tools_creation(self) -> None:
        """RenderedTools stores prompt info and tool schemas."""
        render_id = uuid4()
        session_id = uuid4()
        tool = make_tool_schema("read_file")

        rendered = RenderedTools(
            prompt_ns="myns",
            prompt_key="mykey",
            tools=(tool,),
            render_event_id=render_id,
            session_id=session_id,
            created_at=datetime.now(UTC),
        )

        assert rendered.prompt_ns == "myns"
        assert rendered.prompt_key == "mykey"
        assert rendered.tools == (tool,)
        assert rendered.render_event_id == render_id
        assert rendered.session_id == session_id

    def test_tool_names_property(self) -> None:
        """tool_names returns tuple of tool names."""
        rendered = make_rendered_tools(
            tools=(
                make_tool_schema("read"),
                make_tool_schema("write"),
                make_tool_schema("delete"),
            )
        )

        assert rendered.tool_names == ("read", "write", "delete")

    def test_tool_count_property(self) -> None:
        """tool_count returns number of tools."""
        rendered = make_rendered_tools(
            tools=(make_tool_schema("a"), make_tool_schema("b"))
        )

        assert rendered.tool_count == 2

    def test_tool_count_empty(self) -> None:
        """tool_count returns 0 for empty tools."""
        rendered = make_rendered_tools(tools=())

        assert rendered.tool_count == 0

    def test_get_tool_returns_schema(self) -> None:
        """get_tool returns schema for existing tool."""
        tool = make_tool_schema("target_tool", "Find this one.")
        rendered = make_rendered_tools(
            tools=(
                make_tool_schema("other"),
                tool,
            )
        )

        result = rendered.get_tool("target_tool")

        assert result is tool
        assert result.description == "Find this one."

    def test_get_tool_returns_none_for_missing(self) -> None:
        """get_tool returns None for non-existent tool."""
        rendered = make_rendered_tools(tools=(make_tool_schema("existing"),))

        assert rendered.get_tool("missing") is None

    def test_event_id_is_unique(self) -> None:
        """Each RenderedTools gets a unique event_id."""
        first = make_rendered_tools()
        second = make_rendered_tools()

        assert first.event_id != second.event_id


class TestRenderedToolsSession:
    """Tests for RenderedTools integration with Session."""

    def test_rendered_tools_is_log_slice(self) -> None:
        """RenderedTools is registered as a LOG slice."""
        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)

        # Check the policy is LOG
        assert session._slice_policies.get(RenderedTools) == SlicePolicy.LOG

    def test_rendered_tools_appends_on_dispatch(
        self, session_factory: SessionFactory
    ) -> None:
        """Dispatching RenderedTools appends to session slice."""
        session, dispatcher = session_factory()

        first = make_rendered_tools(prompt_key="first")
        second = make_rendered_tools(prompt_key="second")

        dispatcher.dispatch(first)
        dispatcher.dispatch(second)

        all_tools = session[RenderedTools].all()
        assert len(all_tools) == 2
        assert all_tools[0].prompt_key == "first"
        assert all_tools[1].prompt_key == "second"

    def test_rendered_tools_latest(self, session_factory: SessionFactory) -> None:
        """session[RenderedTools].latest() returns most recent entry."""
        session, dispatcher = session_factory()

        dispatcher.dispatch(make_rendered_tools(prompt_key="old"))
        dispatcher.dispatch(make_rendered_tools(prompt_key="new"))

        latest = session[RenderedTools].latest()
        assert latest is not None
        assert latest.prompt_key == "new"

    def test_rendered_tools_where(self, session_factory: SessionFactory) -> None:
        """session[RenderedTools].where() filters by predicate."""
        session, dispatcher = session_factory()

        dispatcher.dispatch(
            make_rendered_tools(
                prompt_ns="ns1",
                tools=(make_tool_schema("a"),),
            )
        )
        dispatcher.dispatch(
            make_rendered_tools(
                prompt_ns="ns2",
                tools=(make_tool_schema("b"), make_tool_schema("c")),
            )
        )
        dispatcher.dispatch(
            make_rendered_tools(
                prompt_ns="ns1",
                tools=(make_tool_schema("d"),),
            )
        )

        ns1_tools = session[RenderedTools].where(lambda r: r.prompt_ns == "ns1")
        assert len(ns1_tools) == 2

        multi_tool = session[RenderedTools].where(lambda r: r.tool_count > 1)
        assert len(multi_tool) == 1
        assert multi_tool[0].prompt_ns == "ns2"

    def test_rendered_tools_preserved_on_restore_with_preserve_logs(
        self, session_factory: SessionFactory
    ) -> None:
        """RenderedTools are preserved on restore when preserve_logs=True."""
        session, dispatcher = session_factory()

        dispatcher.dispatch(make_rendered_tools(prompt_key="before"))

        snapshot = session.snapshot(include_all=True)

        dispatcher.dispatch(make_rendered_tools(prompt_key="after"))

        # Restore with preserve_logs=True (default)
        session.restore(snapshot, preserve_logs=True)

        # Both entries should still be present (LOG slices preserved)
        all_tools = session[RenderedTools].all()
        assert len(all_tools) == 2

    def test_session_subscribes_to_rendered_tools(self) -> None:
        """Session subscribes to RenderedTools events."""
        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)

        event = make_rendered_tools()
        dispatcher.dispatch(event)

        # Event should be stored in session
        assert session[RenderedTools].latest() == event


class TestRenderedToolsCorrelation:
    """Tests for correlating RenderedTools with PromptRendered."""

    def test_render_event_id_correlation(self, session_factory: SessionFactory) -> None:
        """render_event_id links RenderedTools to PromptRendered."""
        from weakincentives.runtime.events import PromptRendered

        session, dispatcher = session_factory()

        # Simulate what inner_loop does: same event_id for both events
        render_event_id = uuid4()
        session_id = uuid4()
        created_at = datetime.now(UTC)

        prompt_rendered = PromptRendered(
            prompt_ns="test",
            prompt_key="prompt",
            prompt_name="Test Prompt",
            adapter="test",
            session_id=session_id,
            render_inputs=(),
            rendered_prompt="Hello world",
            created_at=created_at,
            event_id=render_event_id,
        )

        rendered_tools = RenderedTools(
            prompt_ns="test",
            prompt_key="prompt",
            tools=(make_tool_schema("tool1"),),
            render_event_id=render_event_id,
            session_id=session_id,
            created_at=created_at,
        )

        dispatcher.dispatch(prompt_rendered)
        dispatcher.dispatch(rendered_tools)

        # Find tools for a specific render
        prompt_event = session[PromptRendered].latest()
        assert prompt_event is not None

        matching_tools = session[RenderedTools].where(
            lambda r: r.render_event_id == prompt_event.event_id
        )
        assert len(matching_tools) == 1
        assert matching_tools[0].tool_names == ("tool1",)
