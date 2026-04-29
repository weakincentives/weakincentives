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

"""Tests for the distilled single-file core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest

from weakincentives.distilled import (
    Append,
    Prompt,
    PromptError,
    Replace,
    Section,
    Session,
    Snapshot,
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
    WinkError,
    execute_tool,
    reducer,
    tool_transaction,
)

# =============================================================================
# Tool / ToolResult
# =============================================================================


class TestToolResult:
    def test_ok_factory(self) -> None:
        result = ToolResult.ok(42, message="done")
        assert result.success is True
        assert result.value == 42
        assert result.message == "done"

    def test_error_factory(self) -> None:
        result = ToolResult.error("boom")
        assert result.success is False
        assert result.value is None
        assert result.message == "boom"


def _noop_handler(params: object, context: ToolContext) -> ToolResult[None]:
    del params, context
    return ToolResult.ok(None)


class TestTool:
    def test_valid_tool_constructs(self) -> None:
        tool = Tool(name="echo", description="Echo input", handler=_noop_handler)
        assert tool.name == "echo"

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ToolError, match="invalid tool name"):
            Tool(name="BAD NAME", description="x", handler=_noop_handler)

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ToolError, match="description must be"):
            Tool(name="t", description="", handler=_noop_handler)

    def test_overlong_description_raises(self) -> None:
        with pytest.raises(ToolError):
            Tool(name="t", description="x" * 201, handler=_noop_handler)


# =============================================================================
# Section / Prompt
# =============================================================================


class TestSection:
    def test_invalid_key_raises(self) -> None:
        with pytest.raises(PromptError, match="invalid section key"):
            Section(title="t", key="UPPER")

    def test_reachable_tools_walks_children(self) -> None:
        leaf_tool = Tool(name="leaf", description="d", handler=_noop_handler)
        root_tool = Tool(name="root", description="d", handler=_noop_handler)
        leaf = Section(title="L", key="leaf", tools=(leaf_tool,))
        root = Section(title="R", key="root", tools=(root_tool,), children=(leaf,))
        names = [tool.name for tool in root.reachable_tools()]
        assert names == ["root", "leaf"]


class TestPromptValidation:
    def test_invalid_ns_raises(self) -> None:
        with pytest.raises(PromptError, match="invalid namespace"):
            Prompt(ns="bad ns", key="k")

    def test_invalid_key_raises(self) -> None:
        with pytest.raises(PromptError, match="invalid key"):
            Prompt(ns="ns", key="bad key")

    def test_duplicate_tool_name_raises(self) -> None:
        tool = Tool(name="dup", description="d", handler=_noop_handler)
        section_a = Section(title="A", key="a", tools=(tool,))
        section_b = Section(title="B", key="b", tools=(tool,))
        with pytest.raises(PromptError, match="duplicate tool name"):
            Prompt(ns="ns", key="k", sections=(section_a, section_b))


class TestPromptRender:
    def test_renders_numbered_headings(self) -> None:
        leaf = Section(title="Leaf", key="leaf", template="leaf body")
        root = Section(
            title="Root",
            key="root",
            template="hello $name",
            children=(leaf,),
        )
        prompt = Prompt(
            ns="demo",
            key="greet",
            sections=(root,),
            params={"root": {"name": "world"}},
        )
        rendered = prompt.render()
        assert "## 1. Root" in rendered.text
        assert "hello world" in rendered.text
        assert "### 1.1. Leaf" in rendered.text
        assert "leaf body" in rendered.text

    def test_section_without_template_renders_heading_only(self) -> None:
        section = Section(title="Header", key="header")
        prompt = Prompt(ns="demo", key="k", sections=(section,))
        text = prompt.render().text
        assert text == "## 1. Header"

    def test_missing_placeholder_raises(self) -> None:
        section = Section(title="T", key="t", template="hi $name")
        prompt = Prompt(ns="demo", key="k", sections=(section,))
        with pytest.raises(PromptError, match="missing placeholder"):
            prompt.render()

    def test_render_collects_tools(self) -> None:
        tool = Tool(name="t", description="d", handler=_noop_handler)
        section = Section(title="S", key="s", tools=(tool,))
        prompt = Prompt(ns="ns", key="k", sections=(section,))
        rendered = prompt.render()
        assert rendered.tools == (tool,)


# =============================================================================
# Sessions, slices, reducers
# =============================================================================


@dataclass(frozen=True)
class AddStep:
    step: str


@dataclass(frozen=True)
class ClearAll:
    pass


@dataclass(frozen=True)
class Plan:
    steps: tuple[str, ...] = ()

    @reducer(on=AddStep)
    def add(self, event: AddStep) -> Replace[Plan]:
        return Replace((replace(self, steps=(*self.steps, event.step)),))

    @reducer(on=ClearAll)
    def clear(self, event: ClearAll) -> Replace[Plan]:
        del event
        return Replace(())


@dataclass(frozen=True)
class LogEntry:
    text: str


@dataclass(frozen=True)
class Log:
    """Demonstrates the Append SliceOp via a free-function reducer."""


def _log_reducer(
    current: tuple[Any, ...], event: AddStep
) -> Replace[LogEntry] | Append[LogEntry]:
    del current
    return Append((LogEntry(text=event.step),))


class TestReducerDecorator:
    def test_marks_method(self) -> None:
        marker = getattr(Plan.add, "__wink_reducer_event__", None)
        assert marker is AddStep


class TestSliceAccessor:
    def test_seed_and_query(self) -> None:
        session = Session()
        plan = Plan(steps=("a",))
        session[Plan].seed(plan)
        assert session[Plan].latest() == plan
        assert session[Plan].all() == (plan,)

    def test_latest_empty(self) -> None:
        session = Session()
        assert session[Plan].latest() is None

    def test_where_filter(self) -> None:
        session = Session()
        session[Plan].seed(Plan(steps=("a",)))
        session[Plan].append(Plan(steps=("a", "b")))
        long_plans = session[Plan].where(lambda p: len(p.steps) > 1)
        assert len(long_plans) == 1
        assert long_plans[0].steps == ("a", "b")

    def test_clear_drops_slice(self) -> None:
        session = Session()
        session[Plan].seed(Plan(steps=("a",)))
        session[Plan].clear()
        assert session[Plan].all() == ()


class TestSessionInstall:
    def test_rejects_non_dataclass(self) -> None:
        class NotADataclass:
            pass

        session = Session()
        with pytest.raises(WinkError, match="frozen dataclass"):
            session.install(NotADataclass)

    def test_rejects_unfrozen_dataclass(self) -> None:
        @dataclass
        class Mutable:
            x: int = 0

        session = Session()
        with pytest.raises(WinkError, match="frozen dataclass"):
            session.install(Mutable)

    def test_rejects_class_without_reducer_methods(self) -> None:
        @dataclass(frozen=True)
        class Empty:
            pass

        session = Session()
        with pytest.raises(WinkError, match="declares no @reducer methods"):
            session.install(Empty)

    def test_rejects_duplicate_reducers(self) -> None:
        @dataclass(frozen=True)
        class Dup:
            @reducer(on=AddStep)
            def first(self, event: AddStep) -> Replace[Dup]:
                del event
                return Replace((self,))

            @reducer(on=AddStep)
            def second(self, event: AddStep) -> Replace[Dup]:
                del event
                return Replace((self,))

        session = Session()
        with pytest.raises(WinkError, match="duplicate reducers"):
            session.install(Dup)


class TestSessionDispatch:
    def test_replace_op_via_method_reducer(self) -> None:
        session = Session()
        session.install(Plan, initial=Plan)
        session.dispatch(AddStep(step="a"))
        session.dispatch(AddStep(step="b"))
        latest = session[Plan].latest()
        assert latest is not None
        assert latest.steps == ("a", "b")

    def test_initial_factory_used_when_empty(self) -> None:
        session = Session()
        session.install(Plan, initial=lambda: Plan(steps=("seed",)))
        session.dispatch(AddStep(step="a"))
        latest = session[Plan].latest()
        assert latest is not None
        assert latest.steps == ("seed", "a")

    def test_no_initial_raises_on_empty(self) -> None:
        session = Session()
        session.install(Plan)
        with pytest.raises(WinkError, match="no state for Plan"):
            session.dispatch(AddStep(step="a"))

    def test_append_op_via_free_reducer(self) -> None:
        session = Session()
        session.register(AddStep, _log_reducer, slice_type=LogEntry)
        session.dispatch(AddStep(step="hello"))
        session.dispatch(AddStep(step="world"))
        items = session[LogEntry].all()
        assert [entry.text for entry in items] == ["hello", "world"]

    def test_event_with_no_reducer_is_noop(self) -> None:
        session = Session()
        session.dispatch(AddStep(step="ignored"))
        assert session[Plan].all() == ()

    def test_clear_via_replace_empty(self) -> None:
        session = Session()
        session.install(Plan, initial=Plan)
        session.dispatch(AddStep(step="a"))
        session.dispatch(ClearAll())
        assert session[Plan].latest() is None


class TestSessionSnapshots:
    def test_round_trip(self) -> None:
        session = Session()
        session[Plan].seed(Plan(steps=("a",)))
        snapshot = session.snapshot()
        session[Plan].seed(Plan(steps=("b",)))
        session.restore(snapshot)
        latest = session[Plan].latest()
        assert latest is not None
        assert latest.steps == ("a",)


# =============================================================================
# Transactions and execute_tool
# =============================================================================


class TestToolTransaction:
    def test_no_rollback_on_success(self) -> None:
        session = Session()
        session[Plan].seed(Plan(steps=("a",)))
        with tool_transaction(session):
            session[Plan].seed(Plan(steps=("a", "b")))
        latest = session[Plan].latest()
        assert latest is not None
        assert latest.steps == ("a", "b")

    def test_rollback_on_exception(self) -> None:
        session = Session()
        session[Plan].seed(Plan(steps=("a",)))
        with pytest.raises(RuntimeError):
            with tool_transaction(session):
                session[Plan].seed(Plan(steps=("a", "b")))
                raise RuntimeError("boom")
        latest = session[Plan].latest()
        assert latest is not None
        assert latest.steps == ("a",)


def _success_handler(params: object, context: ToolContext) -> ToolResult[str]:
    del params
    context.session[Plan].append(Plan(steps=("touched",)))
    return ToolResult.ok("ok")


def _failing_handler(params: object, context: ToolContext) -> ToolResult[str]:
    del params
    context.session[Plan].append(Plan(steps=("dirty",)))
    return ToolResult.error("nope")


def _raising_handler(params: object, context: ToolContext) -> ToolResult[str]:
    del params
    context.session[Plan].append(Plan(steps=("dirty",)))
    raise RuntimeError("kaboom")


def _wink_handler(params: object, context: ToolContext) -> ToolResult[str]:
    del params, context
    raise PromptError("rerender please")


def _make_prompt(*tools: Tool) -> Prompt:
    section = Section(title="Tools", key="tools", tools=tools)
    return Prompt(ns="ns", key="k", sections=(section,))


class TestExecuteTool:
    def test_unknown_tool_raises(self) -> None:
        prompt = _make_prompt()
        with pytest.raises(ToolError, match="no tool named"):
            execute_tool(prompt, Session(), "missing", None)

    def test_success_keeps_state(self) -> None:
        tool = Tool(name="ok", description="d", handler=_success_handler)
        prompt = _make_prompt(tool)
        session = Session()
        session[Plan].seed(Plan(steps=("seed",)))
        result = execute_tool(prompt, session, "ok", None)
        assert result.success is True
        assert session[Plan].latest() == Plan(steps=("touched",))

    def test_error_result_rolls_back(self) -> None:
        tool = Tool(name="err", description="d", handler=_failing_handler)
        prompt = _make_prompt(tool)
        session = Session()
        session[Plan].seed(Plan(steps=("seed",)))
        result = execute_tool(prompt, session, "err", None)
        assert result.success is False
        assert session[Plan].latest() == Plan(steps=("seed",))

    def test_generic_exception_is_wrapped_and_rolled_back(self) -> None:
        tool = Tool(name="boom", description="d", handler=_raising_handler)
        prompt = _make_prompt(tool)
        session = Session()
        session[Plan].seed(Plan(steps=("seed",)))
        result = execute_tool(prompt, session, "boom", None)
        assert result.success is False
        assert "RuntimeError" in result.message
        assert session[Plan].latest() == Plan(steps=("seed",))

    def test_wink_error_propagates_after_rollback(self) -> None:
        tool = Tool(name="wink", description="d", handler=_wink_handler)
        prompt = _make_prompt(tool)
        session = Session()
        session[Plan].seed(Plan(steps=("seed",)))
        with pytest.raises(PromptError):
            execute_tool(prompt, session, "wink", None)
        assert session[Plan].latest() == Plan(steps=("seed",))


# =============================================================================
# Smoke checks for value types
# =============================================================================


class TestValueTypes:
    def test_replace_and_append_are_distinct(self) -> None:
        replace_op: Replace[int] = Replace((1, 2))
        append_op: Append[int] = Append((3,))
        assert isinstance(replace_op, Replace)
        assert isinstance(append_op, Append)
        assert replace_op.items == (1, 2)
        assert append_op.items == (3,)

    def test_snapshot_holds_slice_mapping(self) -> None:
        session = Session()
        session[Plan].seed(Plan(steps=("a",)))
        snapshot: Snapshot = session.snapshot()
        assert Plan in snapshot.slices
