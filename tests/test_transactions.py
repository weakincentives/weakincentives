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

"""Tests for transactional tool execution and the pending-tool tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from weakincentives.core.errors import (
    PolicyDeniedError,
    PromptError,
    ToolError,
    TransactionError,
)
from weakincentives.core.prompt import MarkdownSection, Prompt
from weakincentives.core.session import Session, ToolInvoked
from weakincentives.core.tool import Tool, ToolContext, ToolResult
from weakincentives.core.transactions import (
    PendingToolTracker,
    execute_tool,
    tool_transaction,
)


@dataclass(frozen=True)
class Plan:
    steps: tuple[str, ...] = ()


def _success(params: object, context: ToolContext) -> ToolResult[str]:
    del params
    context.session[Plan].append(Plan(steps=("touched",)))
    return ToolResult.ok("ok")


def _failing(params: object, context: ToolContext) -> ToolResult[str]:
    del params
    context.session[Plan].append(Plan(steps=("dirty",)))
    return ToolResult.error("nope")


def _raising(params: object, context: ToolContext) -> ToolResult[str]:
    del params
    context.session[Plan].append(Plan(steps=("dirty",)))
    raise RuntimeError("kaboom")


def _wink_raising(params: object, context: ToolContext) -> ToolResult[str]:
    del params, context
    raise PromptError("rerender please")


def _make_prompt(*tools: Tool[Any, Any]) -> Prompt:
    section = MarkdownSection[None](title="Tools", key="tools", tools=tools)
    return Prompt(ns="ns", key="k", sections=(section,))


class _RecordingListener:
    def __init__(self) -> None:
        self.events: list[object] = []

    def on_event(self, event: object) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# tool_transaction
# ---------------------------------------------------------------------------


def test_transaction_no_rollback_on_success() -> None:
    session = Session()
    session[Plan].seed(Plan(steps=("a",)))
    with tool_transaction(session):
        session[Plan].seed(Plan(steps=("a", "b")))
    latest = session[Plan].latest()
    assert latest is not None
    assert latest.steps == ("a", "b")


def test_transaction_rollback_on_exception() -> None:
    session = Session()
    session[Plan].seed(Plan(steps=("a",)))
    with pytest.raises(RuntimeError):
        with tool_transaction(session):
            session[Plan].seed(Plan(steps=("dirty",)))
            raise RuntimeError("boom")
    latest = session[Plan].latest()
    assert latest is not None
    assert latest.steps == ("a",)


# ---------------------------------------------------------------------------
# execute_tool
# ---------------------------------------------------------------------------


def test_execute_tool_unknown_raises() -> None:
    prompt = _make_prompt()
    with pytest.raises(ToolError, match="no tool named"):
        execute_tool(prompt, Session(), "missing", None)


def test_execute_tool_success_keeps_state() -> None:
    tool = Tool(name="ok", description="d", handler=_success)
    prompt = _make_prompt(tool)
    session = Session()
    listener = _RecordingListener()
    session.subscribe(listener)
    session[Plan].seed(Plan(steps=("seed",)))
    result = execute_tool(prompt, session, "ok", None)
    assert result.success is True
    assert session[Plan].latest() == Plan(steps=("touched",))
    invoked = [e for e in listener.events if isinstance(e, ToolInvoked)]
    assert invoked == [ToolInvoked(name="ok", success=True, message="")]


def test_execute_tool_error_rolls_back() -> None:
    tool = Tool(name="err", description="d", handler=_failing)
    prompt = _make_prompt(tool)
    session = Session()
    session[Plan].seed(Plan(steps=("seed",)))
    result = execute_tool(prompt, session, "err", None)
    assert result.success is False
    assert session[Plan].latest() == Plan(steps=("seed",))


def test_execute_tool_wraps_generic_exception() -> None:
    tool = Tool(name="boom", description="d", handler=_raising)
    prompt = _make_prompt(tool)
    session = Session()
    session[Plan].seed(Plan(steps=("seed",)))
    result = execute_tool(prompt, session, "boom", None)
    assert result.success is False
    assert "RuntimeError" in result.message
    assert session[Plan].latest() == Plan(steps=("seed",))


def test_execute_tool_wink_error_propagates_after_rollback() -> None:
    tool = Tool(name="wink", description="d", handler=_wink_raising)
    prompt = _make_prompt(tool)
    session = Session()
    session[Plan].seed(Plan(steps=("seed",)))
    with pytest.raises(PromptError):
        execute_tool(prompt, session, "wink", None)
    assert session[Plan].latest() == Plan(steps=("seed",))


def test_execute_tool_runs_policies() -> None:
    calls: list[str] = []

    class TracingPolicy:
        def before_invoke(self, tool: object, context: object) -> None:
            del tool, context
            calls.append("before")

        def on_result(self, tool: object, result: object) -> None:
            del tool, result
            calls.append("after")

    tool = Tool(name="ok", description="d", handler=_success)
    prompt = _make_prompt(tool)
    session = Session()
    execute_tool(prompt, session, "ok", None, policies=(TracingPolicy(),))
    assert calls == ["before", "after"]


def test_execute_tool_policy_denies_call() -> None:
    class Block:
        def before_invoke(self, tool: object, context: object) -> None:
            del tool, context
            raise PolicyDeniedError("nope")

        def on_result(self, tool: object, result: object) -> None:
            del tool, result

    tool = Tool(name="ok", description="d", handler=_success)
    prompt = _make_prompt(tool)
    session = Session()
    session[Plan].seed(Plan(steps=("seed",)))
    with pytest.raises(PolicyDeniedError):
        execute_tool(prompt, session, "ok", None, policies=(Block(),))
    assert session[Plan].latest() == Plan(steps=("seed",))


# ---------------------------------------------------------------------------
# Snapshotable rollback
# ---------------------------------------------------------------------------


class _Counter:
    def __init__(self) -> None:
        self.value = 0
        self._snapshots: list[int] = []

    def snapshot(self) -> int:
        return self.value

    def restore(self, state: int) -> None:
        self.value = state


def test_execute_tool_rolls_back_snapshotables() -> None:
    counter = _Counter()

    def handler(params: object, context: ToolContext) -> ToolResult[None]:
        del params, context
        counter.value = 99
        return ToolResult.error("nope")

    tool = Tool(name="t", description="d", handler=handler)
    prompt = _make_prompt(tool)
    session = Session()
    counter.value = 7
    result = execute_tool(prompt, session, "t", None, snapshotables=(counter,))
    assert result.success is False
    assert counter.value == 7


def test_restore_failure_wraps_in_transaction_error() -> None:
    class Broken:
        def snapshot(self) -> int:
            return 0

        def restore(self, state: int) -> None:
            del state
            raise RuntimeError("cannot restore")

    session = Session()
    with pytest.raises(TransactionError, match="failed to roll back"):
        with tool_transaction(session, snapshotables=(Broken(),)):
            raise RuntimeError("inner")


# ---------------------------------------------------------------------------
# PendingToolTracker
# ---------------------------------------------------------------------------


def test_pending_tracker_rolls_back_on_failure() -> None:
    session = Session()
    session[Plan].seed(Plan(steps=("seed",)))
    tracker = PendingToolTracker(session=session)
    tracker.begin("call-1", "tool")
    session[Plan].append(Plan(steps=("dirty",)))
    rolled = tracker.end("call-1", success=False)
    assert rolled is True
    assert session[Plan].latest() == Plan(steps=("seed",))


def test_pending_tracker_keeps_state_on_success() -> None:
    session = Session()
    session[Plan].seed(Plan(steps=("seed",)))
    tracker = PendingToolTracker(session=session)
    tracker.begin("call-1", "tool")
    session[Plan].append(Plan(steps=("touched",)))
    rolled = tracker.end("call-1", success=True)
    assert rolled is False
    assert session[Plan].latest() == Plan(steps=("touched",))


def test_pending_tracker_unknown_call_returns_false() -> None:
    tracker = PendingToolTracker(session=Session())
    assert tracker.end("nope", success=False) is False
    assert tracker.abort("nope") is False


def test_pending_tracker_abort_restores_state() -> None:
    session = Session()
    session[Plan].seed(Plan(steps=("seed",)))
    tracker = PendingToolTracker(session=session)
    tracker.begin("call-1", "tool")
    session[Plan].append(Plan(steps=("dirty",)))
    assert tracker.abort("call-1") is True
    assert session[Plan].latest() == Plan(steps=("seed",))
