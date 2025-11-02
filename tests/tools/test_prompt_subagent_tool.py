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

"""Tests for the prompt subagent dispatch tool."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import pytest

from weakincentives.events import EventBus, InProcessEventBus, ToolInvoked
from weakincentives.prompt import MarkdownSection, Prompt, Tool, ToolResult, registry
from weakincentives.session import Session, select_all
from weakincentives.tools import (
    DispatchSubagent,
    DispatchSubagentError,
    PromptSubagentToolsSection,
    ToolValidationError,
)
from weakincentives.tools.prompt_subagent import DispatchSubagentResult, SubagentMode


@dataclass(slots=True, frozen=True)
class SampleRecord:
    value: str


@dataclass(slots=True, frozen=True)
class ChildPromptParams:
    instructions: str
    expected_artifacts: tuple[str, ...] = ()


@pytest.fixture(autouse=True)
def _clear_prompt_registry() -> Iterator[None]:
    registry.clear()
    yield
    registry.clear()


def _register_child_prompt(ns: str = "tests/subagent", key: str = "child") -> None:
    def factory(*, session: Session) -> Prompt[Any]:
        section = MarkdownSection[ChildPromptParams](
            title="Child Guidance",
            template="""${instructions}\nExpected: ${expected_artifacts}""",
            key="child-guidance",
            default_params=ChildPromptParams(instructions=""),
        )
        return Prompt(ns=ns, key=key, sections=[section])

    registry.register(ns, key, factory)


class RecordingRunner:
    """Test helper that records runner invocations."""

    def __init__(self, *, expected_mode: SubagentMode) -> None:
        self.expected_mode = expected_mode
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        prompt: Prompt[Any],
        session: Session,
        bus: EventBus,
        instructions: str,
        expected_artifacts: tuple[str, ...],
        mode: SubagentMode,
        plan_step_id: str | None,
    ) -> DispatchSubagentResult:
        self.calls.append(
            {
                "prompt": prompt,
                "session": session,
                "instructions": instructions,
                "expected_artifacts": expected_artifacts,
                "mode": mode,
                "plan_step_id": plan_step_id,
            }
        )
        assert mode == self.expected_mode
        assert select_all(session, SampleRecord) == (SampleRecord("parent"),)

        payload = SampleRecord("child")
        tool_result = cast(
            ToolResult[object],
            ToolResult(message="child tool", value=payload),
        )
        bus.publish(
            ToolInvoked(
                prompt_name="tests/subagent/child",
                adapter="test",
                name="child_tool",
                params=payload,
                result=tool_result,
            )
        )
        return DispatchSubagentResult(
            message_summary=f"done: {instructions}",
            artifacts=("draft.md",),
        )


def _build_tool(
    session: Session, runner: RecordingRunner
) -> PromptSubagentToolsSection:
    return PromptSubagentToolsSection(session=session, runner=runner)


def _get_dispatch_tool(section: PromptSubagentToolsSection) -> Tool[Any, Any]:
    for tool in section.tools():
        if tool.name == "dispatch_subagent":
            assert tool.handler is not None
            return tool
    raise AssertionError("dispatch_subagent tool not found")


def test_dispatch_subagent_runs_child_prompt_and_records_tools() -> None:
    _register_child_prompt()
    bus = InProcessEventBus()
    parent_session = Session(bus=bus)
    parent_session.seed_slice(SampleRecord, (SampleRecord("parent"),))
    runner = RecordingRunner(expected_mode="ad_hoc")
    section = _build_tool(parent_session, runner)
    tool = _get_dispatch_tool(section)

    params = DispatchSubagent(
        mode="ad_hoc",
        prompt_ns="tests/subagent",
        prompt_key="child",
        instructions="  summarize progress  ",
        expected_artifacts=(" report.md ",),
    )

    handler = tool.handler
    assert handler is not None
    result = handler(params)

    assert result.message == "done: summarize progress"
    value = result.value
    assert isinstance(value, DispatchSubagentResult)
    assert value.message_summary == result.message
    assert value.artifacts == ("draft.md",)
    assert value.tools_used == ("child_tool",)

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["instructions"] == "summarize progress"
    assert call["expected_artifacts"] == ("report.md",)
    assert isinstance(call["prompt"], Prompt)
    assert call["session"] is not parent_session

    # Parent session state remains unchanged.
    assert select_all(parent_session, SampleRecord) == (SampleRecord("parent"),)


def test_dispatch_subagent_requires_registered_prompt() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    runner = RecordingRunner(expected_mode="ad_hoc")
    section = _build_tool(session, runner)
    tool = _get_dispatch_tool(section)

    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            DispatchSubagent(
                mode="ad_hoc",
                prompt_ns="missing",
                prompt_key="prompt",
                instructions="do work",
            )
        )


def test_dispatch_subagent_validates_inputs() -> None:
    _register_child_prompt()
    session = Session(bus=InProcessEventBus())
    runner = RecordingRunner(expected_mode="ad_hoc")
    section = _build_tool(session, runner)
    tool = _get_dispatch_tool(section)
    handler = tool.handler
    assert handler is not None

    with pytest.raises(ToolValidationError):
        handler(
            DispatchSubagent(
                mode="ad_hoc",
                prompt_ns="tests/subagent",
                prompt_key="child",
                instructions="   ",
            )
        )

    with pytest.raises(ToolValidationError):
        handler(
            DispatchSubagent(
                mode="ad_hoc",
                prompt_ns="tests/subagent",
                prompt_key="child",
                instructions="ok",
                expected_artifacts=("x" * 161,),
            )
        )

    with pytest.raises(ToolValidationError):
        handler(
            DispatchSubagent(
                mode="plan_step",
                prompt_ns="tests/subagent",
                prompt_key="child",
                instructions="ok",
            )
        )


def test_dispatch_subagent_handles_runtime_errors() -> None:
    _register_child_prompt()
    session = Session(bus=InProcessEventBus())

    def failing_runner(
        *,
        prompt: Prompt[Any],
        session: Session,
        bus: EventBus,
        instructions: str,
        expected_artifacts: tuple[str, ...],
        mode: SubagentMode,
        plan_step_id: str | None,
    ) -> DispatchSubagentResult:
        raise DispatchSubagentError("adapter failed")

    section = PromptSubagentToolsSection(session=session, runner=failing_runner)
    tool = _get_dispatch_tool(section)
    handler = tool.handler
    assert handler is not None

    result = handler(
        DispatchSubagent(
            mode="ad_hoc",
            prompt_ns="tests/subagent",
            prompt_key="child",
            instructions="ok",
        )
    )

    assert result.success is False
    assert result.value is None
    assert "Subagent execution failed" in result.message


def test_dispatch_subagent_handles_unexpected_runner_exceptions() -> None:
    _register_child_prompt()
    session = Session(bus=InProcessEventBus())

    def crashing_runner(
        *,
        prompt: Prompt[Any],
        session: Session,
        bus: EventBus,
        instructions: str,
        expected_artifacts: tuple[str, ...],
        mode: SubagentMode,
        plan_step_id: str | None,
    ) -> DispatchSubagentResult:
        raise RuntimeError("boom")

    section = PromptSubagentToolsSection(session=session, runner=crashing_runner)
    tool = _get_dispatch_tool(section)
    handler = tool.handler
    assert handler is not None

    result = handler(
        DispatchSubagent(
            mode="ad_hoc",
            prompt_ns="tests/subagent",
            prompt_key="child",
            instructions="ok",
        )
    )

    assert result.success is False
    assert result.value is None
    assert "Subagent execution failed" in result.message
