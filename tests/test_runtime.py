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

"""Tests for ``weakincentives.runtime`` and the noop adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import cast

import pytest

from weakincentives.adapters import NoopAdapter, ScriptedResponse
from weakincentives.clock import (
    Budget,
    BudgetTracker,
    Deadline,
    FakeClock,
)
from weakincentives.core import (
    BudgetExceededError,
    DeadlineExceededError,
    MarkdownSection,
    PolicyDeniedError,
    Prompt,
    PromptResponse,
    ProviderAdapter,
    Session,
    Tool,
    ToolCall,
    ToolContext,
    ToolResult,
    Usage,
)
from weakincentives.runtime import (
    AgentIterationCompleted,
    AgentIterationStarted,
    AgentLoop,
    AgentLoopFinished,
    AgentLoopStarted,
    MaxIterationsExceeded,
)
from weakincentives.transcript import TranscriptListener


@dataclass(frozen=True)
class Echo:
    text: str


def _echo(params: object, context: ToolContext) -> ToolResult[str]:
    del context
    if isinstance(params, dict):
        payload = cast("dict[str, object]", params).get("text", "")
    else:
        payload = ""
    return ToolResult.ok(str(payload))


def _make_prompt() -> Prompt:
    section = MarkdownSection[None](
        title="Tools",
        key="tools",
        tools=(Tool(name="echo", description="echo back", handler=_echo),),
    )
    return Prompt(ns="ns", key="k", sections=(section,))


def test_loop_runs_to_terminal_response() -> None:
    adapter = NoopAdapter(
        responses=(ScriptedResponse(text="done", finish_reason="stop"),)
    )
    loop = AgentLoop(adapter=adapter, session=Session())
    result = loop.run(_make_prompt())
    assert result.iterations == 1
    assert result.final_response.finish_reason == "stop"
    assert result.tool_results == ()


def test_loop_dispatches_tool_calls_and_iterates() -> None:
    adapter = NoopAdapter(
        responses=(
            ScriptedResponse(
                text="thinking",
                tool_calls=(ToolCall(name="echo", arguments={"text": "hi"}),),
            ),
            ScriptedResponse(text="ok", finish_reason="stop"),
        )
    )
    loop = AgentLoop(adapter=adapter, session=Session())
    result = loop.run(_make_prompt())
    assert result.iterations == 2
    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is True
    assert result.tool_results[0].value == "hi"


def test_loop_terminates_when_no_tool_calls_and_no_finish_reason() -> None:
    adapter = NoopAdapter(responses=(ScriptedResponse(text="silent"),))
    loop = AgentLoop(adapter=adapter, session=Session())
    result = loop.run(_make_prompt())
    assert result.iterations == 1


def test_loop_emits_lifecycle_events() -> None:
    listener = TranscriptListener(clock=FakeClock())
    session = Session()
    session.subscribe(listener)
    adapter = NoopAdapter(
        responses=(ScriptedResponse(text="done", finish_reason="stop"),)
    )
    AgentLoop(adapter=adapter, session=session).run(_make_prompt())
    kinds = [entry.kind for entry in listener.transcript.entries()]
    for expected in (
        AgentLoopStarted.__name__,
        AgentIterationStarted.__name__,
        AgentIterationCompleted.__name__,
        AgentLoopFinished.__name__,
    ):
        assert expected in kinds


def test_loop_records_usage_in_budget() -> None:
    adapter = NoopAdapter(
        responses=(
            ScriptedResponse(
                text="hi",
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                finish_reason="stop",
            ),
        )
    )
    tracker = BudgetTracker(budget=Budget(max_total_tokens=100))
    AgentLoop(adapter=adapter, session=Session(), budget=tracker).run(_make_prompt())
    assert tracker.consumed.total_tokens == 15


def test_loop_raises_when_budget_exceeded() -> None:
    adapter = NoopAdapter(
        responses=(
            ScriptedResponse(
                text="hi",
                usage=Usage(total_tokens=100),
                finish_reason="stop",
            ),
        )
    )
    tracker = BudgetTracker(budget=Budget(max_total_tokens=10))
    with pytest.raises(BudgetExceededError):
        AgentLoop(adapter=adapter, session=Session(), budget=tracker).run(
            _make_prompt()
        )


def test_loop_raises_when_deadline_expires() -> None:
    clock = FakeClock()
    deadline = Deadline.create(
        expires_at=clock.utcnow() + timedelta(seconds=2),
        clock=clock,
    )

    class StallingAdapter:
        def __init__(self) -> None:
            self._inner = NoopAdapter(
                responses=(ScriptedResponse(text="thinking", finish_reason=None),)
            )

        def evaluate(
            self,
            rendered: object,
            session: object,
            *,
            deadline: Deadline | None = None,
        ) -> PromptResponse:
            clock.advance(5)
            return self._inner.evaluate(rendered, session, deadline=deadline)

    adapter = cast("ProviderAdapter", StallingAdapter())
    with pytest.raises(DeadlineExceededError):
        AgentLoop(
            adapter=adapter,
            session=Session(),
            deadline=deadline,
        ).run(_make_prompt())


def test_loop_max_iterations_guard() -> None:
    adapter = NoopAdapter(
        responses=(
            ScriptedResponse(
                text="loop",
                tool_calls=(ToolCall(name="echo", arguments={"text": "a"}),),
            ),
        )
        * 20
    )
    with pytest.raises(MaxIterationsExceeded):
        AgentLoop(adapter=adapter, session=Session(), max_iterations=3).run(
            _make_prompt()
        )


def test_loop_rejects_non_positive_max_iterations() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        AgentLoop(
            adapter=NoopAdapter(),
            session=Session(),
            max_iterations=0,
        )


def test_loop_runs_policies() -> None:
    seen: list[str] = []

    class TracingPolicy:
        def before_invoke(self, tool: object, context: object) -> None:
            del tool, context
            seen.append("before")

        def on_result(self, tool: object, result: object) -> None:
            del tool, result
            seen.append("after")

    adapter = NoopAdapter(
        responses=(
            ScriptedResponse(
                text="thinking",
                tool_calls=(ToolCall(name="echo", arguments={"text": "x"}),),
            ),
            ScriptedResponse(text="done", finish_reason="stop"),
        )
    )
    AgentLoop(
        adapter=adapter,
        session=Session(),
        policies=(TracingPolicy(),),
    ).run(_make_prompt())
    assert seen == ["before", "after"]


def test_policy_block_aborts_with_policy_denied() -> None:
    class Blocker:
        def before_invoke(self, tool: object, context: object) -> None:
            del tool, context
            raise PolicyDeniedError("nope")

        def on_result(self, tool: object, result: object) -> None:
            del tool, result

    adapter = NoopAdapter(
        responses=(
            ScriptedResponse(
                text="thinking",
                tool_calls=(ToolCall(name="echo", arguments={"text": "x"}),),
            ),
        )
    )
    with pytest.raises(PolicyDeniedError):
        AgentLoop(
            adapter=adapter,
            session=Session(),
            policies=(Blocker(),),
        ).run(_make_prompt())


def test_noop_adapter_returns_stop_after_script_exhausted() -> None:
    adapter = NoopAdapter()
    loop = AgentLoop(adapter=adapter, session=Session())
    result = loop.run(_make_prompt())
    assert result.iterations == 1
    assert result.final_response.finish_reason == "stop"
    assert adapter.calls == 1


def test_noop_adapter_rejects_unknown_tool_calls() -> None:
    adapter = NoopAdapter(
        responses=(
            ScriptedResponse(
                text="oops",
                tool_calls=(ToolCall(name="missing", arguments={}),),
            ),
        )
    )
    with pytest.raises(ValueError, match="unknown tool"):
        AgentLoop(adapter=adapter, session=Session()).run(_make_prompt())


def test_noop_adapter_rejects_wrong_rendered_type() -> None:
    adapter = NoopAdapter(responses=())
    with pytest.raises(TypeError, match="RenderedPrompt"):
        adapter.evaluate("not rendered", Session())


def test_noop_adapter_rejects_wrong_session_type() -> None:
    adapter = NoopAdapter(responses=())
    rendered = _make_prompt().render()
    with pytest.raises(TypeError, match="Session"):
        adapter.evaluate(rendered, "not a session")
