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

"""Unit tests for the dispatch_subagents tool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.events import InProcessEventBus
from weakincentives.prompt import (
    DelegationSummaryParams,
    MarkdownSection,
    Prompt,
    SupportsDataclass,
    ToolContext,
)
from weakincentives.session import Session
from weakincentives.tools.subagents import (
    DispatchSubagentsParams,
    SubagentDispatch,
    dispatch_subagents,
)


@dataclass(slots=True)
class ParentParams:
    instructions: str


@dataclass(slots=True)
class ParentOutput:
    status: str


class RecordingAdapter(ProviderAdapter[object]):
    """Adapter stub that records evaluations for inspection."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def evaluate(
        self,
        prompt: Prompt[object],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: InProcessEventBus,
        session: object | None = None,
    ) -> PromptResponse[object]:
        del parse_output
        rendered = prompt.render(*params)
        call_record = {
            "prompt": prompt,
            "params": params,
            "bus": bus,
            "session": session,
            "rendered": rendered,
        }
        self.calls.append(call_record)
        return PromptResponse(
            prompt_name=prompt.key,
            text=rendered.text,
            output=None,
            tool_results=(),
        )


def _build_context(*, include_response_format: bool = False) -> ToolContext:
    adapter = RecordingAdapter()
    sections = (
        MarkdownSection[ParentParams](
            title="Work",
            key="work",
            template="${instructions}",
        ),
    )
    if include_response_format:
        parent_prompt = Prompt[ParentOutput](
            ns="tests.subagents",
            key="parent",
            sections=sections,
        )
    else:
        parent_prompt = Prompt(
            ns="tests.subagents",
            key="parent",
            sections=sections,
        )
    rendered_parent = parent_prompt.render(ParentParams(instructions="Triage tasks."))
    session_bus = InProcessEventBus()
    session = Session(bus=session_bus)
    context_bus = InProcessEventBus()
    return ToolContext(
        prompt=parent_prompt,
        rendered_prompt=rendered_parent,
        adapter=adapter,
        session=session,
        event_bus=context_bus,
    )


def _make_dispatch(reason: str) -> SubagentDispatch:
    return SubagentDispatch(
        summary=DelegationSummaryParams(
            reason=reason,
            expected_result=f"result for {reason}",
            may_delegate_further="no",
        ),
        recap_lines=("first checkpoint", "second checkpoint"),
    )


def _invoke_tool(
    context: ToolContext, dispatches: Sequence[SubagentDispatch]
) -> dict[str, Any]:
    handler = dispatch_subagents.handler
    assert handler is not None
    params = DispatchSubagentsParams(dispatches=tuple(dispatches))
    result = handler(params, context=context)
    adapter = context.adapter
    assert isinstance(adapter, RecordingAdapter)
    return {"result": result, "calls": adapter.calls}


def test_dispatch_subagents_runs_each_child_in_parallel() -> None:
    context = _build_context()
    dispatches = (_make_dispatch("collect data"), _make_dispatch("summarize"))

    payload = _invoke_tool(context, dispatches)
    result = payload["result"]
    calls = payload["calls"]

    assert result.success is True
    assert result.value is not None
    values = tuple(result.value)
    assert len(values) == 2
    assert [item.success for item in values] == [True, True]
    assert result.value[0] is values[0]
    assert calls[0]["bus"] is not context.event_bus
    assert calls[0]["bus"] is not calls[1]["bus"]
    assert calls[0]["session"] is not context.session
    assert calls[1]["session"] is not context.session
    assert {call["params"][0].reason for call in calls} == {
        "collect data",
        "summarize",
    }
    # Every delegated prompt should contain the parent content markers.
    for call in calls:
        assert "PARENT PROMPT START" in call["rendered"].text


def test_dispatch_subagents_requires_rendered_parent() -> None:
    context = _build_context()
    context = ToolContext(
        prompt=context.prompt,
        rendered_prompt=None,
        adapter=context.adapter,
        session=context.session,
        event_bus=context.event_bus,
    )
    handler = dispatch_subagents.handler
    assert handler is not None
    params = DispatchSubagentsParams(dispatches=())

    result = handler(params, context=context)

    assert result.success is False
    assert result.value is None


def test_dispatch_subagents_marks_failed_children() -> None:
    context = _build_context()
    adapter = context.adapter
    assert isinstance(adapter, RecordingAdapter)

    def _raising_evaluate(
        prompt: Prompt[object],
        *params: DelegationSummaryParams,
        parse_output: bool = True,
        bus: InProcessEventBus,
        session: object | None = None,
    ) -> PromptResponse[object]:
        raise RuntimeError("child failed")

    adapter.evaluate = _raising_evaluate  # type: ignore[method-assign]
    dispatch = _make_dispatch("unstable task")
    handler = dispatch_subagents.handler
    assert handler is not None

    result = handler(DispatchSubagentsParams(dispatches=(dispatch,)), context=context)

    assert result.success is True
    assert result.value is not None
    child_result = next(iter(result.value))
    assert child_result.success is False
    assert child_result.error == "child failed"


def test_dispatch_subagents_propagates_session_clone_errors() -> None:
    context = _build_context()

    class _BadSession:
        def __init__(self) -> None:
            self.clone_called = False

    bad_session = _BadSession()
    context = ToolContext(
        prompt=context.prompt,
        rendered_prompt=context.rendered_prompt,
        adapter=context.adapter,
        session=bad_session,
        event_bus=context.event_bus,
    )
    handler = dispatch_subagents.handler
    assert handler is not None

    params = DispatchSubagentsParams(dispatches=(_make_dispatch("broken"),))

    result = handler(params, context=context)

    assert result.success is False
    assert result.value is None
    assert "Session does not support cloning." in result.message


def test_dispatch_subagents_normalizes_empty_dispatches() -> None:
    context = _build_context()
    handler = dispatch_subagents.handler
    assert handler is not None

    result = handler(DispatchSubagentsParams(dispatches=()), context=context)

    assert result.success is True
    assert result.value is not None
    assert tuple(result.value) == ()
    assert result.message == "Dispatched 0 subagents."


def test_dispatch_subagents_includes_response_format_for_structured_parent() -> None:
    context = _build_context(include_response_format=True)
    dispatch = _make_dispatch("structured work")

    payload = _invoke_tool(context, (dispatch,))
    result = payload["result"]
    call = payload["calls"][0]

    assert result.success is True
    rendered_child = call["rendered"]
    assert "Response Format" in rendered_child.text
