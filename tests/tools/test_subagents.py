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

"""Tests for the subagent dispatch tooling."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, cast

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.events import EventBus, InProcessEventBus
from weakincentives.prompt import DelegationParams, MarkdownSection, Prompt, RecapParams
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.tool import ToolContext
from weakincentives.session import Session
from weakincentives.tools.subagents import (
    DispatchSubagentsParams,
    SubagentIsolationLevel,
    SubagentResult,
    build_dispatch_subagents_tool,
    dispatch_subagents,
)


@dataclass(slots=True)
class ParentSectionParams:
    instructions: str


@dataclass(slots=True)
class ParentOutput:
    summary: str


class RecordingAdapter(ProviderAdapter[Any]):
    def __init__(
        self,
        *,
        failures: set[str] | None = None,
        delays: dict[str, float] | None = None,
        structured_outputs: dict[str, SupportsDataclass] | None = None,
        raw_outputs: dict[str, object] | None = None,
        empty_text: set[str] | None = None,
    ) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.sessions: list[Session | None] = []
        self.buses: list[InProcessEventBus] = []
        self._failures = failures or set()
        self._delays = delays or {}
        self._structured_outputs = structured_outputs or {}
        self._raw_outputs = raw_outputs or {}
        self._empty_text = empty_text or set()

    def evaluate(
        self,
        prompt: Prompt[Any],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: InProcessEventBus,
        session: Session | None = None,
    ) -> PromptResponse[Any]:
        delegation = cast(DelegationParams, params[0])
        recap = (
            cast(RecapParams, params[1]) if len(params) > 1 else RecapParams(bullets=())
        )
        reason = delegation.reason
        self.calls.append((reason, recap.bullets))
        self.sessions.append(session)
        self.buses.append(bus)
        if reason in self._failures:
            raise RuntimeError(f"failure: {reason}")
        delay = self._delays.get(reason, 0.0)
        if delay:
            time.sleep(delay)
        if reason in self._empty_text:
            return PromptResponse(
                prompt_name=prompt.name or prompt.key,
                text="",
                output=None,
                tool_results=(),
            )
        structured = self._structured_outputs.get(reason)
        if structured is not None:
            return PromptResponse(
                prompt_name=prompt.name or prompt.key,
                text="",
                output=structured,
                tool_results=(),
            )
        raw_output = self._raw_outputs.get(reason)
        if raw_output is not None:
            return PromptResponse(
                prompt_name=prompt.name or prompt.key,
                text="",
                output=raw_output,
                tool_results=(),
            )
        return PromptResponse(
            prompt_name=prompt.name or prompt.key,
            text=f"child:{reason}",
            output=None,
            tool_results=(),
        )


def _build_parent_prompt() -> tuple[Prompt[ParentOutput], RenderedPrompt[ParentOutput]]:
    section = MarkdownSection[ParentSectionParams](
        title="Parent",
        key="parent",
        template="${instructions}",
    )
    prompt = Prompt[ParentOutput](
        ns="tests.subagents",
        key="parent",
        sections=(section,),
    )
    rendered = prompt.render(ParentSectionParams(instructions="Document the repo."))
    return prompt, rendered


def test_dispatch_subagents_requires_rendered_prompt() -> None:
    prompt, _ = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=None,
        adapter=adapter,
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="missing",
                expected_result="",
                may_delegate_further="no",
                recap_lines=("recap",),
            ),
        )
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is False
    assert result.value is None
    assert "rendered" in result.message


def test_dispatch_subagents_runs_children_in_parallel() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(delays={"slow": 0.05, "fast": 0.01})
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=rendered,
        adapter=adapter,
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="slow",
                expected_result="slow output",
                may_delegate_further="no",
                recap_lines=("Focus on slow path",),
            ),
            DelegationParams(
                reason="fast",
                expected_result="fast output",
                may_delegate_further="no",
                recap_lines=("Focus on fast path",),
            ),
        )
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert isinstance(result.value, tuple)
    assert [child.output for child in result.value] == [
        "child:slow",
        "child:fast",
    ]
    assert all(child.success for child in result.value)
    assert adapter.calls == [
        ("slow", ("Focus on slow path",)),
        ("fast", ("Focus on fast path",)),
    ]
    assert all(bus is context.event_bus for bus in adapter.buses)
    assert all(s is session for s in adapter.sessions)


def test_dispatch_subagents_isolates_children_when_requested() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=rendered,
        adapter=adapter,
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="first",
                expected_result="one",
                may_delegate_further="no",
                recap_lines=("First",),
            ),
            DelegationParams(
                reason="second",
                expected_result="two",
                may_delegate_further="no",
                recap_lines=("Second",),
            ),
        ),
    )

    isolated_tool = build_dispatch_subagents_tool(
        isolation_level=SubagentIsolationLevel.FULL_ISOLATION
    )
    handler = isolated_tool.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert all(bus is not context.event_bus for bus in adapter.buses)
    assert len({id(bus) for bus in adapter.buses}) == len(adapter.buses)
    assert all(s is not session for s in adapter.sessions)


def test_dispatch_subagents_reports_clone_failures() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()

    class BrokenSession(Session):
        def clone(
            self,
            *,
            bus: EventBus,
            session_id: str | None = None,
            created_at: str | None = None,
        ) -> Session:
            raise RuntimeError("no clones available")

    session = BrokenSession(bus=bus)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=rendered,
        adapter=adapter,
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="fail-clone",
                expected_result="",
                may_delegate_further="no",
                recap_lines=("Recap",),
            ),
        ),
    )

    isolated_tool = build_dispatch_subagents_tool(
        isolation_level=SubagentIsolationLevel.FULL_ISOLATION
    )
    handler = isolated_tool.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is False
    assert result.value is None
    assert result.message is not None
    assert "clone" in result.message.lower()


def test_dispatch_subagents_collects_failures() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(failures={"fail"})
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=rendered,
        adapter=adapter,
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="ok",
                expected_result="success",
                may_delegate_further="no",
                recap_lines=("Keep things tidy",),
            ),
            DelegationParams(
                reason="fail",
                expected_result="error",
                may_delegate_further="no",
                recap_lines=("Handle the failure",),
            ),
        )
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert isinstance(result.value, tuple)
    first, second = result.value
    assert isinstance(first, SubagentResult)
    assert isinstance(second, SubagentResult)
    assert first.success is True
    assert first.error is None
    assert second.success is False
    assert second.error is not None
    assert "fail" in second.error
    assert adapter.calls[1][0] == "fail"


def test_dispatch_subagents_requires_dataclass_output_type() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=RenderedPrompt(
            text=rendered.text,
            output_type=str,
            container=rendered.container,
            allow_extra_keys=rendered.allow_extra_keys,
            _tools=rendered.tools,
            _tool_param_descriptions=rendered.tool_param_descriptions,
        ),
        adapter=adapter,
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="invalid",
                expected_result="",
                may_delegate_further="no",
                recap_lines=("recap",),
            ),
        ),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is False
    assert result.value is None
    assert "dataclass" in result.message


def test_dispatch_subagents_handles_empty_delegations() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=rendered,
        adapter=adapter,
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(delegations=())

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert result.value == ()
    assert "No delegations" in result.message


def test_dispatch_subagents_formats_structured_outputs() -> None:
    @dataclass(slots=True)
    class StructuredChildResult:
        field: str

    class Unserializable:
        def __str__(self) -> str:
            return "fallback-output"

    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(
        structured_outputs={"structured": StructuredChildResult(field="value")},
        raw_outputs={"raw": Unserializable()},
    )
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=rendered,
        adapter=adapter,
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="structured",
                expected_result="json",
                may_delegate_further="no",
                recap_lines=("Render structured",),
            ),
            DelegationParams(
                reason="raw",
                expected_result="fallback",
                may_delegate_further="no",
                recap_lines=("Render raw",),
            ),
        ),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert result.value is not None
    first, second = result.value
    assert json.loads(first.output) == {"field": "value"}
    assert second.output == "fallback-output"


def test_dispatch_subagents_returns_empty_output_when_child_returns_none() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(empty_text={"empty"})
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=rendered,
        adapter=adapter,
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="empty",
                expected_result="",
                may_delegate_further="no",
                recap_lines=("Produce nothing",),
            ),
        ),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert result.value is not None
    child = result.value[0]
    assert child.output == ""


"""Tests for the subagent dispatch tooling."""
