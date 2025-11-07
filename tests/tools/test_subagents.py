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
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.prompt import DelegationParams, MarkdownSection, Prompt, RecapParams
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.tool import ToolContext
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.tools.subagents import (
    DispatchSubagentsParams,
    SubagentResult,
    dispatch_subagents,
)


@dataclass(slots=True)
class ParentSectionParams:
    instructions: str


@dataclass(slots=True)
class ParentOutput:
    summary: str


@dataclass(slots=True)
class DelegationWithDeadline(DelegationParams):
    deadline: datetime | None = None


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
        self.deadlines: list[datetime | None] = []
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
        deadline: datetime | None = None,
    ) -> PromptResponse[Any]:
        delegation = cast(DelegationParams, params[0])
        recap = (
            cast(RecapParams, params[1]) if len(params) > 1 else RecapParams(bullets=())
        )
        reason = delegation.reason
        self.calls.append((reason, recap.bullets))
        self.sessions.append(session)
        self.buses.append(bus)
        self.deadlines.append(deadline)
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


def test_dispatch_subagents_propagates_deadline() -> None:
    prompt, rendered = _build_parent_prompt()
    deadline = datetime.now(UTC) + timedelta(seconds=5)
    rendered_with_deadline = replace(rendered, deadline=deadline)
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=rendered_with_deadline,
        adapter=adapter,
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="one",
                expected_result="",
                may_delegate_further="no",
                recap_lines=("recap",),
            ),
            DelegationParams(
                reason="two",
                expected_result="",
                may_delegate_further="no",
                recap_lines=("recap",),
            ),
        )
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert adapter.deadlines == [deadline, deadline]


def test_dispatch_subagents_prefers_override_deadline() -> None:
    prompt, rendered = _build_parent_prompt()
    parent_deadline = datetime.now(UTC) + timedelta(seconds=20)
    override_deadline = datetime.now(UTC) + timedelta(seconds=5)
    rendered_with_deadline = replace(rendered, deadline=parent_deadline)
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=rendered_with_deadline,
        adapter=adapter,
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationWithDeadline(
                reason="one",
                expected_result="",
                may_delegate_further="no",
                recap_lines=("recap",),
                deadline=override_deadline,
            ),
        )
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    handler(params, context=context)

    assert adapter.deadlines == [override_deadline]


def test_dispatch_subagents_ignores_naive_override() -> None:
    prompt, rendered = _build_parent_prompt()
    parent_deadline = datetime.now(UTC) + timedelta(seconds=15)
    rendered_with_deadline = replace(rendered, deadline=parent_deadline)
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=rendered_with_deadline,
        adapter=adapter,
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationWithDeadline(
                reason="naive",
                expected_result="",
                may_delegate_further="no",
                recap_lines=("recap",),
                deadline=datetime.now(),
            ),
        )
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    handler(params, context=context)

    assert adapter.deadlines == [parent_deadline]


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
            deadline=rendered.deadline,
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
