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
import tempfile
import time
from dataclasses import dataclass, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any, Literal, cast

from tests.helpers.adapters import RECORDING_ADAPTER_NAME
from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.deadlines import Deadline
from weakincentives.prompt import DelegationParams, MarkdownSection, Prompt, RecapParams
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.overrides import PromptOverridesStore
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.protocols import PromptProtocol, ProviderAdapterProtocol
from weakincentives.prompt.structured_output import StructuredOutputConfig
from weakincentives.prompt.tool import ToolContext, ToolResult
from weakincentives.runtime.events import InProcessEventBus, PromptExecuted
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.protocols import SessionProtocol, SnapshotProtocol
from weakincentives.tools.asteval import AstevalSection, EvalParams
from weakincentives.tools.planning import PlanningToolsSection, SetupPlan
from weakincentives.tools.podman import PodmanToolsSection
from weakincentives.tools.subagents import (
    DispatchSubagentsParams,
    SubagentIsolationLevel,
    SubagentResult,
    build_dispatch_subagents_tool,
    dispatch_subagents,
)
from weakincentives.tools.vfs import ListDirectoryParams, VfsToolsSection


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
        self.deadlines: list[Deadline | None] = []
        self._failures = failures or set()
        self._delays = delays or {}
        self._structured_outputs = structured_outputs or {}
        self._raw_outputs = raw_outputs or {}
        self._empty_text = empty_text or set()
        self._lock = Lock()

    def evaluate(
        self,
        prompt: Prompt[Any],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: InProcessEventBus,
        session: Session | None = None,
        deadline: Deadline | None = None,
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str = "latest",
    ) -> PromptResponse[Any]:
        del parse_output
        delegation = cast(DelegationParams, params[0])
        recap = (
            cast(RecapParams, params[1]) if len(params) > 1 else RecapParams(bullets=())
        )
        reason = delegation.reason
        with self._lock:
            self.calls.append((reason, recap.bullets))
            self.sessions.append(session)
            self.buses.append(bus)
            self.deadlines.append(deadline)
        if reason in self._failures:
            raise RuntimeError(f"failure: {reason}")
        delay = self._delays.get(reason, 0.0)
        if delay:
            time.sleep(delay)
        prompt_name = prompt.name or prompt.key

        def _emit(response: PromptResponse[Any]) -> PromptResponse[Any]:
            bus.publish(
                PromptExecuted(
                    prompt_name=prompt_name,
                    adapter=RECORDING_ADAPTER_NAME,
                    result=cast(PromptResponse[object], response),
                    session_id=getattr(session, "session_id", None),
                    created_at=datetime.now(UTC),
                    value=(
                        cast(SupportsDataclass, response.output)
                        if response.output is not None and is_dataclass(response.output)
                        else None
                    ),
                )
            )
            return response

        if reason in self._empty_text:
            return _emit(
                PromptResponse(
                    prompt_name=prompt_name,
                    text="",
                    output=None,
                    tool_results=(),
                )
            )
        structured = self._structured_outputs.get(reason)
        if structured is not None:
            return _emit(
                PromptResponse(
                    prompt_name=prompt_name,
                    text="",
                    output=structured,
                    tool_results=(),
                )
            )
        raw_output = self._raw_outputs.get(reason)
        if raw_output is not None:
            return _emit(
                PromptResponse(
                    prompt_name=prompt_name,
                    text="",
                    output=raw_output,
                    tool_results=(),
                )
            )
        return _emit(
            PromptResponse(
                prompt_name=prompt_name,
                text=f"child:{reason}",
                output=None,
                tool_results=(),
            )
        )


class _ToolInvokingAdapter(ProviderAdapter[Any]):
    def __init__(self, *, tool_name: str, tool_params: SupportsDataclass) -> None:
        self._tool_name = tool_name
        self._tool_params = tool_params
        self.sessions: list[Session] = []
        self.buses: list[InProcessEventBus] = []
        self.results: list[ToolResult[SupportsDataclass]] = []
        self._lock = Lock()

    def evaluate(
        self,
        prompt: Prompt[Any],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: InProcessEventBus,
        session: Session | None = None,
        deadline: Deadline | None = None,
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str = "latest",
    ) -> PromptResponse[Any]:
        del parse_output, overrides_store, overrides_tag
        delegation = cast(DelegationParams, params[0])
        rendered = prompt.render(*params)
        tool = next(tool for tool in rendered.tools if tool.name == self._tool_name)
        handler = tool.handler
        assert handler is not None
        assert session is not None
        with self._lock:
            self.sessions.append(session)
            self.buses.append(bus)
        tool_context = ToolContext(
            prompt=cast(PromptProtocol[Any], prompt),
            rendered_prompt=rendered,
            adapter=cast(ProviderAdapterProtocol[Any], self),
            session=session,
            event_bus=bus,
            deadline=deadline,
        )
        result = handler(self._tool_params, context=tool_context)
        with self._lock:
            self.results.append(result)
        prompt_name = prompt.name or prompt.key
        return PromptResponse(
            prompt_name=prompt_name,
            text=f"child:{delegation.reason}",
            output=None,
            tool_results=(result,),
        )


def _build_parent_prompt(
    *,
    deadline: Deadline | None = None,
    extra_sections: tuple[MarkdownSection[Any], ...] = (),
) -> tuple[Prompt[ParentOutput], RenderedPrompt[ParentOutput]]:
    section = MarkdownSection[ParentSectionParams](
        title="Parent",
        key="parent",
        template="${instructions}",
    )
    prompt = Prompt[ParentOutput](
        ns="tests.subagents",
        key="parent",
        sections=(section, *extra_sections),
    )
    rendered = prompt.render(ParentSectionParams(instructions="Document the repo."))
    if deadline is not None:
        rendered = replace(rendered, deadline=deadline)
    return prompt, rendered


def test_dispatch_subagents_requires_rendered_prompt() -> None:
    prompt, _ = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=None,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
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
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
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
    assert adapter.calls == [
        ("slow", ("Focus on slow path",)),
        ("fast", ("Focus on fast path",)),
    ]
    assert all(bus is context.event_bus for bus in adapter.buses)
    assert all(s is session for s in adapter.sessions)


def test_dispatch_subagents_propagates_deadline() -> None:
    parent_deadline = Deadline(datetime.now(UTC) + timedelta(seconds=5))
    prompt, rendered = _build_parent_prompt(deadline=parent_deadline)
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="one",
                expected_result="",
                may_delegate_further="no",
                recap_lines=(),
            ),
        ),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert adapter.deadlines == [parent_deadline]


def test_dispatch_subagents_collects_failures() -> None:
    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(failures={"fail"})
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
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
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=RenderedPrompt(
            text=rendered.text,
            structured_output=StructuredOutputConfig(
                dataclass_type=cast(type[SupportsDataclass], cast(object, str)),
                container=cast(Literal["object", "array"], rendered.container),
                allow_extra_keys=bool(rendered.allow_extra_keys),
            ),
            _tools=rendered.tools,
            _tool_param_descriptions=rendered.tool_param_descriptions,
        ),
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
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
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
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
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
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
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
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


def test_dispatch_subagents_shares_state_without_isolation() -> None:
    @dataclass(slots=True)
    class ChildRecord:
        field: str

    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(
        structured_outputs={"shared": ChildRecord(field="value")}
    )
    bus = InProcessEventBus()
    session = Session(bus=bus)
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="shared",
                expected_result="capture state",
                may_delegate_further="no",
                recap_lines=("Record to session",),
            ),
        ),
    )

    handler = dispatch_subagents.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    captured = session.select_all(ChildRecord)
    assert captured == (ChildRecord(field="value"),)


def test_dispatch_subagents_full_isolation_clones_state() -> None:
    @dataclass(slots=True)
    class ChildRecord:
        field: str

    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter(
        structured_outputs={"isolated": ChildRecord(field="value")}
    )
    bus = InProcessEventBus()
    session = Session(bus=bus)
    tool = build_dispatch_subagents_tool(
        isolation_level=SubagentIsolationLevel.FULL_ISOLATION
    )
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="isolated",
                expected_result="capture state",
                may_delegate_further="no",
                recap_lines=("Record to clone",),
            ),
        ),
    )

    handler = tool.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert session.select_all(ChildRecord) == ()
    assert adapter.sessions
    assert all(child_session is not session for child_session in adapter.sessions)
    assert all(child_bus is not bus for child_bus in adapter.buses)


def test_dispatch_subagents_full_isolation_requires_clone_support() -> None:
    class NonCloningSession(SessionProtocol):
        def snapshot(self) -> SnapshotProtocol:
            raise NotImplementedError

        def rollback(self, snapshot: SnapshotProtocol) -> None:
            raise NotImplementedError

        def reset(self) -> None:
            raise NotImplementedError

    prompt, rendered = _build_parent_prompt()
    adapter = RecordingAdapter()
    bus = InProcessEventBus()
    session = NonCloningSession()
    tool = build_dispatch_subagents_tool(
        isolation_level=SubagentIsolationLevel.FULL_ISOLATION
    )
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="non-clone",
                expected_result="",
                may_delegate_further="no",
                recap_lines=("Should fail",),
            ),
        ),
    )

    handler = tool.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is False
    assert result.value is None
    assert "cloning" in result.message.lower()


def test_dispatch_subagents_full_isolation_supports_planning_tools() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    planning_section = PlanningToolsSection(session=session)
    prompt, rendered = _build_parent_prompt(extra_sections=(planning_section,))
    adapter = _ToolInvokingAdapter(
        tool_name="planning_setup_plan",
        tool_params=SetupPlan(objective="Isolated", initial_steps=()),
    )
    tool = build_dispatch_subagents_tool(
        isolation_level=SubagentIsolationLevel.FULL_ISOLATION
    )
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="planning",
                expected_result="",
                may_delegate_further="no",
                recap_lines=("exercise planning tools",),
            ),
        )
    )

    handler = tool.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert adapter.sessions
    assert all(child is not session for child in adapter.sessions)
    assert all(
        bus is child.event_bus
        for bus, child in zip(adapter.buses, adapter.sessions, strict=True)
    )


def test_dispatch_subagents_full_isolation_supports_vfs_tools() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    vfs_section = VfsToolsSection(session=session)
    prompt, rendered = _build_parent_prompt(extra_sections=(vfs_section,))
    adapter = _ToolInvokingAdapter(
        tool_name="ls", tool_params=ListDirectoryParams(path=None)
    )
    tool = build_dispatch_subagents_tool(
        isolation_level=SubagentIsolationLevel.FULL_ISOLATION
    )
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="vfs",
                expected_result="",
                may_delegate_further="no",
                recap_lines=("exercise vfs",),
            ),
        )
    )

    handler = tool.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert adapter.sessions
    assert all(child is not session for child in adapter.sessions)
    assert all(
        bus is child.event_bus
        for bus, child in zip(adapter.buses, adapter.sessions, strict=True)
    )


def test_dispatch_subagents_full_isolation_supports_asteval() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    asteval_section = AstevalSection(session=session)
    prompt, rendered = _build_parent_prompt(extra_sections=(asteval_section,))
    adapter = _ToolInvokingAdapter(
        tool_name="evaluate_python",
        tool_params=EvalParams(code="1 + 1"),
    )
    tool = build_dispatch_subagents_tool(
        isolation_level=SubagentIsolationLevel.FULL_ISOLATION
    )
    context = ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=rendered,
        adapter=cast(ProviderAdapterProtocol[Any], adapter),
        session=session,
        event_bus=bus,
    )
    params = DispatchSubagentsParams(
        delegations=(
            DelegationParams(
                reason="asteval",
                expected_result="",
                may_delegate_further="no",
                recap_lines=("exercise asteval",),
            ),
        )
    )

    handler = tool.handler
    assert handler is not None
    result = handler(params, context=context)

    assert result.success is True
    assert adapter.sessions
    assert all(child is not session for child in adapter.sessions)
    assert all(
        bus is child.event_bus
        for bus, child in zip(adapter.buses, adapter.sessions, strict=True)
    )


def test_dispatch_subagents_full_isolation_supports_podman_vfs() -> None:
    class _FakeContainer:
        def __init__(self) -> None:
            self.id = "fake-container"

        def start(self) -> None:
            return None

        def exec_run(
            self, *_args: object, **_kwargs: object
        ) -> tuple[int, tuple[bytes, bytes]]:
            return 0, (b"", b"")

        def stop(self, _timeout: int | None = None) -> None:
            return None

        def remove(self, _force: bool = True) -> None:
            return None

    class _FakeContainers:
        def __init__(self) -> None:
            self._last: _FakeContainer | None = None

        def create(self, **_kwargs: object) -> _FakeContainer:
            self._last = _FakeContainer()
            return self._last

        def get(self, _identifier: object) -> _FakeContainer:
            if self._last is None:
                raise RuntimeError("container missing")
            return self._last

    class _FakeImages:
        def pull(self, _image: object) -> None:
            return None

    class _FakeClient:
        def __init__(self) -> None:
            self.images = _FakeImages()
            self.containers = _FakeContainers()

        def close(self) -> None:
            return None

    bus = InProcessEventBus()
    session = Session(bus=bus)
    with tempfile.TemporaryDirectory() as cache_dir:
        podman_section = PodmanToolsSection(
            session=session,
            base_url="http://podman.invalid",
            identity=None,
            cache_dir=cache_dir,
            client_factory=_FakeClient,
        )
        prompt, rendered = _build_parent_prompt(extra_sections=(podman_section,))
        adapter = _ToolInvokingAdapter(
            tool_name="ls", tool_params=ListDirectoryParams(path=None)
        )
        tool = build_dispatch_subagents_tool(
            isolation_level=SubagentIsolationLevel.FULL_ISOLATION
        )
        context = ToolContext(
            prompt=cast(PromptProtocol[Any], prompt),
            rendered_prompt=rendered,
            adapter=cast(ProviderAdapterProtocol[Any], adapter),
            session=session,
            event_bus=bus,
        )
        params = DispatchSubagentsParams(
            delegations=(
                DelegationParams(
                    reason="podman",
                    expected_result="",
                    may_delegate_further="no",
                    recap_lines=("exercise podman",),
                ),
            )
        )

        handler = tool.handler
        assert handler is not None
        result = handler(params, context=context)

        assert result.success is True
        assert adapter.sessions
        assert all(child is not session for child in adapter.sessions)
        assert all(
            bus is child.event_bus
            for bus, child in zip(adapter.buses, adapter.sessions, strict=True)
        )
        podman_section.close()


def test_build_dispatch_subagents_tool_respects_accepts_overrides() -> None:
    default_tool = build_dispatch_subagents_tool()
    overriding_tool = build_dispatch_subagents_tool(accepts_overrides=True)

    assert default_tool.accepts_overrides is False
    assert overriding_tool.accepts_overrides is True
