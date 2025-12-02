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

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast, override

import pytest

from weakincentives.adapters._names import OPENAI_ADAPTER_NAME
from weakincentives.adapters.core import (
    OptimizationScope,
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
)
from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.overrides import (
    PromptLike,
    PromptOverride,
    PromptOverridesStore,
)
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import ToolInvoked
from weakincentives.runtime.session import Session
from weakincentives.tools.digests import (
    WorkspaceDigest,
    WorkspaceDigestSection,
    latest_workspace_digest,
    set_workspace_digest,
)
from weakincentives.tools.vfs import VfsToolsSection


@dataclass(slots=True, frozen=True)
class _FakeOptimizationOutput:
    digest: str


@dataclass(slots=True, frozen=True)
class _PromptOutput:
    summary: str = "ok"


@dataclass(slots=True, frozen=True)
class _ToolEventParams:
    value: str = "param"


class _RecordingOverridesStore(PromptOverridesStore):
    def __init__(self) -> None:
        self.calls: list[tuple[Prompt[Any], str, tuple[str, ...], str]] = []

    def resolve(
        self,
        descriptor: PromptLike,
        tag: str = "latest",
    ) -> PromptOverride | None:
        self.calls.append((cast(Prompt[Any], descriptor), tag, (), "resolve"))
        return None

    def upsert(
        self,
        descriptor: PromptLike,
        override: PromptOverride,
    ) -> PromptOverride:
        self.calls.append((cast(Prompt[Any], descriptor), "", (), "upsert"))
        return override

    def delete(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None:
        self.calls.append((cast(Prompt[Any], object()), tag, (ns, prompt_key), "delete"))

    def set_section_override(
        self,
        prompt: Prompt[Any],
        *,
        tag: str,
        path: tuple[str, ...],
        body: str,
    ) -> PromptOverride:
        self.calls.append((prompt, tag, path, body))
        return cast(PromptOverride, object())

    def seed(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        self.calls.append((cast(Prompt[Any], prompt), tag, (), "seed"))
        return cast(PromptOverride, object())


class _RecordingAdapter(ProviderAdapter):
    def __init__(self, *, mode: str, emit_tool_event: bool = False) -> None:
        self.mode = mode
        self.rendered_prompts: list[Prompt[Any]] = []
        self.sessions: list[Session | None] = []
        self.buses: list[Any] = []
        self._emit_tool_event = emit_tool_event

    @override
    def evaluate(
        self,
        prompt: Prompt[Any],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: Any = None,
        session: Session | None = None,
        deadline: Any = None,
        overrides_store: Any = None,
        overrides_tag: str = "latest",
    ) -> PromptResponse[Any]:
        del params, parse_output, deadline, overrides_store, overrides_tag
        prompt_name = prompt.name or prompt.key
        self.rendered_prompts.append(prompt)
        self.sessions.append(session)
        self.buses.append(bus)

        if self._emit_tool_event and bus is not None:
            event = ToolInvoked(
                prompt_name=prompt_name,
                adapter=OPENAI_ADAPTER_NAME,
                name="optimize-tool",
                params=_ToolEventParams(),
                result=ToolResult(message="ok", value=None),
                session_id=getattr(session, "session_id", None) if session else None,
                created_at=datetime.now(UTC),
            )
            bus.publish(event)

        digest_value = f"{self.mode}-digest"
        if self.mode == "dataclass":
            output: Any = _FakeOptimizationOutput(digest=digest_value)
            text: str | None = None
        elif self.mode == "string":
            output = digest_value
            text = None
        elif self.mode == "text":
            output = None
            text = digest_value
        else:  # pragma: no cover - mode guarded by tests
            output = None
            text = None

        return PromptResponse(
            prompt_name=prompt.name or prompt.key,
            text=text,
            output=output,
        )


def _build_prompt() -> Prompt[_PromptOutput]:
    session = Session()
    workspace = VfsToolsSection(session=session)
    digest = WorkspaceDigestSection(session=session)
    return Prompt[_PromptOutput](ns="tests", key="opt", sections=(workspace, digest))


def test_optimize_persists_digest_from_output() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    prompt = _build_prompt()
    session = Session()

    result = adapter.optimize(prompt, session=session)

    latest = latest_workspace_digest(session, "workspace-digest")
    assert result.digest == "dataclass-digest"
    assert latest is not None
    assert getattr(latest, "body", None) == "dataclass-digest"


def test_session_clear_slice_removes_entire_digest_slice() -> None:
    session = Session()
    set_workspace_digest(session, "workspace-digest", "value")

    session.clear_slice(WorkspaceDigest)

    assert latest_workspace_digest(session, "workspace-digest") is None


def test_optimize_handles_string_output() -> None:
    adapter = _RecordingAdapter(mode="string")
    prompt = _build_prompt()
    session = Session()

    result = adapter.optimize(prompt, session=session)

    assert result.digest == "string-digest"


def test_optimize_falls_back_to_text() -> None:
    adapter = _RecordingAdapter(mode="text")
    prompt = _build_prompt()
    session = Session()

    result = adapter.optimize(prompt, session=session)

    assert result.digest == "text-digest"


def test_optimize_requires_digest_content() -> None:
    adapter = _RecordingAdapter(mode="none")
    prompt = _build_prompt()
    session = Session()

    with pytest.raises(PromptEvaluationError):
        adapter.optimize(prompt, session=session)


def test_optimize_updates_global_overrides() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    overrides_store = _RecordingOverridesStore()
    prompt = _build_prompt()
    session = Session()

    result = adapter.optimize(
        prompt,
        store_scope=OptimizationScope.GLOBAL,
        overrides_store=overrides_store,
        overrides_tag="tag",
        session=session,
    )

    assert overrides_store.calls
    recorded_prompt, tag, path, body = overrides_store.calls[0]
    assert recorded_prompt is prompt
    assert tag == "tag"
    assert path[-1] == "workspace-digest"
    assert body == result.digest
    assert latest_workspace_digest(session, "workspace-digest") is None


def test_optimize_global_scope_clears_existing_session_digest() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    overrides_store = _RecordingOverridesStore()
    prompt = _build_prompt()
    session = Session()
    _ = set_workspace_digest(session, "workspace-digest", "stale")

    _ = adapter.optimize(
        prompt,
        store_scope=OptimizationScope.GLOBAL,
        overrides_store=overrides_store,
        overrides_tag="tag",
        session=session,
    )

    assert latest_workspace_digest(session, "workspace-digest") is None


def test_optimize_requires_overrides_inputs_for_global_scope() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    prompt = _build_prompt()

    with pytest.raises(PromptEvaluationError):
        adapter.optimize(
            prompt,
            store_scope=OptimizationScope.GLOBAL,
            overrides_tag="tag",
            session=Session(),
        )


def test_optimize_missing_overrides_inputs_preserves_session_digest() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    prompt = _build_prompt()
    session = Session()
    _ = set_workspace_digest(session, "workspace-digest", "existing")

    with pytest.raises(PromptEvaluationError):
        adapter.optimize(
            prompt,
            store_scope=OptimizationScope.GLOBAL,
            overrides_store=None,
            overrides_tag="tag",
            session=session,
        )

    latest = latest_workspace_digest(session, "workspace-digest")
    assert latest is not None
    assert getattr(latest, "body", None) == "existing"


def test_optimize_requires_workspace_section() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    digest_only_prompt = Prompt[_PromptOutput](
        ns="tests",
        key="opt",
        sections=(WorkspaceDigestSection(session=Session()),),
    )

    with pytest.raises(PromptEvaluationError):
        adapter.optimize(digest_only_prompt, session=Session())


def test_optimize_requires_digest_section() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    workspace_only_prompt = Prompt[_PromptOutput](
        ns="tests",
        key="opt",
        sections=(VfsToolsSection(session=Session()),),
    )

    with pytest.raises(PromptEvaluationError):
        adapter.optimize(workspace_only_prompt, session=Session())


def test_optimize_validates_digest_section_type() -> None:
    @dataclass(slots=True)
    class _Params:
        value: str = "v"

    adapter = _RecordingAdapter(mode="dataclass")
    workspace = VfsToolsSection(session=Session())
    fake_digest = MarkdownSection[_Params](
        title="Digest",
        template="${value}",
        key="workspace-digest",
        default_params=_Params(),
    )
    prompt = Prompt[_PromptOutput](
        ns="tests", key="opt", sections=(workspace, fake_digest)
    )

    with pytest.raises(PromptEvaluationError):
        adapter.optimize(prompt, session=Session())


def test_find_section_path_raises_for_missing_section() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    prompt = _build_prompt()

    with pytest.raises(PromptEvaluationError):
        adapter._find_section_path(prompt, "missing")


def test_optimize_uses_isolated_session() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    prompt = _build_prompt()
    outer_session = Session()

    _ = adapter.optimize(prompt, session=outer_session)

    assert adapter.sessions
    inner_session = adapter.sessions[0]
    assert isinstance(inner_session, Session)
    assert inner_session is not outer_session
    assert adapter.buses[0] is inner_session.event_bus


def test_optimize_accepts_provided_session() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    prompt = _build_prompt()
    outer_session = Session()
    provided_session = Session()

    _ = adapter.optimize(
        prompt,
        session=outer_session,
        optimization_session=provided_session,
    )

    assert adapter.sessions
    assert adapter.sessions[0] is provided_session
    assert adapter.buses[0] is provided_session.event_bus
