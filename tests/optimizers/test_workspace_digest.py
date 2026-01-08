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
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
)
from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer
from weakincentives.contrib.tools.digests import (
    WorkspaceDigest,
    WorkspaceDigestSection,
    latest_workspace_digest,
    set_workspace_digest,
)
from weakincentives.contrib.tools.vfs import VfsToolsSection
from weakincentives.deadlines import Deadline
from weakincentives.optimizers import (
    OptimizationContext,
    OptimizerConfig,
    PersistenceScope,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.prompt.overrides import (
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionOverride,
    TaskExampleOverride,
    ToolOverride,
)
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import ToolInvoked
from weakincentives.runtime.session import Session, SessionProtocol


@dataclass(slots=True, frozen=True)
class _FakeOptimizationOutput:
    summary: str
    digest: str


@dataclass(slots=True, frozen=True)
class _PromptOutput:
    summary: str = "ok"


@dataclass(slots=True, frozen=True)
class _ToolEventParams:
    value: str = "param"


class _RecordingOverridesStore(PromptOverridesStore):
    def __init__(self) -> None:
        self.calls: list[tuple[PromptTemplate[Any], str, tuple[str, ...], str]] = []

    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        self.calls.append((cast(PromptTemplate[Any], descriptor), tag, (), "resolve"))
        return None

    def upsert(
        self,
        descriptor: PromptLike,
        override: PromptOverride,
    ) -> PromptOverride:
        self.calls.append((cast(PromptTemplate[Any], descriptor), "", (), "upsert"))
        return override

    def delete(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None:
        self.calls.append(
            (cast(PromptTemplate[Any], object()), tag, (ns, prompt_key), "delete")
        )

    def store(
        self,
        descriptor: PromptDescriptor,
        override: SectionOverride | ToolOverride | TaskExampleOverride,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        if isinstance(override, SectionOverride):
            self.calls.append(
                (
                    cast(PromptTemplate[Any], descriptor),
                    tag,
                    override.path,
                    override.body,
                )
            )
        return cast(PromptOverride, object())

    def seed(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        self.calls.append((cast(PromptTemplate[Any], prompt), tag, (), "seed"))
        return cast(PromptOverride, object())


class _FailingSeedOverridesStore(_RecordingOverridesStore):
    def seed(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        self.calls.append((cast(PromptTemplate[Any], prompt), tag, (), "seed"))
        raise PromptOverridesError("seed failed")


class _RecordingAdapter(ProviderAdapter):
    def __init__(self, *, mode: str, emit_tool_event: bool = False) -> None:
        self.mode = mode
        self.rendered_prompts: list[Prompt[Any]] = []
        self.sessions: list[SessionProtocol] = []
        self._emit_tool_event = emit_tool_event

    @override
    def evaluate(
        self,
        prompt: Prompt[Any],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
    ) -> PromptResponse[Any]:
        del deadline
        prompt_name = prompt.name or prompt.key
        self.rendered_prompts.append(prompt)
        self.sessions.append(session)

        if self._emit_tool_event:
            event = ToolInvoked(
                prompt_name=prompt_name,
                adapter=OPENAI_ADAPTER_NAME,
                name="optimize-tool",
                params=_ToolEventParams(),
                result=ToolResult.ok(None, message="ok"),
                session_id=getattr(session, "session_id", None) if session else None,
                created_at=datetime.now(UTC),
            )
            session.dispatcher.dispatch(event)

        digest_value = f"{self.mode}-digest"
        summary_value = f"{self.mode}-summary"
        if self.mode == "dataclass":
            output: Any = _FakeOptimizationOutput(
                summary=summary_value, digest=digest_value
            )
            text: str | None = None
        elif self.mode == "string":
            output = digest_value
            text = None
        elif self.mode == "text":
            output = None
            text = digest_value
        else:
            output = None
            text = None

        return PromptResponse(
            prompt_name=prompt.name or prompt.key,
            text=text,
            output=output,
        )


def _build_prompt() -> PromptTemplate[_PromptOutput]:
    session = Session()
    workspace = VfsToolsSection(session=session)
    digest = WorkspaceDigestSection(session=session)
    return PromptTemplate[_PromptOutput](
        ns="tests", key="opt", sections=(workspace, digest)
    )


def _create_optimizer(
    adapter: ProviderAdapter[Any],
    *,
    store_scope: PersistenceScope = PersistenceScope.SESSION,
    overrides_store: PromptOverridesStore | None = None,
    overrides_tag: str = "latest",
    accepts_overrides: bool = True,
    optimization_session: Session | None = None,
) -> WorkspaceDigestOptimizer:
    session = Session()
    context = OptimizationContext(
        adapter=adapter,
        dispatcher=session.dispatcher,
        overrides_store=overrides_store,
        overrides_tag=overrides_tag,
        optimization_session=optimization_session,
    )
    config = OptimizerConfig(accepts_overrides=accepts_overrides)
    return WorkspaceDigestOptimizer(context, config=config, store_scope=store_scope)


def test_optimize_persists_digest_from_output() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    optimizer = _create_optimizer(adapter)
    prompt = Prompt(_build_prompt())
    session = Session()

    result = optimizer.optimize(prompt, session=session)

    latest = latest_workspace_digest(session, "workspace-digest")
    assert result.digest == "dataclass-digest"
    assert latest is not None
    assert getattr(latest, "body", None) == "dataclass-digest"
    assert getattr(latest, "summary", None) == "dataclass-summary"


def test_session_clear_slice_removes_entire_digest_slice() -> None:
    session = Session()
    set_workspace_digest(session, "workspace-digest", "value")

    session[WorkspaceDigest].clear()

    assert latest_workspace_digest(session, "workspace-digest") is None


def test_optimize_handles_string_output() -> None:
    adapter = _RecordingAdapter(mode="string")
    optimizer = _create_optimizer(adapter)
    prompt = Prompt(_build_prompt())
    session = Session()

    result = optimizer.optimize(prompt, session=session)

    assert result.digest == "string-digest"


def test_optimize_falls_back_to_text() -> None:
    adapter = _RecordingAdapter(mode="text")
    optimizer = _create_optimizer(adapter)
    prompt = Prompt(_build_prompt())
    session = Session()

    result = optimizer.optimize(prompt, session=session)

    assert result.digest == "text-digest"


def test_optimize_requires_digest_content() -> None:
    adapter = _RecordingAdapter(mode="none")
    optimizer = _create_optimizer(adapter)
    prompt = Prompt(_build_prompt())
    session = Session()

    with pytest.raises(PromptEvaluationError):
        optimizer.optimize(prompt, session=session)


def test_optimize_updates_global_overrides() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    overrides_store = _RecordingOverridesStore()
    optimizer = _create_optimizer(
        adapter,
        store_scope=PersistenceScope.GLOBAL,
        overrides_store=overrides_store,
        overrides_tag="tag",
    )
    prompt = Prompt(_build_prompt())
    session = Session()

    result = optimizer.optimize(prompt, session=session)

    assert overrides_store.calls
    recorded_descriptor, tag, path, body = next(
        call
        for call in overrides_store.calls
        if call[3] not in {"seed", "resolve", "upsert", "delete"}
    )
    # store() now receives a PromptDescriptor, not the Prompt itself
    assert isinstance(recorded_descriptor, PromptDescriptor)
    assert recorded_descriptor.ns == prompt.ns
    assert recorded_descriptor.key == prompt.key
    assert tag == "tag"
    assert path[-1] == "workspace-digest"
    assert body == result.digest
    assert latest_workspace_digest(session, "workspace-digest") is None


def test_optimize_global_scope_clears_existing_session_digest() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    overrides_store = _RecordingOverridesStore()
    optimizer = _create_optimizer(
        adapter,
        store_scope=PersistenceScope.GLOBAL,
        overrides_store=overrides_store,
        overrides_tag="tag",
    )
    prompt = Prompt(_build_prompt())
    session = Session()
    _ = set_workspace_digest(session, "workspace-digest", "stale")

    _ = optimizer.optimize(prompt, session=session)

    assert latest_workspace_digest(session, "workspace-digest") is None


def test_optimize_requires_overrides_inputs_for_global_scope() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    optimizer = _create_optimizer(
        adapter,
        store_scope=PersistenceScope.GLOBAL,
        overrides_store=None,
        overrides_tag="tag",
    )
    prompt = Prompt(_build_prompt())

    with pytest.raises(PromptEvaluationError):
        optimizer.optimize(prompt, session=Session())


def test_optimize_missing_overrides_inputs_preserves_session_digest() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    optimizer = _create_optimizer(
        adapter,
        store_scope=PersistenceScope.GLOBAL,
        overrides_store=None,
        overrides_tag="tag",
    )
    prompt = Prompt(_build_prompt())
    session = Session()
    _ = set_workspace_digest(session, "workspace-digest", "existing")

    with pytest.raises(PromptEvaluationError):
        optimizer.optimize(prompt, session=session)

    latest = latest_workspace_digest(session, "workspace-digest")
    assert latest is not None
    assert getattr(latest, "body", None) == "existing"


def test_optimize_seeds_internal_prompt_overrides() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    overrides_store = _RecordingOverridesStore()
    optimizer = _create_optimizer(adapter, overrides_store=overrides_store)
    prompt = Prompt(_build_prompt())

    _ = optimizer.optimize(prompt, session=Session())

    assert any(call[3] == "seed" for call in overrides_store.calls)


def test_optimize_raises_when_internal_prompt_seeding_fails() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    overrides_store = _FailingSeedOverridesStore()
    optimizer = _create_optimizer(adapter, overrides_store=overrides_store)
    prompt = Prompt(_build_prompt())

    with pytest.raises(PromptEvaluationError):
        optimizer.optimize(prompt, session=Session())


def test_optimize_skips_overrides_when_disabled() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    overrides_store = _RecordingOverridesStore()
    optimizer = _create_optimizer(
        adapter, overrides_store=overrides_store, accepts_overrides=False
    )
    prompt = Prompt(_build_prompt())

    _ = optimizer.optimize(prompt, session=Session())

    assert not overrides_store.calls


def test_optimize_requires_workspace_section() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    optimizer = _create_optimizer(adapter)
    digest_only_prompt = PromptTemplate[_PromptOutput](
        ns="tests",
        key="opt",
        sections=(WorkspaceDigestSection(session=Session()),),
    )

    with pytest.raises(PromptEvaluationError):
        optimizer.optimize(Prompt(digest_only_prompt), session=Session())


def test_optimize_requires_digest_section() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    optimizer = _create_optimizer(adapter)
    workspace_only_prompt = PromptTemplate[_PromptOutput](
        ns="tests",
        key="opt",
        sections=(VfsToolsSection(session=Session()),),
    )

    with pytest.raises(PromptEvaluationError):
        optimizer.optimize(Prompt(workspace_only_prompt), session=Session())


def test_optimize_validates_digest_section_type() -> None:
    @dataclass(slots=True)
    class _Params:
        value: str = "v"

    adapter = _RecordingAdapter(mode="dataclass")
    optimizer = _create_optimizer(adapter)
    workspace = VfsToolsSection(session=Session())
    fake_digest = MarkdownSection[_Params](
        title="Digest",
        template="${value}",
        key="workspace-digest",
        default_params=_Params(),
    )
    prompt = PromptTemplate[_PromptOutput](
        ns="tests", key="opt", sections=(workspace, fake_digest)
    )

    with pytest.raises(PromptEvaluationError):
        optimizer.optimize(Prompt(prompt), session=Session())


def test_find_section_path_raises_for_missing_section() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    optimizer = _create_optimizer(adapter)
    prompt = Prompt(_build_prompt())

    with pytest.raises(PromptEvaluationError):
        optimizer._find_section_path(prompt, "missing")


def test_optimize_uses_isolated_session() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    optimizer = _create_optimizer(adapter)
    prompt = Prompt(_build_prompt())
    outer_session = Session()

    _ = optimizer.optimize(prompt, session=outer_session)

    assert adapter.sessions
    inner_session = adapter.sessions[0]
    assert isinstance(inner_session, Session)
    assert inner_session is not outer_session


def test_optimize_accepts_provided_session() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    provided_session = Session()
    optimizer = _create_optimizer(adapter, optimization_session=provided_session)
    prompt = Prompt(_build_prompt())
    outer_session = Session()

    _ = optimizer.optimize(prompt, session=outer_session)

    assert adapter.sessions
    assert adapter.sessions[0] is provided_session


def test_workspace_digest_result_has_correct_scope() -> None:
    adapter = _RecordingAdapter(mode="dataclass")
    optimizer = _create_optimizer(adapter, store_scope=PersistenceScope.SESSION)
    prompt = Prompt(_build_prompt())
    session = Session()

    result = optimizer.optimize(prompt, session=session)

    assert result.scope == PersistenceScope.SESSION
    assert result.section_key == "workspace-digest"


def test_optimizer_config_defaults() -> None:
    config = OptimizerConfig()
    assert config.accepts_overrides is True


def test_optimize_handles_non_string_digest_attribute() -> None:
    """Test that optimizer handles output with non-string digest attribute."""

    @dataclass(slots=True, frozen=True)
    class _OutputWithNonStringDigest:
        digest: int  # Not a string!

    class _NonStringDigestAdapter(ProviderAdapter):
        @override
        def evaluate(
            self,
            prompt: Prompt[Any],
            *,
            session: SessionProtocol,
            deadline: Deadline | None = None,
        ) -> PromptResponse[Any]:
            del deadline
            # Return output with digest attribute that's not a string
            return PromptResponse(
                prompt_name=prompt.name or prompt.key,
                text="fallback-text",
                output=_OutputWithNonStringDigest(digest=42),
            )

    adapter = _NonStringDigestAdapter()
    optimizer = _create_optimizer(adapter)
    prompt = Prompt(_build_prompt())
    session = Session()

    result = optimizer.optimize(prompt, session=session)

    # Should fall back to text since digest attribute is not a string
    assert result.digest == "fallback-text"


def test_latest_workspace_digest_handles_section_key_mismatch() -> None:
    """Test that latest_workspace_digest handles mismatched section_key."""
    session = Session()

    # Create a digest with a different section_key
    _ = set_workspace_digest(session, "other-digest", "value1")
    _ = set_workspace_digest(session, "workspace-digest", "value2")

    # Should find the matching one
    result = latest_workspace_digest(session, "workspace-digest")
    assert result is not None
    assert getattr(result, "body", None) == "value2"

    # Should return None when no match
    result = latest_workspace_digest(session, "non-existent")
    assert result is None


def test_find_section_hash_raises_for_missing_path() -> None:
    """Test that _find_section_hash raises for unknown paths."""
    adapter = _RecordingAdapter(mode="dataclass")
    optimizer = _create_optimizer(adapter)
    prompt = Prompt(_build_prompt())
    descriptor = PromptDescriptor.from_prompt(prompt)

    with pytest.raises(PromptOverridesError, match="Section hash not found"):
        optimizer._find_section_hash(descriptor, ("nonexistent", "path"))
