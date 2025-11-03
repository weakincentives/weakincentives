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

"""Prompt subagent dispatch tool implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from ..events import EventBus, InProcessEventBus, ToolInvoked
from ..prompt import MarkdownSection, Tool, ToolResult
from ..session import Session, SnapshotRestoreError, SnapshotSerializationError
from .errors import ToolValidationError

SubagentMode = Literal["plan_step", "ad_hoc"]

_ASCII: Final[str] = "ascii"
_MAX_INSTRUCTIONS_LENGTH: Final[int] = 2_000
_MAX_ARTIFACT_LABEL_LENGTH: Final[int] = 160


@dataclass(slots=True, frozen=True)
class DispatchSubagent:
    """Parameters for launching a child prompt session."""

    mode: SubagentMode
    prompt_ns: str
    prompt_key: str
    instructions: str
    expected_artifacts: tuple[str, ...] = field(default_factory=tuple)
    plan_step_id: str | None = None


@dataclass(slots=True, frozen=True)
class DispatchSubagentResult:
    """Structured result returned to the parent prompt."""

    message_summary: str
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    tools_used: tuple[str, ...] = field(default_factory=tuple)


class DispatchSubagentError(RuntimeError):
    """Raised when subagent execution fails for runtime reasons."""


@dataclass(slots=True, frozen=True)
class _PromptSubagentSectionParams:
    """Placeholder params container for the prompt guidance section."""

    pass


_PROMPT_SECTION_TEMPLATE: Final[str] = (
    "Spawn a focused subagent when work requires deep research or a detailed draft,"
    " or to execute a specific plan step.\n"
    "- Provide concise ASCII instructions (<=2000 chars) describing the desired"
    " outcome.\n"
    "- Label the delegated task with a namespace/key so downstream tooling can"
    " categorize the run.\n"
    "- Enumerate any expected artifacts so the subagent can confirm delivery.\n"
    "- The child session inherits every tool available to this prompt but runs in"
    " isolation.\n"
    "- Only the summary message and declared artifacts return to the parent session."
)


class SubagentRunner(Protocol):
    """Callable responsible for evaluating the child prompt."""

    def __call__(
        self,
        *,
        session: Session,
        bus: EventBus,
        params: DispatchSubagent,
    ) -> DispatchSubagentResult: ...


class PromptSubagentToolsSection(MarkdownSection[_PromptSubagentSectionParams]):
    """Prompt section exposing the subagent dispatch tool."""

    def __init__(self, *, session: Session, runner: SubagentRunner) -> None:
        self._session = session
        suite = _PromptSubagentToolSuite(session=session, runner=runner)
        tools = (
            Tool[DispatchSubagent, DispatchSubagentResult](
                name="dispatch_subagent",
                description="Run a prompt in an isolated child session.",
                handler=suite.dispatch_subagent,
            ),
        )
        super().__init__(
            title="Prompt Subagent Dispatch",
            key="prompt-subagent.tools",
            template=_PROMPT_SECTION_TEMPLATE,
            default_params=_PromptSubagentSectionParams(),
            tools=tools,
        )


@dataclass(slots=True, frozen=True)
class _NormalizedDispatch:
    mode: SubagentMode
    prompt_ns: str
    prompt_key: str
    instructions: str
    expected_artifacts: tuple[str, ...]
    plan_step_id: str | None


class _PromptSubagentToolSuite:
    """Tool handlers bound to a parent session instance."""

    def __init__(self, *, session: Session, runner: SubagentRunner) -> None:
        self._session = session
        self._runner = runner

    def dispatch_subagent(
        self, params: DispatchSubagent
    ) -> ToolResult[DispatchSubagentResult]:
        normalized = _normalize_params(params)
        normalized_params = DispatchSubagent(
            mode=normalized.mode,
            prompt_ns=normalized.prompt_ns,
            prompt_key=normalized.prompt_key,
            instructions=normalized.instructions,
            expected_artifacts=normalized.expected_artifacts,
            plan_step_id=normalized.plan_step_id,
        )

        try:
            snapshot = self._session.snapshot()
        except SnapshotSerializationError as error:
            raise ToolValidationError(
                "Unable to capture parent session state."
            ) from error

        child_bus = InProcessEventBus()
        child_session = Session(
            bus=child_bus,
            session_id=self._session.session_id,
            created_at=self._session.created_at,
        )
        _clone_session_structure(self._session, child_session)

        try:
            child_session.rollback(snapshot)
        except SnapshotRestoreError as error:
            raise ToolValidationError(
                "Unable to hydrate child session from snapshot."
            ) from error

        recorder = _ToolUsageRecorder()
        child_bus.subscribe(ToolInvoked, recorder.handle)

        try:
            result = self._runner(
                session=child_session,
                bus=child_bus,
                params=normalized_params,
            )
        except ToolValidationError:
            raise
        except DispatchSubagentError as error:
            message = f"Subagent execution failed: {error}"
            return ToolResult(message=message, value=None, success=False)
        except Exception as error:  # noqa: BLE001
            message = f"Subagent execution failed: {error}"
            return ToolResult(message=message, value=None, success=False)

        if not isinstance(result, DispatchSubagentResult):
            message = "Subagent execution failed: invalid result payload"
            return ToolResult(message=message, value=None, success=False)

        artifacts = tuple(result.artifacts)
        final_result = DispatchSubagentResult(
            message_summary=result.message_summary,
            artifacts=artifacts,
            tools_used=recorder.observed(),
        )
        return ToolResult(
            message=final_result.message_summary,
            value=final_result,
        )


class _ToolUsageRecorder:
    """Capture tool names observed during child execution."""

    def __init__(self) -> None:
        self._names: list[str] = []

    def handle(self, event: object) -> None:
        if isinstance(event, ToolInvoked):
            self._names.append(event.name)

    def observed(self) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        for name in self._names:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        return tuple(ordered)


def _normalize_params(params: DispatchSubagent) -> _NormalizedDispatch:
    instructions = _normalize_text(
        params.instructions,
        field_name="instructions",
        max_length=_MAX_INSTRUCTIONS_LENGTH,
    )
    artifacts = _normalize_artifacts(params.expected_artifacts)
    plan_step_id: str | None = None
    if params.mode == "plan_step":
        plan_step_id = params.plan_step_id
        if plan_step_id is None:
            raise ToolValidationError(
                "plan_step_id is required when mode is 'plan_step'."
            )
        plan_step_id = _normalize_text(
            plan_step_id,
            field_name="plan_step_id",
        )
    return _NormalizedDispatch(
        mode=params.mode,
        prompt_ns=params.prompt_ns,
        prompt_key=params.prompt_key,
        instructions=instructions,
        expected_artifacts=artifacts,
        plan_step_id=plan_step_id,
    )


def _normalize_text(
    value: str,
    *,
    field_name: str,
    max_length: int | None = None,
) -> str:
    stripped = value.strip()
    if not stripped:
        raise ToolValidationError(f"{field_name} must be non-empty ASCII text.")
    if max_length is not None and len(stripped) > max_length:
        raise ToolValidationError(
            f"{field_name} must be <= {max_length} characters of ASCII text."
        )
    try:
        stripped.encode(_ASCII)
    except UnicodeEncodeError as error:
        raise ToolValidationError(
            f"{field_name} must contain only ASCII characters."
        ) from error
    return stripped


def _normalize_artifacts(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        normalized.append(
            _normalize_text(
                value,
                field_name="expected_artifacts",
                max_length=_MAX_ARTIFACT_LABEL_LENGTH,
            )
        )
    return tuple(normalized)


def _clone_session_structure(parent: Session, child: Session) -> None:
    parent_reducers = getattr(parent, "_reducers", {})
    for data_type, registrations in parent_reducers.items():
        for registration in registrations:
            slice_type = registration.slice_type
            kwargs: dict[str, Any] = {}
            if slice_type is not data_type:
                kwargs["slice_type"] = slice_type
            child.register_reducer(data_type, registration.reducer, **kwargs)

    parent_state = getattr(parent, "_state", {})
    for slice_type in parent_state:
        child.seed_slice(slice_type, ())


__all__ = [
    "SubagentMode",
    "DispatchSubagent",
    "DispatchSubagentResult",
    "DispatchSubagentError",
    "PromptSubagentToolsSection",
    "SubagentRunner",
]
