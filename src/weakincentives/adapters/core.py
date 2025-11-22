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

"""Core adapter interfaces shared across provider integrations."""

from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypeVar, cast

from ..deadlines import Deadline
from ..prompt import MarkdownSection
from ..prompt._types import SupportsDataclass
from ..prompt.overrides import PromptLike, PromptOverridesStore
from ..prompt.prompt import Prompt
from ..prompt.section import Section
from ..prompt.workspace_digest import WorkspaceDigestSection
from ..runtime.events._types import EventBus, EventHandler, ToolInvoked
from ..runtime.session import Session
from ..runtime.session.protocols import SessionProtocol
from ..tools.planning import PlanningStrategy, PlanningToolsSection
from ..tools.podman import PodmanSandboxSection
from ..tools.vfs import VfsToolsSection

OutputT = TypeVar("OutputT")


@dataclass(slots=True)
class PromptResponse[OutputT]:
    """Structured result emitted by an adapter evaluation."""

    prompt_name: str
    text: str | None
    output: OutputT | None
    tool_results: tuple[ToolInvoked, ...]
    provider_payload: dict[str, Any] | None = None


class OptimizationScope(Enum):
    """Control where optimized digests are persisted."""

    SESSION = "session"
    GLOBAL = "global"


@dataclass(slots=True)
class OptimizationResult:
    """Capture the outcome of an optimization run."""

    response: PromptResponse[Any]
    digest: str
    scope: OptimizationScope
    section_key: str


@dataclass(slots=True, frozen=True)
class _OptimizationGoalParams:
    """Placeholder params for the optimization goal section."""


@dataclass(slots=True, frozen=True)
class _OptimizationExpectationsParams:
    """Placeholder params for the optimization expectations section."""


@dataclass(slots=True, frozen=True)
class _OptimizationResponse:
    """Structured response emitted by the workspace digest optimization prompt."""

    digest: str


class ProviderAdapter(ABC):
    """Abstract base class describing the synchronous adapter contract."""

    @classmethod
    def __class_getitem__(cls, _: object) -> type[ProviderAdapter[Any]]:
        return cls

    @abstractmethod
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str = "latest",
    ) -> PromptResponse[OutputT]:
        """Evaluate the prompt and return a structured response."""

        ...

    def optimize(
        self,
        prompt: Prompt[OutputT],
        *,
        store_scope: OptimizationScope = OptimizationScope.SESSION,
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str | None = None,
        session: SessionProtocol,
        bus_subscribers: Sequence[tuple[type[object], EventHandler]] | None = None,
    ) -> OptimizationResult:
        """Optimize the workspace digest for the provided prompt."""

        prompt_name = prompt.name or prompt.key
        outer_session = session
        optimization_session = Session()
        digest_section = self._require_workspace_digest_section(
            prompt, prompt_name=prompt_name
        )
        workspace_section = self._resolve_workspace_section(prompt, prompt_name)

        safe_workspace = self._clone_section_for_session(
            workspace_section, session=optimization_session
        )

        optimization_prompt = Prompt[_OptimizationResponse](
            ns=f"{prompt.ns}.optimization",
            key=f"{prompt.key}-workspace-digest",
            name=(f"{prompt.name}_workspace_digest" if prompt.name else None),
            sections=(
                MarkdownSection[_OptimizationGoalParams](
                    title="Optimization Goal",
                    template=(
                        "Summarize the workspace so future prompts can rely on a cached digest."
                    ),
                    key="optimization-goal",
                ),
                MarkdownSection[_OptimizationExpectationsParams](
                    title="Expectations",
                    template=textwrap.dedent(
                        """
                        Explore README/docs/workflow files first. Capture build/test commands,
                        dependency managers, and watchouts. Keep the digest task agnostic.
                        """
                    ).strip(),
                    key="optimization-expectations",
                ),
                PlanningToolsSection(
                    session=optimization_session,
                    strategy=PlanningStrategy.GOAL_DECOMPOSE_ROUTE_SYNTHESISE,
                ),
                safe_workspace,
            ),
        )

        if bus_subscribers:
            for event_type, handler in bus_subscribers:
                optimization_session.event_bus.subscribe(event_type, handler)

        response = self.evaluate(
            optimization_prompt,
            parse_output=True,
            bus=optimization_session.event_bus,
            session=optimization_session,
            overrides_store=overrides_store,
            overrides_tag=overrides_tag or "latest",
        )

        digest = self._extract_digest(response=response, prompt_name=prompt_name)

        if store_scope is OptimizationScope.SESSION:
            _ = outer_session.workspace_digest.set(digest_section.key, digest)

        if store_scope is OptimizationScope.GLOBAL:
            if overrides_store is None or overrides_tag is None:
                message = "Global scope requires overrides_store and overrides_tag."
                raise PromptEvaluationError(
                    message,
                    prompt_name=prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_REQUEST,
                )
            section_path = self._find_section_path(prompt, digest_section.key)
            _ = overrides_store.set_section_override(
                cast(PromptLike, prompt),
                tag=overrides_tag,
                path=section_path,
                body=digest,
            )

        return OptimizationResult(
            response=response,
            digest=digest,
            scope=store_scope,
            section_key=digest_section.key,
        )

    @staticmethod
    def _clone_section_for_session(
        section: Section[SupportsDataclass], *, session: SessionProtocol
    ) -> Section[SupportsDataclass]:
        clone = getattr(section, "clone", None)
        if callable(clone):  # pragma: no cover - helper for optimize
            return cast(
                Section[SupportsDataclass],
                clone(session=session, bus=session.event_bus),
            )
        return section

    def _resolve_workspace_section(
        self, prompt: Prompt[object], prompt_name: str
    ) -> Section[SupportsDataclass]:
        candidates: tuple[object, ...] = (
            PodmanSandboxSection,
            VfsToolsSection,
            "podman.shell",
            "vfs.tools",
        )
        try:
            return prompt.find_section(candidates)
        except KeyError as error:  # pragma: no cover - defensive
            raise PromptEvaluationError(
                "Workspace section required for optimization.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ) from error

    def _require_workspace_digest_section(
        self, prompt: Prompt[object], *, prompt_name: str
    ) -> WorkspaceDigestSection:
        try:
            section = prompt.find_section((WorkspaceDigestSection, "workspace-digest"))
        except KeyError as error:  # pragma: no cover - defensive
            raise PromptEvaluationError(
                "Workspace digest section required for optimization.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ) from error
        if not isinstance(section, WorkspaceDigestSection):
            message = "Workspace digest section has unexpected type."
            raise PromptEvaluationError(
                message,
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            )
        return section

    def _find_section_path(
        self, prompt: Prompt[object], section_key: str
    ) -> tuple[str, ...]:
        for node in prompt.sections:
            if node.section.key == section_key:
                return node.path
        message = f"Section path not found for key: {section_key}"
        raise PromptEvaluationError(
            message,
            prompt_name=prompt.name or prompt.key,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
        )

    def _extract_digest(
        self, *, response: PromptResponse[Any], prompt_name: str
    ) -> str:
        digest: str | None = None
        if isinstance(response.output, str):
            digest = response.output
        elif response.output is not None:
            candidate = getattr(response.output, "digest", None)
            if isinstance(candidate, str):
                digest = candidate
        if digest is None and response.text:
            digest = response.text
        if digest is None:
            raise PromptEvaluationError(
                "Optimization did not return digest content.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_RESPONSE,
            )
        return digest.strip()


class PromptEvaluationError(RuntimeError):
    """Raised when evaluation against a provider fails."""

    def __init__(
        self,
        message: str,
        *,
        prompt_name: str,
        phase: PromptEvaluationPhase,
        provider_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.prompt_name = prompt_name
        self.phase: PromptEvaluationPhase = phase
        self.provider_payload = provider_payload


PromptEvaluationPhase = Literal["request", "response", "tool"]
"""Phases where a prompt evaluation error can occur."""

PROMPT_EVALUATION_PHASE_REQUEST: PromptEvaluationPhase = "request"
"""Prompt evaluation failed while issuing the provider request."""

PROMPT_EVALUATION_PHASE_RESPONSE: PromptEvaluationPhase = "response"
"""Prompt evaluation failed while handling the provider response."""

PROMPT_EVALUATION_PHASE_TOOL: PromptEvaluationPhase = "tool"
"""Prompt evaluation failed while handling a tool invocation."""


__all__ = [
    "PROMPT_EVALUATION_PHASE_REQUEST",
    "PROMPT_EVALUATION_PHASE_RESPONSE",
    "PROMPT_EVALUATION_PHASE_TOOL",
    "OptimizationResult",
    "OptimizationScope",
    "PromptEvaluationError",
    "PromptEvaluationPhase",
    "PromptResponse",
    "ProviderAdapter",
    "SessionProtocol",
]
