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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypeVar, assert_never, cast

from ..deadlines import Deadline
from ..prompt import MarkdownSection
from ..prompt._types import SupportsDataclass
from ..prompt.overrides import PromptLike, PromptOverridesStore
from ..prompt.prompt import Prompt
from ..prompt.section import Section
from ..runtime.events._types import EventBus
from ..runtime.session import Session
from ..runtime.session.protocols import SessionProtocol
from ..tools.asteval import AstevalSection
from ..tools.digests import (
    WorkspaceDigestSection,
    clear_workspace_digest,
    set_workspace_digest,
)
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
        optimization_session: Session | None = None,
    ) -> OptimizationResult:
        """Optimize the workspace digest for the provided prompt."""

        prompt_name = prompt.name or prompt.key
        outer_session = session
        inner_session = optimization_session or Session(
            tags={
                "scope": "workspace_digest_optimization",
                "prompt": prompt_name,
            }
        )
        digest_section = self._require_workspace_digest_section(
            prompt, prompt_name=prompt_name
        )
        workspace_section = self._resolve_workspace_section(prompt, prompt_name)

        safe_workspace = self._clone_section(
            workspace_section,
            session=inner_session,
        )
        tool_sections = self._resolve_tool_sections(prompt)
        safe_tools = tuple(
            self._clone_section(section, session=inner_session)
            for section in tool_sections
        )
        safe_workspace_digest = digest_section.clone(
            session=inner_session,
            bus=inner_session.event_bus,
        )

        optimization_prompt = Prompt[_OptimizationResponse](
            ns=f"{prompt.ns}.optimization",
            key=f"{prompt.key}-workspace-digest",
            name=(f"{prompt.name}_workspace_digest" if prompt.name else None),
            sections=(
                MarkdownSection(
                    title="Optimization Goal",
                    template=(
                        "Summarize the workspace so future prompts can rely on a cached digest."
                    ),
                    key="optimization-goal",
                ),
                MarkdownSection(
                    title="Expectations",
                    template=textwrap.dedent(
                        """
                        Explore README/docs/workflow files first. Capture build/test commands,
                        dependency managers, and watchouts. Keep the digest task agnostic.
                        Capture command exec tools (asteval, Podman exec) plus env caps/
                        versions/libs. Keep it dense.
                        """
                    ).strip(),
                    key="optimization-expectations",
                ),
                PlanningToolsSection(
                    session=inner_session,
                    strategy=PlanningStrategy.GOAL_DECOMPOSE_ROUTE_SYNTHESISE,
                ),
                *safe_tools,
                safe_workspace,
                safe_workspace_digest,
            ),
        )

        response = self.evaluate(
            optimization_prompt,
            parse_output=True,
            bus=inner_session.event_bus,
            session=inner_session,
            overrides_store=overrides_store,
            overrides_tag=overrides_tag or "latest",
        )

        digest = self._extract_digest(response=response, prompt_name=prompt_name)

        match store_scope:
            case OptimizationScope.SESSION:
                _ = set_workspace_digest(outer_session, digest_section.key, digest)

            case OptimizationScope.GLOBAL:
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
                clear_workspace_digest(outer_session, digest_section.key)

            case _:
                assert_never(store_scope)  # pyright: ignore[reportUnreachable]

        return OptimizationResult(
            response=response,
            digest=digest,
            scope=store_scope,
            section_key=digest_section.key,
        )

    def _resolve_workspace_section(
        self, prompt: Prompt[object], prompt_name: str
    ) -> Section[SupportsDataclass]:
        try:
            return prompt.find_section((PodmanSandboxSection, VfsToolsSection))
        except KeyError as error:  # pragma: no cover - defensive
            raise PromptEvaluationError(
                "Workspace section required for optimization.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ) from error

    def _resolve_tool_sections(
        self, prompt: Prompt[object]
    ) -> tuple[Section[SupportsDataclass], ...]:
        sections: list[Section[SupportsDataclass]] = []
        for section_type in (AstevalSection,):
            try:
                sections.append(prompt.find_section(section_type))
            except KeyError:
                continue
        return tuple(sections)

    def _clone_section(
        self, section: Section[SupportsDataclass], *, session: Session
    ) -> Section[SupportsDataclass]:
        return section.clone(session=session, bus=session.event_bus)

    def _require_workspace_digest_section(
        self, prompt: Prompt[object], *, prompt_name: str
    ) -> WorkspaceDigestSection:
        try:
            section = prompt.find_section(WorkspaceDigestSection)
        except KeyError as error:  # pragma: no cover - defensive
            raise PromptEvaluationError(
                "Workspace digest section required for optimization.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ) from error
        return cast(WorkspaceDigestSection, section)

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
