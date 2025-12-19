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

"""Workspace digest optimizer for generating task-agnostic summaries."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, cast, override

from ...adapters.core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PromptEvaluationError,
    PromptResponse,
)
from ...optimizers._base import BasePromptOptimizer, OptimizerConfig
from ...optimizers._context import OptimizationContext
from ...optimizers._results import PersistenceScope, WorkspaceDigestResult
from ...prompt import MarkdownSection, Prompt, PromptTemplate
from ...prompt.overrides import PromptLike, PromptOverridesError
from ...prompt.section import Section
from ...runtime.session import Session
from ...runtime.session.protocols import SessionProtocol
from ...types.dataclass import SupportsDataclass
from ..tools.asteval import AstevalSection
from ..tools.digests import (
    WorkspaceDigestSection,
    clear_workspace_digest,
    set_workspace_digest,
)
from ..tools.planning import PlanningStrategy, PlanningToolsSection
from ..tools.workspace import WorkspaceSection


@dataclass(slots=True, frozen=True)
class _OptimizationResponse:
    """Structured response emitted by the workspace digest optimization prompt."""

    digest: str


class WorkspaceDigestOptimizer(BasePromptOptimizer[object, WorkspaceDigestResult]):
    """Generate a workspace digest for prompts containing WorkspaceDigestSection.

    This optimizer composes an internal prompt that explores the mounted
    workspace and produces a task-agnostic summary. The result can be
    persisted to the session (SESSION scope) or the override store
    (GLOBAL scope).
    """

    def __init__(
        self,
        context: OptimizationContext,
        *,
        config: OptimizerConfig | None = None,
        store_scope: PersistenceScope = PersistenceScope.SESSION,
    ) -> None:
        super().__init__(context, config=config)
        self._store_scope = store_scope

    @property
    @override
    def _optimizer_scope(self) -> str:
        return "workspace_digest"

    @override
    def optimize(  # noqa: PLR0914 - keeping local clarity for optimization flow
        self,
        prompt: Prompt[object],
        *,
        session: SessionProtocol,
    ) -> WorkspaceDigestResult:
        """Generate and persist a workspace digest for the given prompt.

        Raises:
            PromptEvaluationError: If the prompt lacks required sections
                or digest extraction fails.
        """
        prompt_name = prompt.name or prompt.key
        outer_session = session
        inner_session = self._create_optimization_session(prompt)

        digest_section = self._require_workspace_digest_section(
            prompt, prompt_name=prompt_name
        )
        workspace_section = self._resolve_workspace_section(prompt, prompt_name)

        safe_workspace = self._clone_section(workspace_section, session=inner_session)
        tool_sections = self._resolve_tool_sections(prompt)
        safe_tools = tuple(
            self._clone_section(section, session=inner_session)
            for section in tool_sections
        )

        effective_store = self._context.overrides_store or prompt.overrides_store
        effective_tag = self._context.overrides_tag or prompt.overrides_tag
        if not self._config.accepts_overrides:
            effective_store = None
            effective_tag = None

        optimization_prompt_template = PromptTemplate[_OptimizationResponse](
            ns=f"{prompt.ns}.optimization",
            key=f"{prompt.key}-workspace-digest",
            name=(f"{prompt.name}_workspace_digest" if prompt.name else None),
            sections=(
                MarkdownSection(
                    title="Optimization Goal",
                    template=(
                        "Summarize the workspace so future prompts can rely on a "
                        "cached digest."
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
                    accepts_overrides=self._config.accepts_overrides,
                ),
                safe_workspace,
                *safe_tools,
            ),
        )

        normalized_tag = effective_tag or "latest"
        optimization_prompt = Prompt(
            optimization_prompt_template,
            overrides_store=effective_store,
            overrides_tag=normalized_tag,
        )
        if self._config.accepts_overrides and effective_store is not None:
            try:
                _ = effective_store.seed(
                    cast(PromptLike, optimization_prompt), tag=normalized_tag
                )
            except PromptOverridesError as exc:
                raise PromptEvaluationError(
                    "Failed to seed overrides for optimization prompt.",
                    prompt_name=prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_REQUEST,
                ) from exc

        response = self._context.adapter.evaluate(
            optimization_prompt,
            session=inner_session,
            deadline=self._context.deadline,
        )

        digest = self._extract_digest(response=response, prompt_name=prompt_name)

        if self._store_scope is PersistenceScope.SESSION:
            _ = set_workspace_digest(outer_session, digest_section.key, digest)

        if self._store_scope is PersistenceScope.GLOBAL:
            global_store = effective_store
            global_tag = normalized_tag
            if global_store is None:
                message = "Global scope requires overrides_store and overrides_tag."
                raise PromptEvaluationError(
                    message,
                    prompt_name=prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_REQUEST,
                )
            section_path = self._find_section_path(prompt, digest_section.key)
            _ = global_store.set_section_override(
                cast(PromptLike, prompt),
                tag=global_tag,
                path=section_path,
                body=digest,
            )
            clear_workspace_digest(outer_session, digest_section.key)

        return WorkspaceDigestResult(
            response=cast(PromptResponse[object], response),
            digest=digest,
            scope=self._store_scope,
            section_key=digest_section.key,
        )

    def _resolve_workspace_section(  # noqa: PLR6301
        self, prompt: Prompt[object], prompt_name: str
    ) -> Section[SupportsDataclass]:
        """Find a section implementing the WorkspaceSection protocol."""
        for node in prompt.sections:
            if isinstance(node.section, WorkspaceSection):
                return node.section
        raise PromptEvaluationError(
            "Workspace section required for optimization.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
        )

    def _resolve_tool_sections(  # noqa: PLR6301
        self, prompt: Prompt[object]
    ) -> tuple[Section[SupportsDataclass], ...]:
        sections: list[Section[SupportsDataclass]] = []
        for section_type in (AstevalSection,):
            try:
                sections.append(prompt.find_section(section_type))
            except KeyError:
                continue
        return tuple(sections)

    def _clone_section(  # noqa: PLR6301
        self, section: Section[SupportsDataclass], *, session: Session
    ) -> Section[SupportsDataclass]:
        return section.clone(session=session, bus=session.event_bus)

    def _require_workspace_digest_section(  # noqa: PLR6301
        self, prompt: Prompt[object], *, prompt_name: str
    ) -> WorkspaceDigestSection:
        try:
            section = prompt.find_section(WorkspaceDigestSection)
        except KeyError as error:
            raise PromptEvaluationError(
                "Workspace digest section required for optimization.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ) from error
        return cast(WorkspaceDigestSection, section)

    def _find_section_path(  # noqa: PLR6301
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

    def _extract_digest(  # noqa: PLR6301
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


__all__ = ["WorkspaceDigestOptimizer"]
