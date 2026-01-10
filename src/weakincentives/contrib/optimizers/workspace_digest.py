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

from weakincentives.filesystem import Filesystem
from weakincentives.prompt.protocols import WorkspaceSection

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
from ...prompt.overrides import (
    HexDigest,
    PromptDescriptor,
    PromptLike,
    PromptOverridesError,
    SectionOverride,
)
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


@dataclass(slots=True, frozen=True)
class _OptimizationResponse:
    """Structured response emitted by the workspace digest optimization prompt."""

    summary: str
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

        safe_workspace = self._clone_section(
            cast(Section[SupportsDataclass], workspace_section), session=inner_session
        )
        tool_sections = self._resolve_tool_sections(prompt)
        # Pass the workspace filesystem to tool sections so they share the same
        # filesystem instance. This ensures asteval can read/write the same files
        # that VFS tools expose. We use workspace_section.filesystem directly since
        # cloning preserves the same filesystem reference.
        shared_filesystem = workspace_section.filesystem
        safe_tools = tuple(
            self._clone_section(
                section,
                session=inner_session,
                filesystem=shared_filesystem,
            )
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

                        Your output must include:
                        - **summary**: A single paragraph (2-3 sentences) overview of the
                          workspace purpose, primary language/framework, and key capabilities.
                        - **digest**: The full detailed digest with build commands, dependencies,
                          testing instructions, and other technical details.
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

        summary, digest = self._extract_summary_and_digest(
            response=response, prompt_name=prompt_name
        )

        if self._store_scope is PersistenceScope.SESSION:
            _ = set_workspace_digest(
                outer_session, digest_section.key, digest, summary=summary
            )

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
            descriptor = prompt.descriptor
            section_hash = self._find_section_hash(descriptor, section_path)
            override = SectionOverride(
                path=section_path,
                expected_hash=section_hash,
                body=digest,
            )
            _ = global_store.store(descriptor, override, tag=global_tag)
            clear_workspace_digest(outer_session, digest_section.key)

        return WorkspaceDigestResult(
            response=cast(PromptResponse[object], response),
            digest=digest,
            scope=self._store_scope,
            section_key=digest_section.key,
        )

    def _resolve_workspace_section(  # noqa: PLR6301
        self, prompt: Prompt[object], prompt_name: str
    ) -> WorkspaceSection:
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
        self,
        section: Section[SupportsDataclass],
        *,
        session: Session,
        filesystem: Filesystem | None = None,
    ) -> Section[SupportsDataclass]:
        kwargs: dict[str, object] = {
            "session": session,
            "dispatcher": session.dispatcher,
        }
        if filesystem is not None:
            kwargs["filesystem"] = filesystem
        return section.clone(**kwargs)

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

    def _find_section_hash(  # noqa: PLR6301
        self, descriptor: PromptDescriptor, path: tuple[str, ...]
    ) -> HexDigest:
        for section in descriptor.sections:
            if section.path == path:
                return section.content_hash
        msg = f"Section hash not found for path: {path}"
        raise PromptOverridesError(msg)

    def _extract_summary_and_digest(  # noqa: PLR6301
        self, *, response: PromptResponse[Any], prompt_name: str
    ) -> tuple[str, str]:
        """Extract summary and digest from the optimization response.

        Returns:
            Tuple of (summary, digest) strings.
        """
        summary: str | None = None
        digest: str | None = None

        if isinstance(response.output, str):
            # Plain string output - use as digest, generate default summary
            digest = response.output
        elif response.output is not None:
            # Structured output - extract both fields
            summary_candidate = getattr(response.output, "summary", None)
            if isinstance(summary_candidate, str):
                summary = summary_candidate
            digest_candidate = getattr(response.output, "digest", None)
            if isinstance(digest_candidate, str):
                digest = digest_candidate

        # Fall back to response text if no digest extracted
        if digest is None and response.text:
            digest = response.text

        if digest is None:
            raise PromptEvaluationError(
                "Optimization did not return digest content.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_RESPONSE,
            )

        # Use default summary if not provided
        if summary is None:
            summary = (
                "Workspace digest available. Use read_section to view the full details."
            )

        return summary.strip(), digest.strip()


__all__ = ["WorkspaceDigestOptimizer"]
