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

"""Workspace digest optimizer using Claude Agent SDK.

This module provides an optimizer that generates workspace digests using the
Claude Agent SDK adapter. The optimizer creates a prompt that explores the
workspace and produces a task-agnostic summary stored in the session.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
)
from ...dataclasses import FrozenDataclass
from ...prompt import MarkdownSection, Prompt, PromptTemplate, WorkspaceSection
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session import Session
from ..tools.digests import set_workspace_digest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...prompt.workspace import HostMount

__all__ = [
    "WorkspaceDigestOptimizer",
    "WorkspaceDigestResult",
]


_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "optimizers.workspace_digest"}
)


@FrozenDataclass()
class _DigestResponse:
    """Structured response from the workspace digest generation prompt."""

    summary: str
    digest: str


@FrozenDataclass()
class WorkspaceDigestResult:
    """Result of workspace digest optimization.

    Attributes:
        section_key: The key of the workspace digest section that was updated.
        summary: Short summary of the workspace.
        digest: Full workspace digest content.
        success: Whether the optimization completed successfully.
        error: Error message if optimization failed.
    """

    section_key: str
    summary: str = ""
    digest: str = ""
    success: bool = True
    error: str = ""


@dataclass(slots=True)
class WorkspaceDigestOptimizer:
    """Generate workspace digests using Claude Agent SDK.

    This optimizer creates a prompt that explores a workspace and produces
    a task-agnostic digest. The result is stored in the session and can
    be rendered by WorkspaceDigestSection.

    Example::

        from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer
        from weakincentives.adapters.claude_agent_sdk import HostMount
        from weakincentives.runtime import Session

        session = Session()
        optimizer = WorkspaceDigestOptimizer(
            mounts=[HostMount(host_path="/path/to/project")],
        )
        result = optimizer.optimize(session, section_key="workspace-digest")

    Attributes:
        mounts: Host mounts to include in the workspace.
        adapter: Claude Agent SDK adapter for evaluation.
        max_turns: Maximum agentic turns for exploration.
    """

    mounts: Sequence[HostMount]
    adapter: ClaudeAgentSDKAdapter[_DigestResponse] | None = None
    max_turns: int = 10

    def _create_adapter(self) -> ClaudeAgentSDKAdapter[_DigestResponse]:
        """Create or return the adapter for optimization."""
        if self.adapter is not None:
            return self.adapter
        return ClaudeAgentSDKAdapter[_DigestResponse](
            client_config=ClaudeAgentSDKClientConfig(
                permission_mode="bypassPermissions",
                max_turns=self.max_turns,
            )
        )

    def _build_optimization_prompt(self, session: Session) -> Prompt[_DigestResponse]:
        """Build the prompt for workspace exploration and digest generation."""
        workspace = WorkspaceSection(
            session=session,
            mounts=self.mounts,
        )

        template = PromptTemplate[_DigestResponse](
            ns="weakincentives.optimization",
            key="workspace-digest-generator",
            name="Workspace Digest Generator",
            sections=(
                MarkdownSection(
                    title="Optimization Goal",
                    template=(
                        "Explore the workspace and generate a comprehensive, "
                        "task-agnostic digest that helps future agents understand "
                        "the project structure."
                    ),
                    key="goal",
                ),
                MarkdownSection(
                    title="Expectations",
                    template=textwrap.dedent("""
                        1. **Explore README/docs first** - Understand the project purpose
                        2. **Identify build/test commands** - Document how to build and test
                        3. **Map key directories** - Note important paths and their purposes
                        4. **Capture dependencies** - List package managers and key dependencies
                        5. **Note watchouts** - Document any gotchas or special requirements

                        Keep the digest task-agnostic. Focus on facts that any future agent
                        would find useful regardless of their specific task.
                    """).strip(),
                    key="expectations",
                ),
                workspace,
                MarkdownSection(
                    title="Output Format",
                    template=textwrap.dedent("""
                        Return a structured response with:

                        - **summary**: A 1-2 sentence overview of the project
                        - **digest**: A detailed markdown document covering:
                          - Project purpose and description
                          - Directory structure overview
                          - Build and test commands
                          - Key dependencies
                          - Important notes or watchouts
                    """).strip(),
                    key="output-format",
                ),
            ),
        )

        return Prompt(template)

    def optimize(
        self,
        session: Session,
        *,
        section_key: str = "workspace-digest",
    ) -> WorkspaceDigestResult:
        """Generate and store a workspace digest.

        Args:
            session: The session to store the digest in.
            section_key: The key for the WorkspaceDigestSection to update.

        Returns:
            WorkspaceDigestResult with the generated digest or error.
        """
        _LOGGER.info(
            "Starting workspace digest optimization",
            event="optimization.start",
            context={"section_key": section_key, "mounts": len(self.mounts)},
        )

        try:
            adapter = self._create_adapter()
            prompt = self._build_optimization_prompt(session)

            # Evaluate the prompt to explore the workspace
            response = adapter.evaluate(prompt, session=session)

            if response.output is None:
                _LOGGER.warning(
                    "Optimization returned no structured output",
                    event="optimization.no_output",
                    context={"section_key": section_key},
                )
                return WorkspaceDigestResult(
                    section_key=section_key,
                    success=False,
                    error="No structured output returned from optimization",
                )

            # Store the digest in the session
            digest_response = response.output
            _ = set_workspace_digest(
                session,
                section_key,
                digest_response.digest,
                summary=digest_response.summary,
            )

            _LOGGER.info(
                "Workspace digest optimization complete",
                event="optimization.complete",
                context={
                    "section_key": section_key,
                    "summary_length": len(digest_response.summary),
                    "digest_length": len(digest_response.digest),
                },
            )

            return WorkspaceDigestResult(
                section_key=section_key,
                summary=digest_response.summary,
                digest=digest_response.digest,
                success=True,
            )

        except Exception as e:
            _LOGGER.exception(
                "Workspace digest optimization failed",
                event="optimization.error",
                context={"section_key": section_key, "error": str(e)},
            )
            return WorkspaceDigestResult(
                section_key=section_key,
                success=False,
                error=str(e),
            )
