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

"""Prompt scaffolding shared by the code review examples."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

from ..prompt import MarkdownSection, Prompt
from ..session import Session
from ..tools import PlanningToolsSection, VfsToolsSection
from .code_review_tools import build_tools


@dataclass
class ReviewGuidance:
    focus: str = field(
        default=(
            "Identify potential issues, risks, and follow-up questions for the changes "
            "under review."
        ),
        metadata={
            "description": "Default framing instructions for the review assistant.",
        },
    )


@dataclass
class ReviewTurnParams:
    request: str = field(
        metadata={
            "description": "User-provided review task or question to address.",
        }
    )


@dataclass
class ReviewResponse:
    summary: str
    issues: list[str]
    next_steps: list[str]


def build_code_review_prompt(session: Session) -> Prompt[ReviewResponse]:
    tools = build_tools()
    guidance_section = MarkdownSection[ReviewGuidance](
        title="Code Review Brief",
        template=textwrap.dedent(
            """
            You are a code review assistant working in this repository.
            Every response must stay anchored to the specific task described
            in the review request. If the task is unclear, ask for the missing
            details before proceeding.

            Use the available tools to stay grounded:
            - `show_git_log` retrieves commit history relevant to the task.
            - `show_git_branches` lists branches that match specified filters.
            - `show_git_tags` lists tags that match specified filters.
            - `show_current_time` reports the present time (default UTC or a
              requested timezone).
            - `vfs_list_directory` lists directories and files staged in the virtual
              filesystem snapshot.
            - `vfs_read_file` reads staged file contents.
            - `vfs_write_file` stages ASCII edits before applying them to the host
              workspace.
            - `vfs_delete_entry` removes staged files or directories that are no
              longer needed.
            If the task requires information beyond these capabilities, ask the
            user for clarification rather than guessing.

            Maintain a concise working plan for multi-step investigations. Use the
            planning tools to capture the current objective, record step details
            as you gather evidence, and mark tasks complete when finished.

            Always provide a JSON response with the following keys:
            - summary: Single paragraph capturing the overall state of the changes.
            - issues: List of concrete problems, risks, or follow-up questions tied
              to the task.
            - next_steps: List of actionable recommendations or follow-ups that
              help complete the task or mitigate the issues.
            """
        ).strip(),
        default_params=ReviewGuidance(),
        tools=tools,
        key="code-review-brief",
    )
    planning_section = PlanningToolsSection(session=session)
    vfs_section = VfsToolsSection(session=session)
    user_turn_section = MarkdownSection[ReviewTurnParams](
        title="Review Request",
        template="${request}",
        key="review-request",
    )
    return Prompt[ReviewResponse](
        ns="examples/code-review",
        key="code-review-session",
        name="code_review_agent",
        sections=[guidance_section, planning_section, vfs_section, user_turn_section],
    )


__all__ = [
    "ReviewGuidance",
    "ReviewTurnParams",
    "ReviewResponse",
    "build_code_review_prompt",
]
