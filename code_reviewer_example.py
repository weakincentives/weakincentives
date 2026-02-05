#!/usr/bin/env python3
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

"""Code review agent using Claude Agent SDK.

This example demonstrates how to build a code review agent using the Claude
Agent SDK adapter. The agent can explore a workspace, understand its structure,
and provide code review feedback.

Key concepts demonstrated:
- Using ClaudeAgentWorkspaceSection to mount files for review
- Configuring the Claude Agent SDK adapter
- Using WorkspaceDigestSection to cache workspace analysis
- Running optimization to pre-populate workspace digests
- Handling structured output from the agent

Usage:
    python code_reviewer_example.py /path/to/project "Review the main module"

Requirements:
    pip install weakincentives
    # Ensure claude-agent-sdk is installed for actual execution
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from dataclasses import field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    HostMount,
    IsolationConfig,
)
from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer
from weakincentives.contrib.tools import WorkspaceDigestSection
from weakincentives.dataclasses import FrozenDataclass
from weakincentives.deadlines import Deadline
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime import InProcessDispatcher, Session

# Configure logging for the example
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
_LOGGER = logging.getLogger(__name__)

# Default include patterns for code review
DEFAULT_INCLUDE_GLOBS: tuple[str, ...] = (
    "*.py",
    "*.md",
    "*.txt",
    "*.yml",
    "*.yaml",
    "*.toml",
    "*.json",
    "*.cfg",
    "*.ini",
    "*.sh",
)

# Exclude patterns
DEFAULT_EXCLUDE_GLOBS: tuple[str, ...] = (
    "**/__pycache__/**",
    "**/.git/**",
    "**/.venv/**",
    "**/node_modules/**",
    "**/*.pyc",
)

# Maximum bytes to include in workspace
DEFAULT_MAX_BYTES = 500_000

# Default deadline for review operations
DEFAULT_DEADLINE_MINUTES = 5

# Truncation limit for log messages
LOG_SUMMARY_TRUNCATE_LENGTH = 100


@FrozenDataclass()
class ReviewRequest:
    """Input parameters for a code review request."""

    focus: str = field(
        default="Review the code for quality, correctness, and best practices.",
        metadata={"description": "What to focus on in the review."},
    )


@FrozenDataclass()
class ReviewResponse:
    """Structured response from the code review agent."""

    summary: str = field(
        metadata={"description": "Brief summary of the review findings."}
    )
    issues: list[str] = field(
        default_factory=list,
        metadata={"description": "List of identified issues or concerns."},
    )
    suggestions: list[str] = field(
        default_factory=list,
        metadata={"description": "Suggested improvements."},
    )
    positive_notes: list[str] = field(
        default_factory=list,
        metadata={"description": "Positive aspects of the code."},
    )


def create_review_prompt(
    session: Session,
    workspace: ClaudeAgentWorkspaceSection,
    focus: str,
) -> Prompt[ReviewResponse]:
    """Create a code review prompt with workspace context.

    Args:
        session: The session for state management.
        workspace: The workspace section with mounted files.
        focus: What to focus on in the review.

    Returns:
        A configured Prompt for code review.
    """
    digest_section = WorkspaceDigestSection(session=session)

    template = PromptTemplate[ReviewResponse](
        ns="code-review",
        key="review-agent",
        name="Code Review Agent",
        sections=(
            MarkdownSection(
                title="Role",
                template=textwrap.dedent("""
                    You are an expert code reviewer. Your job is to analyze code
                    and provide constructive feedback that helps developers improve
                    their work.

                    Be thorough but fair. Point out issues while also acknowledging
                    good practices. Focus on actionable feedback.
                """).strip(),
                key="role",
            ),
            MarkdownSection(
                title="Review Focus",
                template=f"Focus your review on: {focus}",
                key="focus",
            ),
            digest_section,
            workspace,
            MarkdownSection(
                title="Instructions",
                template=textwrap.dedent("""
                    1. Explore the workspace to understand the project structure
                    2. Read key files to understand the codebase
                    3. Identify issues, risks, or areas for improvement
                    4. Note positive aspects and good practices
                    5. Provide specific, actionable suggestions

                    Use the workspace tools to explore files as needed.
                """).strip(),
                key="instructions",
            ),
            MarkdownSection(
                title="Output Format",
                template=textwrap.dedent("""
                    Return a structured review with:
                    - **summary**: 1-2 sentence overview of findings
                    - **issues**: List of problems or concerns found
                    - **suggestions**: Specific improvement recommendations
                    - **positive_notes**: Good practices observed
                """).strip(),
                key="output-format",
            ),
        ),
    )

    return Prompt(template)


def run_optimization(
    session: Session,
    mounts: list[HostMount],
) -> None:
    """Run workspace digest optimization to pre-populate context.

    Args:
        session: The session to store the digest in.
        mounts: Host mounts for the workspace.
    """
    _LOGGER.info("Running workspace optimization...")

    optimizer = WorkspaceDigestOptimizer(mounts=mounts, max_turns=5)
    result = optimizer.optimize(session)

    if result.success:
        truncated = (
            result.summary[:LOG_SUMMARY_TRUNCATE_LENGTH] + "..."
            if len(result.summary) > LOG_SUMMARY_TRUNCATE_LENGTH
            else result.summary
        )
        _LOGGER.info("Optimization complete: %s", truncated)
    else:
        _LOGGER.warning("Optimization failed: %s", result.error)


def run_review(
    project_path: Path,
    focus: str,
    *,
    optimize: bool = True,
    deadline_minutes: int = DEFAULT_DEADLINE_MINUTES,
) -> ReviewResponse | None:
    """Run a code review on the specified project.

    Args:
        project_path: Path to the project to review.
        focus: What to focus on in the review.
        optimize: Whether to run workspace optimization first.
        deadline_minutes: Deadline for the review operation.

    Returns:
        ReviewResponse with findings, or None if review failed.
    """
    _LOGGER.info("Starting code review for: %s", project_path)

    # Create session and dispatcher
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)

    # Configure workspace mount
    mounts = [
        HostMount(
            host_path=str(project_path),
            include_glob=DEFAULT_INCLUDE_GLOBS,
            exclude_glob=DEFAULT_EXCLUDE_GLOBS,
            max_bytes=DEFAULT_MAX_BYTES,
        )
    ]

    # Optionally run optimization to pre-populate workspace digest
    if optimize:
        run_optimization(session, mounts)

    # Create workspace section
    workspace = ClaudeAgentWorkspaceSection(
        session=session,
        mounts=mounts,
    )

    try:
        # Create the review prompt
        prompt = create_review_prompt(session, workspace, focus)

        # Configure adapter with isolation and reasonable limits
        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                permission_mode="bypassPermissions",
                max_turns=20,
                isolation=IsolationConfig.inherit_host_auth(),
            )
        )

        # Create deadline
        deadline = Deadline(
            expires_at=datetime.now(UTC) + timedelta(minutes=deadline_minutes)
        )

        _LOGGER.info("Running code review...")

        # Evaluate the prompt
        response = adapter.evaluate(prompt, session=session, deadline=deadline)

        if response.output is not None:
            _LOGGER.info("Review complete!")
            return response.output

        _LOGGER.warning("Review returned no structured output")
        return None

    finally:
        # Clean up workspace
        workspace.cleanup()


def format_review(review: ReviewResponse) -> str:
    """Format a review response for display.

    Args:
        review: The review response to format.

    Returns:
        Formatted string for display.
    """
    lines = [
        "=" * 60,
        "CODE REVIEW RESULTS",
        "=" * 60,
        "",
        "SUMMARY",
        "-" * 40,
        review.summary,
        "",
    ]

    if review.issues:
        lines.extend(
            [
                "ISSUES",
                "-" * 40,
            ]
        )
        for i, issue in enumerate(review.issues, 1):
            lines.append(f"  {i}. {issue}")
        lines.append("")

    if review.suggestions:
        lines.extend(
            [
                "SUGGESTIONS",
                "-" * 40,
            ]
        )
        for i, suggestion in enumerate(review.suggestions, 1):
            lines.append(f"  {i}. {suggestion}")
        lines.append("")

    if review.positive_notes:
        lines.extend(
            [
                "POSITIVE NOTES",
                "-" * 40,
            ]
        )
        for i, note in enumerate(review.positive_notes, 1):
            lines.append(f"  {i}. {note}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Code review agent using Claude Agent SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              %(prog)s /path/to/project
              %(prog)s /path/to/project "Review authentication logic"
              %(prog)s /path/to/project --no-optimize
        """),
    )
    parser.add_argument(
        "project_path",
        type=Path,
        help="Path to the project to review",
    )
    parser.add_argument(
        "focus",
        nargs="?",
        default="Review the code for quality, correctness, and best practices.",
        help="What to focus on in the review (default: general review)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip workspace optimization",
    )
    parser.add_argument(
        "--deadline",
        type=int,
        default=DEFAULT_DEADLINE_MINUTES,
        help=f"Deadline in minutes (default: {DEFAULT_DEADLINE_MINUTES})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser


def _validate_project_path(project_path: Path) -> str | None:
    """Validate the project path exists and is a directory.

    Returns:
        Error message if invalid, None if valid.
    """
    if not project_path.exists():
        return f"Project path does not exist: {project_path}"
    if not project_path.is_dir():
        return f"Project path is not a directory: {project_path}"
    return None


def main() -> int:
    """Run the code review example."""
    parser = _create_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    error = _validate_project_path(args.project_path)
    if error:
        _LOGGER.error(error)
        return 1

    try:
        review = run_review(
            args.project_path.resolve(),
            args.focus,
            optimize=not args.no_optimize,
            deadline_minutes=args.deadline,
        )
    except KeyboardInterrupt:
        _LOGGER.info("Review cancelled by user")
        return 130
    except Exception:
        _LOGGER.exception("Review failed")
        if args.verbose:
            raise
        return 1

    if review is not None:
        print(format_review(review))
        return 0
    _LOGGER.error("Review failed to produce results")
    return 1


if __name__ == "__main__":
    sys.exit(main())
