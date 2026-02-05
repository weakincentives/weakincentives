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

"""Code review agent using AgentLoop with in-memory mailbox.

This example demonstrates how to build a code review agent using the WINK
framework's core concepts:

- **AgentLoop**: Durable request processing with visibility timeout semantics
- **InMemoryMailbox**: Thread-safe in-memory queue for request/response routing
- **InProcessDispatcher**: Synchronous event delivery for telemetry
- **ClaudeAgentSDKAdapter**: Provider adapter for Claude evaluation
- **Session**: Event-sourced state container

The agent runs in a single process with the dispatcher, mailboxes, and loop
all in the same address space. This pattern is ideal for development and
testing before moving to distributed processing with Redis mailboxes.

Architecture:

    +-----------+     +----------------+     +-------------+
    |  Client   | --> | Request Mailbox| --> | CodeReview  |
    |           |     |                |     | Loop        |
    +-----------+     +----------------+     +-------------+
         ^                                          |
         |           +----------------+             |
         +---------- | Response Mailbox| <----------+
                     +----------------+

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
import threading
from dataclasses import field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, override

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
from weakincentives.runtime import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
    InMemoryMailbox,
    InProcessDispatcher,
    Session,
)

if TYPE_CHECKING:
    from weakincentives.experiment import Experiment

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

# Wait time for mailbox operations
MAILBOX_WAIT_SECONDS = 1
MAILBOX_VISIBILITY_TIMEOUT = 300


# =============================================================================
# Domain Types
# =============================================================================


@FrozenDataclass()
class ReviewRequest:
    """Input parameters for a code review request."""

    project_path: str = field(
        metadata={"description": "Path to the project to review."}
    )
    focus: str = field(
        default="Review the code for quality, correctness, and best practices.",
        metadata={"description": "What to focus on in the review."},
    )
    optimize: bool = field(
        default=True,
        metadata={"description": "Whether to pre-populate workspace digest."},
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


@FrozenDataclass()
class ReviewParams:
    """Parameters for the review prompt template."""

    focus: str = field(metadata={"description": "What to focus on in the review."})


# =============================================================================
# AgentLoop Implementation
# =============================================================================


class CodeReviewLoop(AgentLoop[ReviewRequest, ReviewResponse]):
    """AgentLoop implementation for code review tasks.

    This loop processes ReviewRequest messages from its request mailbox,
    evaluates them using the Claude Agent SDK, and sends ReviewResponse
    results to the reply mailbox.

    Key features:
    - Durable processing with visibility timeout
    - Automatic workspace digest optimization
    - Structured output parsing
    - Session telemetry via dispatcher
    """

    def __init__(
        self,
        *,
        adapter: ClaudeAgentSDKAdapter[ReviewResponse],
        requests: InMemoryMailbox[
            AgentLoopRequest[ReviewRequest], AgentLoopResult[ReviewResponse]
        ],
        config: AgentLoopConfig | None = None,
        worker_id: str = "code-reviewer",
    ) -> None:
        """Initialize the code review loop.

        Args:
            adapter: Claude Agent SDK adapter for evaluation.
            requests: Mailbox to receive review requests from.
            config: Optional configuration for deadlines/budgets.
            worker_id: Identifier for this worker instance.
        """
        super().__init__(
            adapter=adapter,
            requests=requests,
            config=config,
            worker_id=worker_id,
        )
        self._template = self._create_template()
        self._last_session: Session | None = None

    @property
    def last_session(self) -> Session | None:
        """Access the most recent session for telemetry inspection."""
        return self._last_session

    def _get_section_by_key(self, key: str) -> MarkdownSection:
        """Find a section in the template by its key.

        Args:
            key: The section key to find.

        Returns:
            The section with the matching key.

        Raises:
            KeyError: If no section with the given key exists.
        """
        for node in self._template.sections:
            if node.section.key == key:
                # Cast is safe - we know the template only contains MarkdownSections
                return node.section  # type: ignore[return-value]
        raise KeyError(f"Section with key '{key}' not found in template")

    @staticmethod
    def _create_template() -> PromptTemplate[ReviewResponse]:
        """Create the prompt template for code review."""
        return PromptTemplate[ReviewResponse](
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
                MarkdownSection[ReviewParams](
                    title="Review Focus",
                    template="${focus}",
                    key="focus",
                ),
                # Workspace digest section placeholder - filled during prepare()
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

    def prepare(
        self,
        request: ReviewRequest,
        *,
        experiment: Experiment | None = None,
    ) -> tuple[Prompt[ReviewResponse], Session]:
        """Prepare prompt and session for the review request.

        This method is called by AgentLoop before evaluation. It:
        1. Creates a session with in-process dispatcher
        2. Configures workspace mounts for the project
        3. Optionally runs workspace digest optimization
        4. Builds the complete prompt with workspace sections

        Args:
            request: The review request to process.
            experiment: Optional experiment configuration (unused).

        Returns:
            Tuple of (prompt, session) ready for evaluation.
        """
        _ = experiment  # Not used in this example

        # Create session with in-process dispatcher for telemetry
        dispatcher = InProcessDispatcher()
        session = Session(
            dispatcher=dispatcher,
            tags={"type": "code-review", "project": request.project_path},
        )
        self._last_session = session

        # Configure workspace mounts
        mounts = [
            HostMount(
                host_path=request.project_path,
                include_glob=DEFAULT_INCLUDE_GLOBS,
                exclude_glob=DEFAULT_EXCLUDE_GLOBS,
                max_bytes=DEFAULT_MAX_BYTES,
            )
        ]

        # Optionally run optimization to pre-populate workspace digest
        if request.optimize:
            self._run_optimization(session, mounts)

        # Create workspace sections
        digest_section = WorkspaceDigestSection(session=session)
        workspace_section = ClaudeAgentWorkspaceSection(
            session=session,
            mounts=mounts,
        )

        # Build prompt with all sections
        # Insert digest and workspace sections into the template
        template_with_workspace = PromptTemplate[ReviewResponse](
            ns=self._template.ns,
            key=self._template.key,
            name=self._template.name,
            sections=(
                self._get_section_by_key("role"),
                self._get_section_by_key("focus"),
                digest_section,
                workspace_section,
                self._get_section_by_key("instructions"),
                self._get_section_by_key("output-format"),
            ),
        )

        prompt = Prompt(template_with_workspace).bind(ReviewParams(focus=request.focus))
        return prompt, session

    @staticmethod
    def _run_optimization(
        session: Session,
        mounts: list[HostMount],
    ) -> None:
        """Run workspace digest optimization to pre-populate context."""
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

    @override
    def finalize(
        self,
        prompt: Prompt[ReviewResponse],
        session: Session,
        output: ReviewResponse | None,
    ) -> ReviewResponse | None:
        """Finalize after execution completes.

        Clean up workspace resources and log completion.

        Args:
            prompt: The prompt that was evaluated.
            session: The session used for evaluation.
            output: The parsed output from the model response.

        Returns:
            The output unchanged.
        """
        # Clean up workspace sections
        for node in prompt.template.sections:
            if isinstance(node.section, ClaudeAgentWorkspaceSection):
                node.section.cleanup()

        if output is not None:
            _LOGGER.info("Review complete: %s", output.summary[:50] + "...")

        return output


# =============================================================================
# Mailbox Setup and Processing
# =============================================================================


def create_mailboxes() -> tuple[
    InMemoryMailbox[AgentLoopRequest[ReviewRequest], AgentLoopResult[ReviewResponse]],
    InMemoryMailbox[AgentLoopResult[ReviewResponse], None],
]:
    """Create request and response mailboxes for the review loop.

    Returns:
        Tuple of (requests_mailbox, responses_mailbox).
    """
    requests: InMemoryMailbox[
        AgentLoopRequest[ReviewRequest], AgentLoopResult[ReviewResponse]
    ] = InMemoryMailbox(name="review-requests")

    responses: InMemoryMailbox[AgentLoopResult[ReviewResponse], None] = InMemoryMailbox(
        name="review-responses"
    )

    return requests, responses


def create_adapter() -> ClaudeAgentSDKAdapter[ReviewResponse]:
    """Create the Claude Agent SDK adapter for evaluation."""
    return ClaudeAgentSDKAdapter[ReviewResponse](
        client_config=ClaudeAgentSDKClientConfig(
            permission_mode="bypassPermissions",
            max_turns=20,
            isolation=IsolationConfig.inherit_host_auth(),
        )
    )


def run_loop_worker(
    loop: CodeReviewLoop,
    max_iterations: int = 1,
) -> None:
    """Run the loop worker to process messages.

    Args:
        loop: The code review loop to run.
        max_iterations: Maximum number of messages to process.
    """
    _LOGGER.info("Starting loop worker (max_iterations=%d)...", max_iterations)
    loop.run(
        max_iterations=max_iterations,
        visibility_timeout=MAILBOX_VISIBILITY_TIMEOUT,
        wait_time_seconds=MAILBOX_WAIT_SECONDS,
    )
    _LOGGER.info("Loop worker finished")


# =============================================================================
# Main Entry Point
# =============================================================================


def run_review(
    project_path: Path,
    focus: str,
    *,
    optimize: bool = True,
    deadline_minutes: int = DEFAULT_DEADLINE_MINUTES,
) -> ReviewResponse | None:
    """Run a code review using the AgentLoop pattern.

    This function demonstrates the complete request/response flow:
    1. Create mailboxes for communication
    2. Create the AgentLoop with adapter
    3. Send a request to the request mailbox
    4. Run the loop worker to process the request
    5. Receive the response from the response mailbox

    Args:
        project_path: Path to the project to review.
        focus: What to focus on in the review.
        optimize: Whether to run workspace optimization first.
        deadline_minutes: Deadline for the review operation.

    Returns:
        ReviewResponse with findings, or None if review failed.
    """
    _LOGGER.info("Starting code review for: %s", project_path)

    # Create mailboxes for request/response routing
    requests, responses = create_mailboxes()

    # Create the adapter and loop
    adapter = create_adapter()
    config = AgentLoopConfig(
        deadline=Deadline(
            expires_at=datetime.now(UTC) + timedelta(minutes=deadline_minutes)
        ),
    )
    loop = CodeReviewLoop(
        adapter=adapter,
        requests=requests,
        config=config,
    )

    # Create and send the review request
    review_request = ReviewRequest(
        project_path=str(project_path),
        focus=focus,
        optimize=optimize,
    )
    loop_request = AgentLoopRequest(request=review_request)

    _LOGGER.info("Sending review request to mailbox...")
    _ = requests.send(loop_request, reply_to=responses)

    # Run the loop worker in a separate thread
    worker_thread = threading.Thread(
        target=run_loop_worker,
        args=(loop, 1),
        name="review-worker",
    )
    worker_thread.start()
    worker_thread.join(timeout=deadline_minutes * 60 + 30)

    if worker_thread.is_alive():
        _LOGGER.warning("Worker timed out, requesting shutdown...")
        loop.shutdown(timeout=10.0)
        worker_thread.join(timeout=10.0)

    # Receive the response from the response mailbox
    _LOGGER.info("Checking response mailbox...")
    response_messages = responses.receive(
        max_messages=1,
        visibility_timeout=30,
        wait_time_seconds=MAILBOX_WAIT_SECONDS,
    )

    if not response_messages:
        _LOGGER.error("No response received from review loop")
        return None

    msg = response_messages[0]
    result: AgentLoopResult[ReviewResponse] = msg.body
    msg.acknowledge()

    if result.success and result.output is not None:
        _LOGGER.info("Review completed successfully!")
        return result.output

    _LOGGER.error("Review failed: %s", result.error)
    return None


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
        description="Code review agent using AgentLoop with in-memory mailbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              %(prog)s /path/to/project
              %(prog)s /path/to/project "Review authentication logic"
              %(prog)s /path/to/project --no-optimize

            Architecture:
              This example demonstrates the AgentLoop pattern with:
              - InMemoryMailbox for request/response routing
              - InProcessDispatcher for session telemetry
              - ClaudeAgentSDKAdapter for LLM evaluation
              - Durable processing with visibility timeouts
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
