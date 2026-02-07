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
- **ProviderAdapter**: Provider-agnostic evaluation (Claude SDK or Codex)
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
    python code_reviewer_example.py --adapter codex /path/to/project "Focus area"

Requirements:
    pip install weakincentives
    # Claude SDK adapter requires claude-agent-sdk
    # Codex adapter requires codex CLI on PATH
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
    HostMount as ClaudeHostMount,
    IsolationConfig,
)
from weakincentives.adapters.codex_app_server import (
    CodexAppServerAdapter,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
    CodexWorkspaceSection,
    HostMount as CodexHostMount,
)
from weakincentives.dataclasses import FrozenDataclass
from weakincentives.deadlines import Deadline
from weakincentives.debug import BundleConfig
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.prompt.section import Section
from weakincentives.runtime import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
    InMemoryMailbox,
    InProcessDispatcher,
    Session,
)
from weakincentives.runtime.logging import configure_logging

if TYPE_CHECKING:
    from weakincentives.adapters.core import ProviderAdapter
    from weakincentives.experiment import Experiment

# Adapter name constants
ADAPTER_CLAUDE = "claude"
ADAPTER_CODEX = "codex"

_LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Initialize root logging for the demo with DEBUG on stderr.

    Uses the WINK structured logging so that event names and context
    dicts from StructuredLogger calls are rendered in the output.
    """
    configure_logging(level=logging.DEBUG, force=True)


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
DEFAULT_MAX_BYTES = 200_000

# Default deadline for review operations
DEFAULT_DEADLINE_MINUTES = 15

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


def _create_workspace_section(
    adapter_name: str,
    session: Session,
    project_path: str,
) -> Section:
    """Create the appropriate workspace section for the chosen adapter."""
    if adapter_name == ADAPTER_CODEX:
        return CodexWorkspaceSection(
            session=session,
            mounts=[
                CodexHostMount(
                    host_path=project_path,
                    include_glob=DEFAULT_INCLUDE_GLOBS,
                    exclude_glob=DEFAULT_EXCLUDE_GLOBS,
                    max_bytes=DEFAULT_MAX_BYTES,
                )
            ],
        )
    return ClaudeAgentWorkspaceSection(
        session=session,
        mounts=[
            ClaudeHostMount(
                host_path=project_path,
                include_glob=DEFAULT_INCLUDE_GLOBS,
                exclude_glob=DEFAULT_EXCLUDE_GLOBS,
                max_bytes=DEFAULT_MAX_BYTES,
            )
        ],
    )


class CodeReviewLoop(AgentLoop[ReviewRequest, ReviewResponse]):
    """AgentLoop implementation for code review tasks.

    This loop processes ReviewRequest messages from its request mailbox,
    evaluates them using either the Claude Agent SDK or Codex App Server,
    and sends ReviewResponse results to the reply mailbox.
    """

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[ReviewResponse],
        requests: InMemoryMailbox[
            AgentLoopRequest[ReviewRequest], AgentLoopResult[ReviewResponse]
        ],
        adapter_name: str = ADAPTER_CLAUDE,
        config: AgentLoopConfig | None = None,
        worker_id: str = "code-reviewer",
    ) -> None:
        super().__init__(
            adapter=adapter,
            requests=requests,
            config=config,
            worker_id=worker_id,
        )
        self._adapter_name = adapter_name
        self._last_session: Session | None = None

    @property
    def last_session(self) -> Session | None:
        """Access the most recent session for telemetry inspection."""
        return self._last_session

    def prepare(
        self,
        request: ReviewRequest,
        *,
        experiment: Experiment | None = None,
    ) -> tuple[Prompt[ReviewResponse], Session]:
        """Prepare prompt and session for the review request."""
        _ = experiment

        dispatcher = InProcessDispatcher()
        session = Session(
            dispatcher=dispatcher,
            tags={"type": "code-review", "project": request.project_path},
        )
        self._last_session = session

        workspace = _create_workspace_section(
            self._adapter_name, session, request.project_path
        )

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
                MarkdownSection[ReviewParams](
                    title="Review Focus",
                    template="${focus}",
                    key="focus",
                ),
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

        prompt = Prompt(template).bind(ReviewParams(focus=request.focus))
        return prompt, session

    @override
    def finalize(
        self,
        prompt: Prompt[ReviewResponse],
        session: Session,
        output: ReviewResponse | None,
    ) -> ReviewResponse | None:
        """Log review completion. Cleanup is handled by the framework."""
        _ = (prompt, session)
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
    """Create request and response mailboxes for the review loop."""
    requests: InMemoryMailbox[
        AgentLoopRequest[ReviewRequest], AgentLoopResult[ReviewResponse]
    ] = InMemoryMailbox(name="review-requests")

    responses: InMemoryMailbox[AgentLoopResult[ReviewResponse], None] = InMemoryMailbox(
        name="review-responses"
    )

    return requests, responses


def create_adapter(adapter_name: str) -> ProviderAdapter[ReviewResponse]:
    """Create the appropriate adapter for evaluation."""
    if adapter_name == ADAPTER_CODEX:
        return CodexAppServerAdapter(
            model_config=CodexAppServerModelConfig(),
            client_config=CodexAppServerClientConfig(approval_policy="never"),
        )
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
    """Run the loop worker to process messages."""
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
    adapter_name: str = ADAPTER_CLAUDE,
    deadline_minutes: int = DEFAULT_DEADLINE_MINUTES,
) -> ReviewResponse | None:
    """Run a code review using the AgentLoop pattern."""
    _LOGGER.info(
        "Starting code review for: %s (adapter=%s)", project_path, adapter_name
    )

    requests, responses = create_mailboxes()

    adapter = create_adapter(adapter_name)
    config = AgentLoopConfig(
        deadline=Deadline(
            expires_at=datetime.now(UTC) + timedelta(minutes=deadline_minutes)
        ),
        debug_bundle=BundleConfig(target=Path("debug_bundles/")),
    )
    loop = CodeReviewLoop(
        adapter=adapter,
        requests=requests,
        adapter_name=adapter_name,
        config=config,
    )

    review_request = ReviewRequest(
        project_path=str(project_path),
        focus=focus,
    )
    loop_request = AgentLoopRequest(request=review_request)

    _LOGGER.info("Sending review request to mailbox...")
    _ = requests.send(loop_request, reply_to=responses)

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
    """Format a review response for display."""
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
        lines.extend(["ISSUES", "-" * 40])
        for i, issue in enumerate(review.issues, 1):
            lines.append(f"  {i}. {issue}")
        lines.append("")

    if review.suggestions:
        lines.extend(["SUGGESTIONS", "-" * 40])
        for i, suggestion in enumerate(review.suggestions, 1):
            lines.append(f"  {i}. {suggestion}")
        lines.append("")

    if review.positive_notes:
        lines.extend(["POSITIVE NOTES", "-" * 40])
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
              %(prog)s --adapter codex /path/to/project "Review auth logic"

            Architecture:
              This example demonstrates the AgentLoop pattern with:
              - InMemoryMailbox for request/response routing
              - InProcessDispatcher for session telemetry
              - Provider-agnostic adapter (Claude SDK or Codex)
              - Durable processing with visibility timeouts
        """),
    )
    parser.add_argument(
        "--adapter",
        choices=[ADAPTER_CLAUDE, ADAPTER_CODEX],
        default=ADAPTER_CLAUDE,
        help=f"Which adapter to use (default: {ADAPTER_CLAUDE})",
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
        "--deadline",
        type=int,
        default=DEFAULT_DEADLINE_MINUTES,
        help=f"Deadline in minutes (default: {DEFAULT_DEADLINE_MINUTES})",
    )
    return parser


def _validate_project_path(project_path: Path) -> str | None:
    """Validate the project path exists and is a directory."""
    if not project_path.exists():
        return f"Project path does not exist: {project_path}"
    if not project_path.is_dir():
        return f"Project path is not a directory: {project_path}"
    return None


def main() -> int:
    """Run the code review example."""
    parser = _create_argument_parser()
    args = parser.parse_args()

    _configure_logging()

    error = _validate_project_path(args.project_path)
    if error:
        _LOGGER.error(error)
        return 1

    try:
        review = run_review(
            args.project_path.resolve(),
            args.focus,
            adapter_name=args.adapter,
            deadline_minutes=args.deadline,
        )
    except KeyboardInterrupt:
        _LOGGER.info("Review cancelled by user")
        return 130
    except Exception:
        _LOGGER.exception("Review failed")
        return 1

    if review is not None:
        print(format_review(review))
        return 0
    _LOGGER.error("Review failed to produce results")
    return 1


if __name__ == "__main__":
    sys.exit(main())
