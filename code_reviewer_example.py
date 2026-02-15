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
- **ProviderAdapter**: Provider-agnostic evaluation (Claude SDK, Codex, or OpenCode)
- **Session**: Event-sourced state container
- **Tool**: Custom tool registration with typed params, results, and examples
- **SkillMount**: Skill composition via the Agent Skills specification

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
    python code_reviewer_example.py --adapter opencode --model openai/gpt-5.3-codex /path/to/project

Requirements:
    pip install weakincentives
    # Claude SDK adapter requires claude-agent-sdk
    # Codex adapter requires codex CLI on PATH
    # OpenCode adapter requires opencode CLI on PATH + agent-client-protocol
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
    IsolationConfig,
)
from weakincentives.adapters.codex_app_server import (
    CodexAppServerAdapter,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
)
from weakincentives.adapters.opencode_acp import (
    OpenCodeACPAdapter,
    OpenCodeACPAdapterConfig,
    OpenCodeACPClientConfig,
)
from weakincentives.dataclasses import FrozenDataclass
from weakincentives.deadlines import Deadline
from weakincentives.debug import BundleConfig
from weakincentives.prompt import (
    HostMount,
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolExample,
    ToolResult,
    WorkspaceSection,
)
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
from weakincentives.skills import SkillMount

if TYPE_CHECKING:
    from weakincentives.adapters.core import ProviderAdapter
    from weakincentives.experiment import Experiment

# Adapter name constants
ADAPTER_CLAUDE = "claude"
ADAPTER_CODEX = "codex"
ADAPTER_OPENCODE = "opencode"

_LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Initialize root logging for the demo with DEBUG on stderr.

    Uses the WINK structured logging so that event names and context
    dicts from StructuredLogger calls are rendered in the output.
    """
    configure_logging(level=logging.DEBUG, force=True)


# Sunfish mount configuration — include source and docs, exclude binary assets.
SUNFISH_MOUNT_INCLUDE_GLOBS: tuple[str, ...] = (
    "*.py",
    "*.md",
    "*.txt",
    "*.yml",
    "*.yaml",
    "*.toml",
    ".gitignore",
    "*.json",
    "*.cfg",
    "*.ini",
    "*.sh",
    "*.6",
)

SUNFISH_MOUNT_EXCLUDE_GLOBS: tuple[str, ...] = (
    "**/__pycache__/**",
    "**/.git/**",
    "**/.venv/**",
    "**/node_modules/**",
    "**/*.pyc",
    "**/*.pickle",
    "**/*.png",
    "**/*.bmp",
)

SUNFISH_MOUNT_MAX_BYTES = 600_000

# Skills directory (ships alongside this script)
SKILLS_DIR = Path(__file__).resolve().parent / "demo-skills"

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
# Custom Tool: count_lines
# =============================================================================


@FrozenDataclass()
class CountLinesParams:
    """Parameters for the count_lines tool."""

    path: str = field(
        metadata={"description": "Relative file path within the project."}
    )


@FrozenDataclass()
class CountLinesResult:
    """Line-count breakdown for a single file."""

    path: str
    total: int
    code: int
    blank: int
    comment: int

    def render(self) -> str:
        return (
            f"{self.path}: {self.total} total, {self.code} code, "
            f"{self.blank} blank, {self.comment} comment"
        )


def _create_count_lines_tool(
    project_path: str,
) -> Tool[CountLinesParams, CountLinesResult]:
    """Create a ``count_lines`` tool bound to *project_path*."""

    def count_lines(
        params: CountLinesParams, *, context: ToolContext
    ) -> ToolResult[CountLinesResult]:
        """Count lines of code, blanks, and comments in a project file."""
        _ = context
        root = Path(project_path).resolve()
        target = (root / params.path).resolve()
        if not target.is_relative_to(root):
            return ToolResult.error("Path escapes project directory")
        if not target.is_file():
            return ToolResult.error(f"File not found: {params.path}")
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult.error(f"Cannot read {params.path}: {exc}")
        lines = text.splitlines()
        total = len(lines)
        blank = sum(1 for line in lines if not line.strip())
        comment = sum(1 for line in lines if line.strip().startswith("#"))
        code = total - blank - comment
        result = CountLinesResult(
            path=params.path,
            total=total,
            code=code,
            blank=blank,
            comment=comment,
        )
        return ToolResult.ok(result, message=result.render())

    return Tool[CountLinesParams, CountLinesResult](
        name="count_lines",
        description="Count lines of code, blanks, and comments in a project file.",
        handler=count_lines,
        examples=(
            ToolExample(
                description="Count lines in the main engine file",
                input=CountLinesParams(path="sunfish.py"),
                output=CountLinesResult(
                    path="sunfish.py", total=425, code=298, blank=47, comment=80
                ),
            ),
        ),
    )


# =============================================================================
# AgentLoop Implementation
# =============================================================================


def _create_workspace_section(
    session: Session,
    project_path: str,
) -> Section:
    """Create the workspace section with sunfish-specific mount config."""
    return WorkspaceSection(
        session=session,
        mounts=[
            HostMount(
                host_path=project_path,
                include_glob=SUNFISH_MOUNT_INCLUDE_GLOBS,
                exclude_glob=SUNFISH_MOUNT_EXCLUDE_GLOBS,
                max_bytes=SUNFISH_MOUNT_MAX_BYTES,
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
        config: AgentLoopConfig | None = None,
        worker_id: str = "code-reviewer",
    ) -> None:
        super().__init__(
            adapter=adapter,
            requests=requests,
            config=config,
            worker_id=worker_id,
        )
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

        workspace = _create_workspace_section(session, request.project_path)

        # Custom tool — attach to any section via tools=(...).
        count_lines_tool = _create_count_lines_tool(request.project_path)

        # Skills — mount SKILL.md directories via skills=(...).
        code_review_skill = SkillMount(source=SKILLS_DIR / "code-review")
        python_style_skill = SkillMount(source=SKILLS_DIR / "python-style")

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
                    skills=(code_review_skill, python_style_skill),
                ),
                MarkdownSection[ReviewParams](
                    title="Review Focus",
                    template="${focus}",
                    key="focus",
                ),
                workspace,
                MarkdownSection(
                    title="Analysis",
                    template=textwrap.dedent("""
                        1. Explore the workspace to understand the project structure
                        2. Use the ``count_lines`` tool to gauge file size and composition
                        3. Read key files to understand the codebase
                        4. Identify issues, risks, or areas for improvement
                        5. Note positive aspects and good practices
                        6. Provide specific, actionable suggestions
                    """).strip(),
                    key="analysis",
                    tools=(count_lines_tool,),
                ),
                MarkdownSection(
                    title="Output Format",
                    template=textwrap.dedent("""
                        Return your review as a JSON object with these fields:
                        - **summary** (string): 1-2 sentence overview of findings
                        - **issues** (list of strings): Problems or concerns found
                        - **suggestions** (list of strings): Specific improvement recommendations
                        - **positive_notes** (list of strings): Good practices observed

                        Example:
                        ```json
                        {
                          "summary": "...",
                          "issues": ["..."],
                          "suggestions": ["..."],
                          "positive_notes": ["..."]
                        }
                        ```
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


def create_adapter(
    adapter_name: str,
    *,
    model_id: str | None = None,
) -> ProviderAdapter[ReviewResponse]:
    """Create the appropriate adapter for evaluation."""
    if adapter_name == ADAPTER_CODEX:
        return CodexAppServerAdapter(
            model_config=CodexAppServerModelConfig(),
            client_config=CodexAppServerClientConfig(approval_policy="never"),
        )
    if adapter_name == ADAPTER_OPENCODE:
        return OpenCodeACPAdapter(
            adapter_config=OpenCodeACPAdapterConfig(model_id=model_id),
            client_config=OpenCodeACPClientConfig(
                permission_mode="auto",
                allow_file_reads=True,
                allow_file_writes=False,
            ),
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
    model_id: str | None = None,
    deadline_minutes: int = DEFAULT_DEADLINE_MINUTES,
) -> ReviewResponse | None:
    """Run a code review using the AgentLoop pattern."""
    _LOGGER.info(
        "Starting code review for: %s (adapter=%s)", project_path, adapter_name
    )
    requests, responses = create_mailboxes()

    adapter = create_adapter(adapter_name, model_id=model_id)
    config = AgentLoopConfig(
        debug_bundle=BundleConfig(target=Path("debug_bundles/")),
    )
    loop = CodeReviewLoop(
        adapter=adapter,
        requests=requests,
        config=config,
    )

    review_request = ReviewRequest(
        project_path=str(project_path),
        focus=focus,
    )
    loop_request = AgentLoopRequest(
        request=review_request,
        deadline=Deadline(
            expires_at=datetime.now(UTC) + timedelta(minutes=deadline_minutes)
        ),
    )

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
              %(prog)s --adapter opencode --model openai/gpt-5.3-codex /path/to/project
        """),
    )
    parser.add_argument(
        "--adapter",
        choices=[ADAPTER_CLAUDE, ADAPTER_CODEX, ADAPTER_OPENCODE],
        default=ADAPTER_CLAUDE,
        help=f"Which adapter to use (default: {ADAPTER_CLAUDE})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID for the adapter (e.g. openai/gpt-5.3-codex)",
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
            model_id=args.model,
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
