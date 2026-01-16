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

"""Interactive walkthrough showcasing a minimalist code review agent."""

from __future__ import annotations

import argparse
import logging
import os
import textwrap
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

from examples import (
    build_logged_session,
    configure_logging,
    render_plan_snapshot,
    resolve_override_tag,
)
from weakincentives.adapters import ProviderAdapter
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    HostMount as ClaudeHostMount,
    IsolationAuthError,
    IsolationConfig,
    NetworkPolicy,
    PlanBasedChecker,
    SandboxConfig,
    get_default_model,
)
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer
from weakincentives.contrib.tools import (
    AstevalSection,
    HostMount,
    Plan,
    PlanningStrategy,
    PlanningToolsSection,
    PodmanSandboxConfig,
    PodmanSandboxSection,
    VfsPath,
    VfsToolsSection,
    WorkspaceDigest,
    WorkspaceDigestSection,
)
from weakincentives.deadlines import Deadline
from weakincentives.debug import (
    archive_filesystem,
    archive_metrics,
    collect_all_logs,
    dump_session as dump_session_tree,
)
from weakincentives.metrics import InMemoryMetricsCollector
from weakincentives.optimizers import (
    OptimizationContext,
    PersistenceScope,
)
from weakincentives.prompt import (
    DeadlineFeedback,
    FeedbackProviderConfig,
    FeedbackTrigger,
    MarkdownSection,
    Prompt,
    PromptTemplate,
    SectionVisibility,
)
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptOverridesError,
)
from weakincentives.runtime import (
    DLQPolicy,
    InMemoryMailbox,
    LeaseExtenderConfig,
    MainLoop,
    MainLoopConfig,
    MainLoopRequest,
    MainLoopResult,
    Session,
    ShutdownCoordinator,
)
from weakincentives.skills import SkillConfig, SkillMount
from weakincentives.types import SupportsDataclass

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_SKILLS_ROOT = PROJECT_ROOT / "demo-skills"

TEST_REPOSITORIES_ROOT = (PROJECT_ROOT / "test-repositories").resolve()
SNAPSHOT_DIR = PROJECT_ROOT / "snapshots"
PROMPT_OVERRIDES_TAG_ENV = "CODE_REVIEW_PROMPT_TAG"
SUNFISH_MOUNT_INCLUDE_GLOBS: tuple[str, ...] = (
    "*.md",
    "*.py",
    "*.txt",
    "*.yml",
    "*.yaml",
    "*.toml",
    "*.gitignore",
    "*.json",
    "*.cfg",
    "*.ini",
    "*.sh",
    "*.6",
)
SUNFISH_MOUNT_EXCLUDE_GLOBS: tuple[str, ...] = (
    "**/*.pickle",
    "**/*.png",
    "**/*.bmp",
)
SUNFISH_MOUNT_MAX_BYTES = 600_000
DEFAULT_DEADLINE_MINUTES = 5
_LOGGER = logging.getLogger(__name__)

# Domains allowed for code review reference documentation
CODE_REVIEW_ALLOWED_DOMAINS: tuple[str, ...] = (
    "api.anthropic.com",  # Required for API access
    "peps.python.org",  # PEP documentation
    "docs.python.org",  # Python standard library docs
    "typing.readthedocs.io",  # typing module documentation
    "mypy.readthedocs.io",  # mypy type checker docs
)


def _default_deadline() -> Deadline:
    """Create a fresh default deadline for each request."""

    return Deadline(
        expires_at=datetime.now(UTC) + timedelta(minutes=DEFAULT_DEADLINE_MINUTES)
    )


@dataclass(slots=True, frozen=True)
class ReviewGuidance:
    """Default guidance for the review agent."""

    focus: str = field(
        default=(
            "Identify potential issues, risks, and follow-up questions for the changes "
            "under review."
        ),
        metadata={
            "description": "Default framing instructions for the review assistant."
        },
    )


@dataclass(slots=True, frozen=True)
class ReviewTurnParams:
    """Dataclass for dynamic parameters provided at runtime."""

    request: str = field(metadata={"description": "User-provided review request."})


@dataclass(slots=True, frozen=True)
class ReviewResponse:
    """Structured response emitted by the agent."""

    summary: str
    issues: list[str]
    next_steps: list[str]


@dataclass(slots=True, frozen=True)
class ReferenceParams:
    """Parameters for the reference documentation section."""

    project_name: str = field(
        default="sunfish",
        metadata={"description": "Name of the project being reviewed."},
    )


class CodeReviewLoop(MainLoop[ReviewTurnParams, ReviewResponse]):
    """MainLoop implementation for code review with optional optimization.

    This loop runs as a background worker processing requests from a mailbox.
    It maintains a persistent session across all requests and optionally
    runs workspace digest optimization on first use when enabled.

    Supports optional dead letter queue (DLQ) configuration for handling
    poison messages that fail repeatedly.

    Example::

        responses: InMemoryMailbox[MainLoopResult[ReviewResponse], None] = InMemoryMailbox(
            name="responses"
        )
        requests: InMemoryMailbox[
            MainLoopRequest[ReviewTurnParams], MainLoopResult[ReviewResponse]
        ] = InMemoryMailbox(name="requests")
        loop = CodeReviewLoop(adapter=adapter, requests=requests)
        # Run in background thread
        thread = threading.Thread(target=lambda: loop.run(max_iterations=None))
        thread.start()
        # Send request with reply_to mailbox instance
        params = ReviewTurnParams(request="Review the latest changes")
        requests.send(MainLoopRequest(request=params), reply_to=responses)

    Example with DLQ::

        from weakincentives.runtime import DeadLetter, DLQPolicy

        # Create a DLQ mailbox for failed messages
        dlq_mailbox = InMemoryMailbox(name="review-dlq")
        dlq = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=3,  # Dead-letter after 3 failures
        )
        loop = CodeReviewLoop(adapter=adapter, requests=requests, dlq=dlq)
    """

    _persistent_session: Session
    _template: PromptTemplate[ReviewResponse]
    _overrides_store: LocalPromptOverridesStore
    _override_tag: str
    _use_podman: bool
    _use_claude_agent: bool
    _enable_optimization: bool
    _optimization_done: bool

    def __init__(  # noqa: PLR0913
        self,
        *,
        adapter: ProviderAdapter[ReviewResponse],
        requests: InMemoryMailbox[
            MainLoopRequest[ReviewTurnParams], MainLoopResult[ReviewResponse]
        ],
        overrides_store: LocalPromptOverridesStore | None = None,
        override_tag: str | None = None,
        use_podman: bool = False,
        use_claude_agent: bool = False,
        workspace_section: ClaudeAgentWorkspaceSection | None = None,
        enable_optimization: bool = False,
        dlq: DLQPolicy[
            MainLoopRequest[ReviewTurnParams], MainLoopResult[ReviewResponse]
        ]
        | None = None,
    ) -> None:
        # Configure lease extender to extend message visibility during long tool execution.
        # Extends by 5 minutes every 60 seconds of active work (heartbeats from tools).
        config = MainLoopConfig(
            lease_extender=LeaseExtenderConfig(
                interval=60.0,  # Rate-limit extensions to once per minute
                extension=300,  # Extend by 5 minutes on each extension
            ),
        )
        super().__init__(adapter=adapter, requests=requests, config=config, dlq=dlq)
        self._overrides_store = overrides_store or LocalPromptOverridesStore()
        self._override_tag = resolve_override_tag(
            override_tag, env_var=PROMPT_OVERRIDES_TAG_ENV
        )
        self._use_podman = use_podman
        self._use_claude_agent = use_claude_agent
        self._enable_optimization = enable_optimization
        self._optimization_done = False
        # Create persistent session at construction time
        self._persistent_session = build_logged_session(tags={"app": "code-reviewer"})
        self._template = build_task_prompt(
            session=self._persistent_session,
            use_podman=use_podman,
            use_claude_agent=use_claude_agent,
            workspace_section=workspace_section,
        )
        # Seed overrides for all modes
        self._seed_overrides()

    def _seed_overrides(self) -> None:
        """Initialize prompt overrides store."""
        try:
            self._overrides_store.seed(self._template, tag=self._override_tag)
        except PromptOverridesError as exc:  # pragma: no cover
            raise SystemExit(f"Failed to initialize prompt overrides: {exc}") from exc

    def prepare(
        self,
        request: ReviewTurnParams,
        *,
        experiment: object = None,
    ) -> tuple[Prompt[ReviewResponse], Session]:
        """Prepare prompt and session for the given request.

        Runs workspace optimization on first request if enabled, then creates
        the review prompt and returns the persistent session.
        """
        _ = experiment  # Experiment support not yet implemented
        # Run optimization once on first request (if enabled)
        if self._enable_optimization and not self._optimization_done:
            needs_optimization = (
                self._persistent_session[WorkspaceDigest].latest() is None
            )
            if needs_optimization:
                self._run_optimization()
            self._optimization_done = True

        prompt = Prompt(
            self._template,
            overrides_store=self._overrides_store,
            overrides_tag=self._override_tag,
        ).bind(request)
        return prompt, self._persistent_session

    def _run_optimization(self) -> None:
        """Run workspace digest optimization."""
        from weakincentives.runtime.events import InProcessDispatcher

        _LOGGER.info("Running workspace digest optimization...")
        optimization_session = build_logged_session(parent=self._persistent_session)
        context = OptimizationContext(
            adapter=self._adapter,
            dispatcher=InProcessDispatcher(),
            overrides_store=self._overrides_store,
            overrides_tag=self._override_tag,
            optimization_session=optimization_session,
        )
        optimizer = WorkspaceDigestOptimizer(
            context,
            store_scope=PersistenceScope.SESSION,
        )
        prompt = Prompt(
            self._template,
            overrides_store=self._overrides_store,
            overrides_tag=self._override_tag,
        )
        result = optimizer.optimize(prompt, session=self._persistent_session)
        _LOGGER.info("Workspace digest optimization complete.")
        _LOGGER.debug("Digest: %s", result.digest.strip())

    @property
    def session(self) -> Session:
        """Expose session for external inspection."""
        return self._persistent_session

    @property
    def override_tag(self) -> str:
        """Expose override tag for display."""
        return self._override_tag

    @property
    def use_podman(self) -> bool:
        """Expose workspace mode for display."""
        return self._use_podman


class CodeReviewApp:
    """Owns the REPL loop and user interaction.

    Runs the MainLoop in a background thread while the main thread handles
    user input. Requests are sent via the request mailbox and results are
    received from the response mailbox via reply routing.
    """

    _requests: InMemoryMailbox[
        MainLoopRequest[ReviewTurnParams], MainLoopResult[ReviewResponse]
    ]
    _responses: InMemoryMailbox[MainLoopResult[ReviewResponse], None]
    _loop: CodeReviewLoop
    _worker_thread: threading.Thread | None
    _pending_requests: dict[UUID, str]  # request_id -> user prompt for display
    _use_claude_agent: bool
    _enable_optimization: bool
    _workspace_section: ClaudeAgentWorkspaceSection | None
    _shutdown_requested: bool
    _metrics: InMemoryMetricsCollector

    def __init__(  # noqa: PLR0913
        self,
        adapter: ProviderAdapter[ReviewResponse],
        *,
        overrides_store: LocalPromptOverridesStore | None = None,
        override_tag: str | None = None,
        use_podman: bool = False,
        use_claude_agent: bool = False,
        workspace_section: ClaudeAgentWorkspaceSection | None = None,
        enable_optimization: bool = False,
    ) -> None:
        self._use_claude_agent = use_claude_agent
        self._enable_optimization = enable_optimization
        self._workspace_section = workspace_section
        self._worker_thread = None
        self._pending_requests = {}
        self._shutdown_requested = False
        # Create metrics collector for observability
        self._metrics = InMemoryMetricsCollector(worker_id="code-reviewer")
        # Create mailboxes with reply routing
        self._responses = InMemoryMailbox(name="code-review-responses")
        self._requests = InMemoryMailbox(name="code-review-requests")
        self._loop = CodeReviewLoop(
            adapter=adapter,
            requests=self._requests,
            overrides_store=overrides_store,
            override_tag=override_tag,
            use_podman=use_podman,
            use_claude_agent=use_claude_agent,
            workspace_section=workspace_section,
            enable_optimization=enable_optimization,
        )

    def _run_worker(self) -> None:
        """Background worker that processes requests from the mailbox."""
        # Collect all logs during prompt evaluations to a session-specific file
        log_path = SNAPSHOT_DIR / f"{self._loop.session.session_id}.log"
        with collect_all_logs(log_path):
            # Run indefinitely until mailbox is closed
            self._loop.run(max_iterations=None, wait_time_seconds=5)
        _LOGGER.debug("Worker thread exiting")

    def _render_result(self, result: MainLoopResult[ReviewResponse]) -> None:
        """Render the result to console."""
        print("\n--- Agent Response ---")
        if result.success and result.output is not None:
            answer = _render_response(result.output)
            print(answer)
        else:
            print(f"Error: {result.error}")
        print("\n--- Plan Snapshot ---")
        print(render_plan_snapshot(self._loop.session))
        print("-" * 23 + "\n")

    def _wait_for_response(
        self, request_id: UUID
    ) -> MainLoopResult[ReviewResponse] | None:
        """Poll the response mailbox until we get a response for our request.

        Returns None if mailbox is closed before response received.
        """
        while not self._responses.closed:
            msgs = self._responses.receive(max_messages=1, wait_time_seconds=1)
            if msgs:
                msg = msgs[0]
                result = msg.body
                msg.acknowledge()
                if result.request_id == request_id:
                    return result
                # Not our response - this shouldn't happen in single-user mode
                _LOGGER.warning(
                    "Received response for unknown request: %s", result.request_id
                )
        return None

    def _request_shutdown(self) -> None:
        """Signal handler callback to request shutdown."""
        self._shutdown_requested = True
        self._loop.shutdown(timeout=0)  # Non-blocking, just set the flag

    def run(self) -> None:
        """Start the interactive review session with background worker."""
        print(
            _build_intro(
                self._loop.override_tag,
                use_podman=self._loop.use_podman,
                use_claude_agent=self._use_claude_agent,
                enable_optimization=self._enable_optimization,
            )
        )
        print("Type a review prompt to begin. (Type 'exit' to quit.)")

        # Install signal handlers for graceful shutdown
        coordinator = ShutdownCoordinator.install()
        coordinator.register(self._request_shutdown)

        # Start background worker thread
        self._worker_thread = threading.Thread(
            target=self._run_worker,
            name="code-review-worker",
        )
        self._worker_thread.start()
        _LOGGER.info("Started background worker thread")

        try:
            while not self._shutdown_requested:
                try:
                    user_prompt = input("Review prompt: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if self._shutdown_requested:
                    break

                if not user_prompt or user_prompt.lower() in {"exit", "quit"}:
                    break

                # Send request to mailbox
                request = ReviewTurnParams(request=user_prompt)
                request_event = MainLoopRequest(
                    request=request,
                    deadline=_default_deadline(),
                )
                self._pending_requests[request_event.request_id] = user_prompt
                self._requests.send(request_event, reply_to=self._responses)
                print("Processing request...")

                # Wait for response
                result = self._wait_for_response(request_event.request_id)
                if result is None or self._shutdown_requested:
                    if not self._shutdown_requested:
                        print("Worker stopped unexpectedly.")
                    break
                del self._pending_requests[request_event.request_id]
                self._render_result(result)
        finally:
            self._cleanup()

        print("Goodbye.")
        session_id = self._loop.session.session_id
        dump_session_tree(self._loop.session, SNAPSHOT_DIR)
        print(f"Debug artifacts saved to {SNAPSHOT_DIR}/:")
        print(f"  - {session_id}.jsonl (session snapshots)")
        print(f"  - {session_id}.log (prompt evaluation logs)")
        print("  - .weakincentives/debug/metrics/*.json (metrics snapshots)")

    def _cleanup(self) -> None:
        """Clean up resources."""
        # Archive workspace before cleanup removes the temp directory
        self._archive_workspace()

        # Archive metrics snapshot
        self._archive_metrics()

        # Gracefully shutdown the loop - completes in-flight work
        if self._loop.shutdown(timeout=5.0):
            _LOGGER.info("Worker loop stopped cleanly")
        else:
            _LOGGER.warning("Worker loop did not stop within timeout")

        # Close mailboxes
        self._requests.close()
        self._responses.close()

        # Wait for worker thread to exit
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)

        if self._workspace_section is not None:
            self._workspace_section.cleanup()
            _LOGGER.info("Cleaned up Claude Agent workspace.")

    def _archive_metrics(self) -> None:
        """Archive the metrics snapshot to the debug directory."""
        snapshot = self._metrics.snapshot()
        archive_path = archive_metrics(snapshot, base_dir=SNAPSHOT_DIR.parent)
        _LOGGER.info("Metrics archived to %s", archive_path)

    def _archive_workspace(self) -> None:
        """Archive the workspace filesystem to a zip file.

        Only archives for Claude Agent mode where workspace_section is stored.
        VFS mode uses InMemoryFilesystem with host file copies that already
        exist on disk, so archiving would be redundant.
        """
        if self._workspace_section is None:
            return
        archive_path = archive_filesystem(
            self._workspace_section.filesystem,
            SNAPSHOT_DIR,
            archive_id=self._loop.session.session_id,
        )
        if archive_path is not None:
            _LOGGER.info("Workspace archived to %s", archive_path)
        else:
            _LOGGER.debug("Workspace archive skipped (empty filesystem)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive code review agent example."
    )
    parser.add_argument(
        "--podman",
        action="store_true",
        help="Use Podman sandbox instead of VFS + Asteval (requires Podman connection).",
    )
    parser.add_argument(
        "--claude-agent",
        action="store_true",
        help=(
            "Use Claude Agent SDK adapter with native agentic capabilities. "
            "Requires ANTHROPIC_API_KEY. Uses SDK's built-in tools."
        ),
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable workspace digest optimization on first request.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point used by the `weakincentives` CLI harness."""
    args = parse_args()
    configure_logging()

    if args.claude_agent:
        adapter, workspace_section = build_claude_agent_adapter()
        app = CodeReviewApp(
            adapter,
            use_podman=False,
            use_claude_agent=True,
            workspace_section=workspace_section,
            enable_optimization=args.optimize,
        )
    else:
        adapter = build_adapter()
        app = CodeReviewApp(
            adapter,
            use_podman=args.podman,
            enable_optimization=args.optimize,
        )

    app.run()


def build_adapter() -> ProviderAdapter[ReviewResponse]:
    """Build the OpenAI adapter, checking for the required API key."""
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    return cast(ProviderAdapter[ReviewResponse], OpenAIAdapter(model=model))


def build_claude_agent_adapter() -> tuple[
    ProviderAdapter[ReviewResponse], ClaudeAgentWorkspaceSection
]:
    """Build the Claude Agent SDK adapter with hermetic isolation.

    Creates a workspace section with the test repository mounted, and configures
    the adapter to use the SDK's native agentic capabilities with isolation.

    Uses ``IsolationConfig.inherit_host_auth()`` to validate authentication is
    configured and inherit it from the host environment. Supports both Anthropic
    API (via ANTHROPIC_API_KEY) and AWS Bedrock (via CLAUDE_CODE_USE_BEDROCK=1
    + AWS_REGION).

    Isolation guarantees:
    - Ephemeral HOME directory (no access to ~/.claude)
    - Sandboxed execution with OS-level enforcement
    - Network restricted to Python documentation domains only
    - Skills mounted from demo-skills/ directory
    - AWS config (~/.aws) copied into ephemeral home for Bedrock authentication

    Returns:
        Tuple of (adapter, workspace_section). The workspace section should be
        cloned with the real session before use in prompts.

    Raises:
        SystemExit: If no valid authentication method is available.
    """

    _ensure_test_repository_available()

    # Create a temporary session for workspace materialization
    temp_session = Session(tags={"purpose": "workspace-materialization"})

    # Create workspace section with test repository mounted
    sunfish_path = TEST_REPOSITORIES_ROOT / "sunfish"
    workspace_section = ClaudeAgentWorkspaceSection(
        session=temp_session,
        mounts=(
            ClaudeHostMount(
                host_path=str(sunfish_path),
                mount_path="sunfish",
                include_glob=SUNFISH_MOUNT_INCLUDE_GLOBS,
                exclude_glob=SUNFISH_MOUNT_EXCLUDE_GLOBS,
                max_bytes=SUNFISH_MOUNT_MAX_BYTES,
            ),
        ),
        allowed_host_roots=(str(TEST_REPOSITORIES_ROOT),),
    )

    # Auto-discover and mount all skills from demo-skills/
    skill_mounts = (
        tuple(
            SkillMount(source=skill_dir)
            for skill_dir in DEMO_SKILLS_ROOT.iterdir()
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists()
        )
        if DEMO_SKILLS_ROOT.exists()
        else ()
    )

    # Configure isolation with fail-fast validation
    # IsolationConfig.inherit_host_auth() validates that auth is configured
    try:
        isolation = IsolationConfig.inherit_host_auth(
            network_policy=NetworkPolicy(
                allowed_domains=CODE_REVIEW_ALLOWED_DOMAINS,
            ),
            sandbox=SandboxConfig(
                enabled=True,
                # Allow reading workspace directory
                readable_paths=(str(workspace_section.temp_dir),),
                # Auto-approve bash commands in sandbox (safe with network restrictions)
                bash_auto_allow=True,
            ),
            skills=SkillConfig(skills=skill_mounts),
        )
    except IsolationAuthError as e:
        raise SystemExit(str(e)) from None

    # Get the default model (Opus 4.5) in the appropriate format for the auth mode
    # Environment variable CLAUDE_MODEL can override for both modes
    model = os.getenv("CLAUDE_MODEL", get_default_model())

    adapter = ClaudeAgentSDKAdapter(
        model=model,
        client_config=ClaudeAgentSDKClientConfig(
            permission_mode="bypassPermissions",
            cwd=str(workspace_section.temp_dir),
            isolation=isolation,
            # Ensure all plan steps are completed before the agent finishes
            task_completion_checker=PlanBasedChecker(plan_type=Plan),
        ),
    )

    # Display adapter configuration
    is_bedrock = model.startswith("us.anthropic.")
    auth_mode = "AWS Bedrock" if is_bedrock else "Anthropic API"
    print("\n[Claude Agent SDK Adapter]")
    print(f"  Model: {model}")
    print(f"  Auth:  {auth_mode}")
    print(f"  CWD:   {workspace_section.temp_dir}")

    return cast(ProviderAdapter[ReviewResponse], adapter), workspace_section


def build_task_prompt(
    *,
    session: Session,
    use_podman: bool = False,
    use_claude_agent: bool = False,
    workspace_section: ClaudeAgentWorkspaceSection | None = None,
) -> PromptTemplate[ReviewResponse]:
    """Builds the main prompt template for the code review agent.

    This prompt demonstrates progressive disclosure: the Reference Documentation
    section starts summarized and can be expanded on demand via `open_sections`.

    All modes now support full features including workspace optimization.

    Args:
        session: Session for state management.
        use_podman: If True, use Podman sandbox instead of VFS.
        use_claude_agent: If True, use Claude Agent SDK mode.
        workspace_section: Pre-created workspace section (for Claude Agent mode).
            Will be cloned with the provided session.
    """
    _ensure_test_repository_available()

    if use_claude_agent:
        # Claude Agent SDK mode: full features with ClaudeAgentWorkspaceSection.
        # Streaming mode enables hooks, MCP tool bridging, and workspace optimization.
        # Clone the workspace section with the real session.
        if workspace_section is None:
            msg = "workspace_section is required when use_claude_agent=True"
            raise ValueError(msg)
        cloned_workspace = workspace_section.clone(session=session)
        sections = (
            _build_claude_agent_guidance_section(),
            WorkspaceDigestSection(session=session),
            _build_reference_section(),  # Progressive disclosure section
            PlanningToolsSection(
                session=session,
                strategy=PlanningStrategy.PLAN_ACT_REFLECT,
                accepts_overrides=True,
            ),
            cloned_workspace,
            MarkdownSection[ReviewTurnParams](
                title="Review Request",
                template="${request}",
                key="review-request",
            ),
        )
    else:
        # Standard mode: full prompt with VFS or Podman workspace sections
        workspace_sections = _build_workspace_section(
            session=session, use_podman=use_podman
        )
        sections = (
            _build_review_guidance_section(),
            WorkspaceDigestSection(session=session),
            _build_reference_section(),  # Progressive disclosure section
            PlanningToolsSection(
                session=session,
                strategy=PlanningStrategy.PLAN_ACT_REFLECT,
                accepts_overrides=True,
            ),
            *workspace_sections,
            MarkdownSection[ReviewTurnParams](
                title="Review Request",
                template="${request}",
                key="review-request",
            ),
        )

    # Configure deadline feedback provider to remind the agent of time constraints.
    # - Triggers every 30 seconds or every 5 tool calls (whichever comes first)
    # - Warns when less than 60 seconds remain (severity changes to "warning")
    feedback_providers = (
        FeedbackProviderConfig(
            provider=DeadlineFeedback(warning_threshold_seconds=60),
            trigger=FeedbackTrigger(every_n_seconds=30, every_n_calls=5),
        ),
    )

    return PromptTemplate[ReviewResponse](
        ns="examples/code-review",
        key="code-review-session",
        name="sunfish_code_review_agent",
        sections=sections,
        feedback_providers=feedback_providers,
    )


def _ensure_test_repository_available() -> None:
    if TEST_REPOSITORIES_ROOT.exists():
        return
    raise SystemExit(
        f"Expected test repositories under {TEST_REPOSITORIES_ROOT!s},"
        " but the directory is missing."
    )


def _build_review_guidance_section() -> MarkdownSection[ReviewGuidance]:
    return MarkdownSection[ReviewGuidance](
        title="Code Review Brief",
        template=textwrap.dedent(
            """
            You are a code review assistant exploring the mounted workspace.
            Access the repository under the `sunfish/` directory.

            Use the available tools to stay grounded:
            - Planning tools help you capture multi-step investigations; keep the
              plan updated as you explore.
            - Filesystem tools list directories, read files, and stage edits.
              When available, the `shell_execute` command runs short Podman
              commands (no network access). Mounted files are read-only; use
              writes to stage new snapshots.

            Respond with JSON containing:
            - summary: One paragraph describing your findings so far.
            - issues: List concrete risks, questions, or follow-ups you found.
            - next_steps: Actionable recommendations to progress the task.
            """
        ).strip(),
        default_params=ReviewGuidance(),
        key="code-review-brief",
    )


def _build_claude_agent_guidance_section() -> MarkdownSection[ReviewGuidance]:
    """Build guidance section for Claude Agent SDK mode.

    This version is tailored for the SDK's native agentic capabilities
    and includes references to accessible Python documentation.
    """
    return MarkdownSection[ReviewGuidance](
        title="Code Review Brief",
        template=textwrap.dedent(
            """
            You are a code review assistant. The repository has been mounted
            in your current working directory under `sunfish/`.

            Use your native tools to explore the codebase:
            - Read files to understand the code structure
            - Use Bash to run commands like `find`, `grep`, or `git`
            - Write files if you need to suggest changes

            ## Code Quality References

            You have network access to Python documentation for code quality guidance:
            - **PEP 8** (https://peps.python.org/pep-0008/): Style guide for Python code
            - **PEP 484** (https://peps.python.org/pep-0484/): Type hints
            - **PEP 257** (https://peps.python.org/pep-0257/): Docstring conventions
            - **PEP 20** (https://peps.python.org/pep-0020/): The Zen of Python
            - **Python docs** (https://docs.python.org/): Standard library reference

            When reviewing code, consider citing relevant PEPs for style or design issues.

            Respond with JSON containing:
            - summary: One paragraph describing your findings so far.
            - issues: List concrete risks, questions, or follow-ups you found.
            - next_steps: Actionable recommendations to progress the task.
            """
        ).strip(),
        default_params=ReviewGuidance(),
        key="code-review-brief",
    )


def _build_reference_section() -> MarkdownSection[ReferenceParams]:
    """Build a reference documentation section with progressive disclosure.

    This section starts summarized. The model can call `open_sections` to
    expand it when detailed documentation is needed.
    """
    return MarkdownSection[ReferenceParams](
        title="Reference Documentation",
        template=textwrap.dedent(
            """
            Detailed documentation for the ${project_name} project:

            ## Architecture Overview
            - The project follows a modular architecture with clear separation of concerns.
            - Core components are organized into discrete packages.
            - Dependencies flow inward toward the domain layer.

            ## Code Conventions
            - Follow PEP 8 style guidelines.
            - Use type annotations for all public functions.
            - Document public APIs with docstrings.
            - Prefer composition over inheritance.

            ## Review Checklist
            - Verify that new code includes appropriate tests.
            - Check for security vulnerabilities in user input handling.
            - Ensure error handling follows project conventions.
            - Validate that changes are backward compatible.
            """
        ).strip(),
        summary="Documentation for ${project_name} is available. Request expansion if needed.",
        default_params=ReferenceParams(),
        key="reference-docs",
        visibility=SectionVisibility.SUMMARY,
    )


def _sunfish_mounts() -> tuple[HostMount, ...]:
    return (
        HostMount(
            host_path="sunfish",
            mount_path=VfsPath(("sunfish",)),
            include_glob=SUNFISH_MOUNT_INCLUDE_GLOBS,
            exclude_glob=SUNFISH_MOUNT_EXCLUDE_GLOBS,
            max_bytes=SUNFISH_MOUNT_MAX_BYTES,
        ),
    )


def _build_workspace_section(
    *,
    session: Session,
    use_podman: bool = False,
) -> tuple[MarkdownSection[SupportsDataclass], ...]:
    """Build workspace sections based on the selected sandbox mode.

    By default, returns VFS + Asteval sections. When ``use_podman`` is True,
    attempts to use the Podman sandbox (falling back to VFS+Asteval if unavailable).
    """
    mounts = _sunfish_mounts()
    allowed_roots = (TEST_REPOSITORIES_ROOT,)

    if use_podman:
        connection = PodmanSandboxSection.resolve_connection()
        if connection is None:
            _LOGGER.warning(
                "Podman requested but connection unavailable; "
                "falling back to VFS + Asteval."
            )
        else:
            return (
                PodmanSandboxSection(
                    session=session,
                    config=PodmanSandboxConfig(
                        mounts=mounts,
                        allowed_host_roots=allowed_roots,
                        base_url=connection.get("base_url"),
                        identity=connection.get("identity"),
                        connection_name=connection.get("connection_name"),
                        accepts_overrides=True,
                    ),
                ),
            )

    # Default: VFS + Asteval
    return (
        VfsToolsSection(
            session=session,
            mounts=mounts,
            allowed_host_roots=allowed_roots,
            accepts_overrides=True,
        ),
        AstevalSection(session=session, accepts_overrides=True),
    )


def _build_intro(
    override_tag: str,
    *,
    use_podman: bool,
    use_claude_agent: bool = False,
    enable_optimization: bool = False,
) -> str:
    optimization_status = "enabled (--optimize)" if enable_optimization else "disabled"
    if use_claude_agent:
        return textwrap.dedent(
            f"""
            Launching example code reviewer agent with Claude Agent SDK.
            - Adapter: Claude Agent SDK (native agentic capabilities)
            - Isolation: Hermetic sandbox with ephemeral home directory
            - Repository: test-repositories/sunfish mounted in workspace
            - Tools: SDK's native Read, Write, Bash + custom MCP tools (planning, open_sections)
            - Skills: Auto-discovered from demo-skills/ (code-review, python-style, ascii-art)
            - Network: Access to peps.python.org, docs.python.org for code quality reference
            - Optimization: {optimization_status}
            - Metrics: Enabled (archived to .weakincentives/debug/metrics/)
            - Overrides: Using tag '{override_tag}' (set {PROMPT_OVERRIDES_TAG_ENV} to change).

            Note: Custom MCP tools are bridged via streaming mode for full feature parity.
            """
        ).strip()

    workspace_mode = "Podman sandbox" if use_podman else "VFS + Asteval"
    return textwrap.dedent(
        f"""
        Launching example code reviewer agent.
        - Repository: test-repositories/sunfish mounted under virtual path 'sunfish/'.
        - Workspace: {workspace_mode} (use --podman flag to enable Podman sandbox).
        - Optimization: {optimization_status}
        - Metrics: Enabled (archived to .weakincentives/debug/metrics/)
        - Overrides: Using tag '{override_tag}' (set {PROMPT_OVERRIDES_TAG_ENV} to change).

        Note: Full prompt text and tool calls will be logged to the console for observability.
        """
    ).strip()


def _render_response(output: ReviewResponse) -> str:
    """Render a ReviewResponse as formatted text."""
    lines = [f"Summary: {output.summary}"]
    if output.issues:
        lines.append("Issues:")
        lines.extend(f"- {issue}" for issue in output.issues)
    if output.next_steps:
        lines.append("Next Steps:")
        lines.extend(f"- {step}" for step in output.next_steps)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
