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
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from examples import (
    build_logged_session,
    configure_logging,
    render_plan_snapshot,
    resolve_override_tag,
)
from weakincentives.adapters import PromptResponse, ProviderAdapter
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.deadlines import Deadline
from weakincentives.debug import dump_session as dump_session_tree
from weakincentives.optimizers import (
    OptimizationContext,
    PersistenceScope,
    WorkspaceDigestOptimizer,
)
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    SectionVisibility,
    SupportsDataclass,
)
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptOverridesError,
)
from weakincentives.runtime import (
    EventBus,
    MainLoop,
    MainLoopCompleted,
    MainLoopRequest,
    Session,
)
from weakincentives.tools import (
    AstevalSection,
    HostMount,
    PlanningStrategy,
    PlanningToolsSection,
    PodmanSandboxConfig,
    PodmanSandboxSection,
    VfsPath,
    VfsToolsSection,
    WorkspaceDigest,
    WorkspaceDigestSection,
)

PROJECT_ROOT = Path(__file__).resolve().parent

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
    """MainLoop implementation for code review with auto-optimization.

    This loop maintains a persistent session across all execute() calls and
    automatically runs workspace digest optimization on first use.
    """

    _session: Session
    _template: PromptTemplate[ReviewResponse]
    _overrides_store: LocalPromptOverridesStore
    _override_tag: str
    _use_podman: bool

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[ReviewResponse],
        bus: EventBus,
        overrides_store: LocalPromptOverridesStore | None = None,
        override_tag: str | None = None,
        use_podman: bool = False,
    ) -> None:
        super().__init__(adapter=adapter, bus=bus)
        self._overrides_store = overrides_store or LocalPromptOverridesStore()
        self._override_tag = resolve_override_tag(
            override_tag, env_var=PROMPT_OVERRIDES_TAG_ENV
        )
        self._use_podman = use_podman
        # Create persistent session at construction time
        self._session = build_logged_session(tags={"app": "code-reviewer"})
        self._template = build_task_prompt(session=self._session, use_podman=use_podman)
        self._seed_overrides()

    def _seed_overrides(self) -> None:
        """Initialize prompt overrides store."""
        try:
            self._overrides_store.seed(self._template, tag=self._override_tag)
        except PromptOverridesError as exc:  # pragma: no cover
            raise SystemExit(f"Failed to initialize prompt overrides: {exc}") from exc

    def create_prompt(self, request: ReviewTurnParams) -> Prompt[ReviewResponse]:
        """Create and bind the review prompt for the given request."""
        return Prompt(
            self._template,
            overrides_store=self._overrides_store,
            overrides_tag=self._override_tag,
        ).bind(request)

    def create_session(self) -> Session:
        """Return the persistent session (reused across all turns)."""
        return self._session

    def execute(
        self,
        request: ReviewTurnParams,
        *,
        budget: None = None,
        deadline: Deadline | None = None,
    ) -> tuple[PromptResponse[ReviewResponse], Session]:
        """Execute with auto-optimization for workspace digest.

        If no WorkspaceDigest exists in the session, runs optimization first.
        """
        if self._session.query(WorkspaceDigest).latest() is None:
            self._run_optimization()
        effective_deadline = deadline or _default_deadline()
        return super().execute(request, budget=budget, deadline=effective_deadline)

    def _run_optimization(self) -> None:
        """Run workspace digest optimization."""
        _LOGGER.info("Running workspace digest optimization...")
        optimization_session = build_logged_session(parent=self._session)
        context = OptimizationContext(
            adapter=self._adapter,
            event_bus=self._bus,
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
        result = optimizer.optimize(prompt, session=self._session)
        _LOGGER.info("Workspace digest optimization complete.")
        _LOGGER.debug("Digest: %s", result.digest.strip())

    @property
    def session(self) -> Session:
        """Expose session for external inspection."""
        return self._session

    @property
    def override_tag(self) -> str:
        """Expose override tag for display."""
        return self._override_tag

    @property
    def use_podman(self) -> bool:
        """Expose workspace mode for display."""
        return self._use_podman


class CodeReviewApp:
    """Owns the REPL loop and user interaction."""

    _bus: EventBus
    _loop: CodeReviewLoop

    def __init__(
        self,
        adapter: ProviderAdapter[ReviewResponse],
        *,
        overrides_store: LocalPromptOverridesStore | None = None,
        override_tag: str | None = None,
        use_podman: bool = False,
    ) -> None:
        bus = _create_bus_with_logging()
        self._bus = bus
        self._loop = CodeReviewLoop(
            adapter=adapter,
            bus=bus,
            overrides_store=overrides_store,
            override_tag=override_tag,
            use_podman=use_podman,
        )
        bus.subscribe(MainLoopCompleted, self._on_loop_completed)

    def _on_loop_completed(self, event: object) -> None:
        """Handle completed response from event bus."""
        completed: MainLoopCompleted[ReviewResponse] = event  # type: ignore[assignment]
        answer = _render_response_payload(completed.response)

        print("\n--- Agent Response ---")
        print(answer)
        print("\n--- Plan Snapshot ---")
        print(render_plan_snapshot(self._loop.session))
        print("-" * 23 + "\n")

    def run(self) -> None:
        """Start the interactive review session."""
        print(_build_intro(self._loop.override_tag, use_podman=self._loop.use_podman))
        print("Type a review prompt to begin. (Type 'exit' to quit.)")

        while True:
            try:
                user_prompt = input("Review prompt: ").strip()
            except EOFError:  # pragma: no cover - interactive convenience
                print()
                break

            if not user_prompt or user_prompt.lower() in {"exit", "quit"}:
                break

            request = ReviewTurnParams(request=user_prompt)
            request_event = MainLoopRequest(
                request=request,
                deadline=_default_deadline(),
            )
            self._bus.publish(request_event)

        print("Goodbye.")
        dump_session_tree(self._loop.session, SNAPSHOT_DIR)


def _create_bus_with_logging() -> EventBus:
    """Create an event bus with logging subscribers attached."""
    from examples.logging import attach_logging_subscribers
    from weakincentives.runtime.events import InProcessEventBus

    bus = InProcessEventBus()
    attach_logging_subscribers(bus)
    return bus


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
    return parser.parse_args()


def main() -> None:
    """Entry point used by the `weakincentives` CLI harness."""
    args = parse_args()
    configure_logging()
    adapter = build_adapter()
    app = CodeReviewApp(adapter, use_podman=args.podman)
    app.run()


def build_adapter() -> ProviderAdapter[ReviewResponse]:
    """Build the OpenAI adapter, checking for the required API key."""
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")
    model = os.getenv("OPENAI_MODEL", "gpt-5.1")
    return cast(ProviderAdapter[ReviewResponse], OpenAIAdapter(model=model))


def build_task_prompt(
    *, session: Session, use_podman: bool = False
) -> PromptTemplate[ReviewResponse]:
    """Builds the main prompt template for the code review agent.

    This prompt demonstrates progressive disclosure: the Reference Documentation
    section starts summarized and can be expanded on demand via `open_sections`.
    """
    _ensure_test_repository_available()
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
    return PromptTemplate[ReviewResponse](
        ns="examples/code-review",
        key="code-review-session",
        name="sunfish_code_review_agent",
        sections=sections,
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


def _build_intro(override_tag: str, *, use_podman: bool) -> str:
    workspace_mode = "Podman sandbox" if use_podman else "VFS + Asteval"
    return textwrap.dedent(
        f"""
        Launching example code reviewer agent.
        - Repository: test-repositories/sunfish mounted under virtual path 'sunfish/'.
        - Workspace: {workspace_mode} (use --podman flag to enable Podman sandbox).
        - Overrides: Using tag '{override_tag}' (set {PROMPT_OVERRIDES_TAG_ENV} to change).
        - Auto-optimization: Workspace digest generated on first request.

        Note: Full prompt text and tool calls will be logged to the console for observability.
        """
    ).strip()


def _render_response_payload(response: PromptResponse[ReviewResponse]) -> str:
    if response.output is not None:
        output = response.output
        lines = [f"Summary: {output.summary}"]
        if output.issues:
            lines.append("Issues:")
            lines.extend(f"- {issue}" for issue in output.issues)
        if output.next_steps:
            lines.append("Next Steps:")
            lines.extend(f"- {step}" for step in output.next_steps)
        return "\n".join(lines)
    if response.text:
        return response.text
    return "(no response from assistant)"


if __name__ == "__main__":
    main()
