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
    ChecklistItem,
    ChecklistSection,
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

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[ReviewResponse],
        bus: EventBus,
        overrides_store: LocalPromptOverridesStore | None = None,
        override_tag: str | None = None,
    ) -> None:
        super().__init__(adapter=adapter, bus=bus)
        self._overrides_store = overrides_store or LocalPromptOverridesStore()
        self._override_tag = resolve_override_tag(
            override_tag, env_var=PROMPT_OVERRIDES_TAG_ENV
        )
        # Create persistent session at construction time
        self._session = build_logged_session(tags={"app": "code-reviewer"})
        self._template = build_task_prompt(session=self._session)
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
    ) -> None:
        bus = _create_bus_with_logging()
        self._bus = bus
        self._loop = CodeReviewLoop(
            adapter=adapter,
            bus=bus,
            overrides_store=overrides_store,
            override_tag=override_tag,
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
        print(_build_intro(self._loop.override_tag))
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


def main() -> None:
    """Entry point used by the `weakincentives` CLI harness."""
    configure_logging()
    adapter = build_adapter()
    app = CodeReviewApp(adapter)
    app.run()


def build_adapter() -> ProviderAdapter[ReviewResponse]:
    """Build the OpenAI adapter, checking for the required API key."""
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")
    model = os.getenv("OPENAI_MODEL", "gpt-5.1")
    return cast(ProviderAdapter[ReviewResponse], OpenAIAdapter(model=model))


def build_task_prompt(*, session: Session) -> PromptTemplate[ReviewResponse]:
    """Builds the main prompt template for the code review agent.

    This prompt demonstrates progressive disclosure: the Reference Documentation
    section and domain-specific checklists start summarized and can be expanded
    on demand via `open_sections`.
    """
    _ensure_test_repository_available()
    workspace_section = _build_workspace_section(session=session)
    sections = (
        _build_review_guidance_section(),
        WorkspaceDigestSection(session=session),
        _build_reference_section(),  # Progressive disclosure section
        # Domain-specific review checklists (progressive disclosure)
        _build_security_checklist(),
        _build_performance_checklist(),
        _build_api_checklist(),
        _build_test_checklist(),
        PlanningToolsSection(
            session=session,
            strategy=PlanningStrategy.PLAN_ACT_REFLECT,
            accepts_overrides=True,
        ),
        workspace_section,
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


# =============================================================================
# Domain-Specific Review Checklists
# =============================================================================


def _build_security_checklist() -> ChecklistSection:
    """Build a security review checklist based on OWASP Top 10."""
    items = [
        # Injection
        ChecklistItem(
            "Verify parameterized queries for all database operations",
            category="Injection Prevention",
            severity="critical",
        ),
        ChecklistItem(
            "Check for command injection in shell/system calls",
            category="Injection Prevention",
            severity="critical",
        ),
        ChecklistItem(
            "Validate LDAP/XPath/NoSQL query construction",
            category="Injection Prevention",
            severity="high",
        ),
        # Authentication
        ChecklistItem(
            "Ensure password hashing uses bcrypt/argon2 with proper cost",
            category="Authentication",
            severity="critical",
        ),
        ChecklistItem(
            "Verify multi-factor authentication for sensitive operations",
            category="Authentication",
            severity="high",
        ),
        ChecklistItem(
            "Check session token generation uses cryptographically secure randomness",
            category="Authentication",
            severity="high",
        ),
        ChecklistItem(
            "Validate session timeout and invalidation on logout",
            category="Authentication",
            severity="medium",
        ),
        # Data Exposure
        ChecklistItem(
            "Confirm sensitive data encrypted at rest (AES-256 or equivalent)",
            category="Data Protection",
            severity="critical",
        ),
        ChecklistItem(
            "Verify TLS 1.2+ for all data in transit",
            category="Data Protection",
            severity="critical",
        ),
        ChecklistItem(
            "Check for accidental logging of sensitive data (PII, credentials)",
            category="Data Protection",
            severity="high",
        ),
        ChecklistItem(
            "Validate secrets management (no hardcoded credentials)",
            category="Data Protection",
            severity="critical",
        ),
        # Access Control
        ChecklistItem(
            "Verify authorization checks on all protected endpoints",
            category="Access Control",
            severity="critical",
        ),
        ChecklistItem(
            "Check for IDOR vulnerabilities in resource access",
            category="Access Control",
            severity="high",
        ),
        ChecklistItem(
            "Validate principle of least privilege in role assignments",
            category="Access Control",
            severity="medium",
        ),
        # Security Misconfiguration
        ChecklistItem(
            "Ensure security headers present (CSP, X-Frame-Options, etc.)",
            category="Security Configuration",
            severity="high",
        ),
        ChecklistItem(
            "Check for exposed debug endpoints or verbose error messages",
            category="Security Configuration",
            severity="high",
        ),
        ChecklistItem(
            "Verify default credentials changed and unnecessary features disabled",
            category="Security Configuration",
            severity="medium",
        ),
        # XSS
        ChecklistItem(
            "Validate output encoding for all user-controlled content",
            category="XSS Prevention",
            severity="critical",
        ),
        ChecklistItem(
            "Check for DOM-based XSS in client-side JavaScript",
            category="XSS Prevention",
            severity="high",
        ),
        # Deserialization
        ChecklistItem(
            "Avoid deserializing untrusted data or use safe alternatives",
            category="Deserialization",
            severity="critical",
        ),
        # Logging & Monitoring
        ChecklistItem(
            "Ensure security events are logged (auth failures, access denials)",
            category="Logging & Monitoring",
            severity="medium",
        ),
    ]

    return ChecklistSection(
        title="Security Review Checklist",
        key="checklist.security",
        domain="security",
        items=items,
        preamble=textwrap.dedent(
            """
            Review the code against these security criteria based on OWASP Top 10.
            Mark items as verified or flag concerns for follow-up.
            """
        ).strip(),
        visibility=SectionVisibility.SUMMARY,
    )


def _build_performance_checklist() -> ChecklistSection:
    """Build a performance review checklist."""
    items = [
        # Database Performance
        ChecklistItem(
            "Check for N+1 query patterns in ORM usage",
            category="Database Queries",
            severity="critical",
        ),
        ChecklistItem(
            "Verify indexes exist for frequent query predicates",
            category="Database Queries",
            severity="high",
        ),
        ChecklistItem(
            "Review query plans for expensive operations (full scans, sorts)",
            category="Database Queries",
            severity="high",
        ),
        ChecklistItem(
            "Check for unbounded queries (missing LIMIT clauses)",
            category="Database Queries",
            severity="high",
        ),
        ChecklistItem(
            "Validate connection pooling configuration",
            category="Database Queries",
            severity="medium",
        ),
        # Memory Management
        ChecklistItem(
            "Check for memory leaks in long-running processes",
            category="Memory Management",
            severity="critical",
        ),
        ChecklistItem(
            "Verify large collections are processed in batches/streams",
            category="Memory Management",
            severity="high",
        ),
        ChecklistItem(
            "Check for circular references preventing garbage collection",
            category="Memory Management",
            severity="medium",
        ),
        ChecklistItem(
            "Review buffer sizes for I/O operations",
            category="Memory Management",
            severity="medium",
        ),
        # Caching
        ChecklistItem(
            "Identify cacheable operations (expensive computations, remote calls)",
            category="Caching",
            severity="medium",
        ),
        ChecklistItem(
            "Verify cache invalidation strategy is correct",
            category="Caching",
            severity="high",
        ),
        ChecklistItem(
            "Check cache TTLs align with data freshness requirements",
            category="Caching",
            severity="medium",
        ),
        # Concurrency
        ChecklistItem(
            "Verify thread-safe access to shared mutable state",
            category="Concurrency",
            severity="critical",
        ),
        ChecklistItem(
            "Check for deadlock potential in lock ordering",
            category="Concurrency",
            severity="high",
        ),
        ChecklistItem(
            "Review async/await patterns for blocking operations",
            category="Concurrency",
            severity="high",
        ),
        ChecklistItem(
            "Validate thread pool sizing for workload characteristics",
            category="Concurrency",
            severity="medium",
        ),
        # Network & I/O
        ChecklistItem(
            "Check for appropriate timeouts on external calls",
            category="Network & I/O",
            severity="high",
        ),
        ChecklistItem(
            "Verify retry logic with exponential backoff",
            category="Network & I/O",
            severity="medium",
        ),
        ChecklistItem(
            "Review payload sizes for API calls (compression, pagination)",
            category="Network & I/O",
            severity="medium",
        ),
        # Algorithms
        ChecklistItem(
            "Verify algorithm complexity is appropriate for data scale",
            category="Algorithms",
            severity="high",
        ),
        ChecklistItem(
            "Check for unnecessary object allocations in hot paths",
            category="Algorithms",
            severity="medium",
        ),
    ]

    return ChecklistSection(
        title="Performance Review Checklist",
        key="checklist.performance",
        domain="performance",
        items=items,
        preamble=textwrap.dedent(
            """
            Review the code for performance concerns. Focus on database access patterns,
            memory usage, caching opportunities, and concurrency correctness.
            """
        ).strip(),
        visibility=SectionVisibility.SUMMARY,
    )


def _build_api_checklist() -> ChecklistSection:
    """Build an API review checklist."""
    items = [
        # Breaking Changes
        ChecklistItem(
            "Check for removed or renamed endpoints",
            category="Breaking Changes",
            severity="critical",
        ),
        ChecklistItem(
            "Verify no required fields added to request schemas",
            category="Breaking Changes",
            severity="critical",
        ),
        ChecklistItem(
            "Check for changed response field types or removal",
            category="Breaking Changes",
            severity="critical",
        ),
        ChecklistItem(
            "Validate HTTP method changes maintain semantics",
            category="Breaking Changes",
            severity="high",
        ),
        ChecklistItem(
            "Review changes to authentication/authorization requirements",
            category="Breaking Changes",
            severity="critical",
        ),
        # Versioning
        ChecklistItem(
            "Verify API version is incremented for breaking changes",
            category="Versioning",
            severity="high",
        ),
        ChecklistItem(
            "Check deprecated endpoints have sunset timeline",
            category="Versioning",
            severity="medium",
        ),
        ChecklistItem(
            "Validate version negotiation works correctly",
            category="Versioning",
            severity="medium",
        ),
        # Request/Response Design
        ChecklistItem(
            "Verify RESTful conventions (resource naming, HTTP verbs)",
            category="API Design",
            severity="medium",
        ),
        ChecklistItem(
            "Check for consistent naming conventions (camelCase/snake_case)",
            category="API Design",
            severity="medium",
        ),
        ChecklistItem(
            "Validate pagination for list endpoints",
            category="API Design",
            severity="high",
        ),
        ChecklistItem(
            "Review response envelope structure consistency",
            category="API Design",
            severity="medium",
        ),
        # Error Handling
        ChecklistItem(
            "Verify appropriate HTTP status codes for error conditions",
            category="Error Handling",
            severity="high",
        ),
        ChecklistItem(
            "Check error response includes actionable details",
            category="Error Handling",
            severity="medium",
        ),
        ChecklistItem(
            "Validate rate limit responses include retry-after",
            category="Error Handling",
            severity="medium",
        ),
        # Documentation
        ChecklistItem(
            "Verify OpenAPI/Swagger spec updated for changes",
            category="Documentation",
            severity="high",
        ),
        ChecklistItem(
            "Check request/response examples are accurate",
            category="Documentation",
            severity="medium",
        ),
        ChecklistItem(
            "Validate changelog entry for API changes",
            category="Documentation",
            severity="medium",
        ),
        # Security (API-specific)
        ChecklistItem(
            "Verify input validation on all request parameters",
            category="API Security",
            severity="high",
        ),
        ChecklistItem(
            "Check for rate limiting on public endpoints",
            category="API Security",
            severity="high",
        ),
        ChecklistItem(
            "Validate CORS configuration is appropriate",
            category="API Security",
            severity="medium",
        ),
    ]

    return ChecklistSection(
        title="API Review Checklist",
        key="checklist.api",
        domain="API",
        items=items,
        preamble=textwrap.dedent(
            """
            Review API changes for backward compatibility, versioning correctness,
            and documentation completeness. Pay special attention to breaking changes.
            """
        ).strip(),
        visibility=SectionVisibility.SUMMARY,
    )


def _build_test_checklist() -> ChecklistSection:
    """Build a test review checklist."""
    items = [
        # Edge Cases
        ChecklistItem(
            "Verify null/None input handling is tested",
            category="Edge Cases",
            severity="high",
        ),
        ChecklistItem(
            "Check empty collection edge cases covered",
            category="Edge Cases",
            severity="high",
        ),
        ChecklistItem(
            "Validate boundary values tested (min, max, overflow)",
            category="Edge Cases",
            severity="high",
        ),
        ChecklistItem(
            "Test concurrent access scenarios if applicable",
            category="Edge Cases",
            severity="medium",
        ),
        ChecklistItem(
            "Verify error/exception paths are tested",
            category="Edge Cases",
            severity="high",
        ),
        # Mocking Patterns
        ChecklistItem(
            "Check mocks verify interaction contracts",
            category="Mocking",
            severity="high",
        ),
        ChecklistItem(
            "Verify external dependencies are isolated",
            category="Mocking",
            severity="high",
        ),
        ChecklistItem(
            "Avoid over-mocking (testing implementation vs behavior)",
            category="Mocking",
            severity="medium",
        ),
        ChecklistItem(
            "Check mock return values match real implementation contracts",
            category="Mocking",
            severity="high",
        ),
        ChecklistItem(
            "Verify time-dependent tests use controlled clocks",
            category="Mocking",
            severity="medium",
        ),
        # Test Structure
        ChecklistItem(
            "Verify test names describe behavior being tested",
            category="Test Quality",
            severity="medium",
        ),
        ChecklistItem(
            "Check tests follow Arrange-Act-Assert pattern",
            category="Test Quality",
            severity="medium",
        ),
        ChecklistItem(
            "Validate tests are independent and repeatable",
            category="Test Quality",
            severity="high",
        ),
        ChecklistItem(
            "Review test data setup for clarity and maintainability",
            category="Test Quality",
            severity="medium",
        ),
        # Coverage
        ChecklistItem(
            "Verify new code has corresponding tests",
            category="Coverage",
            severity="high",
        ),
        ChecklistItem(
            "Check branch coverage for conditional logic",
            category="Coverage",
            severity="high",
        ),
        ChecklistItem(
            "Validate integration tests for cross-component flows",
            category="Coverage",
            severity="medium",
        ),
        # Assertions
        ChecklistItem(
            "Verify assertions are specific (not just 'not null')",
            category="Assertions",
            severity="medium",
        ),
        ChecklistItem(
            "Check for appropriate use of assertion messages",
            category="Assertions",
            severity="low",
        ),
        ChecklistItem(
            "Validate exception assertions check type and message",
            category="Assertions",
            severity="medium",
        ),
        # Performance Tests
        ChecklistItem(
            "Check performance-critical paths have benchmarks",
            category="Performance Testing",
            severity="low",
        ),
    ]

    return ChecklistSection(
        title="Test Review Checklist",
        key="checklist.test",
        domain="testing",
        items=items,
        preamble=textwrap.dedent(
            """
            Review tests for completeness, quality, and correctness. Ensure edge cases
            are covered and mocking patterns follow best practices.
            """
        ).strip(),
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
) -> MarkdownSection[SupportsDataclass]:
    mounts = _sunfish_mounts()
    allowed_roots = (TEST_REPOSITORIES_ROOT,)
    connection = PodmanSandboxSection.resolve_connection()
    if connection is None:
        _LOGGER.info(
            "Podman connection unavailable; falling back to VFS tools for the code reviewer example."
        )
        return VfsToolsSection(
            session=session,
            mounts=mounts,
            allowed_host_roots=allowed_roots,
            accepts_overrides=True,
        )

    return PodmanSandboxSection(
        session=session,
        config=PodmanSandboxConfig(
            mounts=mounts,
            allowed_host_roots=allowed_roots,
            base_url=connection.get("base_url"),
            identity=connection.get("identity"),
            connection_name=connection.get("connection_name"),
            accepts_overrides=True,
        ),
    )


def _build_intro(override_tag: str) -> str:
    return textwrap.dedent(
        f"""
        Launching example code reviewer agent.
        - Repository: test-repositories/sunfish mounted under virtual path 'sunfish/'.
        - Tools: Planning and a filesystem workspace (with a Podman shell if available).
        - Checklists: Security, Performance, API, and Test review checklists (expand on demand).
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
