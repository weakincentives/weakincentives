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

"""Textual-based code review agent with MainLoop and EvalLoop modes.

This demo showcases:
- Interactive code review via MainLoop with a rich TUI
- Evaluation mode via EvalLoop on datasets derived from past snapshots
- Snapshot-to-dataset conversion for regression testing

Run with:
    uv run python code_reviewer_example.py              # Interactive mode
    uv run python code_reviewer_example.py --eval       # Evaluation mode
    uv run python code_reviewer_example.py --convert    # Convert snapshots to dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar, cast
from uuid import UUID

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    ProgressBar,
    Static,
    TabbedContent,
    TabPane,
)

from examples import (
    build_logged_session,
    configure_logging,
    render_plan_snapshot,
    resolve_override_tag,
)
from weakincentives.adapters import ProviderAdapter
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer
from weakincentives.contrib.tools import (
    AstevalSection,
    HostMount,
    PlanningStrategy,
    PlanningToolsSection,
    VfsPath,
    VfsToolsSection,
    WorkspaceDigest,
    WorkspaceDigestSection,
)
from weakincentives.deadlines import Deadline
from weakincentives.debug import dump_session as dump_session_tree
from weakincentives.evals import (
    Dataset,
    EvalLoop,
    EvalReport,
    EvalRequest,
    EvalResult,
    Sample,
    Score,
    submit_dataset,
)
from weakincentives.optimizers import (
    OptimizationContext,
    PersistenceScope,
)
from weakincentives.prompt import (
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
    InMemoryMailbox,
    MainLoop,
    MainLoopRequest,
    MainLoopResult,
    Session,
)
from weakincentives.runtime.session.snapshots import Snapshot
from weakincentives.types import SupportsDataclass

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_REPOSITORIES_ROOT = (PROJECT_ROOT / "test-repositories").resolve()
SNAPSHOT_DIR = PROJECT_ROOT / "snapshots"
DATASET_DIR = PROJECT_ROOT / "datasets"
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
MIN_KEYWORD_LENGTH = 3
_LOGGER = logging.getLogger(__name__)


def _default_deadline() -> Deadline:
    """Create a fresh default deadline for each request."""
    return Deadline(
        expires_at=datetime.now(UTC) + timedelta(minutes=DEFAULT_DEADLINE_MINUTES)
    )


# -----------------------------------------------------------------------------
# Data Types
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Expected Response for Evaluation
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ExpectedReviewResponse:
    """Expected output for evaluation - allows partial matching."""

    keywords: tuple[str, ...] = ()
    min_issues: int = 0
    min_next_steps: int = 0


# -----------------------------------------------------------------------------
# Evaluator for Code Review Responses
# -----------------------------------------------------------------------------


def review_response_evaluator(
    output: ReviewResponse, expected: ExpectedReviewResponse
) -> Score:
    """Evaluate a review response against expected criteria."""
    issues_found = len(output.issues) >= expected.min_issues
    steps_found = len(output.next_steps) >= expected.min_next_steps

    all_text = (
        output.summary.lower()
        + " ".join(output.issues).lower()
        + " ".join(output.next_steps).lower()
    )
    keywords_found = all(kw.lower() in all_text for kw in expected.keywords)

    passed = issues_found and steps_found and keywords_found
    score = (
        (1.0 if issues_found else 0.0)
        + (1.0 if steps_found else 0.0)
        + (1.0 if keywords_found else 0.0)
    ) / 3.0

    reasons: list[str] = []
    if not issues_found:
        reasons.append(f"expected >= {expected.min_issues} issues")
    if not steps_found:
        reasons.append(f"expected >= {expected.min_next_steps} next steps")
    if not keywords_found:
        missing = [kw for kw in expected.keywords if kw.lower() not in all_text]
        reasons.append(f"missing keywords: {missing}")

    return Score(
        value=score,
        passed=passed,
        reason="; ".join(reasons) if reasons else "all criteria met",
    )


# -----------------------------------------------------------------------------
# MainLoop Implementation
# -----------------------------------------------------------------------------


class CodeReviewLoop(MainLoop[ReviewTurnParams, ReviewResponse]):
    """MainLoop implementation for code review with auto-optimization.

    This loop runs as a background worker processing requests from a mailbox.
    It maintains a persistent session across all requests and automatically
    runs workspace digest optimization on first use.
    """

    _persistent_session: Session
    _template: PromptTemplate[ReviewResponse]
    _overrides_store: LocalPromptOverridesStore
    _override_tag: str
    _optimization_done: bool

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[ReviewResponse],
        requests: InMemoryMailbox[MainLoopRequest[ReviewTurnParams]],
        responses: InMemoryMailbox[MainLoopResult[ReviewResponse]],
        overrides_store: LocalPromptOverridesStore | None = None,
        override_tag: str | None = None,
    ) -> None:
        super().__init__(adapter=adapter, requests=requests, responses=responses)
        self._overrides_store = overrides_store or LocalPromptOverridesStore()
        self._override_tag = resolve_override_tag(
            override_tag, env_var=PROMPT_OVERRIDES_TAG_ENV
        )
        self._optimization_done = False
        self._persistent_session = build_logged_session(tags={"app": "code-reviewer"})
        self._template = build_task_prompt(session=self._persistent_session)
        self._seed_overrides()

    def _seed_overrides(self) -> None:
        """Initialize prompt overrides store."""
        try:
            self._overrides_store.seed(self._template, tag=self._override_tag)
        except PromptOverridesError as exc:  # pragma: no cover
            raise SystemExit(f"Failed to initialize prompt overrides: {exc}") from exc

    def prepare(
        self, request: ReviewTurnParams
    ) -> tuple[Prompt[ReviewResponse], Session]:
        """Prepare prompt and session for the given request."""
        if not self._optimization_done:
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


# -----------------------------------------------------------------------------
# Snapshot to Dataset Conversion
# -----------------------------------------------------------------------------


def load_snapshots_from_directory(snapshot_dir: Path) -> list[Snapshot]:
    """Load all snapshots from JSONL files in a directory."""
    snapshots: list[Snapshot] = []
    for jsonl_file in sorted(snapshot_dir.glob("*.jsonl")):
        try:
            with jsonl_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        snapshot = Snapshot.from_json(line)
                        snapshots.append(snapshot)
                    except Exception as e:
                        _LOGGER.warning("Failed to parse snapshot: %s", e)
        except Exception as e:
            _LOGGER.warning("Failed to read %s: %s", jsonl_file, e)
    return snapshots


def extract_review_requests_from_snapshots(
    snapshots: list[Snapshot],
) -> list[tuple[str, ReviewResponse | None]]:
    """Extract review request/response pairs from snapshots.

    Returns a list of (request_text, response_or_none) tuples.
    """
    pairs: list[tuple[str, ReviewResponse | None]] = []

    for snapshot in snapshots:
        # Look for ReviewTurnParams and ReviewResponse in the slices
        request_text: str | None = None
        response: ReviewResponse | None = None

        for slice_type, items in snapshot.slices.items():
            type_name = slice_type.__name__

            if type_name == "ReviewTurnParams" and items:
                item = items[-1]
                if hasattr(item, "request"):
                    request_text = str(item.request)  # type: ignore[attr-defined]

            if type_name == "ReviewResponse" and items:
                item = items[-1]
                if (
                    hasattr(item, "summary")
                    and hasattr(item, "issues")
                    and hasattr(item, "next_steps")
                ):
                    response = ReviewResponse(
                        summary=str(item.summary),  # type: ignore[attr-defined]
                        issues=list(item.issues),  # type: ignore[attr-defined]
                        next_steps=list(item.next_steps),  # type: ignore[attr-defined]
                    )

        if request_text:
            pairs.append((request_text, response))

    return pairs


def create_dataset_from_snapshots(
    snapshot_dir: Path,
    output_path: Path,
    *,
    min_issues: int = 1,
    min_next_steps: int = 1,
) -> int:
    """Convert snapshots to a JSONL dataset for evaluation.

    Returns the number of samples written.
    """
    snapshots = load_snapshots_from_directory(snapshot_dir)
    pairs = extract_review_requests_from_snapshots(snapshots)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w") as f:
        for idx, (request, response) in enumerate(pairs):
            # Create expected response based on what we saw
            if response:
                keywords = tuple(
                    word
                    for word in response.summary.split()[:5]
                    if len(word) > MIN_KEYWORD_LENGTH and word.isalpha()
                )
            else:
                keywords = ()

            sample = {
                "id": f"snapshot-{idx}",
                "input": {"request": request},
                "expected": {
                    "keywords": keywords,
                    "min_issues": min(
                        min_issues, len(response.issues) if response else 0
                    ),
                    "min_next_steps": min(
                        min_next_steps, len(response.next_steps) if response else 0
                    ),
                },
            }
            f.write(json.dumps(sample) + "\n")
            count += 1

    return count


def load_review_dataset(
    path: Path,
) -> Dataset[ReviewTurnParams, ExpectedReviewResponse]:
    """Load a review dataset from JSONL."""
    samples: list[Sample[ReviewTurnParams, ExpectedReviewResponse]] = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            input_obj = obj["input"]
            expected_obj = obj["expected"]
            samples.append(
                Sample(
                    id=obj["id"],
                    input=ReviewTurnParams(request=input_obj["request"]),
                    expected=ExpectedReviewResponse(
                        keywords=tuple(expected_obj.get("keywords", [])),
                        min_issues=expected_obj.get("min_issues", 0),
                        min_next_steps=expected_obj.get("min_next_steps", 0),
                    ),
                )
            )
    return Dataset(samples=tuple(samples))


# -----------------------------------------------------------------------------
# Prompt Building
# -----------------------------------------------------------------------------


def build_task_prompt(
    *,
    session: Session,
) -> PromptTemplate[ReviewResponse]:
    """Builds the main prompt template for the code review agent."""
    _ensure_test_repository_available()

    workspace_sections = _build_workspace_section(session=session)
    sections = (
        _build_review_guidance_section(),
        WorkspaceDigestSection(session=session),
        _build_reference_section(),
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
    """Build a reference documentation section with progressive disclosure."""
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
) -> tuple[MarkdownSection[SupportsDataclass], ...]:
    """Build VFS + Asteval workspace sections."""
    mounts = _sunfish_mounts()
    allowed_roots = (TEST_REPOSITORIES_ROOT,)

    return (
        VfsToolsSection(
            session=session,
            mounts=mounts,
            allowed_host_roots=allowed_roots,
            accepts_overrides=True,
        ),
        AstevalSection(session=session, accepts_overrides=True),
    )


def build_adapter() -> ProviderAdapter[ReviewResponse]:
    """Build the OpenAI adapter, checking for the required API key."""
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    return cast(ProviderAdapter[ReviewResponse], OpenAIAdapter(model=model))


def _render_response(output: ReviewResponse) -> str:
    """Render a ReviewResponse as formatted markdown."""
    lines = [f"## Summary\n\n{output.summary}\n"]
    if output.issues:
        lines.append("## Issues\n")
        lines.extend(f"- {issue}" for issue in output.issues)
        lines.append("")
    if output.next_steps:
        lines.append("## Next Steps\n")
        lines.extend(f"- {step}" for step in output.next_steps)
        lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Textual TUI Components
# -----------------------------------------------------------------------------


class ReviewMessage(Message):
    """Message sent when a review is completed."""

    def __init__(self, result: MainLoopResult[ReviewResponse]) -> None:
        super().__init__()
        self.result = result


class EvalProgressMessage(Message):
    """Message for evaluation progress updates."""

    def __init__(self, current: int, total: int, latest: EvalResult | None) -> None:
        super().__init__()
        self.current = current
        self.total = total
        self.latest = latest


class EvalCompleteMessage(Message):
    """Message sent when evaluation is complete."""

    def __init__(self, report: EvalReport) -> None:
        super().__init__()
        self.report = report


class ReviewPanel(Container):
    """Panel showing the latest review response."""

    def compose(self) -> ComposeResult:
        yield Label("Review Response", classes="panel-title")
        yield VerticalScroll(
            Markdown("*Submit a review request to get started...*", id="response-md"),
            id="response-scroll",
        )


class PlanPanel(Container):
    """Panel showing the current plan state."""

    def compose(self) -> ComposeResult:
        yield Label("Plan Snapshot", classes="panel-title")
        yield VerticalScroll(
            Static("No plan yet", id="plan-content"),
            id="plan-scroll",
        )


class StatusBar(Static):
    """Status bar showing current state."""

    status_text: reactive[str] = reactive("Ready")

    def render(self) -> str:
        return f"Status: {self.status_text}"


class HistoryItem(ListItem):
    """A review history item."""

    def __init__(
        self, request: str, response: ReviewResponse, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)
        self.request = request
        self.response = response

    def compose(self) -> ComposeResult:
        preview = self.request[:50] + "..." if len(self.request) > 50 else self.request
        yield Label(preview)


class EvalResultItem(ListItem):
    """An evaluation result item."""

    def __init__(self, result: EvalResult, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.result = result

    def compose(self) -> ComposeResult:
        status = "[green]PASS[/]" if self.result.score.passed else "[red]FAIL[/]"
        yield Label(f"{status} {self.result.sample_id}: {self.result.score.value:.2f}")


# -----------------------------------------------------------------------------
# Main Textual App
# -----------------------------------------------------------------------------


class CodeReviewApp(App[None]):
    """Textual app for interactive code review with MainLoop and EvalLoop support."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-rows: 1fr 3;
    }

    .panel-title {
        background: $primary;
        color: $text;
        padding: 0 1;
        text-style: bold;
    }

    ReviewPanel {
        column-span: 1;
        row-span: 1;
        border: solid $primary;
    }

    PlanPanel {
        column-span: 1;
        row-span: 1;
        border: solid $secondary;
    }

    #input-container {
        column-span: 2;
        height: 3;
        layout: horizontal;
    }

    #review-input {
        width: 1fr;
    }

    #submit-btn {
        width: auto;
        min-width: 12;
    }

    StatusBar {
        dock: bottom;
        height: 1;
        background: $surface;
        padding: 0 1;
    }

    #response-scroll {
        height: 100%;
    }

    #plan-scroll {
        height: 100%;
    }

    #eval-tab {
        height: 100%;
    }

    #eval-progress {
        margin: 1;
    }

    #eval-results-list {
        height: 1fr;
        border: solid $secondary;
    }

    #eval-summary {
        margin: 1;
        height: auto;
    }

    .history-panel {
        border: solid $secondary;
        height: 100%;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+e", "switch_mode", "Switch Mode"),
        Binding("ctrl+r", "run_eval", "Run Eval"),
    ]

    def __init__(
        self,
        adapter: ProviderAdapter[ReviewResponse],
        *,
        eval_mode: bool = False,
        dataset_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._eval_mode = eval_mode
        self._dataset_path = dataset_path
        self._history: list[tuple[str, ReviewResponse]] = []

        # MainLoop setup
        self._requests: InMemoryMailbox[MainLoopRequest[ReviewTurnParams]] = (
            InMemoryMailbox(name="code-review-requests")
        )
        self._responses: InMemoryMailbox[MainLoopResult[ReviewResponse]] = (
            InMemoryMailbox(name="code-review-responses")
        )
        self._loop = CodeReviewLoop(
            adapter=adapter,
            requests=self._requests,
            responses=self._responses,
        )
        self._worker_thread: threading.Thread | None = None
        self._pending_request_id: UUID | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with TabbedContent():
            with TabPane("Review", id="review-tab"):
                with Container(id="review-container"):
                    yield ReviewPanel()
                    yield PlanPanel()
                    with Horizontal(id="input-container"):
                        yield Input(
                            placeholder="Enter your review request...",
                            id="review-input",
                        )
                        yield Button("Submit", id="submit-btn", variant="primary")

            with TabPane("Evaluation", id="eval-tab"):
                with Vertical():
                    yield Label("Evaluation Progress", classes="panel-title")
                    yield ProgressBar(id="eval-progress", total=100, show_eta=True)
                    yield Label("Results:", classes="panel-title")
                    yield ListView(id="eval-results-list")
                    yield Static("", id="eval-summary")
                    with Horizontal():
                        yield Button(
                            "Run Evaluation", id="run-eval-btn", variant="primary"
                        )
                        yield Button(
                            "Load Dataset", id="load-dataset-btn", variant="default"
                        )

            with TabPane("History", id="history-tab"):
                with Vertical(classes="history-panel"):
                    yield Label("Review History", classes="panel-title")
                    yield ListView(id="history-list")

        yield StatusBar()
        yield Footer()

    def on_mount(self) -> None:
        """Start the background worker when the app mounts."""
        self._start_worker()
        if self._eval_mode and self._dataset_path:
            self.call_later(self._run_evaluation)

    def _start_worker(self) -> None:
        """Start the MainLoop worker thread."""
        self._worker_thread = threading.Thread(
            target=self._run_worker,
            name="code-review-worker",
            daemon=True,
        )
        self._worker_thread.start()
        self.query_one(StatusBar).status_text = "Worker started"

    def _run_worker(self) -> None:
        """Background worker that processes requests from the mailbox."""
        self._loop.run(max_iterations=None, wait_time_seconds=1)

    @on(Button.Pressed, "#submit-btn")
    def handle_submit(self) -> None:
        """Handle the submit button press."""
        input_widget = self.query_one("#review-input", Input)
        request_text = input_widget.value.strip()
        if not request_text:
            return

        input_widget.value = ""
        self._submit_review(request_text)

    @on(Input.Submitted, "#review-input")
    def handle_input_submit(self, event: Input.Submitted) -> None:
        """Handle Enter key in the input field."""
        request_text = event.value.strip()
        if not request_text:
            return

        event.input.value = ""
        self._submit_review(request_text)

    def _submit_review(self, request_text: str) -> None:
        """Submit a review request to the MainLoop."""
        self.query_one(StatusBar).status_text = "Processing..."
        self.query_one("#submit-btn", Button).disabled = True

        request = ReviewTurnParams(request=request_text)
        request_event = MainLoopRequest(
            request=request,
            deadline=_default_deadline(),
        )
        self._pending_request_id = request_event.request_id
        self._requests.send(request_event)

        self._wait_for_response(request_event.request_id, request_text)

    @work(thread=True)
    def _wait_for_response(self, request_id: UUID, request_text: str) -> None:
        """Wait for a response from the MainLoop in a background thread."""
        while not self._responses.closed:
            msgs = self._responses.receive(max_messages=1, wait_time_seconds=1)
            if msgs:
                msg = msgs[0]
                result = msg.body
                msg.acknowledge()
                if result.request_id == request_id:
                    self.post_message(ReviewMessage(result))
                    if result.output:
                        self._history.append((request_text, result.output))
                    return

    @on(ReviewMessage)
    def handle_review_result(self, message: ReviewMessage) -> None:
        """Handle a completed review."""
        result = message.result
        self.query_one("#submit-btn", Button).disabled = False

        if result.success and result.output is not None:
            response_md = _render_response(result.output)
            self.query_one("#response-md", Markdown).update(response_md)
            self.query_one(StatusBar).status_text = "Review complete"

            # Update history
            history_list = self.query_one("#history-list", ListView)
            if self._history:
                req, resp = self._history[-1]
                history_list.append(HistoryItem(req, resp))
        else:
            self.query_one("#response-md", Markdown).update(
                f"**Error:** {result.error}"
            )
            self.query_one(StatusBar).status_text = "Review failed"

        # Update plan snapshot
        plan_text = render_plan_snapshot(self._loop.session)
        self.query_one("#plan-content", Static).update(plan_text or "No plan")

    @on(Button.Pressed, "#run-eval-btn")
    def handle_run_eval(self) -> None:
        """Handle the run evaluation button press."""
        self._run_evaluation()

    @work(thread=True)
    def _run_evaluation(self) -> None:
        """Run evaluation on the dataset."""
        dataset_path = self._dataset_path or (DATASET_DIR / "reviews.jsonl")

        if not dataset_path.exists():
            # Create a sample dataset if none exists
            _LOGGER.info("Creating sample evaluation dataset...")
            sample_data = [
                {
                    "id": "sample-1",
                    "input": {"request": "Review the README.md file for clarity"},
                    "expected": {"keywords": [], "min_issues": 0, "min_next_steps": 1},
                },
                {
                    "id": "sample-2",
                    "input": {"request": "Check the main Python files for code style"},
                    "expected": {"keywords": [], "min_issues": 0, "min_next_steps": 1},
                },
            ]
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with dataset_path.open("w") as f:
                for item in sample_data:
                    f.write(json.dumps(item) + "\n")

        dataset = load_review_dataset(dataset_path)
        total = len(dataset)

        # Create mailboxes for EvalLoop
        eval_requests: InMemoryMailbox[
            EvalRequest[ReviewTurnParams, ExpectedReviewResponse]
        ] = InMemoryMailbox(name="eval-requests")
        eval_results: InMemoryMailbox[EvalResult] = InMemoryMailbox(name="eval-results")

        # Create a fresh MainLoop for evaluation
        eval_loop_requests: InMemoryMailbox[MainLoopRequest[ReviewTurnParams]] = (
            InMemoryMailbox(name="eval-mainloop-requests")
        )
        eval_loop_responses: InMemoryMailbox[MainLoopResult[ReviewResponse]] = (
            InMemoryMailbox(name="eval-mainloop-responses")
        )
        eval_main_loop = CodeReviewLoop(
            adapter=self._adapter,
            requests=eval_loop_requests,
            responses=eval_loop_responses,
        )

        # Create EvalLoop
        eval_loop = EvalLoop(
            loop=eval_main_loop,
            evaluator=review_response_evaluator,
            requests=eval_requests,
            results=eval_results,
        )

        # Submit dataset
        submit_dataset(dataset, eval_requests)

        # Run evaluation in a thread
        def run_eval() -> None:
            eval_loop.run(max_iterations=total)

        eval_thread = threading.Thread(target=run_eval, daemon=True)
        eval_thread.start()

        # Collect results with progress updates
        collected: list[EvalResult] = []
        while len(collected) < total:
            msgs = eval_results.receive(max_messages=1, wait_time_seconds=1)
            for msg in msgs:
                collected.append(msg.body)
                msg.acknowledge()
                self.post_message(
                    EvalProgressMessage(len(collected), total, collected[-1])
                )

        eval_thread.join(timeout=5)
        report = EvalReport(results=tuple(collected))
        self.post_message(EvalCompleteMessage(report))

    @on(EvalProgressMessage)
    def handle_eval_progress(self, message: EvalProgressMessage) -> None:
        """Update the evaluation progress bar."""
        progress = self.query_one("#eval-progress", ProgressBar)
        progress.total = message.total
        progress.progress = message.current

        self.query_one(
            StatusBar
        ).status_text = f"Evaluating {message.current}/{message.total}"

        if message.latest:
            results_list = self.query_one("#eval-results-list", ListView)
            results_list.append(EvalResultItem(message.latest))

    @on(EvalCompleteMessage)
    def handle_eval_complete(self, message: EvalCompleteMessage) -> None:
        """Handle evaluation completion."""
        report = message.report
        summary_text = (
            f"**Evaluation Complete**\n\n"
            f"- Total samples: {report.total}\n"
            f"- Successful: {report.successful}\n"
            f"- Pass rate: {report.pass_rate:.1%}\n"
            f"- Mean score: {report.mean_score:.2f}\n"
            f"- Mean latency: {report.mean_latency_ms:.0f}ms\n"
        )
        self.query_one("#eval-summary", Static).update(summary_text)
        self.query_one(StatusBar).status_text = "Evaluation complete"

    @on(ListView.Selected, "#history-list")
    def handle_history_select(self, event: ListView.Selected) -> None:
        """Handle history item selection."""
        if isinstance(event.item, HistoryItem):
            response_md = _render_response(event.item.response)
            self.query_one("#response-md", Markdown).update(response_md)

    def action_quit(self) -> None:
        """Quit the application."""
        self._cleanup()
        self.exit()

    def action_switch_mode(self) -> None:
        """Switch between Review and Eval tabs."""
        tabbed = self.query_one(TabbedContent)
        if tabbed.active == "review-tab":
            tabbed.active = "eval-tab"
        else:
            tabbed.active = "review-tab"

    def action_run_eval(self) -> None:
        """Trigger evaluation from keyboard shortcut."""
        self._run_evaluation()

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._loop.shutdown(timeout=2.0):
            _LOGGER.info("Worker loop stopped cleanly")
        else:
            _LOGGER.warning("Worker loop did not stop within timeout")

        self._requests.close()
        self._responses.close()

        # Dump session for debugging
        dump_session_tree(self._loop.session, SNAPSHOT_DIR)


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Textual-based code review agent with MainLoop and EvalLoop modes."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Start in evaluation mode instead of interactive mode.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to evaluation dataset (JSONL format).",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert snapshots to a dataset and exit.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=SNAPSHOT_DIR,
        help="Directory containing snapshot JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_DIR / "reviews.jsonl",
        help="Output path for converted dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point used by the `weakincentives` CLI harness."""
    args = parse_args()
    configure_logging()

    if args.convert:
        # Convert snapshots to dataset
        count = create_dataset_from_snapshots(
            args.snapshot_dir,
            args.output,
        )
        print(f"Created dataset with {count} samples at {args.output}")
        return

    adapter = build_adapter()
    app = CodeReviewApp(
        adapter,
        eval_mode=args.eval,
        dataset_path=args.dataset,
    )
    app.run()


if __name__ == "__main__":
    main()
