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

"""Code review agent with Textual TUI.

Demonstrates MainLoop for interactive reviews and EvalLoop for evaluation.
Everything is driven through the TUI - no CLI flags needed.

    uv run python code_reviewer_example.py
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar, cast
from uuid import UUID

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
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
    EvalResult,
    Sample,
    Score,
    submit_dataset,
)
from weakincentives.optimizers import OptimizationContext, PersistenceScope
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

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
TEST_REPOSITORIES_ROOT = PROJECT_ROOT / "test-repositories"
SNAPSHOT_DIR = PROJECT_ROOT / "snapshots"
DATASET_PATH = PROJECT_ROOT / "datasets" / "reviews.jsonl"

# Config
DEADLINE_MINUTES = 5
MOUNT_MAX_BYTES = 600_000
MOUNT_INCLUDE = ("*.md", "*.py", "*.txt", "*.yml", "*.yaml", "*.toml", "*.json", "*.sh")
MOUNT_EXCLUDE = ("**/*.pickle", "**/*.png", "**/*.bmp")

_LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Types
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ReviewRequest:
    """User request for code review."""

    request: str


@dataclass(slots=True, frozen=True)
class ReviewResponse:
    """Agent response with findings."""

    summary: str
    issues: list[str]
    next_steps: list[str]


@dataclass(slots=True, frozen=True)
class ExpectedResponse:
    """Expected output for evaluation."""

    min_issues: int = 0
    min_next_steps: int = 0


@dataclass(slots=True, frozen=True)
class _EmptyParams:
    """Empty params for static sections."""

    pass


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------


def evaluate_response(output: ReviewResponse, expected: ExpectedResponse) -> Score:
    """Score a review response against expectations."""
    has_issues = len(output.issues) >= expected.min_issues
    has_steps = len(output.next_steps) >= expected.min_next_steps
    passed = has_issues and has_steps
    score = (1.0 if has_issues else 0.0) + (1.0 if has_steps else 0.0)
    return Score(
        value=score / 2, passed=passed, reason="ok" if passed else "incomplete"
    )


# -----------------------------------------------------------------------------
# MainLoop
# -----------------------------------------------------------------------------


class ReviewLoop(MainLoop[ReviewRequest, ReviewResponse]):
    """MainLoop for code review with auto-optimization."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[ReviewResponse],
        requests: InMemoryMailbox[MainLoopRequest[ReviewRequest]],
        responses: InMemoryMailbox[MainLoopResult[ReviewResponse]],
    ) -> None:
        super().__init__(adapter=adapter, requests=requests, responses=responses)
        self._store = LocalPromptOverridesStore()
        self._tag = resolve_override_tag(None, env_var="CODE_REVIEW_PROMPT_TAG")
        self._session = build_logged_session(tags={"app": "code-reviewer"})
        self._template = _build_prompt(self._session)
        self._optimized = False
        self._seed_overrides()

    def _seed_overrides(self) -> None:
        try:
            self._store.seed(self._template, tag=self._tag)
        except PromptOverridesError as e:
            raise SystemExit(f"Failed to seed overrides: {e}") from e

    def prepare(self, request: ReviewRequest) -> tuple[Prompt[ReviewResponse], Session]:
        if not self._optimized:
            if self._session[WorkspaceDigest].latest() is None:
                self._optimize()
            self._optimized = True
        prompt = Prompt(
            self._template, overrides_store=self._store, overrides_tag=self._tag
        )
        return prompt.bind(request), self._session

    def _optimize(self) -> None:
        from weakincentives.runtime.events import InProcessDispatcher

        _LOG.info("Optimizing workspace digest...")
        ctx = OptimizationContext(
            adapter=self._adapter,
            dispatcher=InProcessDispatcher(),
            overrides_store=self._store,
            overrides_tag=self._tag,
            optimization_session=build_logged_session(parent=self._session),
        )
        optimizer = WorkspaceDigestOptimizer(ctx, store_scope=PersistenceScope.SESSION)
        prompt = Prompt(
            self._template, overrides_store=self._store, overrides_tag=self._tag
        )
        optimizer.optimize(prompt, session=self._session)
        _LOG.info("Optimization complete")

    @property
    def session(self) -> Session:
        return self._session


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------


def _build_prompt(session: Session) -> PromptTemplate[ReviewResponse]:
    if not TEST_REPOSITORIES_ROOT.exists():
        raise SystemExit(f"Missing test-repositories at {TEST_REPOSITORIES_ROOT}")

    mount = HostMount(
        host_path="sunfish",
        mount_path=VfsPath(("sunfish",)),
        include_glob=MOUNT_INCLUDE,
        exclude_glob=MOUNT_EXCLUDE,
        max_bytes=MOUNT_MAX_BYTES,
    )

    return PromptTemplate[ReviewResponse](
        ns="examples/code-review",
        key="code-review-session",
        name="sunfish_code_review_agent",
        sections=(
            MarkdownSection[_EmptyParams](
                title="Instructions",
                template=textwrap.dedent("""
                    You are a code review assistant. Explore the sunfish/ workspace.

                    Use planning tools to track your investigation.
                    Use filesystem tools to read files.

                    Respond with JSON:
                    - summary: One paragraph of findings
                    - issues: List of concrete problems found
                    - next_steps: Actionable recommendations
                """).strip(),
                key="instructions",
            ),
            WorkspaceDigestSection(session=session),
            MarkdownSection[_EmptyParams](
                title="Reference",
                template="Project documentation is available on request.",
                summary="Documentation available.",
                key="reference",
                visibility=SectionVisibility.SUMMARY,
            ),
            PlanningToolsSection(
                session=session,
                strategy=PlanningStrategy.PLAN_ACT_REFLECT,
                accepts_overrides=True,
            ),
            VfsToolsSection(
                session=session,
                mounts=(mount,),
                allowed_host_roots=(TEST_REPOSITORIES_ROOT,),
                accepts_overrides=True,
            ),
            AstevalSection(session=session, accepts_overrides=True),
            MarkdownSection[ReviewRequest](
                title="Request",
                template="${request}",
                key="request",
            ),
        ),
    )


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


def load_dataset(path: Path) -> Dataset[ReviewRequest, ExpectedResponse]:
    """Load evaluation dataset from JSONL."""
    samples: list[Sample[ReviewRequest, ExpectedResponse]] = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            samples.append(
                Sample(
                    id=obj["id"],
                    input=ReviewRequest(request=obj["input"]["request"]),
                    expected=ExpectedResponse(
                        min_issues=obj["expected"].get("min_issues", 0),
                        min_next_steps=obj["expected"].get("min_next_steps", 0),
                    ),
                )
            )
    return Dataset(samples=tuple(samples))


def create_sample_dataset(path: Path) -> None:
    """Create a minimal sample dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        {
            "id": "1",
            "input": {"request": "Review README.md"},
            "expected": {"min_issues": 0, "min_next_steps": 1},
        },
        {
            "id": "2",
            "input": {"request": "Check code style"},
            "expected": {"min_issues": 0, "min_next_steps": 1},
        },
    ]
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# -----------------------------------------------------------------------------
# TUI Messages
# -----------------------------------------------------------------------------


class ReviewDone(Message):
    def __init__(self, result: MainLoopResult[ReviewResponse]) -> None:
        super().__init__()
        self.result = result


class EvalProgress(Message):
    def __init__(self, current: int, total: int, latest: EvalResult | None) -> None:
        super().__init__()
        self.current = current
        self.total = total
        self.latest = latest


class EvalDone(Message):
    def __init__(self, report: EvalReport) -> None:
        super().__init__()
        self.report = report


# -----------------------------------------------------------------------------
# TUI Widgets
# -----------------------------------------------------------------------------


class ResultItem(ListItem):
    def __init__(self, result: EvalResult) -> None:
        super().__init__()
        self.result = result

    def compose(self) -> ComposeResult:
        icon = "[green]✓[/]" if self.result.score.passed else "[red]✗[/]"
        yield Label(f"{icon} {self.result.sample_id}: {self.result.score.value:.0%}")


# -----------------------------------------------------------------------------
# TUI App
# -----------------------------------------------------------------------------


class ReviewApp(App[None]):
    """Code review TUI with MainLoop and EvalLoop."""

    CSS = """
    #main { height: 1fr; }
    #left { width: 2fr; border: solid $primary; }
    #right { width: 1fr; border: solid $secondary; }
    #input-row { height: 3; dock: bottom; }
    #input { width: 1fr; }
    #eval-panel { height: 1fr; display: none; }
    #eval-panel.visible { display: block; }
    #results { height: 1fr; }
    #summary { height: auto; padding: 1; }
    .title { background: $primary; padding: 0 1; text-style: bold; }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+e", "toggle_eval", "Toggle Eval"),
        Binding("ctrl+r", "run_eval", "Run Eval"),
    ]

    def __init__(self, adapter: ProviderAdapter[ReviewResponse]) -> None:
        super().__init__()
        self._adapter = adapter
        self._requests: InMemoryMailbox[MainLoopRequest[ReviewRequest]] = (
            InMemoryMailbox(name="requests")
        )
        self._responses: InMemoryMailbox[MainLoopResult[ReviewResponse]] = (
            InMemoryMailbox(name="responses")
        )
        self._loop = ReviewLoop(
            adapter=adapter, requests=self._requests, responses=self._responses
        )
        self._worker: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Label("Response", classes="title")
                yield VerticalScroll(
                    Markdown("*Enter a review request below*", id="response")
                )
            with Vertical(id="right"):
                yield Label("Plan", classes="title")
                yield VerticalScroll(Static("No plan", id="plan"))
        with Vertical(id="eval-panel"):
            yield Label("Evaluation", classes="title")
            yield ProgressBar(id="progress", total=100)
            yield ListView(id="results")
            yield Static("", id="summary")
            yield Button("Run Evaluation", id="run-btn")
        with Horizontal(id="input-row"):
            yield Input(placeholder="Enter review request...", id="input")
            yield Button("Submit", id="submit-btn")
        yield Footer()

    def on_mount(self) -> None:
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def _run_loop(self) -> None:
        self._loop.run(max_iterations=None, wait_time_seconds=1)

    @on(Input.Submitted, "#input")
    @on(Button.Pressed, "#submit-btn")
    def submit_review(self) -> None:
        inp = self.query_one("#input", Input)
        text = inp.value.strip()
        if not text:
            return
        inp.value = ""
        self.query_one("#submit-btn", Button).disabled = True
        request = MainLoopRequest(
            request=ReviewRequest(request=text),
            deadline=Deadline(
                expires_at=datetime.now(UTC) + timedelta(minutes=DEADLINE_MINUTES)
            ),
        )
        self._requests.send(request)
        self._wait_response(request.request_id)

    @work(thread=True)
    def _wait_response(self, request_id: UUID) -> None:
        while not self._responses.closed:
            msgs = self._responses.receive(max_messages=1, wait_time_seconds=1)
            for msg in msgs:
                if msg.body.request_id == request_id:
                    msg.acknowledge()
                    self.post_message(ReviewDone(msg.body))
                    return

    @on(ReviewDone)
    def show_result(self, event: ReviewDone) -> None:
        self.query_one("#submit-btn", Button).disabled = False
        result = event.result
        if result.success and result.output:
            out = result.output
            md = f"## Summary\n{out.summary}\n\n"
            if out.issues:
                md += "## Issues\n" + "\n".join(f"- {i}" for i in out.issues) + "\n\n"
            if out.next_steps:
                md += "## Next Steps\n" + "\n".join(f"- {s}" for s in out.next_steps)
            self.query_one("#response", Markdown).update(md)
        else:
            self.query_one("#response", Markdown).update(f"**Error:** {result.error}")
        plan = render_plan_snapshot(self._loop.session) or "No plan"
        self.query_one("#plan", Static).update(plan)

    def action_toggle_eval(self) -> None:
        panel = self.query_one("#eval-panel")
        panel.toggle_class("visible")

    def action_run_eval(self) -> None:
        self._run_evaluation()

    @on(Button.Pressed, "#run-btn")
    def handle_run_btn(self) -> None:
        self._run_evaluation()

    @work(thread=True)
    def _run_evaluation(self) -> None:
        if not DATASET_PATH.exists():
            create_sample_dataset(DATASET_PATH)

        dataset = load_dataset(DATASET_PATH)
        total = len(dataset)

        req_mb: InMemoryMailbox[MainLoopRequest[ReviewRequest]] = InMemoryMailbox(
            name="eval-req"
        )
        res_mb: InMemoryMailbox[MainLoopResult[ReviewResponse]] = InMemoryMailbox(
            name="eval-res"
        )
        loop = ReviewLoop(adapter=self._adapter, requests=req_mb, responses=res_mb)

        from weakincentives.evals import EvalRequest

        eval_req: InMemoryMailbox[EvalRequest[ReviewRequest, ExpectedResponse]] = (
            InMemoryMailbox(name="eval-in")
        )
        eval_res: InMemoryMailbox[EvalResult] = InMemoryMailbox(name="eval-out")
        eval_loop = EvalLoop(
            loop=loop, evaluator=evaluate_response, requests=eval_req, results=eval_res
        )

        submit_dataset(dataset, eval_req)

        def run() -> None:
            eval_loop.run(max_iterations=total)

        t = threading.Thread(target=run, daemon=True)
        t.start()

        collected: list[EvalResult] = []
        while len(collected) < total:
            msgs = eval_res.receive(max_messages=1, wait_time_seconds=1)
            for msg in msgs:
                collected.append(msg.body)
                msg.acknowledge()
                self.post_message(EvalProgress(len(collected), total, collected[-1]))

        t.join(timeout=5)
        self.post_message(EvalDone(EvalReport(results=tuple(collected))))

    @on(EvalProgress)
    def update_progress(self, event: EvalProgress) -> None:
        self.query_one("#progress", ProgressBar).update(
            total=event.total, progress=event.current
        )
        if event.latest:
            self.query_one("#results", ListView).append(ResultItem(event.latest))

    @on(EvalDone)
    def show_eval_done(self, event: EvalDone) -> None:
        r = event.report
        self.query_one("#summary", Static).update(
            f"Done: {r.total} samples, {r.pass_rate:.0%} pass, {r.mean_score:.2f} avg"
        )

    def action_quit(self) -> None:
        self._loop.shutdown(timeout=2.0)
        self._requests.close()
        self._responses.close()
        dump_session_tree(self._loop.session, SNAPSHOT_DIR)
        self.exit()


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------


def main() -> None:
    configure_logging()
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    adapter = cast(ProviderAdapter[ReviewResponse], OpenAIAdapter(model=model))
    ReviewApp(adapter).run()


if __name__ == "__main__":
    main()
