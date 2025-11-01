# Weak Incentives

**Lean, typed building blocks for side-effect-free background agents.**
Compose deterministic prompts, run typed tools, and parse strict JSON replies without
heavy dependencies. Optional adapters snap in when you need a model provider.

## Highlights

- Namespaced prompt trees with deterministic Markdown renders, placeholder
  verification, and tool-aware versioning metadata.
- Stdlib-only dataclass serde (`parse`, `dump`, `clone`, `schema`) keeps request and
  response types honest end-to-end.
- Session state container and event bus collect prompt and tool telemetry for
  downstream automation.
- Built-in planning and virtual filesystem tool suites give agents durable plans and
  sandboxed edits backed by reducers and selectors.
- Optional OpenAI and LiteLLM adapters integrate structured output parsing, tool
  orchestration, and telemetry hooks.

## Requirements

- Python 3.12+ (the repository pins 3.14 in `.python-version` for development)
- [`uv`](https://github.com/astral-sh/uv) CLI

## Install

```bash
uv add weakincentives
# optional provider adapters
uv add "weakincentives[openai]"
uv add "weakincentives[litellm]"
# cloning the repo? use: uv sync --extra openai --extra litellm
```


## Tutorial: Build a Stateful Code-Reviewing Agent

Use Weak Incentives to assemble a reproducible reviewer that tracks every
decision, stages file edits in a sandbox, and evaluates quick calculations when
the diff raises questions. Compared to LangGraph you do not need to bolt on a
custom state store—the `Session` captures prompt and tool telemetry out of the
box. Unlike DSPy, prompt sections already expose versioning and override hooks
so optimizers can swap instructions without rewriting the runtime.

### 1. Model review data and tool contracts

Typed dataclasses keep inputs, tool payloads, and outputs honest. Tools return
`ToolResult` objects so the adapter can emit consistent telemetry.

```python
from dataclasses import dataclass
from weakincentives import Tool, ToolResult


@dataclass
class PullRequestContext:
    repository: str
    title: str
    body: str
    files_summary: str


@dataclass
class FileDiffRequest:
    path: str


@dataclass
class FileDiffSnapshot:
    path: str
    patch: str


@dataclass
class ReviewComment:
    file_path: str
    line: int
    severity: str
    summary: str
    rationale: str


@dataclass
class ReviewBundle:
    comments: tuple[ReviewComment, ...]
    overall_assessment: str


def fetch_diff(params: FileDiffRequest) -> ToolResult[FileDiffSnapshot]:
    # Replace this stub with a Git provider lookup in production.
    diff = FileDiffSnapshot(path=params.path, patch="@@ -1 +1 @@ ...")
    return ToolResult(message=f"Loaded diff for {params.path}", value=diff)


diff_tool = Tool[FileDiffRequest, FileDiffSnapshot](
    name="fetch_diff",
    description="Retrieve the unified diff for a repository path.",
    handler=fetch_diff,
)
```

### 2. Create a session and surface built-in tool suites

Planning, virtual filesystem, and Python-evaluation sections register reducers on
the provided session. Introducing them early keeps every evaluation capable of
multi-step plans, staged edits, and quick calculations.

```python
from weakincentives.events import InProcessEventBus, PromptExecuted
from weakincentives.session import Session
from weakincentives.tools import AstevalSection, PlanningToolsSection, VfsToolsSection


bus = InProcessEventBus()
session = Session(bus=bus)


planning_section = PlanningToolsSection(session=session)
vfs_section = VfsToolsSection(session=session)
asteval_section = AstevalSection(session=session)


def log_prompt(event: PromptExecuted) -> None:
    print(
        f"Prompt {event.prompt_name} completed with "
        f"{len(event.result.tool_results)} tool calls"
    )


bus.subscribe(PromptExecuted, log_prompt)
```

### 3. Compose the prompt with deterministic sections

Sections rely on `string.Template`, so prepare readable placeholders up front.
Combine your review instructions with the built-in tool suites to publish a
single, auditable prompt tree.

```python
from weakincentives import MarkdownSection, Prompt


@dataclass
class ReviewGuidance:
    severity_scale: str = "minor | major | critical"
    output_schema: str = "ReviewBundle with comments[] and overall_assessment"
    focus_areas: str = (
        "Security regressions, concurrency bugs, test coverage gaps, and"
        " ambiguous logic should be escalated."
    )


overview_section = MarkdownSection[PullRequestContext](
    title="Repository Overview",
    key="review.overview",
    template="""
    You are a principal engineer reviewing a pull request.
    Repository: ${repository}
    Title: ${title}

    Pull request summary:
    ${body}

    Files touched: ${files_summary}
    """,
)


analysis_section = MarkdownSection[ReviewGuidance](
    title="Review Directives",
    key="review.directives",
    template="""
    - Classify findings using this severity scale: ${severity_scale}.
    - Emit output that matches ${output_schema}; missing fields fail the run.
    - Investigation focus:
      ${focus_areas}
    - Call `fetch_diff` before commenting on any unfamiliar hunk.
    """,
    default_params=ReviewGuidance(),
    tools=(diff_tool,),
)


review_prompt = Prompt[ReviewBundle](
    ns="tutorial/code_review",
    key="review.generate",
    name="code_review_agent",
    sections=(
        overview_section,
        planning_section,
        vfs_section,
        asteval_section,
        analysis_section,
    ),
)


rendered = review_prompt.render(
    PullRequestContext(
        repository="octo/widgets",
        title="Add caching layer",
        body="Introduces memoization to reduce redundant IO while preserving correctness.",
        files_summary="loader.py, cache.py",
    ),
)


print(rendered.text)
print([tool.name for tool in rendered.tools])
```

### 4. Evaluate the prompt with an adapter

Adapters send the rendered prompt to a provider and publish telemetry to the
event bus. The session subscribed above automatically ingests each
`PromptExecuted` and `ToolInvoked` event.

```python
from weakincentives.adapters.openai import OpenAIAdapter


adapter = OpenAIAdapter(
    model="gpt-4o-mini",
    client_kwargs={"api_key": "sk-..."},
)


response = adapter.evaluate(
    review_prompt,
    PullRequestContext(
        repository="octo/widgets",
        title="Add caching layer",
        body="Introduces memoization to reduce redundant IO while preserving correctness.",
        files_summary="loader.py, cache.py",
    ),
    bus=bus,
)


bundle = response.output
if bundle is None:
    raise RuntimeError("Structured parsing failed")


for comment in bundle.comments:
    print(f"{comment.file_path}:{comment.line} → {comment.summary}")
```

If the model omits a required field, `OpenAIAdapter` raises `PromptEvaluationError`
with provider context rather than silently degrading.

### 5. Mine session state for downstream automation

Built-in selectors expose the data collected by reducers that each tool suite
registered. This gives you ready-to-ship audit logs without building LangGraph
callbacks or DSPy side channels.

```python
from weakincentives.session import select_all, select_latest
from weakincentives.tools import Plan, VirtualFileSystem


plan_history = select_all(session, Plan)
latest_plan = select_latest(session, Plan)
vfs_snapshot = select_latest(session, VirtualFileSystem)


print(f"Plan steps recorded: {len(plan_history)}")
if latest_plan:
    for step in latest_plan.steps:
        print(f"- [{step.status}] {step.title}")


if vfs_snapshot:
    for file in vfs_snapshot.files:
        print(f"Staged file {file.path.segments} (version {file.version})")
```

### 6. Override sections with a version store

DSPy-style optimizers can persist improved instructions and let the runtime swap
them in without re-deploying code. Implement the `PromptVersionStore` protocol to
serve overrides by namespace, key, and tag.

```python
from dataclasses import dataclass
from weakincentives.prompt.versioning import (
    PromptDescriptor,
    PromptOverride,
    PromptVersionStore,
)


@dataclass
class StaticVersionStore(PromptVersionStore):
    override: PromptOverride | None = None


    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        if (
            self.override
            and self.override.ns == descriptor.ns
            and self.override.prompt_key == descriptor.key
            and self.override.tag == tag
        ):
            return self.override
        return None


overrides = PromptOverride(
    ns=review_prompt.ns,
    prompt_key=review_prompt.key,
    tag="assertive-feedback",
    overrides={
        ("review.directives",): """
        - Classify findings using this severity scale: minor | major | critical.
        - Always cite the exact diff hunk when raising a major or critical issue.
        - Respond with ReviewBundle JSON. Missing fields terminate the run.
        """,
    },
)


store = StaticVersionStore(override=overrides)
rendered_with_override = review_prompt.render_with_overrides(
    PullRequestContext(
        repository="octo/widgets",
        title="Add caching layer",
        body="Introduces memoization to reduce redundant IO while preserving correctness.",
        files_summary="loader.py, cache.py",
    ),
    version_store=store,
    tag="assertive-feedback",
)


print(rendered_with_override.text)
```

Because sections expose stable `(ns, key, path)` identifiers, overrides stay scoped
to the intended content. That means optimizers can explore new directives without
risking accidental prompt drift elsewhere in the tree.

### 7. Ship it

You now have a deterministic reviewer that:

1. Enforces typed contracts for inputs, tools, and outputs.
2. Persists multi-step plans, VFS edits, and evaluation transcripts inside a
   session without custom plumbing.
3. Supports optimizer-driven overrides that slot cleanly into CI, evaluation
   harnesses, or on-call tuning workflows.

Drop the agent into a queue worker, Slack bot, or scheduled job. Every evaluation
is replayable thanks to the captured session state, so postmortems start with
facts—not speculation.


## Sessions and Built-in Tools

Session state turns prompt output and tool calls into durable data. Built-in planning
and virtual filesystem sections register reducers on the provided session.

```python
from weakincentives.session import Session, select_latest
from weakincentives.tools import (
    PlanningToolsSection,
    Plan,
    VfsToolsSection,
    VirtualFileSystem,
)

session = Session()
planning_section = PlanningToolsSection(session=session)
vfs_section = VfsToolsSection(session=session)

prompt = Prompt[ResearchSummary](
    ns="examples/research",
    key="research.session",
    sections=[task_section, planning_section, vfs_section],
)

active_plan = select_latest(session, Plan)
vfs_snapshot = select_latest(session, VirtualFileSystem)
```

Use `session.select_all(...)` or the helpers in `weakincentives.session` to drive UI
state, persistence, or audits after each adapter run.

## Adapter Integrations

Adapters stay optional and only load their dependencies when you import them.

```python
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.events import InProcessEventBus
from weakincentives.session import Session
from weakincentives.tools import Plan

bus = InProcessEventBus()
session = Session(bus=bus)

adapter = OpenAIAdapter(
    model="gpt-4o-mini",
    client_kwargs={"api_key": "sk-..."},
)

response = adapter.evaluate(
    prompt,
    ResearchGuidance(topic="Ada Lovelace"),
    bus=bus,
)

plan_history = session.select_all(Plan)
```

`InProcessEventBus` publishes `ToolInvoked` and `PromptExecuted` events for the
session (or any other subscriber) to consume.

## Development Setup

1. Install Python 3.14 (for example with `pyenv install 3.14.0`).

1. Install `uv`, then bootstrap the environment and hooks:

   ```bash
   uv sync
   ./install-hooks.sh
   ```

1. Run checks with `uv run` so everything shares the managed virtualenv:

   - `make format` / `make format-check`
   - `make lint` / `make lint-fix`
   - `make typecheck` (Ty + Pyright, warnings fail the build)
   - `make test` (pytest via `build/run_pytest.py`, 100% coverage enforced)
   - `make check` (aggregates the quiet checks above plus Bandit, Deptry, pip-audit,
     and markdown linting)

## Documentation

- `AGENTS.md` — operational handbook and contributor workflow.
- `specs/` — design docs for prompts, planning tools, and adapters.
- `ROADMAP.md` — upcoming feature sketches.
- `docs/api/` — API reference material.

## License

Apache 2.0 • Status: Alpha (APIs may change between releases)
