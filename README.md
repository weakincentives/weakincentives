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

## Tutorial: Build a Code-Reviewing Agent

This tutorial walks through building a production-ready, stateful code-reviewing
agent that reads pull requests, drafts comments, and records rationale. The
workflow uses the same primitives as the
[`code_reviewer_example.py`](code_reviewer_example.py) script but goes well beyond
it:

- **Deterministic prompts** keep every model call reproducible without juggling
  ad-hoc f-strings.
- **Session-backed state** lets you coordinate multiple model rounds in a single
  review, something that requires custom middleware in LangGraph’s edge-based
  planner.
- **Optimizable prompt overrides** wire directly into DSPy-style optimizers,
  giving you pragmatic hooks for experimentation without forking the agent.

> **Why Weak Incentives instead of LangGraph or DSPy?**
>
> - LangGraph focuses on graph orchestration; you still have to invent a custom
>   state store to keep review history coherent. Weak Incentives ships a typed
>   `Session` with reducers, selectors, and an event bus so every prompt render
>   and tool execution is automatically captured.
> - DSPy’s strength is prompt optimization, but its Modules assume you’ll wire
>   the prompting and state plumbing yourself. Here, prompt sections embed rich
>   metadata, so an optimizer can override just the Markdown template or tool
>   policy without rewriting the rest of the agent.

### 1. Define review types and telemetry

Start by modeling the structured inputs, tool payloads, and agent outputs. The
library’s serde helpers guarantee that model replies match these dataclasses.

```python
from dataclasses import dataclass
from weakincentives import ToolResult


@dataclass
class PullRequestSummary:
    repository: str
    title: str
    body: str
    files: list[str]


@dataclass
class FileDiff:
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
    comments: list[ReviewComment]
    overall_assessment: str


def fetch_diff(diff: FileDiff) -> ToolResult[FileDiff]:
    # In production you would hydrate from the provider API.
    return ToolResult(
        message=f"Loaded diff for {diff.path}",
        value=diff,
        telemetry={"bytes": len(diff.patch)},
    )
```

Here we also return telemetry metadata from the tool handler. The event bus will
log that alongside the prompt render, giving you reproducible audit trails.

### 2. Compose review prompt sections

Break the agent prompt into sections so optimizers can swap one piece at a time.
Every section is namespaced and versioned, which is how Weak Incentives keeps
state consistent without the edge-level bookkeeping LangGraph requires.

```python
from weakincentives import MarkdownSection, Prompt, Tool

summary_section = MarkdownSection[PullRequestSummary](
    title="Repository Overview",
    template=(
        "You are a senior reviewer.",
        "Repository: ${repository}",
        "PR Title: ${title}",
        "PR Body:\n${body}",
        "Files touched: ${', '.join(files)}",
    ),
    key="review.summary",
)

diff_tool = Tool[FileDiff, FileDiff](
    name="fetch_diff",
    description="Retrieve the unified diff for a file to analyze context",
    handler=fetch_diff,
)

analysis_section = MarkdownSection[PullRequestSummary](
    title="Diff Analysis",
    template=(
        "Review each file. Call `fetch_diff` when you need the patch.",
        "Propose comments in JSON with fields: file_path, line, severity, summary, rationale.",
        "Flag risky refactors, data races, or missing tests.",
    ),
    key="review.analysis",
    tools=[diff_tool],
)

prompt = Prompt[ReviewBundle](
    ns="tutorial/code_review",
    key="review.generate",
    name="code_review_prompt",
    sections=[summary_section, analysis_section],
)
```

Because each section is independently versioned, you can roll back just the
analysis template if an optimizer explores a bad variant.

### 3. Persist state with a session and event bus

Sessions capture every render and tool call so later steps can reference prior
actions. Unlike DSPy, you do not need custom callback plumbing—events are
broadcast automatically.

```python
from weakincentives.events import InProcessEventBus, PromptExecuted
from weakincentives.session import Session, select_latest

bus = InProcessEventBus()
session = Session(bus=bus)


@bus.subscribe
def on_prompt_executed(event: PromptExecuted) -> None:
    print(f"Rendered {event.prompt.key} with tools {event.tools}")


rendered = prompt.render(
    PullRequestSummary(
        repository="octo/widgets",
        title="Add caching layer",
        body="Refactors the data loader with memoization",
        files=["loader.py", "cache.py"],
    ),
    session=session,
)

print(rendered.text)
print(session.select_all(PromptExecuted))
```

The session now contains structured history that you can query with selectors or
persist to your own store. LangGraph workflows typically require mutating a
global dict—here the state container enforces types and isolation.

### 4. Run the agent with an adapter

Adapters handle the transport. Once you have the rendered prompt, evaluate it
through your model provider. The adapter emits events so your session stays in
sync.

```python
from weakincentives.adapters.openai import OpenAIAdapter

adapter = OpenAIAdapter(
    model="gpt-4o-mini",
    client_kwargs={"api_key": "sk-..."},
)

response = adapter.evaluate(
    prompt=prompt,
    params=PullRequestSummary(
        repository="octo/widgets",
        title="Add caching layer",
        body="Refactors the data loader with memoization",
        files=["loader.py", "cache.py"],
    ),
    session=session,
    bus=bus,
)

review_bundle = response.value
for comment in review_bundle.comments:
    print(f"{comment.file_path}:{comment.line} — {comment.summary}")
```

Your `ReviewBundle` is fully typed. If the model forgets a field, the structured
parser raises immediately, giving you deterministic failure modes.

### 5. Layer in optimizer-driven overrides

Weak Incentives exposes prompt sections as plain objects, so you can experiment
with DSPy-style optimizers while keeping the deterministic runtime. For example:

```python
from weakincentives import clone


def strengthen_defect_language(section: MarkdownSection[PullRequestSummary]) -> MarkdownSection[PullRequestSummary]:
    updated = clone(section)
    updated.template = (*section.template, "Always cite the specific diff hunk.")
    updated.version = section.version.increment(minor=1)
    return updated


optimized_prompt = clone(prompt)
optimized_prompt.sections = [
    summary_section,
    strengthen_defect_language(analysis_section),
]
```

Because sections carry versions, you can store optimizer trials alongside their
metadata, diff them, and roll back without retraining a graph. Try that in
LangGraph without a pile of custom tracking code!

### 6. Extend with planning and a virtual filesystem

The review agent can coordinate multi-step workflows (draft → refine → approve)
by layering in the built-in planning tools. These reducers store plans inside the
session so the next evaluator call has the full conversation context.

```python
from weakincentives.tools import PlanningToolsSection, Plan

planning_section = PlanningToolsSection(session=session)

prompt.sections.append(planning_section)

# Later, inspect the evolving plan:
active_plan = select_latest(session, Plan)
print(active_plan.steps if active_plan else "No plan yet")
```

The planning reducer will emit consistent plan states across retries, something
graph-based frameworks often leave to the user.

### 7. Ship it

At this point you have:

1. Reproducible prompt renders with structured sections and tool metadata.
2. Stateful reviews captured in a type-safe session.
3. Hooks for DSPy-style optimizers to propose new templates without touching the
   rest of the agent.

Whether you host it in a background worker or a Slack bot, the deterministic
architecture keeps reviews explainable—and your future self can replay every
step when something looks off.

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
