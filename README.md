# Weak Incentives

**Lean, typed building blocks for side-effect-free background agents.**
Compose deterministic prompts, run typed tools, and parse strict JSON replies without
heavy dependencies. Optional adapters snap in when you need a model provider.

## Why now?

This library was built out of frustration with LangGraph and DSPy to explore
better ways to do state and context management when building apps with LLMs
while allowing the prompts to be automatically optimized.

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

### 1. Model review data and expected outputs

Typed dataclasses keep inputs and outputs honest so adapters can emit consistent
telemetry and structured responses stay predictable.

```python
from dataclasses import dataclass


@dataclass
class PullRequestContext:
    repository: str
    title: str
    body: str
    files_summary: str


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
```

### 2. Create a session, surface built-in tool suites, and mount diffs

Planning, virtual filesystem, and Python-evaluation sections register reducers on
the provided session. Introducing them early keeps every evaluation capable of
multi-step plans, staged edits, and quick calculations. Host mounts feed the
reviewer precomputed diffs before the run begins so it can read them through the
virtual filesystem tools without calling back to your orchestrator. Because each
tool suite records its activity on the session, selectors later in the tutorial
can recover audit logs without extra plumbing.

```python
from pathlib import Path

from weakincentives.events import InProcessEventBus, PromptExecuted
from weakincentives.session import Session
from weakincentives.tools import (
    AstevalSection,
    HostMount,
    PlanningToolsSection,
    VfsPath,
    VfsToolsSection,
)


bus = InProcessEventBus()
session = Session(bus=bus)


diff_root = Path("/srv/agent-mounts")
diff_root.mkdir(parents=True, exist_ok=True)
vfs_section = VfsToolsSection(
    session=session,
    allowed_host_roots=(diff_root,),
    mounts=(
        HostMount(
            host_path="octo_widgets/cache-layer.diff",
            mount_path=VfsPath(("diffs", "cache-layer.diff")),
        ),
    ),
)
planning_section = PlanningToolsSection(session=session)
asteval_section = AstevalSection(session=session)


def log_prompt(event: PromptExecuted) -> None:
    print(
        f"Prompt {event.prompt_name} completed with "
        f"{len(event.result.tool_results)} tool calls"
    )


bus.subscribe(PromptExecuted, log_prompt)
```

Copy unified diff files into `/srv/agent-mounts` before launching the run. The
host mount resolves `octo_widgets/cache-layer.diff` relative to that directory
and exposes it to the agent as `diffs/cache-layer.diff` inside the virtual
filesystem snapshot.

### 3. Define a symbol search helper tool

Weak Incentives tools are typed functions that return structured results. Use
them to expose deterministic helpers alongside the built-in suites. A reviewer
benefits from a lightweight code-search utility that surfaces the context around
symbols referenced in a diff. Mount a checkout of the repository under
`/srv/agent-repo` before launching the run so the tool can read from it.

```python
from dataclasses import dataclass
from pathlib import Path

from weakincentives.prompt.tool import Tool, ToolResult


@dataclass
class SymbolSearchRequest:
    query: str
    file_glob: str = "*.py"
    max_results: int = 5


@dataclass
class SymbolMatch:
    file_path: str
    line: int
    snippet: str


@dataclass
class SymbolSearchResult:
    matches: tuple[SymbolMatch, ...]


repo_root = Path("/srv/agent-repo")


def find_symbol(params: SymbolSearchRequest) -> ToolResult[SymbolSearchResult]:
    if not repo_root.exists():
        raise FileNotFoundError(
            "Mount a repository checkout at /srv/agent-repo before running the agent."
        )

    matches: list[SymbolMatch] = []
    for file_path in repo_root.rglob(params.file_glob):
        if not file_path.is_file():
            continue
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if params.query in line:
                    matches.append(
                        SymbolMatch(
                            file_path=str(file_path.relative_to(repo_root)),
                            line=line_number,
                            snippet=line.strip(),
                        )
                    )
                    if len(matches) >= params.max_results:
                        break
        if len(matches) >= params.max_results:
            break

    return ToolResult(
        message=f"Found {len(matches)} matching snippets.",
        value=SymbolSearchResult(matches=tuple(matches)),
    )


symbol_search_tool = Tool[SymbolSearchRequest, SymbolSearchResult](
    name="symbol_search",
    description=(
        "Search the repository checkout for a symbol and return file snippets."
    ),
    handler=find_symbol,
)
```

Attach custom tools to sections (next step) so the adapter can call them and
record their outputs on the session alongside built-in reducers. The prompt can
now chase suspicious references without delegating work back to the orchestrator.

### 4. Compose the prompt with deterministic sections

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
    - Inspect mounted diffs under `diffs/` with `vfs_read_file` before
      commenting on unfamiliar hunks.
    - Reach for `symbol_search` when you need surrounding context from the
      repository checkout.
    """,
    tools=(symbol_search_tool,),
    default_params=ReviewGuidance(),
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

### 5. Evaluate the prompt with an adapter

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

### 6. Mine session state for downstream automation

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

### 7. Override sections with an overrides store

DSPy-style optimizers can persist improved instructions and let the runtime swap
them in without re-deploying code. Implement the `PromptOverridesStore` protocol
to serve overrides by namespace, key, and tag.

```python
from dataclasses import dataclass
from weakincentives.prompt.versioning import (
    PromptDescriptor,
    PromptOverride,
    PromptOverridesStore,
)


@dataclass
class StaticOverridesStore(PromptOverridesStore):
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


overrides_store = StaticOverridesStore(override=overrides)
rendered_with_override = review_prompt.render_with_overrides(
    PullRequestContext(
        repository="octo/widgets",
        title="Add caching layer",
        body="Introduces memoization to reduce redundant IO while preserving correctness.",
        files_summary="loader.py, cache.py",
    ),
    overrides_store=overrides_store,
    tag="assertive-feedback",
)


print(rendered_with_override.text)
```

Because sections expose stable `(ns, key, path)` identifiers, overrides stay scoped
to the intended content. That means optimizers can explore new directives without
risking accidental prompt drift elsewhere in the tree.

### 8. Ship it

You now have a deterministic reviewer that:

1. Enforces typed contracts for inputs, tools, and outputs.
1. Persists multi-step plans, VFS edits, and evaluation transcripts inside a
   session without custom plumbing.
1. Supports optimizer-driven overrides that slot cleanly into CI, evaluation
   harnesses, or on-call tuning workflows.

Drop the agent into a queue worker, Slack bot, or scheduled job. Every evaluation
is replayable thanks to the captured session state, so postmortems start with
facts—not speculation.

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
