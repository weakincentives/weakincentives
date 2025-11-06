# Weak Incentives

**Lean, typed building blocks for side-effect-free background agents.**
Compose deterministic prompts, run typed tools, and parse strict JSON replies without
heavy dependencies. Optional adapters snap in when you need a model provider.

## What's novel?

- **Observable session state with reducer support.** A Redux-like session state
  manager keeps every tool invocation and prompt interaction observable,
  replayable, and ready for automation while an in-process event bus publishes
  `ToolInvoked` and `PromptExecuted` events. Built-in planning, virtual
  filesystem, and Python-eval sections register reducers, enforce
  domain-specific validation, and expose guided Markdown so stateful runs stay
  deterministic.
- **Composable prompt blueprints with strict contracts.** Prompt composition
  primitives let you assemble deterministic, typed sections into reusable
  blueprints. Prompt objects compose trees of dataclass-backed sections, render
  Markdown with validated placeholders, and automatically surface tool
  contracts so every render stays predictable.
- **Override-friendly workflows that scale into optimization.** First-class
  support for prompt overrides lays the groundwork for an optimizer that plugs
  into your development cycle. Prompt definitions ship with hash-based
  descriptors plus on-disk overrides that stay in sync through schema
  validation and Git-root discovery. Prompt optimizers become important as your
  evaluation suite matures, but they are not a day-one requirement—start by
  writing prompts manually and add automation once you have robust evals.
- **Provider adapters that standardize tool negotiation.** Provider adapters
  share a conversation loop that negotiates tool calls, applies JSON-schema
  response formats, and normalizes structured payloads, making the runtime
  model-agnostic.
- **Local-first, deterministic execution.** Everything runs locally without
  mandatory APIs or hosted services, and every render stays
  version-control-friendly so diffs capture intent instead of churn. The
  code-review example ties it together with override-aware prompts, session
  telemetry, and replayable tooling for deterministic agent runs.

## Requirements

- Python 3.12+ (the repository pins 3.12 in `.python-version` for development)
- [`uv`](https://github.com/astral-sh/uv) CLI

## Install

```bash
uv add weakincentives
# optional tool extras
uv add "weakincentives[asteval]"
# optional provider adapters
uv add "weakincentives[openai]"
uv add "weakincentives[litellm]"
# cloning the repo? use: uv sync --extra asteval --extra openai --extra litellm
```

## Tutorial: Build a Stateful Code-Reviewing Agent

Use Weak Incentives to assemble a reproducible reviewer that tracks every
decision, stages edits safely, and answers quick calculations inline. The
runtime already ships with a session ledger and override-aware prompts, so you
avoid custom state stores or ad-hoc optimizers.

### 1. Model review data and expected outputs

Typed dataclasses keep inputs and outputs honest so adapters emit consistent
telemetry and structured responses stay predictable. See
[Dataclass Serde Utilities](specs/DATACLASS_SERDE.md) and
[Structured Output via `Prompt[OutputT]`](specs/STRUCTURED_OUTPUT.md) for the
validation and JSON-contract details behind this snippet.

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
the session so every run supports plans, staged edits, and quick calculations.
Mount diffs ahead of time so the agent can read them through the virtual
filesystem without extra callbacks. Install the `asteval` extra
(`uv add "weakincentives[asteval]"`) before instantiating `AstevalSection` so the
sandbox is available at runtime. Specs worth skimming:
[Session State](specs/SESSIONS.md), [Prompt Event Emission](specs/EVENTS.md),
[Virtual Filesystem Tools](specs/VFS_TOOLS.md), [Planning Tools](specs/PLANNING_TOOL.md),
and [Asteval Integration](specs/ASTEVAL.md).

```python
from pathlib import Path

from weakincentives.runtime.events import InProcessEventBus, PromptExecuted
from weakincentives.runtime.session import Session
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
    allowed_host_roots=(diff_root,),
    mounts=(
        HostMount(
            host_path="octo_widgets/cache-layer.diff",
            mount_path=VfsPath(("diffs", "cache-layer.diff")),
        ),
    ),
)
planning_section = PlanningToolsSection()
asteval_section = AstevalSection()


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
filesystem snapshot. `PlanningToolsSection`, `AstevalSection` and `VfsToolsSection` pull the
active `Session` from `ToolContext.session` during tool execution, so adapters
must populate that field before dispatching calls.

### 3. Define a symbol search helper tool

Tools are typed callables that return structured results. Add lightweight
helpers alongside the built-in suites—in this case, a symbol searcher that reads
from a repo mounted at `/srv/agent-repo`. Review the
[Tool Registration](specs/TOOLS.md) and [Tool Error Handling](specs/TOOL_ERROR_HANDLING.md)
specs to match the handler and `ToolResult` contracts.

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

Session reducers accumulate structured state across prompt and tool events.
When the `symbol_search` tool returns results, register a reducer that records
the queries the reviewer explored along with the snippets that satisfied each
one. Downstream sections can inspect this slice with
`session.select_all(ReviewedSymbol)` to summarize the investigation history.

```python
from dataclasses import dataclass

from weakincentives.runtime.session import ToolData


@dataclass
class ReviewedSymbol:
    query: str
    matches: tuple[SymbolMatch, ...]


def track_reviewed_symbols(
    reviewed: tuple[ReviewedSymbol, ...],
    event: ToolData,
) -> tuple[ReviewedSymbol, ...]:
    if event.value is None or not isinstance(event.value, SymbolSearchResult):
        return reviewed

    params = event.source.params
    reviewed_symbol = ReviewedSymbol(
        query=params.query,
        matches=event.value.matches,
    )
    return (*reviewed, reviewed_symbol)


session.register_reducer(
    SymbolSearchResult,
    track_reviewed_symbols,
    slice_type=ReviewedSymbol,
)
```

Attach custom tools to sections (next step) so the adapter can call them and
record their outputs on the session alongside built-in reducers. The prompt can
now chase suspicious references without delegating work back to the orchestrator.

### 4. Compose the prompt with deterministic sections

Sections render through `string.Template`, so keep placeholders readable and
combine guidance with the tool suites into one auditable prompt tree. See the
[Prompt Class](specs/PROMPTS.md) and
[Prompt Versioning & Persistence](specs/PROMPTS_VERSIONING.md) specs for the
rendering and hashing rules that stabilize this structure.

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
    ReviewGuidance(),
)


print(rendered.text)
print([tool.name for tool in rendered.tools])
```

### 5. Evaluate the prompt with an adapter

Adapters send the rendered prompt to a provider and publish telemetry to the
event bus; the session wiring above captures `PromptExecuted` and `ToolInvoked`
events automatically. For payload formats and parsing guarantees see
[Adapter Evaluation](specs/ADAPTERS.md) and
[Native OpenAI Structured Outputs](specs/NATIVE_OPENAI_STRUCTURED_OUTPUTS.md).

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

Selectors expose reducer output so you can ship audit logs without extra
plumbing. Planning reducers keep only the latest `Plan`; register a custom
reducer before `PlanningToolsSection` if you need history. See
[Session State](specs/SESSIONS.md) and
[Session Snapshots](specs/SESSION_SNAPSHOTS.md) for selector and rollback rules.

```python
from weakincentives.runtime.session import select_latest
from weakincentives.tools import Plan, VirtualFileSystem


latest_plan = select_latest(session, Plan)
vfs_snapshot = select_latest(session, VirtualFileSystem)


if latest_plan:
    print(f"Plan objective: {latest_plan.objective}")
    for step in latest_plan.steps:
        print(f"- [{step.status}] {step.title}")
else:
    print("No plan recorded yet.")


if vfs_snapshot:
    for file in vfs_snapshot.files:
        print(f"Staged file {file.path.segments} (version {file.version})")
```

### 7. Override sections with an overrides store

Persist optimizer output so the runtime can swap in tuned sections without a
redeploy. `LocalPromptOverridesStore` is the default choice: it discovers the
workspace root, enforces descriptors, and reads JSON overrides from
`.weakincentives/prompts/overrides/`. Pair the
[Local Prompt Overrides Store](specs/LOCAL_PROMPT_OVERRIDES_STORE.md) and
[Prompt Versioning & Persistence](specs/PROMPTS_VERSIONING.md) specs to keep
namespace, key, and tag hashes aligned.

```python
from pathlib import Path

from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptDescriptor,
    PromptOverride,
    SectionOverride,
)


workspace_root = Path("/srv/agent-workspace")
overrides_store = LocalPromptOverridesStore(root_path=workspace_root)

descriptor = PromptDescriptor.from_prompt(review_prompt)
seed_override = overrides_store.seed_if_necessary(
    review_prompt, tag="assertive-feedback"
)

section_path = ("review", "directives")
section_descriptor = next(
    section
    for section in descriptor.sections
    if section.path == section_path
)

custom_override = PromptOverride(
    ns=descriptor.ns,
    prompt_key=descriptor.key,
    tag="assertive-feedback",
    sections={
        **seed_override.sections,
        section_path: SectionOverride(
            expected_hash=section_descriptor.content_hash,
            body="\n".join(
                (
                    "- Classify findings using this severity scale: minor | major | critical.",
                    "- Always cite the exact diff hunk when raising a major or critical issue.",
                    "- Respond with ReviewBundle JSON. Missing fields terminate the run.",
                )
            ),
        ),
    },
    tool_overrides=seed_override.tool_overrides,
)

persisted_override = overrides_store.upsert(descriptor, custom_override)

rendered_with_override = review_prompt.render_with_overrides(
    PullRequestContext(
        repository="octo/widgets",
        title="Add caching layer",
        body="Introduces memoization to reduce redundant IO while preserving correctness.",
        files_summary="loader.py, cache.py",
    ),
    overrides_store=overrides_store,
    tag=persisted_override.tag,
)


print(rendered_with_override.text)
```

The overrides store writes atomically to
`.weakincentives/prompts/overrides/{ns}/{prompt_key}/{tag}.json` inside the
workspace described in the
[Local Prompt Overrides Store Specification](specs/LOCAL_PROMPT_OVERRIDES_STORE.md).
Optimizers and prompt engineers can still drop JSON overrides into that tree by
hand—checked into source control or generated during evaluations—without
subclassing `PromptOverridesStore`. Because sections expose stable `(ns, key, path)` identifiers, overrides stay scoped to the intended content so teams can
iterate on directives without risking accidental drift elsewhere in the tree.

### 8. Ship it

You now have a deterministic reviewer that:

1. Enforces typed contracts for inputs, tools, and outputs.
1. Persists plans, VFS edits, and evaluation transcripts inside a session.
1. Supports optimizer-driven overrides that fit neatly into CI or evaluation
   harnesses.

Run it inside a worker, bot, or scheduler; the captured session state keeps each
evaluation replayable. For long-lived deployments, follow
[Tool-Aware Prompt Versioning](specs/TOOL_AWARE_PROMPT_VERSIONING.md) to keep
overrides and tool descriptors in sync.

## Logging

Weak Incentives ships a structured logging adapter so hosts can add contextual
metadata to every record without manual dictionary plumbing. Call
`configure_logging()` during startup to install the default handler and then
bind logger instances wherever you need telemetry:

```python
from weakincentives.runtime.logging import configure_logging, get_logger

configure_logging(json_mode=True)
logger = get_logger("demo").bind(component="cli")
logger.info("boot", event="demo.start", context={"attempt": 1})
```

The helper respects any existing root handlers—omit `force=True` if your
application already configures logging and you only want Weak Incentives to
honor the selected level. When you do want to take over the pipeline, call
`configure_logging(..., force=True)` and then customize the root handler list
with additional sinks (for example, forwarding records to Cloud Logging or a
structured log shipper). Each emitted record contains an `event` field plus a
`context` mapping, so downstream processors can make routing decisions without
parsing raw message strings.

## Development Setup

1. Install Python 3.12 (for example with `pyenv install 3.12.0`).

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

### Integration tests

Provider integrations require live credentials, so the suite stays opt-in. Export the
necessary OpenAI configuration and then run the dedicated `make` target, which disables
coverage enforcement automatically:

```bash
export OPENAI_API_KEY="sk-your-key"
# Optionally override the default model (`gpt-4.1`).
export OPENAI_TEST_MODEL="gpt-4.1-mini"

make integration-tests
```

`make integration-tests` forwards `--no-cov` to pytest so you can exercise the adapter
scenarios without tripping the 100% coverage gate configured for the unit test suite. The
tests remain skipped when `OPENAI_API_KEY` is not present.

## Documentation

- `AGENTS.md` — operational handbook and contributor workflow.
- `specs/` — design docs for prompts, planning tools, and adapters.
- `ROADMAP.md` — upcoming feature sketches.
- `docs/api/` — API reference material.

## License

Apache 2.0 • Status: Alpha (APIs may change between releases)
