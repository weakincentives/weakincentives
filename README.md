# Weak Incentives

**Lean, typed building blocks for side-effect-free background agents.**
Compose deterministic prompts, run typed tools, and parse strict JSON replies without
heavy dependencies. Optional adapters snap in when you need a model provider.

## What's novel?

- **Observable session state with reducer hooks.** A Redux-like session ledger
  and in-process event bus keep every tool call and prompt render replayable.
  Built-in planning, virtual filesystem, and Python-evaluation sections ship
  with reducers that enforce domain rules while emitting structured telemetry.
  See [Session State](https://github.com/weakincentives/weakincentives/blob/main/specs/SESSIONS.md), [Prompt Event Emission](https://github.com/weakincentives/weakincentives/blob/main/specs/EVENTS.md),
  [Planning Tools](https://github.com/weakincentives/weakincentives/blob/main/specs/PLANNING_TOOL.md), [Virtual Filesystem Tools](https://github.com/weakincentives/weakincentives/blob/main/specs/VFS_TOOLS.md),
  and [Asteval Integration](https://github.com/weakincentives/weakincentives/blob/main/specs/ASTEVAL.md).
- **Composable prompt blueprints with strict contracts.** Dataclass-backed
  sections compose into reusable blueprints that render validated Markdown and
  expose tool contracts automatically. Specs: [Prompt Overview](https://github.com/weakincentives/weakincentives/blob/main/specs/PROMPTS.md),
  [Prompt Composition](https://github.com/weakincentives/weakincentives/blob/main/specs/PROMPTS_COMPOSITION.md), and
  [Structured Output](https://github.com/weakincentives/weakincentives/blob/main/specs/STRUCTURED_OUTPUT.md).
- **Chapter-driven visibility controls.** Chapters gate when prompt regions
  enter the model context, defaulting to closed until runtime policies open
  them. Expansion strategies and lifecycle guidance live in
  [Chapters Specification](https://github.com/weakincentives/weakincentives/blob/main/specs/CHAPTERS.md).
- **Override-friendly workflows that scale into optimization.** Prompt
  definitions ship with hash-based descriptors and on-disk overrides that stay
  in sync through schema validation and Git-root discovery, laying the
  groundwork for iterative optimization. Review
  [Prompt Overrides](https://github.com/weakincentives/weakincentives/blob/main/specs/PROMPT_OVERRIDES.md) for the full contract.
- **Provider adapters standardize tool negotiation.** Shared conversation
  loops negotiate tool calls, apply JSON-schema response formats, and normalize
  structured payloads so the runtime stays model-agnostic. See
  [Adapter Specification](https://github.com/weakincentives/weakincentives/blob/main/specs/ADAPTERS.md) and provider-specific docs such as
  [LiteLLM Adapter](https://github.com/weakincentives/weakincentives/blob/main/specs/LITE_LLM_ADAPTER.md).
- **Local-first, deterministic execution.** Everything runs locally without
  hosted dependencies, and prompt renders stay diff-friendly so version control
  captures intent instead of churn. The code-review example ties it together
  with override-aware prompts, session telemetry, and replayable tooling.

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
[Dataclass Serde Utilities](https://github.com/weakincentives/weakincentives/blob/main/specs/DATACLASS_SERDE.md) and
[Structured Output via `Prompt[OutputT]`](https://github.com/weakincentives/weakincentives/blob/main/specs/STRUCTURED_OUTPUT.md) for the
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
[Session State](https://github.com/weakincentives/weakincentives/blob/main/specs/SESSIONS.md), [Prompt Event Emission](https://github.com/weakincentives/weakincentives/blob/main/specs/EVENTS.md),
[Virtual Filesystem Tools](https://github.com/weakincentives/weakincentives/blob/main/specs/VFS_TOOLS.md), [Planning Tools](https://github.com/weakincentives/weakincentives/blob/main/specs/PLANNING_TOOL.md),
and [Asteval Integration](https://github.com/weakincentives/weakincentives/blob/main/specs/ASTEVAL.md).

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
[Tool Runtime Specification](https://github.com/weakincentives/weakincentives/blob/main/specs/TOOLS.md) to match the handler,
`ToolContext`, and `ToolResult` contracts.

```python
from dataclasses import dataclass
from pathlib import Path

from weakincentives.prompt import Tool, ToolResult


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

from weakincentives.runtime.events import ToolInvoked


@dataclass
class ReviewedSymbol:
    query: str
    matches: tuple[SymbolMatch, ...]


def track_reviewed_symbols(
    reviewed: tuple[ReviewedSymbol, ...],
    event: ToolInvoked,
    *,
    context: object,
) -> tuple[ReviewedSymbol, ...]:
    del context
    if event.value is None or not isinstance(event.value, SymbolSearchResult):
        return reviewed

    params = event.params
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

### 4. Compose the prompt with deterministic sections and chapters

Sections render through `string.Template`, so keep placeholders readable and
combine guidance with the tool suites into one auditable prompt tree. Long-form
checklists or escalation playbooks often span many pages and only matter for
specialized reviews; wrap them in a chapter so adapters can toggle visibility
based on the user prompt. See the [Prompt Class](https://github.com/weakincentives/weakincentives/blob/main/specs/PROMPTS.md),
[Prompt Versioning & Persistence](https://github.com/weakincentives/weakincentives/blob/main/specs/PROMPTS_VERSIONING.md), and
[Chapters Specification](https://github.com/weakincentives/weakincentives/blob/main/specs/CHAPTERS.md) for the rendering, hashing, and
visibility rules that stabilize this structure.

```python
from dataclasses import dataclass

from weakincentives import MarkdownSection, Prompt
from weakincentives.prompt import Chapter, ChaptersExpansionPolicy


@dataclass
class ReviewGuidance:
    severity_scale: str = "minor | major | critical"
    output_schema: str = "ReviewBundle with comments[] and overall_assessment"
    focus_areas: str = (
        "Security regressions, concurrency bugs, test coverage gaps, and"
        " ambiguous logic should be escalated."
    )


@dataclass
class ComplianceChapterParams:
    required: bool = False
    primary_jurisdictions: str = ""
    regulation_matrix_summary: str = ""
    escalation_contact: str = "compliance@octo.widgets"
    evidence_workspace: str = "gs://audit-artifacts"


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
    chapters=(
        Chapter[ComplianceChapterParams](
            key="review.compliance",
            title="Compliance Deep Dive",
            description=(
                "Multi-page regulations guidance that only opens when the "
                "request demands a compliance audit."
            ),
            sections=(
                MarkdownSection[ComplianceChapterParams](
                    title="Regulatory Background",
                    key="review.compliance.background",
                    template="""
                    Compliance review requested.
                    Focus jurisdictions: ${primary_jurisdictions}

                    The attached regulation matrix may span many pages. Only
                    cite sections that apply to this pull request.
                    """,
                    default_params=ComplianceChapterParams(),
                ),
                MarkdownSection[ComplianceChapterParams](
                    title="Compliance Checklist",
                    key="review.compliance.checklist",
                    template="""
                    - Summarize gaps against: ${regulation_matrix_summary}
                    - Escalate urgent findings to: ${escalation_contact}
                    - Link all evidence in: ${evidence_workspace}
                    """,
                    default_params=ComplianceChapterParams(),
                ),
            ),
            default_params=ComplianceChapterParams(),
            enabled=lambda params: params.required,
        ),
    ),
)


requires_compliance_review = True  # derived from user metadata
compliance_params = ComplianceChapterParams(
    required=requires_compliance_review,
    primary_jurisdictions="SOX §404, PCI-DSS",
    regulation_matrix_summary="See 12-page compliance dossier in appendix.",
    evidence_workspace="gs://audit-artifacts/octo-widgets",
)

expanded_prompt = review_prompt.expand_chapters(
    ChaptersExpansionPolicy.ALL_INCLUDED,
    chapter_params={
        "review.compliance": compliance_params,
    },
)


rendered = expanded_prompt.render(
    PullRequestContext(
        repository="octo/widgets",
        title="Add caching layer",
        body="Introduces memoization to reduce redundant IO while preserving correctness.",
        files_summary="loader.py, cache.py",
    ),
    ReviewGuidance(),
    compliance_params,
)


print(rendered.text)
print([tool.name for tool in rendered.tools])
```

Set `requires_compliance_review = False` (and skip the chapter parameters) when
the user prompt does not request a regulated-industry audit—the compliance
chapter stays closed and the oversized guidance never reaches the model.

### 5. Evaluate the prompt with an adapter

Adapters send the rendered prompt to a provider and publish telemetry to the
event bus; the session wiring above captures `PromptExecuted` and `ToolInvoked`
events automatically. Pass the chapter-expanded prompt plus the same parameter
dataclasses you used for rendering so the adapter sees the specialized
compliance guidance. For payload formats and parsing guarantees see
[Adapter Evaluation](https://github.com/weakincentives/weakincentives/blob/main/specs/ADAPTERS.md) and
[Native OpenAI Structured Outputs](https://github.com/weakincentives/weakincentives/blob/main/specs/NATIVE_OPENAI_STRUCTURED_OUTPUTS.md).

```python
from weakincentives.adapters.openai import OpenAIAdapter


adapter = OpenAIAdapter(
    model="gpt-4o-mini",
    client_kwargs={"api_key": "sk-..."},
)


response = adapter.evaluate(
    expanded_prompt,
    PullRequestContext(
        repository="octo/widgets",
        title="Add caching layer",
        body="Introduces memoization to reduce redundant IO while preserving correctness.",
        files_summary="loader.py, cache.py",
    ),
    ReviewGuidance(),
    compliance_params,
    bus=bus,
    session=session,
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
[Session State](https://github.com/weakincentives/weakincentives/blob/main/specs/SESSIONS.md) and
[Snapshot Capture and Rollback](https://github.com/weakincentives/weakincentives/blob/main/specs/SESSIONS.md#snapshot-capture-and-rollback)
for selector and rollback rules.

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
`.weakincentives/prompts/overrides/`. Refer to the
[Prompt Overrides specification](https://github.com/weakincentives/weakincentives/blob/main/specs/PROMPT_OVERRIDES.md) to keep namespace,
key, and tag hashes aligned.

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

rendered_with_override = review_prompt.render(
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
workspace described in the [Prompt Overrides specification](https://github.com/weakincentives/weakincentives/blob/main/specs/PROMPT_OVERRIDES.md).
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
evaluation replayable. For long-lived deployments, follow the
[Prompt Overrides specification](https://github.com/weakincentives/weakincentives/blob/main/specs/PROMPT_OVERRIDES.md) to keep overrides
and tool descriptors in sync.

## Logging

Weak Incentives ships a structured logging adapter so hosts can add contextual
metadata to every record without manual dictionary plumbing. Call
`configure_logging()` during startup to install the default handler and then
bind logger instances wherever you need telemetry:

```python
from weakincentives.runtime.logging import (
    StructuredLogPayload,
    configure_logging,
    get_logger,
)

configure_logging(json_mode=True)
logger = get_logger("demo").bind(context={"component": "cli"})
logger.info(
    "boot",
    payload=StructuredLogPayload(event="demo.start", context={"attempt": 1}),
)
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
