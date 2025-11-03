# Weak Incentives

**Lean, typed building blocks for side-effect-free background agents.**  
Compose deterministic prompts, run typed tools, and parse strict JSON replies without heavy dependencies. Optional adapters snap in when you need a model provider.

- **Homepage:** https://weakincentives.com
- **License:** Apache-2.0
- **Status:** Alpha (APIs may change between releases)

---

## Quickstart (5 minutes)

Run the included code-reviewer example end-to-end:

```bash
# 0) Clone and enter the repo
git clone https://github.com/weakincentives/weakincentives.git
cd weakincentives

# 1) Create a local environment with the OpenAI adapter extra
uv sync --extra openai

# 2) Provide a model key (example: OpenAI)
export OPENAI_API_KEY=sk-...

# 3) Run the example
uv run python code_reviewer_example.py
```

**More examples**

```bash
# Native OpenAI structured outputs example
uv run python openai_example.py

# LiteLLM adapter example
uv sync --extra litellm
uv run python litellm_example.py
```

---

## Why now?

This library was built out of frustration with LangGraph and DSPy to explore better patterns for state and context management when building apps with LLMs—while still allowing prompts to be automatically optimized.

---

## Highlights

* **Deterministic prompt trees**
  Namespaced sections, Markdown rendering, placeholder verification, and **tool-aware versioning metadata**.
* **Stdlib-only dataclass serde**
  `parse`, `dump`, `clone`, `schema` keep request/response types honest across model calls.
* **Session state + event bus**
  Collect prompt and tool telemetry for downstream automation (no separate state store).
* **Built-in tool suites**
  Planning and Virtual Filesystem tools provide durable plans and sandboxed edits backed by reducers/selectors.
* **Provider adapters**
  Optional OpenAI and LiteLLM adapters integrate structured output parsing, tool orchestration, and telemetry hooks.

---

## Requirements

* Python **3.12+** (repo pins **3.14** in `.python-version` for development)
* [`uv`](https://github.com/astral-sh/uv) CLI

## Install (as a dependency)

```bash
uv add weakincentives
# optional provider adapters
uv add "weakincentives[openai]"
uv add "weakincentives[litellm]"
# cloning the repo? use: uv sync --extra openai --extra litellm
```

---

## Tutorial: Build a Stateful Code-Reviewing Agent

Use Weak Incentives to assemble a reproducible reviewer that tracks every decision, stages file edits in a sandbox, and evaluates quick calculations when the diff raises questions. Compared to LangGraph you don’t need to bolt on a state store—the `Session` captures prompt/tool telemetry. Unlike DSPy, sections expose versioning and override hooks so optimizers can swap instructions without touching the runtime.

### 1) Model the review input and the structured output

Typed dataclasses make adapters’ structured parsing predictable.

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

### 2) Create a session, enable built-in tools, and mount diffs

Planning, virtual filesystem, and Python-evaluation sections register reducers on the session so every run can plan, stage edits, and compute. Copy your unified diff files into `/srv/agent-mounts` before running.

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

> The host mount resolves `octo_widgets/cache-layer.diff` relative to `/srv/agent-mounts` and exposes it to the agent as `diffs/cache-layer.diff` through the VFS snapshot.

### 3) Add a small helper tool (symbol search)

Custom tools are typed functions that return structured results. This one scans a repo checkout you mount at `/srv/agent-repo`.

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
    description="Search the repository checkout for a symbol and return file snippets.",
    handler=find_symbol,
)
```

### 4) Compose a deterministic prompt from sections

Sections use `string.Template` placeholders. The prompt tree is auditable and versioned.

```python
from dataclasses import dataclass
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

### 5) Evaluate the prompt with an adapter

Use the OpenAI adapter (or the LiteLLM adapter if that’s your stack).

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
    bus=bus,  # publishes PromptExecuted / ToolInvoked events
)

bundle = response.output
if bundle is None:
    raise RuntimeError("Structured parsing failed")

for comment in bundle.comments:
    print(f"{comment.file_path}:{comment.line} → {comment.summary}")
```

> If a required field is missing, `OpenAIAdapter` raises a `PromptEvaluationError` with provider context rather than silently degrading.

### 6) Mine session state for automation & audit

Reducers from the built-in suites record their data on the session. Selectors recover the latest snapshots.

```python
from weakincentives.session import select_latest
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

### 7) Override sections via a local overrides store

Persist optimizer-tuned instructions and have the runtime pick them up without redeploying.

```python
from pathlib import Path
from weakincentives.prompt.local_prompt_overrides_store import LocalPromptOverridesStore
from weakincentives.prompt.versioning import PromptDescriptor, PromptOverride, SectionOverride

workspace_root = Path("/srv/agent-workspace")
overrides_store = LocalPromptOverridesStore(root_path=workspace_root)

descriptor = PromptDescriptor.from_prompt(review_prompt)
seed_override = overrides_store.seed_if_necessary(review_prompt, tag="assertive-feedback")

section_path = ("review", "directives")
section_descriptor = next(s for s in descriptor.sections if s.path == section_path)

custom_override = PromptOverride(
    ns=descriptor.ns,
    prompt_key=descriptor.key,
    tag="assertive-feedback",
    sections={
        **seed_override.sections,
        section_path: SectionOverride(
            expected_hash=section_descriptor.content_hash,
            body="\n".join((
                "- Classify findings using this severity scale: minor | major | critical.",
                "- Always cite the exact diff hunk when raising a major or critical issue.",
                "- Respond with ReviewBundle JSON. Missing fields terminate the run.",
            )),
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

> The store writes to `.weakincentives/prompts/overrides/{ns}/{prompt_key}/{tag}.json` under the workspace root.

### 8) Ship it

You now have a deterministic reviewer that:

1. Enforces typed contracts for inputs, tools, and outputs.
2. Persists multi-step plans, VFS edits, and evaluation transcripts inside a session.
3. Supports optimizer-driven overrides for CI, evaluation harnesses, or on-call tuning.

For long-lived deployments, pair this with **tool-aware prompt versioning** to keep overrides and tool descriptions aligned as code evolves.

---

## Troubleshooting

* **“Structured parsing failed.”**
  The model didn’t return valid `ReviewBundle` JSON. Tighten the `analysis_section` guidance or enforce schema reminders via the overrides store.
* **Tool can’t read diffs.**
  Ensure unified diff files exist under `/srv/agent-mounts` and `HostMount.host_path` is relative to that directory.
* **OpenAI auth errors.**
  Confirm `OPENAI_API_KEY` is set and that the `openai` extra is installed (`uv sync --extra openai` or `uv add "weakincentives[openai]"`).

---

## Development Setup

1. Install Python **3.14** (e.g., `pyenv install 3.14.0`).

2. Install `uv`, then bootstrap the environment and hooks:

   ```bash
   uv sync
   ./install-hooks.sh
   ```

3. Run checks via the managed virtualenv:

   * `make format` / `make format-check`
   * `make lint` / `make lint-fix`
   * `make typecheck` (Ty + Pyright; warnings fail the build)
   * `make test` (pytest via `build/run_pytest.py`, **100% coverage enforced**)
   * `make check` (aggregates the above plus Bandit, Deptry, pip-audit, Markdown lint)

---

## Documentation

* `AGENTS.md` — operational handbook & contributor workflow
* `specs/` — design docs for prompts, planning tools, adapters, and overrides
* `ROADMAP.md` — upcoming feature sketches
* `docs/api/` — API reference

---

## License

Apache 2.0 • Status: Alpha

```

---

### Notes on alignment with `main` (for maintainers)

- The examples referenced above (`code_reviewer_example.py`, `openai_example.py`, `litellm_example.py`) and the extras instructions are present on the repository’s **main** branch. 
- The imports, class names, and paths used in the tutorial (e.g., `weakincentives.adapters.openai.OpenAIAdapter`, `weakincentives.tools.*`, and `specs/*` docs) match what’s published in the README and tree on **main**. 

If you want me to convert this into a PR body (including commit message + diff preview), say the word and I’ll prepare the patch text.
```
