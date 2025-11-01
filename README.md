# Weak Incentives

**Lean, typed building blocks for side-effect-free background agents.**
Compose deterministic prompts, define typed tools, and get strict JSON back—without heavy dependencies. Optional adapters (OpenAI, LiteLLM) snap in when you need a provider.

______________________________________________________________________

## Requirements

- Python **3.12+**
- Astral’s **uv** CLI (all commands below assume `uv`)

______________________________________________________________________

## Install

```bash
uv add weakincentives
# optional: provider adapters
uv add "weakincentives[openai]"    # OpenAI SDK support
uv add "weakincentives[litellm]"   # LiteLLM compatibility
# (or, when cloning: uv sync --extra openai --extra litellm)
```

______________________________________________________________________

## 60-Second Quickstart

````python
from dataclasses import dataclass
from weakincentives import (
    MarkdownSection,
    Prompt,
    Tool,
    ToolResult,
    parse_structured_output,
)

# ---- Inputs & outputs (dataclasses = your contract) ----
@dataclass
class ResearchGuidance:
    topic: str

@dataclass
class SourceLookup:
    source_id: str

@dataclass
class SourceDetails:
    source_id: str
    title: str

@dataclass
class ResearchSummary:
    summary: str
    citations: list[str]

# ---- Local tool (typed params → typed result) ----
def lookup_source(params: SourceLookup) -> ToolResult[SourceDetails]:
    details = SourceDetails(source_id=params.source_id, title="Ada Lovelace Archive")
    return ToolResult(message=f"Loaded {details.title}", value=details)

catalog_tool = Tool[SourceLookup, SourceDetails](
    name="catalog_lookup",
    description="Look up a primary source identifier and return structured details.",
    handler=lookup_source,
)

# ---- Prompt with a tool and a typed JSON output ----
research_section = MarkdownSection[ResearchGuidance](
    title="Task",
    template=(
        "Research ${topic}. Use the `catalog_lookup` tool for citations and return "
        "a JSON summary with citations."
    ),
    tools=[catalog_tool],
)

prompt = Prompt[ResearchSummary](
    name="research_run",
    sections=[research_section],
)

# Render once and pass to your model adapter of choice
rendered = prompt.render(ResearchGuidance(topic="Ada Lovelace"))
print(rendered.text)                 # deterministic markdown
print([t.name for t in rendered.tools])  # ('catalog_lookup',)

# Later: parse the model’s reply strictly into your dataclass
reply = """```json
{
  "summary": "Ada Lovelace pioneered computing...",
  "citations": ["catalog_lookup:ada-archive"]
}
```"""
parsed = parse_structured_output(reply, rendered)
print(parsed.summary)
print(parsed.citations)
````

______________________________________________________________________

## Key Ideas (at a glance)

- **Typed prompts, deterministic renders**
  `Prompt` + `MarkdownSection` compose markdown with strict placeholder checks. Missing or unknown fields error early.
- **Tools live where they’re documented**
  Attach `Tool[Params, Result]` to any section. Handlers run **locally**; models only see the short tool message.
- **Strict JSON output**
  Declare `Prompt[T]` or `Prompt[list[T]]` and get a built-in “Response Format” section plus a strict `parse_structured_output(...)` helper.
- **Stdlib serde, zero heavy deps**
  `parse`, `dump`, `clone`, `schema` give Pydantic-like ergonomics with dataclasses only.
- **Quiet, predictable runtime**
  Minimal imports; adapters use a **blocking single-turn** API and emit in-process telemetry events.
- **Session-scoped planning tools**
  Drop `PlanningToolsSection` into a prompt to expose plan setup, edits, and
  status tracking for the active `Session`. Read the
  [planning tool spec](specs/PLANNING_TOOL.md) for details.

______________________________________________________________________

## Optional Adapters

```python
from weakincentives.adapters import (
    LiteLLMAdapter,
    OpenAIAdapter,
)

# OpenAI SDK (requires `uv add "weakincentives[openai]"`)
openai_adapter = OpenAIAdapter(
    model="gpt-4o-mini",
    client_kwargs={"api_key": "sk-..."},
)

# LiteLLM shim (requires `uv add "weakincentives[litellm]"`)
litellm_adapter = LiteLLMAdapter(
    model="gpt-4o-mini",
    completion_kwargs={"api_key": "sk-..."},
)

# response = openai_adapter.evaluate(prompt, ResearchGuidance(...), bus=...)
# response = litellm_adapter.evaluate(prompt, ResearchGuidance(...), bus=...)
```

Both adapters raise a clear runtime error with install guidance if the optional dependency is missing.

See `examples/openai_runner.py` and `examples/litellm_runner.py` for runnable demos.

______________________________________________________________________

## When to Use / When to Skip

**Use it when you need:**

- Background jobs/cron workers that **must** return well-typed JSON or fail fast
- Local, audited tool calls with typed params/results
- Minimal runtime footprint and deterministic prompts

**Probably skip if you need:**

- **Streaming** tokens or async/concurrency out of the box
- RAG/connectors/graph composers
- Multi-turn agent orchestration today
  *(these are discussed in the roadmap/specs, but not shipped yet)*

______________________________________________________________________

## For Library Developers

### Environment Setup

1. Install Python 3.12 (e.g., `pyenv install 3.12.0`)

1. Install [`uv`](https://github.com/astral-sh/uv)

1. Sync the env and (optional) git hooks:

   ```bash
   uv sync
   ./install-hooks.sh  # wires git hooks that call `make check`
   ```

1. Use `uv run ...` for ad-hoc scripts to stay inside the managed env.

### Development Workflow

- `make format` / `make format-check` — Ruff formatting
- `make lint` / `make lint-fix` — Ruff linting
- `make typecheck` — Ty with warnings as errors
- `make test` — pytest via `build/run_pytest.py` with `--cov-fail-under=100`
- `make bandit`, `make deptry`, `make pip-audit` — security & deps checks
- `make check` — aggregate gate (enforced by git hooks)

### Project Layout

- `src/weakincentives/` — package root
- `src/weakincentives/prompt/` — prompt, section, tool primitives
- `src/weakincentives/adapters/` — optional provider integrations
- `tests/` — coverage for prompts, tools, adapters
- `specs/` — design docs (see `specs/PROMPTS.md` for prompt requirements)
- `build/` — quiet wrappers for CLI tools under `uv`
- `hooks/` — symlink-friendly git hooks

### Releases

Tag git as `vX.Y.Z`. Version is derived from tags.

### More Docs

See `AGENTS.md` for the agent handbook and workflows.
`ROADMAP.md` and `specs/` cover upcoming work and formal requirements.

______________________________________________________________________

**License:** Apache-2.0 • **Status:** Alpha (APIs may change)
