# Weak Incentives

Tools for developing and optimizing side effect free background agents. The library ships prompt composition primitives, structured tool metadata, and optional provider adapters so you can scaffold deterministic automation flows quickly. All commands below assume Astral's `uv` CLI.

## For Library Users

### Installation

- `uv add weakincentives`
- Add `uv add "weakincentives[openai]"` (or `uv sync --extra openai` when cloning the repo) to enable the OpenAI adapter helpers.

### Key Features

- Fully typed prompt composition primitives (`Prompt`, `Section`, `TextSection`, `Tool`, `ToolResult`) for assembling deterministic Markdown prompts with attached tool metadata.
- Stdlib-only dataclass serde utilities (`parse`, `dump`, `clone`, `schema`) for Pydantic-like ergonomics without third-party dependencies.
- Optional OpenAI adapter that gates imports behind a friendly error and returns the SDK client when the extra is present.
- Quiet-by-default package with minimal runtime dependencies so background agents stay lean and predictable.

### Quickstart

````python
from dataclasses import dataclass

from weakincentives.prompts import (
    Prompt,
    TextSection,
    Tool,
    ToolResult,
    parse_output,
)


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


def lookup_source(params: SourceLookup) -> ToolResult[SourceDetails]:
    details = SourceDetails(source_id=params.source_id, title="Ada Lovelace Archive")
    return ToolResult(message=f"Loaded {details.title}", payload=details)


catalog_tool = Tool[SourceLookup, SourceDetails](
    name="catalog_lookup",
    description="Look up a primary source identifier and return structured details.",
    handler=lookup_source,
)

research_section = TextSection[ResearchGuidance](
    title="Task",
    body=(
        "Research ${topic}. Use the `catalog_lookup` tool for citations and return"
        " a JSON summary with citations."
    ),
    tools=[catalog_tool],
)

prompt = Prompt[ResearchSummary](
    name="research_run",
    sections=[research_section],
)

rendered = prompt.render(ResearchGuidance(topic="Ada Lovelace"))
print(rendered.text)
print([tool.name for tool in rendered.tools])

reply = """```json
{
  "summary": "Ada Lovelace pioneered computing...",
  "citations": ["catalog_lookup:ada-archive"]
}
```"""

parsed = parse_output(reply, rendered)
print(parsed.summary)
print(parsed.citations)
````

### Optional Extras

Use the OpenAI helpers once the extra is installed:

```python
from weakincentives.adapters import OpenAIAdapter

adapter = OpenAIAdapter(model="gpt-4o-mini", client_kwargs={"api_key": "sk-..."})
```

If the dependency is missing, the adapter raises a runtime error with installation guidance.

## For Library Developers

### Environment Setup

1. Install Python 3.14 (pyenv users can run `pyenv install 3.14.0`).
1. Install [`uv`](https://github.com/astral-sh/uv).
1. Sync the virtualenv and optional git hooks:
   ```bash
   uv sync
   ./install-hooks.sh  # optional – wires git hooks that call make check
   ```
1. Use `uv run ...` when invoking ad-hoc scripts so everything stays inside the managed environment.

### Development Workflow

- `make format` / `make format-check` — run Ruff formatters.
- `make lint` / `make lint-fix` — lint with Ruff.
- `make typecheck` — execute Ty with warnings promoted to errors.
- `make test` — run pytest via `build/run_pytest.py` with `--cov-fail-under=100`.
- `make bandit`, `make deptry`, `make pip-audit` — security, dependency, and vulnerability audits.
- `make check` — aggregate the quiet quality gate; run this before every commit (git hooks enforce it).

### Project Layout

- `src/weakincentives/` — package root for the Python module.
- `src/weakincentives/prompts/` — prompt, section, and tool primitives.
- `src/weakincentives/adapters/` — optional provider integrations.
- `tests/` — pytest suites covering prompts, tools, and adapters.
- `specs/` — design docs; see `specs/PROMPTS.md` for prompt requirements.
- `build/` — thin wrappers that keep CLI tools quiet inside `uv`.
- `hooks/` — symlink-friendly git hooks (install via `./install-hooks.sh`).

### Release Notes

Version numbers come from git tags named `vX.Y.Z`. Tag the repository manually before pushing a release.

### More Documentation

`AGENTS.md` captures the full agent handbook, workflows, and TDD expectations. `ROADMAP.md` and `specs/` document upcoming work and prompt requirements.
