# Weak Incentives

Tools for developing and optimizing side effect free background agents. The codebase is intentionally small and pre-alpha; expect the public API to move quickly while we refine the prompt abstractions.

## What's Inside

- Fully typed prompt composition primitives (`Prompt`, `Section`, `TextSection`, `Tool`, `ToolResult`) for assembling deterministic Markdown prompts with attached tool metadata.
- Optional OpenAI adapter that gates imports behind a friendly error and returns the SDK client when the extra dependency is present.
- A quiet quality gate wired through Ruff, Ty, pytest (100% coverage), Bandit, Deptry, and pip-audit, all executed via `uv`.
- Stdlib-only dataclass serde utilities (`parse`, `dump`, `clone`, `schema`) for Pydantic-like ergonomics without third-party dependencies.

## Getting Started

1. Install Python 3.14 (pyenv users can run `pyenv install 3.14.0`).
1. Install [`uv`](https://github.com/astral-sh/uv).
1. Sync the virtualenv and tooling:
   ```bash
   uv sync
   ./install-hooks.sh  # optional – sets up git hooks that call make check
   ```

Use `uv run ...` when invoking ad-hoc scripts so everything runs inside the managed environment.

## Development Workflow

- `make format` / `make format-check` — run Ruff formatters.
- `make lint` / `make lint-fix` — lint with Ruff.
- `make typecheck` — execute Ty with warnings promoted to errors.
- `make test` — run pytest via `tools/run_pytest.py` with `--cov-fail-under=100`.
- `make bandit`, `make deptry`, `make pip-audit` — security, dependency, and vulnerability audits.
- `make check` — aggregate the quiet quality gate; prefer this before commits.

## Usage Example

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

## Optional Extras

Install the OpenAI client helpers with:

```bash
uv sync --extra openai
```

or

```bash
pip install weakincentives[openai]
```

Then create a client:

```python
from weakincentives.adapters.openai import create_openai_client

client = create_openai_client(api_key="sk-...")
```

If the dependency is missing, the helper raises a runtime error with install instructions.

## Project Layout

- `src/weakincentives/` — package root for the Python module.
- `src/weakincentives/prompts/` — prompt, section, and tool primitives.
- `src/weakincentives/adapters/` — optional provider integrations.
- `tests/` — pytest suites covering prompts, tools, and adapters.
- `specs/` — design docs; see `specs/PROMPTS.md` for prompt requirements.
- `tools/` — thin wrappers that keep CLI tools quiet inside `uv`.
- `hooks/` — symlink-friendly git hooks (install via `./install-hooks.sh`).

## Releases

Version numbers come from git tags named `vX.Y.Z`. Tag the repository manually before pushing a release.

## More Documentation

See `AGENTS.md` for the full agent handbook, workflows, and TDD expectations.
