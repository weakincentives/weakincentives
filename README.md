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

## Quickstart

````python
from dataclasses import dataclass
from weakincentives import (
    MarkdownSection,
    Prompt,
    Tool,
    ToolResult,
    parse_structured_output,
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
    return ToolResult(message=f"Loaded {details.title}", value=details)

catalog_tool = Tool[SourceLookup, SourceDetails](
    name="catalog_lookup",
    description="Look up a primary source identifier and return details.",
    handler=lookup_source,
)

task_section = MarkdownSection[ResearchGuidance](
    title="Task",
    template=(
        "Research ${topic}. Use `catalog_lookup` for citations and reply with a "
        "JSON summary."
    ),
    key="research.task",
    tools=[catalog_tool],
)

prompt = Prompt[ResearchSummary](
    ns="examples/research",
    key="research.run",
    name="research_prompt",
    sections=[task_section],
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
result = parse_structured_output(reply, rendered)
print(result.summary)
print(result.citations)
````

The rendered prompt text stays deterministic, tool metadata travels with the prompt,
and `parse_structured_output` enforces your dataclass contract.

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
