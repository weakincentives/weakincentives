# Weak Incentives Is All You Need

Tools for developing and optimizing side effect free background agents.

**Built by coding agents, for coding agents.**

## Overview
- Python 3.14 library with a growing prompt-composition toolkit.
- `Prompt`, `Section`, `TextSection`, and `ToolsSection` abstractions help build structured, parameterised Markdown prompts.
- Strict typing, Ruff formatting/linting, and pytest coverage keep the surface reliable while the API evolves.

## Guiding Principles
- Dataclasses anchor every contract so schema issues surface immediately.
- Fail loudly with contextual diagnostics instead of masking invalid templates or sections.
- Keep markdown structure deterministic and readable with disciplined section hierarchies.
- Prefer declarative configuration over embedded control flow within prompts and tools.
- Reuse shared dataclasses freely, while keeping tool names unique and descriptive.
- Document behavior in the prompt layer and leave provider-specific execution to adapters.

## Quick Start
```bash
# Install dependencies and the managed virtualenv
uv sync

# (Optional) install git hooks that run the quality gate before commits
./install-hooks.sh
```

## Optional Extras
- `openai`: install with `uv sync --extra openai` or `pip install weakincentives[openai]` to enable the OpenAI adapter utilities.
- `make test` and downstream aggregates run with all extras enabled, so adapter coverage stays green if an optional provider regresses.

## Using the Prompt Toolkit
```python
from dataclasses import dataclass

from weakincentives.prompts import Prompt, TextSection

@dataclass
class GreetingParams:
    name: str

section = TextSection(
    title="Greeting",
    body="Hello ${name}!",
    params=GreetingParams,
)
prompt = Prompt(sections=[section])

print(prompt.render(GreetingParams(name="Ada")))
# -> '## Greeting\n\nHello Ada!'
```

Sections can declare child sections, defaults, and `enabled` predicates. The prompt prevents placeholder/parameter mismatches and reports rich context when validation or rendering fails.

## Working with Tools
```python
from dataclasses import dataclass

from weakincentives.prompts import Prompt, TextSection, Tool, ToolResult, ToolsSection

@dataclass
class GuidanceParams:
    primary_tool: str

@dataclass
class ToolDescriptionParams:
    primary_tool: str = "lookup_entity"

@dataclass
class LookupParams:
    entity_id: str

@dataclass
class LookupResult:
    entity_id: str
    document_url: str

def lookup_handler(params: LookupParams) -> ToolResult[LookupResult]:
    result = LookupResult(entity_id=params.entity_id, document_url="https://example.com")
    return ToolResult(message=f"Fetched {result.entity_id}.", payload=result)

lookup_tool = Tool[LookupParams, LookupResult](
    name="lookup_entity",
    description="Fetch structured information for a given entity id.",
    handler=lookup_handler,
)

prompt = Prompt(
    sections=[
        TextSection(
            title="Guidance",
            body="Use ${primary_tool} for critical lookups.",
            params=GuidanceParams,
            children=[
                ToolsSection(
                    title="Available Tools",
                    tools=[lookup_tool],
                    params=ToolDescriptionParams,
                    defaults=ToolDescriptionParams(),
                    description="Invoke ${primary_tool} whenever you need fresh context.",
                )
            ],
        )
    ]
)

tools = prompt.tools(GuidanceParams(primary_tool="lookup_entity"))
assert tools[0].handler is lookup_handler
```

`Prompt.tools()` returns the tools contributed by enabled `ToolsSection`s in depth-first order, making it easy to hand descriptors and handlers to an orchestration layer without duplicating configuration.

## Development Workflow
- `make format` / `make format-check` – auto-format or audit with Ruff.
- `make lint` / `make lint-fix` – linting with Ruff.
- `make typecheck` – run the Python 3.14 type checker (`ty`).
- `make test` – execute pytest with coverage thresholds.
- `make check` – aggregate format-check, lint, typecheck, and tests.

All tasks run inside the `uv` virtualenv, so prefix ad-hoc commands with `uv run ...` when needed.

## Project Structure
- `src/weakincentives/` – library source, including the prompt module.
- `tests/` – pytest suites covering prompt validation and rendering edge cases.
- `specs/` – design notes; see `PROMPTS.md` for prompt requirements and terminology.

The library is pre-alpha: expect the public API to change while we refine the prompt abstractions and add agent tooling.
