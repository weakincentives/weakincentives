# Weak Incentives

Tools for developing and optimizing side effect free background agents. The codebase is intentionally small and pre-alpha; expect the public API to move quickly while we refine the prompt abstractions.

## What's Inside
- Fully typed prompt composition primitives (`Prompt`, `Section`, `TextSection`, `Tool`, `ToolResult`) for assembling deterministic Markdown prompts with attached tool metadata.
- Optional OpenAI adapter that gates imports behind a friendly error and returns the SDK client when the extra dependency is present.
- A quiet quality gate wired through Ruff, Ty, pytest (100% coverage), Bandit, Deptry, and pip-audit, all executed via `uv`.
- Stdlib-only dataclass serde utilities (`parse`, `dump`, `clone`, `schema`) for Pydantic-like ergonomics without third-party dependencies.

## Getting Started
1. Install Python 3.14 (pyenv users can run `pyenv install 3.14.0`).
2. Install [`uv`](https://github.com/astral-sh/uv).
3. Sync the virtualenv and tooling:
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
```python
from dataclasses import dataclass

from weakincentives.prompts import Prompt, TextSection, Tool, ToolResult

@dataclass
class GuidanceParams:
    name: str

@dataclass
class LookupParams:
    entity_id: str

@dataclass
class LookupResult:
    entity_id: str


def lookup_handler(params: LookupParams) -> ToolResult[LookupResult]:
    result = LookupResult(entity_id=params.entity_id)
    return ToolResult(message=f"Fetched {result.entity_id}.", payload=result)


lookup_tool = Tool[LookupParams, LookupResult](
    name="lookup_entity",
    description="Fetch information for an entity id.",
    handler=lookup_handler,
)

overview = TextSection[GuidanceParams](
    title="Guidance",
    body="Use ${name}'s lookup tool when you need context.",
    tools=[lookup_tool],
)

prompt = Prompt(name="demo", sections=[overview])
print(prompt.render(GuidanceParams(name="Ada")))
print(prompt.tools(GuidanceParams(name="Ada")))
```

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
