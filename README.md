# Weak Incentives Is All You Need

Tools for developing and optimizing side effect free background agents.

## Overview
- Python 3.14 library with a growing prompt-composition toolkit.
- `Prompt`, `Section`, and `TextSection` abstractions help build structured, parameterised Markdown prompts.
- Strict typing, Ruff formatting/linting, and pytest coverage keep the surface reliable while the API evolves.

## Quick Start
```bash
# Install dependencies and the managed virtualenv
uv sync

# (Optional) install git hooks that run the quality gate before commits
./install-hooks.sh
```

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
prompt = Prompt(root_sections=[section])

print(prompt.render(GreetingParams(name="Ada")))
# -> '## Greeting\n\nHello Ada!'
```

Sections can declare child sections, defaults, and `enabled` predicates. The prompt prevents placeholder/parameter mismatches and reports rich context when validation or rendering fails.

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
