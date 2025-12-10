# CLAUDE.md

Quick-reference guide for Claude and other AI assistants working in the
`weakincentives` repository.

## Project Overview

WINK (Weak Incentives) is a Python library for building deterministic,
side-effect-free background agents. It provides:

- **Redux-style sessions** with immutable event ledgers and pure reducers
- **Typed prompt composition** using dataclass-backed sections
- **Provider-agnostic adapters** (OpenAI, LiteLLM)
- **Design-by-contract enforcement** via decorators
- **Sandboxed tooling** (VFS, asteval, Podman)

## Essential Commands

```bash
# Setup
uv sync && ./install-hooks.sh

# Development workflow
make format          # Format code (ruff, 88-char lines)
make lint            # Lint with ruff --preview
make typecheck       # ty check + pyright (strict mode)
make test            # Pytest with 100% coverage requirement
make check           # ALL checks - run before every commit

# Additional checks
make bandit          # Security scanning
make deptry          # Dependency analysis
make pip-audit       # Vulnerability scanning
make markdown-check  # Markdown formatting
make mutation-test   # Mutation testing (80% score gate)

# Integration tests (requires OPENAI_API_KEY)
make integration-tests
```

**Critical**: Always run `make check` before committing. The pre-commit hooks
enforce a clean run.

## Architecture

```
src/weakincentives/
├── adapters/        # Provider integrations (OpenAI, LiteLLM)
├── cli/             # wink CLI entrypoints
├── dataclasses/     # FrozenDataclass utilities
├── dbc/             # Design-by-contract decorators
├── prompt/          # Section/Prompt composition, overrides, tools
├── runtime/         # Session, events, logging
├── serde/           # Dataclass serialization (no Pydantic)
├── tools/           # Planning, VFS, asteval, Podman
└── types/           # JSON type aliases
```

## Code Conventions

### Type Annotations

- Strict pyright mode is enforced
- Type annotations are the source of truth; avoid redundant runtime guards
- Use `FrozenDataclass` for immutable data structures

### Design-by-Contract

Public APIs use DbC decorators from `weakincentives.dbc`:

```python
from weakincentives.dbc import require, ensure, invariant, pure

@require(lambda x: x > 0, "x must be positive")
@ensure(lambda result: result >= 0, "result must be non-negative")
def compute(x: int) -> int:
    ...
```

Always read `specs/DBC.md` before modifying DbC-decorated modules.

### Dataclass Patterns

```python
from dataclasses import dataclass

# Preferred: slots and frozen for immutability
@dataclass(slots=True, frozen=True)
class MyConfig:
    name: str
    count: int = 0
```

### Tool Handlers

Tool handlers follow a consistent signature:

```python
def my_handler(params: ParamsType, *, context: ToolContext) -> ToolResult[ResultType]:
    return ToolResult(message="done", value=ResultType(...), success=True)
```

### Imports

- First-party imports use `weakincentives.*`
- Keep `__init__.py` exports curated and minimal
- Favor composition over inheritance

## Testing Requirements

- **100% coverage** required for `src/weakincentives/`
- Pytest runs with `--strict-config --strict-markers`
- Flaky tests retry twice automatically
- Add fixtures to `tests/helpers/` when needed
- Mutation testing protects hotspots in `runtime/session/` and `serde/`

Run focused tests during development:

```bash
uv run pytest tests/path/to/test.py -v
```

Always finish with `make test` to verify coverage.

## Spec Documents

Consult these specs before modifying related code:

| Spec | When to Read |
|------|--------------|
| `specs/ADAPTERS.md` | Provider adapters, structured output, throttling |
| `specs/DATACLASSES.md` | Serde utilities or frozen dataclass patterns |
| `specs/DBC.md` | Editing DbC-decorated modules (required) |
| `specs/HOSTED_TOOLS.md` | Provider-executed tools (web search, code interpreter) |
| `specs/MAIN_LOOP.md` | Main loop orchestration, visibility handling, event-driven execution |
| `specs/PROMPTS.md` | Prompt system, composition, structured output |
| `specs/PROMPT_OPTIMIZATION.md` | Override system or optimizer logic |
| `specs/SESSIONS.md` | Session lifecycle, events, deadlines, budgets |
| `specs/TOOLS.md` | Adding/modifying tools, planning tools |
| `specs/UNSAFE_LOCAL_SANDBOX.md` | Local sandbox for containerized environments |
| `specs/WORKSPACE.md` | VFS, Podman, asteval, workspace digest |
| `specs/THREAD_SAFETY.md` | Concurrency or shared state |
| `specs/LOGGING.md` | Logging surfaces |

Full spec index in `AGENTS.md`.

## Key Files

- `README.md` - Public overview and tutorial
- `AGENTS.md` - Canonical contributor handbook
- `llms.md` - PyPI README with full public API reference
- `GLOSSARY.md` - Terminology definitions
- `CHANGELOG.md` - Track changes under "Unreleased"
- `code_reviewer_example.py` - Complete working example

## Common Patterns

### Creating a Prompt

```python
from weakincentives import Prompt, MarkdownSection, Tool

prompt = Prompt[OutputType](
    ns="my-namespace",
    key="my-prompt",
    name="my_prompt",
    sections=[
        MarkdownSection(
            title="Instructions",
            template="Do something with $param",
            key="instructions",
            tools=(my_tool,),
        ),
    ],
)
```

### Session State

```python
from weakincentives.runtime import Session, InProcessEventBus

bus = InProcessEventBus()
session = Session(bus=bus)

# Query state
plan = session.query(Plan).latest()

# Mutate state (fluent API mirrors query)
session.mutate(Plan).seed(initial_plan)           # Initialize slice
session.mutate(Plan).dispatch(AddStep(...))       # Event through reducers
session.mutate(Plan).register(AddStep, reducer)   # Register reducer
session.mutate().reset()                          # Clear all slices
```

### Provider Adapter

```python
from weakincentives.adapters.openai import OpenAIAdapter

adapter = OpenAIAdapter(model="gpt-4o")
response = adapter.evaluate(prompt, bus=bus, session=session)
output = response.output  # Typed result
```

## Stability Notice

This is **alpha software**. All APIs may change without backward compatibility.
Do not add backward-compatibility shims or deprecation warnings—delete unused
code completely.

## Quick Checklist

Before submitting changes:

- [ ] `make check` passes
- [ ] New public APIs have DbC decorators where appropriate
- [ ] Tests cover new code paths (100% coverage)
- [ ] Relevant spec documents consulted and updated if needed
- [ ] `CHANGELOG.md` updated for user-visible changes
