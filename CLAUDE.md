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
├── adapters/        # Provider integrations (OpenAI, LiteLLM, Claude Agent SDK)
├── cli/             # wink CLI entrypoints
├── contrib/         # Batteries for specific agent styles
│   ├── tools/       # Planning, VFS, asteval, Podman, workspace digest
│   └── optimizers/  # Workspace digest optimizer
├── dataclasses/     # FrozenDataclass utilities
├── dbc/             # Design-by-contract decorators
├── optimizers/      # Optimizer framework and protocols
├── prompt/          # Section/Prompt composition, overrides, tools
├── runtime/         # Session, events, logging
├── serde/           # Dataclass serialization (no Pydantic)
└── types/           # JSON type aliases
```

The library is organized as "core primitives" + "batteries for specific agent
styles":

- **Core** (`weakincentives.*`): Prompt composition, sessions, adapters,
  serde, dbc
- **Contrib** (`weakincentives.contrib.*`): Planning tools, VFS, Podman,
  asteval, workspace optimizers

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
| `specs/CLAUDE_AGENT_SDK.md` | Claude Agent SDK adapter, MCP tool bridging |
| `specs/DATACLASSES.md` | Serde utilities or frozen dataclass patterns |
| `specs/DBC.md` | Editing DbC-decorated modules (required) |
| `specs/EVALS.md` | Evaluation framework, datasets, evaluators, metrics |
| `specs/EXECUTION_STATE.md` | ExecutionState root, transactional tool execution, snapshot/rollback |
| `specs/FILESYSTEM.md` | Filesystem protocol, backend implementations, ToolContext integration |
| `specs/MAIN_LOOP.md` | Main loop orchestration, visibility handling, event-driven execution |
| `specs/PROMPTS.md` | Prompt system, composition, structured output |
| `specs/PROMPT_OPTIMIZATION.md` | Override system or optimizer logic |
| `specs/SESSIONS.md` | Session lifecycle, events, deadlines, budgets |
| `specs/TESTING.md` | Test harnesses, fault injection, fuzzing, coverage standards |
| `specs/TOOLS.md` | Adding/modifying tools, planning tools |
| `specs/WORKSPACE.md` | VFS, Podman, asteval, workspace digest |
| `specs/THREAD_SAFETY.md` | Concurrency or shared state |
| `specs/LOGGING.md` | Logging surfaces |
| `specs/LANGSMITH.md` | LangSmith telemetry, prompt hub, evaluation integration |

Full spec index in `AGENTS.md`.

## Guides

User-facing how-to material lives in `guides/`:

| Guide | Description |
|-------|-------------|
| `guides/code-review-agent.md` | End-to-end walkthrough of the code reviewer example |

Guides cover quickstarts, patterns, recipes, and best practices. Design specs
(what the system guarantees) remain in `specs/`.

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
from weakincentives.contrib.tools import Plan, AddStep

bus = InProcessEventBus()
session = Session(bus=bus)

# Query state via indexing
session[Plan].latest()                            # Most recent value
session[Plan].all()                               # All values in slice
session[Plan].where(lambda p: p.active)           # Filter by predicate

# Dispatch events (through reducers)
session.broadcast(AddStep(...))                   # Broadcast to all reducers

# Direct mutations via indexing (bypass reducers)
session[Plan].seed(initial_plan)                  # Initialize slice
session[Plan].register(AddStep, reducer)          # Register reducer
session[Plan].clear()                             # Clear slice

# Global mutations
session.reset()                                   # Clear all slices
session.rollback(snapshot)                        # Restore from snapshot
```

### Provider Adapter

```python
from weakincentives.adapters.openai import OpenAIAdapter

adapter = OpenAIAdapter(model="gpt-4o")
response = adapter.evaluate(prompt, bus=bus, session=session)
output = response.output  # Typed result
```

## Stability Notice

This is **alpha software**. All APIs may change without backward
compatibility. Do not add backward-compatibility shims or deprecation
warnings—delete unused code completely.

## Quick Checklist

Before submitting changes:

- [ ] `make check` passes
- [ ] New public APIs have DbC decorators where appropriate
- [ ] Tests cover new code paths (100% coverage)
- [ ] Relevant spec documents consulted and updated if needed
- [ ] `CHANGELOG.md` updated for user-visible changes
