# CLAUDE.md

Quick-reference guide for Claude and other AI assistants working in the
`weakincentives` repository.

## Project Overview

WINK (Weak Incentives) is a Python library for building deterministic,
side-effect-free background agents. It provides:

- **Redux-style sessions** with immutable event ledgers and pure reducers
- **Typed prompt composition** using dataclass-backed sections
- **Provider-agnostic adapters** (OpenAI, LiteLLM, Claude Agent SDK)
- **Design-by-contract enforcement** via decorators
- **Sandboxed tooling** (VFS, asteval, Podman)
- **Evaluation framework** with session-aware evaluators
- **Formal verification support** via TLA+ spec embedding

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
│   ├── optimizers/  # Workspace digest optimizer
│   └── mailbox/     # Redis mailbox implementation
├── dataclasses/     # FrozenDataclass utilities
├── dbc/             # Design-by-contract decorators
├── evals/           # Evaluation framework (datasets, evaluators, EvalLoop)
├── filesystem/      # Filesystem protocol and types (core abstraction)
├── formal/          # TLA+ formal specification support
├── optimizers/      # Optimizer framework and protocols
├── skills/          # Skill types, validation (Agent Skills spec support)
├── prompt/          # Section/Prompt composition, overrides, tools
├── resources/       # Dependency injection with scoped lifecycles
├── runtime/         # Session, events, lifecycle, mailbox, transactions
├── serde/           # Dataclass serialization (no Pydantic)
└── types/           # JSON type aliases
```

The library is organized as "core primitives" + "batteries for specific agent
styles":

- **Core** (`weakincentives.*`): Prompt composition, sessions, adapters, serde,
  dbc, filesystem protocols, skills, resource registry
- **Contrib** (`weakincentives.contrib.*`): Planning tools, VFS, Podman,
  asteval, workspace optimizers, Redis mailbox

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
    # Success case
    return ToolResult.ok(ResultType(...), message="done")
    # Or failure case
    return ToolResult.error("Something went wrong")
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

Run focused tests during development:

```bash
uv run pytest tests/path/to/test.py -v
```

Always finish with `make test` to verify coverage.

## Spec Documents

Consult these specs before modifying related code:

| Spec | When to Read |
| ------------------------------ | --------------------------------------------------------------------- |
| `specs/ADAPTERS.md` | Provider adapters, structured output, throttling |
| `specs/CLAUDE_AGENT_SDK.md` | Claude Agent SDK adapter, MCP tool bridging, skill mounting |
| `specs/DATACLASSES.md` | Serde utilities or frozen dataclass patterns |
| `specs/DBC.md` | Editing DbC-decorated modules (required) |
| `specs/EVALS.md` | Evaluation framework, datasets, evaluators, session evaluators |
| `specs/EXAMPLES.md` | Code review agent reference implementation |
| `specs/EXHAUSTIVENESS.md` | Union type totality, `assert_never` patterns, match statement coverage |
| `specs/FILESYSTEM.md` | Filesystem protocol, backend implementations, ToolContext integration |
| `specs/FORMAL_VERIFICATION.md` | Embedding TLA+ in Python, `@formal_spec` decorator, TLC verification |
| `specs/HEALTH.md` | Health endpoints, watchdog, stuck worker detection, process termination |
| `specs/LIFECYCLE.md` | LoopGroup, ShutdownCoordinator, graceful shutdown patterns |
| `specs/LEASE_EXTENDER.md` | Automatic message visibility extension during processing |
| `specs/LOGGING.md` | Logging surfaces |
| `specs/MAILBOX.md` | Message queue abstraction, SQS/Redis semantics, MainLoop integration |
| `specs/MAILBOX_RESOLVER.md` | Mailbox routing, reply-to patterns, resolver configuration |
| `specs/MAIN_LOOP.md` | Main loop orchestration, visibility handling, event-driven execution |
| `specs/POLICIES_OVER_WORKFLOWS.md` | Philosophy of declarative policies vs rigid workflows for unattended agents |
| `specs/PROMPTS.md` | Prompt system, composition, structured output, resource lifecycle |
| `specs/PROMPT_OPTIMIZATION.md` | Override system or optimizer logic |
| `specs/RESOURCE_REGISTRY.md` | Dependency injection, resource scopes, transactional snapshots |
| `specs/SESSIONS.md` | Session lifecycle, events, deadlines, budgets |
| `specs/SKILLS.md` | Agent Skills specification and WINK skill mounting |
| `specs/SLICES.md` | Slice storage backends, factory configuration, JSONL persistence |
| `specs/TASK_COMPLETION.md` | Task completion checking, PlanBasedChecker, composite verification |
| `specs/TESTING.md` | Test harnesses, fault injection, fuzzing, coverage standards |
| `specs/THREAD_SAFETY.md` | Concurrency or shared state |
| `specs/TOOLS.md` | Adding/modifying tools, planning tools |
| `specs/TOOL_POLICIES.md` | Sequential dependencies, read-before-write, keyed constraints |
| `specs/TRAJECTORY_OBSERVERS.md` | Ongoing progress assessment, stall/drift detection, feedback injection |
| `specs/VERIFICATION.md` | Redis mailbox detailed specification, invariants, property tests |
| `specs/WINK_DEBUG.md` | Debug web UI, snapshot explorer, session inspection |
| `specs/WINK_DOCS.md` | CLI docs command, bundled documentation access |
| `specs/WORKSPACE.md` | VFS, Podman, asteval, workspace digest |

Full spec index in `AGENTS.md`.

## Guides

User-facing how-to material lives in `guides/`:

| Guide | Description |
| ----------------------------- | --------------------------------------------------- |
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
from weakincentives.prompt import PromptTemplate

template = PromptTemplate[OutputType](
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
prompt = Prompt(template)  # Bind params with prompt.bind(MyParams(...))
```

### Session State

```python
from weakincentives.runtime import Session, InProcessDispatcher
from weakincentives.runtime.session import InitializeSlice, ClearSlice
from weakincentives.contrib.tools import Plan, AddStep

bus = InProcessDispatcher()
session = Session(bus=bus)

# Query state via indexing
session[Plan].latest()                            # Most recent value
session[Plan].all()                               # All values in slice
session[Plan].where(lambda p: p.active)           # Filter by predicate

# All mutations go through dispatch (unified, auditable)
session.dispatch(AddStep(...))                    # Dispatch to reducers

# Convenience methods (dispatch events internally)
session[Plan].seed(initial_plan)                  # → InitializeSlice
session[Plan].clear()                             # → ClearSlice
session[Plan].register(AddStep, reducer)          # Register reducer

# Direct system event dispatch (equivalent to methods above)
session.dispatch(InitializeSlice(Plan, (initial_plan,)))
session.dispatch(ClearSlice(Plan))

# Global mutations
session.reset()                                   # Clear all slices
session.restore(snapshot)                         # Restore from snapshot
```

### Provider Adapter

```python
from weakincentives.adapters.openai import OpenAIAdapter

adapter = OpenAIAdapter(model="gpt-4o")
response = adapter.evaluate(prompt, session=session)
output = response.output  # Typed result
```

### Prompt Resource Lifecycle

Access resources through the prompt's resource context:

```python
# Resources bound to prompt require context manager
with prompt.resources:
    fs = prompt.resources.get(Filesystem)
    # Resources available within context
# Resources cleaned up automatically
```

### Reducers with SliceOp

Reducers receive `SliceView[S]` and return `SliceOp[S]` operations:

```python
from dataclasses import dataclass, replace
from weakincentives.runtime.session import SliceView, Append, Replace, reducer

# Append a single item (Append takes one item)
def add_step_reducer(state: SliceView[Plan], event: AddStep) -> Append[Plan]:
    return Append(Plan(steps=(event.step,)))

# Replace entire slice (Replace takes a tuple of items)
def reset_plan_reducer(state: SliceView[Plan], event: ResetPlan) -> Replace[Plan]:
    return Replace((Plan(steps=()),))

# Declarative reducer on dataclass
@dataclass(frozen=True)
class AgentPlan:
    steps: tuple[str, ...] = ()

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> Replace["AgentPlan"]:
        return Replace((replace(self, steps=(*self.steps, event.step)),))
```

### Resource Registry

Dependency injection with scoped lifecycles:

```python
from weakincentives.resources import Binding, ResourceRegistry, Scope

registry = ResourceRegistry.of(
    Binding(Config, lambda r: Config.from_env()),
    Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
    Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
)

with registry.open() as ctx:
    http = ctx.get(HTTPClient)  # Lazily constructed singleton
    with ctx.tool_scope() as resolver:
        tracer = resolver.get(Tracer)  # Fresh per tool call
```

### LoopGroup for Production

Coordinate multiple loops with graceful shutdown:

```python
from weakincentives.runtime import LoopGroup

group = LoopGroup(
    loops=[main_loop, eval_loop],
    health_port=8080,        # Kubernetes probes at /health/live, /health/ready
    watchdog_threshold=720.0,  # Terminate stuck workers
)
group.run()  # Blocks until SIGTERM/SIGINT
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
