# CLAUDE.md

Quick-reference for AI assistants working in the `weakincentives` repository.

______________________________________________________________________

## Core Philosophy

**The prompt is the agent.** Prompts are hierarchical documents where sections
bundle instructions and tools together. No separate tool registry; capabilities
live in the prompt definition.

**Event-driven state.** All mutations flow through pure reducers processing
typed events. State is immutable and inspectable via snapshots.

**Provider-agnostic.** Same agent definition works across OpenAI, LiteLLM, and
Claude Agent SDK via adapter abstraction.

______________________________________________________________________

## Guiding Principles

### Definition vs Harness

WINK separates what you own from what the runtime provides:

**Agent Definition (you own):** Prompt, Tools, Policies, Feedback

**Execution Harness (runtime-owned):** Planning loop, sandboxing, retries,
throttling, crash recovery, deadlines, budgets

The harness keeps changing; your agent definition should not. WINK makes the
definition a first-class artifact you can version, review, test, and port.

### Policies Over Workflows

**Prefer declarative policies over prescriptive workflows.**

A workflow encodes _how_ to accomplish a goal—a predetermined sequence that
fractures when encountering unexpected situations. A policy encodes _what_ the
goal requires—constraints the agent must satisfy while remaining free to find
any valid path.

| Aspect | Workflow | Policy |
|--------|----------|--------|
| Specifies | Steps to execute | Constraints to satisfy |
| On unexpected | Fails or branches | Agent reasons |
| Composability | Sequential coupling | Independent conjunction |
| Agent role | Executor | Reasoner |

**Key policy characteristics:** Declarative, Composable, Fail-closed, Observable

### Transactional Tools

Tool calls are atomic transactions. When a tool fails:

1. Session state rolls back to pre-call state
1. Filesystem changes revert
1. Error result returned to LLM with guidance

Failed tools don't leave partial state. This enables aggressive retry and
recovery strategies.

______________________________________________________________________

## MANDATORY: Definition of Done

**No work is considered complete until `make check` passes with zero errors.**

This is non-negotiable. Do not claim a task is complete, do not move on to the
next task, and do not commit until:

```bash
make check  # Must exit 0 with no errors
```

If `make check` fails: fix errors, re-run, repeat until all checks pass.

______________________________________________________________________

## Commands

```bash
uv sync && ./install-hooks.sh   # Setup

make format      # Ruff format (88-char lines)
make lint        # Ruff lint --preview
make typecheck   # ty + pyright (strict)
make test        # Pytest, 100% coverage required
make check       # ALL checks - MANDATORY before any commit
```

### Efficient Testing Workflow

`make check` and `make test` automatically detect local vs CI execution:

- **In CI:** Full test suite with 100% coverage enforcement
- **Locally:** Only tests affected by changes (uses testmon coverage database)

The first local run builds a coverage database (`.testmondata`). Subsequent
runs use this database to identify which tests cover changed code and skip the rest.
This dramatically reduces iteration time when working on focused changes.

## Architecture

```
src/weakincentives/
├── adapters/     # OpenAI, LiteLLM, Claude Agent SDK
├── contrib/      # Tools (VFS, Podman, asteval), mailbox
├── dbc/          # Design-by-contract decorators
├── evals/        # Evaluation framework
├── prompt/       # Section/Prompt composition
├── resources/    # Dependency injection
├── runtime/      # Session, events, lifecycle
├── serde/        # Dataclass serialization
└── ...
```

## Style Patterns

### Types & Dataclasses

- Strict pyright; annotations are source of truth—no redundant runtime guards
- Use `@dataclass(slots=True, frozen=True)` or `@FrozenDataclass()`
- Use `assert_never()` with `# pragma: no cover` for union exhaustiveness
- Use `TYPE_CHECKING` blocks to avoid circular imports

### Design-by-Contract

- Public APIs: `@require`, `@ensure`, `@invariant`, `@pure` from `weakincentives.dbc`
- Preconditions validate input; postconditions validate `result`
- Messages: return `(bool, message)` tuple for custom diagnostics

### Prompts & Sections

- `PromptTemplate[OutputType]` with `ns`, `key`, `sections`
- Section keys: `^[a-z0-9][a-z0-9._-]{0,63}$`
- Tools declared on sections in `tools=(...)` tuple
- Resources accessed via `with prompt.resources:` context

### Sessions & Reducers

- Pure reducers return `SliceOp[T]` (Append, Replace, Clear)—never mutate
- All mutations via `session.dispatch(event)` by concrete dataclass type
- Use `@reducer(on=EventType)` decorator on frozen dataclass methods
- Access: `session[T].latest()`, `.all()`, `.where(predicate)`

### Tools

- Signature: `def handler(params: P, *, context: ToolContext) -> ToolResult[R]:`
- Use `ToolResult.ok(value, message="...")` or `ToolResult.error("message")`
- Tool names: `^[a-z0-9_-]{1,64}$`; descriptions 1-200 chars
- Failed tools return errors (never abort); rollback is automatic

### Resources

- Scope: `SINGLETON`, `TOOL_CALL`, `PROTOTYPE` per `Binding`
- Factory: `Binding(protocol, lambda resolver: Value(resolver.get(Dep)))`
- Lifecycle: implement `Closeable`, `PostConstruct`, `Snapshotable` as needed

### Serialization

- Use `serde.parse(cls, data)` and `serde.dump(obj)`—no Pydantic
- Constraints via `Annotated[type, {"ge": 0, "pattern": "..."}]`
- `__type__` field for polymorphic union deserialization

### Time

- Depend on narrow protocols: `WallClock`, `MonotonicClock`, `Sleeper`
- Inject `clock` parameter (default `SYSTEM_CLOCK`); use `FakeClock` in tests
- Deadlines use `datetime(..., tzinfo=UTC)`

### Module Layers

- Foundation → Core → Adapters → High-level; no reverse imports
- Private `_foo.py` modules never imported outside their package
- Use protocols or `TYPE_CHECKING` to break circular dependencies

### Avoid

- Mutable defaults (`[]`, `{}`)
- Global state—inject dependencies explicitly
- Monkeypatching—use FakeClock/FakeFS instead
- Cross-layer imports outside `TYPE_CHECKING`
- Redundant type narrowing after type guards

## Testing

- 100% coverage required for `src/weakincentives/`
- Run focused: `uv run pytest tests/path/to/test.py -v`
- Always finish with `make check`

## Documentation

- **Specs**: `specs/` - design specs (PROMPTS, SESSIONS, TOOLS, ADAPTERS, etc.)
- **Guides**: `guides/` - how-to material; see `guides/README.md`
- **Key files**: `README.md`, `llms.md` (API reference), `CHANGELOG.md`
- **CLI docs**: `wink docs --reference` (API), `--specs` (design), `--guide`

### Key Specs

Read before modifying related code:

| Spec | Topic |
|------|-------|
| `PROMPTS.md` | Prompt system, sections, composition |
| `SESSIONS.md` | Session lifecycle, events, budgets |
| `TOOLS.md` | Tool registration, planning tools |
| `GUARDRAILS.md` | Tool policies, feedback providers, task completion |
| `ADAPTERS.md` | Provider adapters, throttling |
| `CLAUDE_AGENT_SDK.md` | SDK adapter, isolation, MCP |
| `OPENCODE_ACP_ADAPTER.md` | OpenCode ACP adapter, workspace |
| `AGENT_LOOP.md` | AgentLoop orchestration |
| `POLICIES_OVER_WORKFLOWS.md` | Design philosophy |
| `MODULE_BOUNDARIES.md` | Layer architecture |

## Stability

Alpha software. APIs may change. Delete unused code completely; no
backward-compatibility shims.

______________________________________________________________________

## Final Checklist

**Before ANY commit or claiming work is done:**

- [ ] `make check` passes with zero errors (MANDATORY)
- [ ] Tests cover new code paths (100% coverage)
- [ ] Relevant specs consulted/updated
- [ ] `CHANGELOG.md` updated for user-visible changes

**If `make check` fails, the work is not done.**
