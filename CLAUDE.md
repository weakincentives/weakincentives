# CLAUDE.md

Quick-reference for AI assistants working in the `weakincentives` repository.

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

- **Specs**: `specs/` - design specs; see `AGENTS.md` for full index
- **Guides**: `guides/` - how-to material; see `guides/README.md`
- **Key files**: `README.md`, `AGENTS.md`, `llms.md`, `CHANGELOG.md`

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
