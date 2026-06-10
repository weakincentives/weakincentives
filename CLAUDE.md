# CLAUDE.md

Quick-reference for AI assistants working in the `weakincentives` repository.

______________________________________________________________________

## Core Philosophy

**The prompt is the agent.** Prompts are hierarchical documents where sections
bundle instructions and tools together. No separate tool registry; capabilities
live in the prompt definition.

**Event-driven state.** All mutations flow through pure reducers processing
typed events. State is immutable and inspectable via snapshots.

**Provider-agnostic.** Same agent definition works across agentic harnesses
(Claude Agent SDK, Codex App Server, ACP) via adapter abstraction.

______________________________________________________________________

## Guiding Principles

Full rationale lives in `llms.md` (Guiding Principles) and
`specs/POLICIES_OVER_WORKFLOWS.md`. The short version:

- **Definition vs harness.** You own the agent definition (prompt, tools,
  policies, feedback); the runtime owns the harness (planning loop,
  sandboxing, retries, budgets). Keep definitions portable across runtimes.
- **Policies over workflows.** Encode constraints to satisfy, not steps to
  execute. Policies are declarative, composable, fail-closed, observable.
- **Transactional tools.** A failed tool call rolls back session and
  filesystem state and returns an error result to the LLM; partial state
  never leaks.

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
uv sync && ./install-hooks.sh   # Setup - BOTH STEPS ARE MANDATORY

make format      # Ruff format (88-char lines)
make lint        # Ruff lint --preview
make typecheck   # ty + pyright (strict)
make test        # Pytest, 100% coverage required
make check       # ALL checks - MANDATORY before any commit
```

______________________________________________________________________

## MANDATORY: Git Hooks Installation

**Git hooks MUST be installed in every new development environment.**

```bash
./install-hooks.sh   # Run this after cloning or in any new environment
```

The pre-commit hook runs `CI=true make check`: the full test suite with 100%
coverage enforcement, exactly what CI runs on pull requests. A plain local
`make check` uses testmon to run only tests affected by your changes (the
first run builds the `.testmondata` database), so code can pass locally yet
fail CI — the hook closes that gap.

______________________________________________________________________

## Repository Map

Where each area lives. Source paths are relative to `src/weakincentives/`,
test paths to `tests/`, specs to `specs/`. Read the spec before modifying
the code it covers.

| Area | Source | Tests | Spec |
|------|--------|-------|------|
| Prompts, sections, rendering | `prompt/` (`prompt.py`, `section.py`, `markdown.py`) | `prompts/`, `prompt/` | `PROMPTS.md` |
| Tools | `prompt/tool.py`, `prompt/tool_result.py` | `prompt/` | `TOOLS.md` |
| Policies, feedback, task completion | `prompt/policy.py`, `prompt/feedback.py`, `prompt/task_completion.py` | `prompt/` | `GUARDRAILS.md` |
| Task examples | `prompt/task_examples.py` | `prompts/` | `EXAMPLES.md` |
| Workspace sections, digest tools | `prompt/workspace.py`, `contrib/tools/` | `prompt/`, `tools/` | `WORKSPACE.md` |
| Sessions, events, budgets | `runtime/session/`, `runtime/events/`, `budget.py`, `deadlines.py` | `runtime/`, `test_budget.py`, `test_deadlines.py` | `SESSIONS.md` |
| Slice storage | `runtime/session/slices/` | `runtime/test_slices.py`, `runtime/test_state_slice.py` | `SLICES.md` |
| Agent loop | `runtime/agent_loop.py` | `runtime/agent_loop/` | `AGENT_LOOP.md` |
| Mailbox, DLQ | `runtime/mailbox/`, `runtime/dlq.py`, `contrib/mailbox/` | `runtime/`, `contrib/` | `MAILBOX.md`, `DLQ.md` |
| Lifecycle, health, watchdog | `runtime/lifecycle.py`, `runtime/watchdog.py` | `runtime/` | `LIFECYCLE.md`, `HEALTH.md` |
| Lease extender | `runtime/lease_extender.py` | `runtime/` | `LEASE_EXTENDER.md` |
| Transcript | `runtime/transcript.py` | `runtime/` | `TRANSCRIPT.md` |
| Run context | `runtime/run_context.py` | `runtime/` | `RUN_CONTEXT.md` |
| Logging | `runtime/logging.py` | `runtime/` | `LOGGING.md` |
| Adapter core, throttling | `adapters/` (`core.py`, `config.py`, `_shared/`) | `adapters/` | `ADAPTERS.md` |
| Claude Agent SDK adapter | `adapters/claude_agent_sdk/` | `adapters/claude_agent_sdk/` | `CLAUDE_AGENT_SDK.md` |
| Codex App Server adapter | `adapters/codex_app_server/` | `adapters/codex_app_server/` | `CODEX_APP_SERVER.md` |
| ACP adapters (generic, Gemini, OpenCode) | `adapters/acp/`, `adapters/gemini_acp/`, `adapters/opencode_acp/` | `adapters/acp/`, `adapters/gemini_acp/`, `adapters/opencode_acp/` | `ACP_ADAPTER.md`, `GEMINI_ACP_ADAPTER.md`, `OPENCODE_ACP_ADAPTER.md` |
| Resources (DI) | `resources/` | `resources/` | `RESOURCE_REGISTRY.md` |
| Filesystem | `filesystem/` | `filesystem/` | `FILESYSTEM.md` |
| Serde, dataclasses | `serde/`, `dataclasses/`, `types/` | `serde/`, `test_dataclass_serialization.py` | `DATACLASSES.md` |
| Design-by-contract | `dbc/` | `test_dbc_contracts.py` | `DBC.md` |
| Clock | `clock.py` | `test_clock.py` | `CLOCK.md` |
| Evals, experiments | `evals/`, `experiment.py` | `evals/` | `EVALS.md`, `EXPERIMENTS.md` |
| Debug bundles | `debug/` | `debug/` | `DEBUG_BUNDLE.md` |
| CLI (`wink`) | `cli/` | `cli/` | `WINK_DOCS.md`, `WINK_QUERY.md`, `WINK_DEBUG.md` |
| Skills | `skills/` | `skills/` | `SKILLS.md` |
| Formal verification (TLA+) | `formal/` | `formal/` | `FORMAL_VERIFICATION.md`, `VERIFICATION.md` |

Process/design specs without a single code home: `POLICIES_OVER_WORKFLOWS.md`
(philosophy), `MODULE_BOUNDARIES.md` (layering), `TESTING.md`,
`THREAD_SAFETY.md`, `VERIFICATION_TOOLBOX.md` (`check.py`, `toolchain/`),
`ACK.md` (`integration-tests/ack/`), `ANALYSIS_LOOP.md` (not yet implemented).

### Layout Conventions

- `tests/` mirrors `src/weakincentives/` package-for-package, with
  exceptions: small foundation modules (`dbc`, `clock`, `budget`,
  `deadlines`, `dataclasses`) are covered by top-level `tests/test_*.py`
  files; prompt tests are split across `tests/prompt/` and `tests/prompts/`;
  `tests/helpers/` and `tests/plugins/` are shared fixtures and pytest
  plugins, not mirrors.
- Every top-level `src/weakincentives/` package is covered by at least one
  spec — find it in the table above.

______________________________________________________________________

## Change Playbooks

`guides/contributing-playbooks.md` has step-by-step recipes — files to
touch, canonical example to copy, test pattern to follow — for the common
change shapes: add a tool, an event + reducer, a prompt section type, a
provider adapter, or a resource binding. Start there before exploring.

## Style Patterns

### Types & Dataclasses

- Strict pyright; annotations are source of truth—no redundant runtime guards
- Use `@dataclass(slots=True, frozen=True)` or `@FrozenDataclass()`
- Use `assert_never()` with `# pragma: no cover` for union exhaustiveness
- Use `TYPE_CHECKING` blocks to avoid circular imports

### Design-by-Contract

- Public APIs: `@require`, `@ensure`, `@invariant` from `weakincentives.dbc`
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
- **10-second timeout enforced per test** (`--timeout=10 --timeout-method=thread`)
- Every unit test MUST complete in under 10 seconds. If a test exceeds this,
  refactor it — do not raise the timeout. Use mocks, fakes, or smaller inputs.
- Use `@pytest.mark.timeout(N)` only for integration tests that genuinely need
  more time. Never on unit tests.

## Documentation

- **Specs**: `specs/` - design contracts; mapped to code in the
  [Repository Map](#repository-map) above
- **Guides**: `guides/` - how-to material; see `guides/README.md`
- **Key files**: `README.md`, `llms.md` (agent-oriented API reference with a
  table of contents — read sections, not the whole file), `CHANGELOG.md`
- **CLI docs**: `wink docs --reference` (API), `--specs` (design), `--guide`

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
