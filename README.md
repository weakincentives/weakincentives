# weakincentives

A small, modular spine for building reliable AI agents in Python.

The package is layered. The base install ships only the **spine**
(`weakincentives.core`):

- Hierarchical, typed prompts that bundle tools (`Section[ParamsT]`, `Prompt`).
- Event-sourced sessions with pure reducers and atomic slice access (`Session`,
  `@reducer`).
- Snapshot/restore with JSON persistence and a registry-based polymorphic
  encoding (`Snapshot`, `TypeRegistry`).
- Transactional tool execution that rolls session state and registered
  resources back on any failure (`tool_transaction`, `execute_tool`).
- A narrow set of extension protocols — `ProviderAdapter`, `EventListener`,
  `ResourceProvider`, `Snapshotable`, `ToolPolicy`, `Deadline`, `Budget` — so
  higher-level layers can plug in without forking the core.

Layered extras built on top, all stdlib-only:

**Layer 1 — foundations:**

- `weakincentives.clock` — `SystemClock`, `FakeClock`, concrete `Deadline`,
  `Budget`, `BudgetTracker`.
- `weakincentives.serde` — `parse(cls, data)` / `dump(value)` with
  `Annotated` constraints and polymorphic union encoding.
- `weakincentives.dbc` — `@require`, `@ensure`, `@invariant` runtime
  contracts.
- `weakincentives.filesystem` — `InMemoryFilesystem` implementing
  `Snapshotable` so transactional tools roll filesystem state back too.
- `weakincentives.resources` — scoped DI container that satisfies
  `core.ResourceProvider` (`SINGLETON`, `TOOL_CALL`, `PROTOTYPE`).

**Layer 2 — state & IO:**

- `weakincentives.filesystem` — `Filesystem` protocol + `InMemoryFilesystem`.
- `weakincentives.resources` — see above.

**Layer 3 — orchestration:**

- `weakincentives.transcript` — `Transcript`, `TranscriptListener`
  (implements `core.EventListener`) for unified event logging.
- `weakincentives.runtime` — `AgentLoop` that drives a
  `core.ProviderAdapter` through render → call → tool → repeat with
  optional deadlines, budgets, and tool policies.

**Layer 4 — quality & observability:**

- `weakincentives.evals` — `Dataset`, `Evaluator` protocol, built-in
  evaluators (`ExactMatch`, `Contains`, `ToolCalled`, `AllToolsSucceeded`)
  and a `run_evaluation` driver.
- `weakincentives.debug` — `DebugBundle` (snapshot + transcript +
  metadata) with JSON round-trip via `core.TypeRegistry`.
- `weakincentives.skills` — `Skill` + `SkillMount`; load skills from
  markdown files with TOML frontmatter (no third-party dependencies).
- `weakincentives.formal` — `@formal_spec` decorator that attaches
  metadata for downstream verifiers; ships a registry callers can iterate.

**Layer 5 — provider integrations:**

- `weakincentives.adapters.noop` — `NoopAdapter` + `ScriptedResponse` for
  deterministic tests and demos.
- `weakincentives.adapters.openai_compatible` —
  `OpenAICompatibleAdapter` against a structural `ChatClient` protocol;
  works with `openai`, `mistralai`, `together`, etc., or with hand-rolled
  fakes in tests.

**Layer 6 — CLI:**

- `weakincentives.cli` — the `wink` command. Today: `wink debug <bundle.json>` pretty-prints a debug bundle's metadata, transcript,
  and slice contents.

See `specs/ARCHITECTURE.md` for the full layered design and remaining
work.

## Status

Pre-1.0. The spine API is the published surface
(`weakincentives.core.__all__`), and once tagged v1.0.0 it follows semver.
Extras live in their own subpackages and ship behind pip extras as they land.

## Install

```sh
pip install weakincentives
```

The base install requires only the standard library.

## Quickstart

```python
from dataclasses import dataclass, replace

from weakincentives import (
    MarkdownSection,
    Prompt,
    Replace,
    Session,
    Tool,
    ToolContext,
    ToolResult,
    execute_tool,
    reducer,
)


# A typed dataclass that becomes a session slice.
@dataclass(frozen=True)
class AddNote:
    text: str


@dataclass(frozen=True)
class Notes:
    items: tuple[str, ...] = ()

    @reducer(on=AddNote)
    def add(self, event: AddNote) -> Replace["Notes"]:
        return Replace((replace(self, items=(*self.items, event.text)),))


# A typed parameter block for a section.
@dataclass(frozen=True)
class Greeting:
    name: str


# A tool that mutates session state.
def remember(text: str, context: ToolContext) -> ToolResult[None]:
    context.session.dispatch(AddNote(text=text))
    return ToolResult.ok(None, message=f"noted: {text}")


# Wire it together.
note_tool = Tool(name="note", description="Record a note", handler=remember)

prompt = Prompt(
    ns="demo",
    key="notes",
    sections=(
        MarkdownSection[Greeting](
            title="Greet",
            key="greet",
            template="Hello $name. You can call `note(...)`.",
            params_type=Greeting,
            default_params=Greeting(name="world"),
            tools=(note_tool,),
        ),
    ),
)

session = Session()
session.install(Notes, initial=Notes)

print(prompt.render().text)

result = execute_tool(prompt, session, "note", "remember the milk")
assert result.success
assert session[Notes].latest() == Notes(items=("remember the milk",))
```

## Development

```sh
uv sync
make check    # ruff format + lint, pyright strict, ty, markdown, pytest
make test     # full pytest with 100% coverage requirement
```

The spine is tested under strict pyright, `ty`, ruff, and 100% line/branch
coverage.

## License

Apache-2.0. See `LICENSE`.
