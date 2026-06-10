# Contributing Playbooks

Step-by-step recipes for the most common change shapes in this repository.
Each playbook names the canonical example to copy from, the files a change
touches, and the test pattern to follow. They are aimed at contributors
(human or AI) modifying `weakincentives` itself, not at library users.

Every playbook ends the same way:

1. Tests cover the new code paths (100% coverage is enforced).
1. The relevant spec in `specs/` is updated if you changed a contract.
1. `CHANGELOG.md` gains a bullet under `## Unreleased` for user-visible
   changes.
1. `make check` passes with zero errors.

(Adding a doc instead? New specs and guides must be registered in
`src/weakincentives/cli/docs_metadata.py` — a test enforces this — and new
guides also belong in the `guides/README.md` index.)

______________________________________________________________________

## Add a Tool

Spec: `specs/TOOLS.md`. Canonical example: the `open_sections` and
`read_section` tools in `src/weakincentives/prompt/progressive_disclosure.py`.
Test patterns: `tests/prompt/test_tool.py`.

1. Define frozen param and result dataclasses. Document fields with
   `field(metadata={"description": ...})`; constraints go in `Annotated`
   metadata (see `specs/DATACLASSES.md`).
1. Write the handler:
   `def handler(params: P, *, context: ToolContext) -> ToolResult[R]:`.
   The context provides `context.session`, `context.resources`, and
   `context.filesystem`. Return `ToolResult.ok(value, message="...")` or
   `ToolResult.error("message")` — never raise for expected failures;
   rollback of session and filesystem state is automatic.
1. Create the tool with `Tool[P, R].create(name=..., description=..., handler=..., examples=(...))`. Names must match `^[a-z0-9_-]{1,64}$`;
   descriptions are 1–200 ASCII chars. Add `ToolExample` entries — they
   render into the prompt.
1. Attach it to a section via `tools=(...)` — there is no separate tool
   registry; the section is the registration point.
1. Test by invoking the handler directly with a constructed `ToolContext`
   (see `tests/prompt/test_tool.py`), asserting on the `ToolResult` and on
   any session slices the tool dispatched to.

______________________________________________________________________

## Add an Event + Reducer (Session Slice)

Specs: `specs/SESSIONS.md` (lifecycle) and `specs/SLICES.md` (slice ops).
Canonical example: the small slices in `tests/runtime/test_state_slice.py`.

1. Define a frozen event dataclass (plain `@dataclass(frozen=True)`).
1. Define a frozen state-slice dataclass and put reducer methods on it:
   `@reducer(on=MyEvent) def apply(self, event: MyEvent) -> SliceOp[Self]`.
   Reducers are pure — return `Replace`, `Append`, `Extend`, or `Clear`
   (from `weakincentives.runtime.session`); never mutate.
1. Register the slice with `session.install(MySlice)` (optionally with an
   `initial=` factory). Installation auto-registers all `@reducer` methods.
1. Mutate only via `session.dispatch(MyEvent(...))`; query via
   `session[MySlice].latest()`, `.all()`, or `.where(predicate)`.
1. Export new public types from
   `src/weakincentives/runtime/session/__init__.py` if they are part of the
   API.
1. Test with the `session_factory` fixture: install, `seed(...)` an initial
   value, `dispatch(...)`, then assert on `session[MySlice].latest()`.

______________________________________________________________________

## Add a Prompt Section Type

Spec: `specs/PROMPTS.md`. Canonical examples:
`src/weakincentives/prompt/markdown.py` (`MarkdownSection`, the simple
case) and `src/weakincentives/prompt/workspace.py` (`WorkspaceSection`,
which contributes resources and needs cleanup). Tests:
`tests/prompts/test_text_section.py` and `tests/prompt/test_workspace.py`.

1. Subclass `Section` (from `src/weakincentives/prompt/section.py`) and
   call `super().__init__(title=..., key=..., tools=(...))`. Keys must
   match `^[a-z0-9][a-z0-9._-]{0,63}$`.
1. Implement `render_body(...)` returning the section's markdown body
   (the heading is rendered for you).
1. Implement `clone(**kwargs)` so the section can be reused across prompts.
1. Override `resources()` to contribute bindings
   (`ResourceRegistry.build({Protocol: instance})`) and `cleanup()` —
   idempotent — if the section owns external state. `WorkspaceSection`
   shows both.
1. Export the class from `src/weakincentives/prompt/__init__.py`
   (`__all__`).
1. Test rendering output directly (`section.render(...)`), and test
   `cleanup()` in a `try/finally` if you implemented it.

______________________________________________________________________

## Add a Provider Adapter

Specs: `specs/ADAPTERS.md` plus a new per-adapter spec (follow
`specs/GEMINI_ACP_ADAPTER.md`). Canonical template:
`src/weakincentives/adapters/gemini_acp/` — the smallest adapter
(~120-line `adapter.py` subclassing the generic `ACPAdapter`). Tests:
`tests/adapters/gemini_acp/test_adapter.py`.

1. Create `src/weakincentives/adapters/<name>/` with `config.py`
   (`@FrozenDataclass()` config classes extending the patterns in
   `adapters/config.py`), `adapter.py`, and `__init__.py` re-exports.
1. Subclass `ProviderAdapter` from `adapters/core.py` and implement
   `evaluate()` — or subclass `ACPAdapter` for ACP-speaking CLIs and
   override only the quirks. Override the `adapter_name` property.
1. Add a `<NAME>_ADAPTER_NAME` constant to
   `src/weakincentives/types/adapter.py` and its `__all__`.
1. If the adapter needs new third-party packages, add an optional
   dependency group to `pyproject.toml` (see the `claude-agent-sdk`
   extra) — core never grows hard dependencies.
1. Adapters are not auto-registered; callers instantiate them directly,
   so no factory or CLI wiring is required.
1. Mirror the test layout: a `test_adapter.py` in a `tests/adapters/`
   subdirectory named after the adapter, covering adapter name, config
   defaults, and every overridden hook. Shared cross-adapter code belongs
   in `adapters/_shared/`.

______________________________________________________________________

## Add a Resource Binding

Spec: `specs/RESOURCE_REGISTRY.md`. Core types live in
`src/weakincentives/resources/` (`binding.py`, `scope.py`, `registry.py`,
`protocols.py`). Test patterns: `tests/resources/test_resources.py`.

1. Define the protocol (a `typing.Protocol`) the binding satisfies, and
   the implementation class.
1. Pick a scope: `SINGLETON` (default, one per session), `TOOL_CALL`
   (fresh per tool invocation, disposed after), or `PROTOTYPE` (fresh per
   access). Implement lifecycle protocols as needed: `Closeable.close()`,
   `PostConstruct.post_construct()`, `Snapshotable.snapshot()/restore()`
   (required for transactional rollback of stateful resources).
1. Declare the binding:
   `Binding(Protocol, lambda r: Impl(r.get(Dep)), scope=...)` for
   factories (dependencies resolved through `r`), or
   `Binding.instance(Protocol, value)` for pre-built instances.
1. Register it where it belongs: a section's `resources()` override for
   section-owned resources, or `ResourceRegistry.of(...)`/`.build({...})`
   at prompt construction.
1. Consume via `with prompt.resources:` then `prompt.resources.get(Protocol)`
   (tools use `context.resources`).
1. Test scope behavior and lifecycle ordering directly against a registry:
   `with registry.open() as ctx: ctx.get(Protocol)` — see
   `tests/resources/test_resources.py`.
