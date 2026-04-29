# Distilled WINK Specification

## Purpose

`weakincentives.distilled` is a **single-file** distillation of the WINK core,
modeled after Bottle's "all-in-one-file" philosophy. It exists to:

- Make the genuinely novel ideas in WINK easy to read end-to-end
- Provide a sandbox for iterating on the abstractions themselves
- Serve as an executable diagram — testable in isolation, no subpackages

It is **not** a drop-in replacement for the full package. It deliberately drops
provider adapters, dependency-injected resources, debug bundles, evaluations,
formal verification, skills, feedback providers, progressive disclosure,
overrides, design-by-contract, mailbox/dispatcher infrastructure, structured
output parsing, and clock/budget/deadline machinery.

## Scope

Located at `src/weakincentives/distilled.py`. Self-contained: imports only the
Python standard library. The companion test suite is `tests/test_distilled.py`.

| File | Hard limit |
| --- | --- |
| `distilled.py` | 720 lines (project code-length cap) |
| Functions/methods | 120 lines each |

## What's In

The distilled core captures four interlocking ideas:

### 1. Hierarchical sections that bundle tools

`Section` is a frozen dataclass with `title`, `key`, `template`, `children`,
and `tools`. The same primitive serves as both a documentation node and a tool
attachment point — "the prompt is the agent."

`Prompt` wires sections together with a namespace/key pair and optional
per-section parameter dicts. `render()` walks the tree depth-first, producing
markdown with deterministic numbered headings (`## 1.`, `### 1.1.`, …) and a
flat tuple of every reachable tool. Duplicate tool names raise
`PromptValidationError` at render time.

Templates use `string.Template` (`$field` substitution) and are dedented before
rendering. Missing placeholders raise `PromptRenderError`.

### 2. Tools as transactional functions

`Tool` packages a `name` (regex `^[a-z0-9_-]{1,64}$`), a 1-200 char
description, and a handler:

```python
def handler(params: ParamsT, context: ToolContext) -> ToolResult[ResultT]
```

`ToolResult.ok(value, message="")` and `ToolResult.error(message)` are the only
factories. `ToolContext` exposes the active `Session` plus the rendered prompt.

`execute_tool(prompt, session, name, params)` looks up the tool by name, takes
a session snapshot, runs the handler, and on failure (raised exception **or**
`ToolResult.error`) restores the snapshot. This is the entire transactional
contract.

### 3. Event-sourced sessions

`Session` is a `dict[type, tuple[object, ...]]` plus a reducer registry.
Mutations only happen through `dispatch(event)`, which routes to every reducer
registered for `type(event)`. Reducers are pure: they receive the current
slice tuple and the event, and return a `SliceOp` (`Replace` or `Append`).

Query API:

```python
session[Plan].latest()        # most recent value or None
session[Plan].all()           # tuple of all values
session[Plan].where(pred)     # filtered tuple
session[Plan].seed(value)     # initialize-or-replace (single-element)
session[Plan].clear()         # drop the slice
session[Plan].append(value)   # bypass reducers for ad-hoc data
```

### 4. Declarative state slices

The `@reducer(on=Event)` decorator marks methods on a frozen dataclass as
reducers. `session.install(StateClass)` scans for decorated methods and
registers each one against the appropriate event type. Methods receive
`(self, event)` and return a `SliceOp`.

```python
@dataclass(frozen=True)
class Plan:
    steps: tuple[str, ...] = ()

    @reducer(on=AddStep)
    def add(self, event: AddStep) -> Replace["Plan"]:
        return Replace((replace(self, steps=(*self.steps, event.step)),))
```

`session.install(Plan, initial=lambda: Plan())` lets reducers run even when
the slice is empty.

### Snapshot / restore

`session.snapshot()` returns an immutable `Snapshot` capturing all slice
state. `session.restore(snapshot)` reinstalls it. The `tool_transaction`
context manager calls `snapshot()` on entry and `restore()` on any raised
exception, which is what makes `execute_tool` atomic.

## What's Out (and where to look in the real package)

| Dropped | Real location |
| --- | --- |
| Provider adapters (OpenAI, LiteLLM, Claude SDK) | `src/weakincentives/adapters/` |
| Resource registry, scoped DI | `src/weakincentives/resources/` |
| Filesystem sandbox | `src/weakincentives/filesystem/` |
| Skills, feedback, policies, overrides | `src/weakincentives/prompt/` |
| Debug bundles, transcript, logging | `src/weakincentives/debug/`, `runtime/` |
| Evaluation framework | `src/weakincentives/evals/` |
| Formal verification | `src/weakincentives/formal/` |
| Design-by-contract | `src/weakincentives/dbc/` |
| Serde, structured output | `src/weakincentives/serde/`, `prompt/structured_output.py` |
| Budgets, deadlines, clocks | `src/weakincentives/budget.py`, `deadlines.py`, `clock.py` |

## Stability

Experimental. The distilled module is a teaching artifact and a design
playground. It is allowed to drift ahead of (or behind) the full package while
ideas are being explored, and may be deleted entirely if it stops earning its
keep.

## Testing

`tests/test_distilled.py` covers every line and branch of `distilled.py`.
Tests must complete within the project-wide 10-second per-test timeout.

## Limitations

- Single-threaded; no locking
- No deadlines/budgets
- No structured output parsing
- No progressive disclosure or section overrides
- Section parameters are plain `dict[str, str]`, not typed dataclasses
- No provider integration — the loop driving the model is the caller's problem
