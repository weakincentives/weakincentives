# Sessions

*Canonical spec: [specs/SESSIONS.md](../specs/SESSIONS.md)*

A `Session` is WINK's answer to "agent memory", with a constraint:

> **Memory must be deterministic and inspectable.**

Instead of "a magic dict" you mutate, sessions store typed slices managed by
pure reducers. Every mutation flows through a reducer, and every change is
recorded as an event.

## Session as Deterministic Memory

A session is a container keyed by dataclass type:

- Each type has a **slice**: `tuple[T, ...]`
- **Reducers** update slices in response to events
- The session subscribes to the event dispatcher and records telemetry

Mental model: **"events in, new immutable slices out"**.

The session never mutates in place. Reducers return new tuples. This makes
snapshots trivial (just serialize the current tuples) and restoration
straightforward.

## Queries

Use the slice accessor to read state:

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Fact:
    key: str
    value: str

facts: tuple[Fact, ...] = session[Fact].all()
latest_fact: Fact | None = session[Fact].latest()
selected: tuple[Fact, ...] = session[Fact].where(lambda f: f.key == "repo_root")
```

The slice accessor `session[T]` gives you a `QueryBuilder` with fluent methods.
Queries are read-only; they never mutate the session.

## What Is a Reducer?

A **reducer** is a pure function that takes the current state and an event, and
returns the new state. The name comes from functional programming (and was
popularized by Redux in frontend development), but the concept is simple:

```
new_state = reducer(current_state, event)
```

Reducers never mutate state directly. They always return a new value. This makes
state changes predictable: given the same inputs, you always get the same
output. It also makes debugging easier—you can log every event and trace the
exact sequence that led to any state.

In WINK, reducers receive a `SliceView[S]` (read-only access to current values)
and return a `SliceOp[S]` (describing the mutation to apply):

```python
from weakincentives.runtime.session import SliceView, Append

def my_reducer(state: SliceView[Plan], event: AddStep) -> Append[Plan]:
    return Append(Plan(steps=(event.step,)))
```

**SliceOp variants:**

- `Append[T]`: Add a single value to the slice
- `Extend[T]`: Add multiple values to the slice
- `Replace[T]`: Replace all values in the slice
- `Clear`: Remove all values from the slice

WINK ships helper reducers:

- `append_all`: append the event to the slice
- `replace_latest`: replace the most recent value
- `replace_latest_by`: replace by key
- `upsert_by`: insert or update by key

**Example**: keep only the latest plan:

```python
from dataclasses import dataclass
from weakincentives.runtime import replace_latest

@dataclass(slots=True, frozen=True)
class Plan:
    steps: tuple[str, ...]

session[Plan].register(Plan, replace_latest)
session.dispatch(Plan(steps=("step 1",)))
session.dispatch(Plan(steps=("step 2",)))
assert session[Plan].all() == (Plan(steps=("step 2",)),)
```

## Declarative Reducers with @reducer

For complex slices, attach reducers as methods using `@reducer`. Methods must
return `Replace[T]` wrapping the new value:

```python
from dataclasses import dataclass, replace
from weakincentives.runtime.session import reducer, Replace

@dataclass(slots=True, frozen=True)
class AddStep:
    step: str

@dataclass(slots=True, frozen=True)
class AgentPlan:
    steps: tuple[str, ...]

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> Replace["AgentPlan"]:
        return Replace((replace(self, steps=(*self.steps, event.step)),))

session.install(AgentPlan, initial=lambda: AgentPlan(steps=()))
session.dispatch(AddStep(step="read README"))
session.dispatch(AddStep(step="run tests"))
```

This pattern keeps reducer logic close to the data it operates on. The
`@reducer` decorator is just metadata; the actual reducer registration happens
in `session.install()`.

## Snapshots and Restore

Sessions can be snapshotted and restored:

```python
snapshot = session.snapshot()
# ... do work ...
session.restore(snapshot)
```

**Typical use cases:**

- Store a JSONL flight recorder for debugging
- Implement "rollback" on risky operations
- Attach snapshots to bug reports for reproduction

Snapshots serialize to JSON. You can persist them to disk and reload them later.
This is how the debug UI works: it reads snapshot files and displays the session
state at each point.

## SlicePolicy: State vs Logs

Not all slices should roll back the same way.

WINK distinguishes between:

- `SlicePolicy.STATE`: working state that should be restored on rollback
- `SlicePolicy.LOG`: append-only history that should be preserved

By default, `session.snapshot()` captures only `STATE` slices.

If you want everything (including logs), use:

```python
snapshot = session.snapshot(include_all=True)
```

This distinction matters for debugging. You often want to preserve the full
event log even when rolling back working state.

## Dispatching Events

All mutations go through dispatch:

```python
from weakincentives.runtime.session import InitializeSlice, ClearSlice

# Dispatch to reducers
session.dispatch(AddStep(step="do something"))

# Convenience methods (dispatch events internally)
session[Plan].seed(initial_plan)   # → InitializeSlice
session[Plan].clear()              # → ClearSlice

# Direct system event dispatch (equivalent to methods above)
session.dispatch(InitializeSlice(Plan, (initial_plan,)))
session.dispatch(ClearSlice(Plan))

# Global mutations
session.reset()                    # Clear all slices
session.restore(snapshot)          # Restore from snapshot
```

The unified dispatch mechanism ensures all state changes are auditable. You can
subscribe to the dispatcher and log every event that flows through the system.

## Next Steps

- [Tools](tools.md): Learn about transactional tool execution
- [Orchestration](orchestration.md): Use MainLoop for request handling
- [Debugging](debugging.md): Inspect sessions and dump snapshots
