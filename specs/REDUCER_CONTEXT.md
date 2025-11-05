# Reducer Context Specification

## Overview

Reducers execute inside the session orchestration loop. They frequently need access to
shared runtime services such as the active `Session` instance and the event bus used
for telemetry. Today each reducer receives only the event payload, which forces them
either to reach into global state or accept bespoke constructor parameters. This spec
introduces a lightweight `ReducerContext` object that injects the runtime references
reducers need while keeping the contract deterministic and test-friendly.

## Goals

- **Centralize runtime access**: Provide reducers a single object that exposes the
  session, event bus, and other shared services they commonly reach for.
- **Stay ergonomic for implementers**: Make it trivial for reducers to import and use
  the context without digging through orchestrator internals.
- **Keep the contract explicit**: Thread the context through reducer invocations so
  call sites document the available services and tests can substitute fakes.

## Non-goals

- Introducing mutable global state or singleton accessors.
- Redesigning the reducer invocation lifecycle or event model.
- Exposing provider-specific adapters; reducers focus on session state, not prompt
  execution.

## Shape of `ReducerContext`

`ReducerContext` is a frozen dataclass. It carries the shared references reducers need
for coordination but does not add new stateful behaviour.

```python
from __future__ import annotations

from dataclasses import dataclass

from weakincentives.events import EventBus
from weakincentives.session.session import Session


@dataclass(slots=True, frozen=True)
class ReducerContext:
    session: Session
    event_bus: EventBus
```

- `session` is always present because reducers operate within an active session.
  Supplying it ensures reducers can inspect selectors, enqueue mutations, or fork new
  work.
- `event_bus` is the in-process event bus the orchestrator already uses. Reducers can
  emit additional instrumentation or forward events derived from their work.

Future additions should follow the same pattern: explicit, immutable references that
mirror data already available in the orchestrator.

## Construction and Threading

1. The session runtime resolves the reducer to invoke for the current event.
1. Immediately before calling the reducer it constructs a `ReducerContext` with the
   active session and event bus.
1. The context is passed as a keyword-only argument when invoking the reducer.
1. After the reducer returns the context is discarded; it is never cached or reused
   across invocations.

Example signature:

```python
def my_reducer(event: SessionEvent, *, context: ReducerContext) -> None:
    context.session.upsert_selector("task", event.payload)
    context.event_bus.publish("reducer.ran", {"name": "my_reducer"})
```

Reducers that do not need the context can still accept the keyword argument and ignore
it. Callers must always supply `context=` so type checkers flag mismatches.

## Testing Guidance

- Unit tests should instantiate `ReducerContext` with lightweight fakes or fixtures for
  the session and event bus. Because the dataclass is frozen, accidental mutations will
  raise immediately.
- Prefer building helper factories (for example `build_reducer_context(...)`) in test
  utilities to keep test setup consistent.

## Migration Plan

1. Implement the dataclass in `weakincentives.session.reducer_context` alongside
   helper constructors.
1. Update the reducer dispatcher so it builds and passes a `ReducerContext` to every
   reducer invocation.
1. Migrate existing reducers to accept `*, context: ReducerContext` and remove any
   ad-hoc wiring they use today.
1. Document the new context in developer-facing guides (for example `SESSIONS.md`) so
   contributors understand the entry point.
