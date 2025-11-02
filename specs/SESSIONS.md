# Session State Specification

## Overview

Sessions provide a tiny, deterministic state container for agent runs. They listen to
the in-process event bus, collect structured tool payloads and prompt outputs, and
expose the accumulated data through a typed API. Sessions never execute side effects;
all state transitions are pure functions driven by the event stream.

## Minimal Requirements

1. **Typed routing** – Dataclass payloads/outputs map 1:1 to dedicated slices, while
   every tool invocation is also recorded in the generic `ToolData` slice even when
   the handler fails and returns `value=None`.
1. **Pure reducers** – State transitions are pure `(tuple[T, ...], DataEvent[T]) -> tuple[T, ...]`
   functions. Reducers run synchronously on the publisher thread.
1. **Event bus integration** – Sessions subscribe to `ToolInvoked` and `PromptExecuted`
   from `weakincentives.events`. They ignore every other event type.
1. **Default behavior that works** – Without custom reducers, the Session appends new
   dataclass values (deduping by equality) and keeps results immutable.
1. **Flexible slices** – Reducers may manage a slice whose element type differs from
   the incoming dataclass value so Sessions can maintain derived aggregates.

## Terminology

- `DataEvent[T]` – Normalized wrapper around `ToolInvoked` payloads and
  `PromptExecuted` outputs whose value is a dataclass of type `T`.
- **Slice** – The tuple of accumulated dataclass instances managed for a reducer.
  The slice element type may be the same as `T` or a separate dataclass.
- **Reducer** – A pure function responsible for producing a new slice when a
  matching `DataEvent[T]` arrives.

## Data Model

```python
from weakincentives.prompt._types import SupportsDataclass

@dataclass(slots=True, frozen=True)
class ToolData:
    value: SupportsDataclass | None
    source: ToolInvoked

@dataclass(slots=True, frozen=True)
class PromptData(Generic[T]):
    value: T
    source: PromptExecuted

DataEvent = ToolData | PromptData[SupportsDataclass]

TypedReducer[S] = Callable[[tuple[S, ...], DataEvent], tuple[S, ...]]
```

Session state is an immutable mapping from a **slice type** to a tuple of dataclass
instances. When the slice type matches the data type, the tuple contains the raw
event values. Reducers may also register an explicit slice type to store derived
structures.

## Public API Surface

```
src/weakincentives/session/
  session.py      # Session, DataEvent, reducer plumbing
  reducers.py     # append (default), helper factories
  selectors.py    # optional helpers built on top of Session.select_all
```

### Session

```python
class Session:
    def __init__(self, *, bus: EventBus | None = None,
                 session_id: str | None = None,
                 created_at: str | None = None) -> None: ...

    def register_reducer[T, S](
        self,
        data_type: type[T],
        reducer: TypedReducer[T, S],
        *,
        slice_type: type[S] | None = None,
    ) -> None: ...

    def select_all[S](self, slice_type: type[S]) -> tuple[S, ...]: ...
```

- Constructing with a bus subscribes to `ToolInvoked` and `PromptExecuted` immediately.
- Multiple reducers may register for the same data type; they run in registration
  order and each maintains its own slice.
- `slice_type` defaults to `data_type`. When provided it controls the tuple type the
  reducer receives and returns.
- Reducers never mutate the previous tuple; they always return a new tuple instance.

### Reducers

Provide three reducers:

1. `append` – Default behavior, dedupes by equality.
1. `upsert_by(key_fn)` – Replaces items that share the same derived key.
1. `replace_latest` – Stores only the most recent value.

### Selectors

A minimal helper module offers:

- `select_all(session, slice_type)` – Returns the tuple slice for the registered type.
- `select_latest(session, slice_type)` – Returns the last item or `None`.
- `select_where(session, slice_type, pred)` – Filters the slice by predicate.

These helpers delegate to `Session.select_all` and perform no caching.

## Event Handling Rules

1. On `ToolInvoked`, record the `ToolData` event regardless of success. When
   `result.value` is a dataclass, also dispatch the same `ToolData` to the payload's
   concrete type.
1. On `PromptExecuted`, normalize structured dataclass outputs. If the output is a
   list of dataclasses, emit one `PromptData` per item.
1. Every normalized `DataEvent` is passed to the reducer chain registered for its
   target type. Each reducer operates on the tuple registered for its `slice_type`.
   If no reducer exists, use the default append reducer.
1. Reducer failures are caught and logged; the slice stays unchanged.

## Testing Checklist

- Publishing `ToolInvoked` / `PromptExecuted` populates the correct slice exactly once.
- Reducer registration order is respected, and absence of reducers falls back to the
  default append implementation.
- Non-dataclass payloads populate the generic `ToolData` slice while leaving
  dataclass-specific slices untouched.
- Reducers may register a distinct `slice_type` and still receive the correct tuple.

## Usage Sketch

```python
bus = InProcessEventBus()
session = Session(bus=bus)

# Optional reducer wiring
session.register_reducer(SourceDetails, upsert_by(lambda d: d.source_id))
session.register_reducer(
    ResearchSummary,
    reducer_that_updates_metrics,
    slice_type=ResearchMetrics,
)

# Evaluate prompts/tools with adapters that publish events...

latest_summary = select_latest(session, ResearchSummary)
metrics = select_all(session, ResearchMetrics)
```

Sessions keep the orchestration loop deterministic and easy to test while staying as
small as possible.
