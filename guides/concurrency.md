# Concurrency

*Canonical spec: [specs/THREAD_SAFETY.md](../specs/THREAD_SAFETY.md)*

This guide explains WINK's thread-safety guarantees and the mental
models you need when running agents across multiple threads. The short
version: sessions are safe to share, but you should understand why.

## The Concurrency Model

WINK uses threads, not async. Tool adapters and orchestration loops run
in threads so they can share session state without copying it between
processes. The design is deliberately simple: a small number of locks
protect shared structures, and immutable data does the rest.

The key insight: **most concurrency bugs come from shared mutable
state.** WINK sidesteps this by making state transitions produce new
immutable tuples rather than mutating existing ones. Locks protect the
brief window where a new tuple replaces the old one.

## What Is Safe to Share

| Component | Shareable? | Why |
|-----------|------------|-----|
| `Session` | Yes | All operations protected by RLock |
| `InProcessDispatcher` | Yes | Handler list protected by RLock |
| `PromptTemplate` | Yes | Immutable after construction |
| `Prompt` | Yes | Immutable after construction |
| Adapter instances | No | Adapters may hold per-call state |
| Tool handlers | Depends | Safe if they have no mutable state |

**Sessions** are the main shared structure. Multiple threads can
dispatch events, read slices, take snapshots, and restore state
concurrently. Every operation is serialized through a lock internally.

**Dispatchers** are also shared. You can subscribe and unsubscribe
handlers from any thread. Handler execution itself happens outside the
lock, so slow handlers don't block other dispatches.

**Adapters** should not be shared across threads. Create one adapter
per thread or per worker. Adapters may hold connection state, retry
counters, or other per-call context that is not thread-safe.

## Why RLock

WINK uses `threading.RLock` (reentrant lock) rather than a plain
`Lock`. The difference matters: an RLock allows the same thread to
acquire the lock multiple times without deadlocking.

This is necessary because dispatch can trigger reducers, which may
read session state (acquiring the lock again from the same thread). A
plain Lock would deadlock in this scenario. RLock makes the common
case safe.

The trade-off: RLock is slightly slower than Lock, and it permits
recursive dispatch (dispatching a new event from inside a reducer).
This won't deadlock, but it can cause unbounded recursion or
surprising state interleaving. More on this in the anti-patterns
section.

## Copy-on-Write Semantics

When a reducer runs, it returns a new tuple describing the updated
slice. The session then atomically replaces the old tuple with the
new one under the lock. This is copy-on-write: readers see either
the old state or the new state, never a partially updated slice.

```
Thread A: dispatch(event) -> reducer returns new tuple -> lock -> swap -> unlock
Thread B: session[T].all() -> lock -> read current tuple -> unlock
```

Thread B always gets a consistent snapshot. The tuple it receives is
immutable, so it remains valid even if Thread A dispatches more events
afterward.

This is why WINK insists on frozen dataclasses for slice contents.
Mutable objects in slices would break the immutability guarantee that
makes lock-free reads safe after the initial copy.

## One Session per Workflow

The recommended pattern is one session per logical workflow (one agent
run, one evaluation, one request). This is not a technical
requirement--sessions are thread-safe--but a practical one.

**Why it matters:**

- **Isolation.** Separate workflows don't interfere with each other's
  state. A failing agent run can't corrupt the session of a healthy
  one.
- **Snapshots.** Snapshotting a session captures one workflow's state.
  Mixing workflows in a single session makes snapshots meaningless.
- **Debugging.** When something goes wrong, you want to inspect the
  exact sequence of events for one workflow. Interleaved events from
  multiple workflows are hard to untangle.
- **Lifecycle.** Sessions can be reset or restored. These operations
  affect the entire session. If two workflows share a session, one
  workflow's restore clobbers the other's state.

**In multi-worker setups** (like `LoopGroup` running multiple loops),
each worker should create its own session for each unit of work. The
dispatcher can be shared (it's thread-safe), but sessions should not
be shared across independent workflows.

```python nocheck
# Good: one session per request
def handle_request(request, dispatcher):
    session = Session(dispatcher=dispatcher)
    # ... process request with this session ...

# Avoid: shared session across requests
shared_session = Session(dispatcher=dispatcher)
def handle_request(request):
    # Multiple requests interleave in the same session
    shared_session.dispatch(...)  # Which request does this belong to?
```

## Anti-Patterns

### Blocking in Reducers

Reducers run under the session lock. A slow reducer blocks all other
threads from dispatching events or reading state.

```python nocheck
# WRONG: blocks all session operations for 10 seconds
@reducer(on=MyEvent)
def bad_reducer(self, event: MyEvent) -> Replace["MyState"]:
    time.sleep(10)  # Every other thread waits here
    return Replace((new_state,))
```

**Fix:** Keep reducers fast and pure. If you need to do I/O or heavy
computation, do it in a tool handler and dispatch the result as an
event.

### Mutable State in Slices

Frozen dataclasses exist for a reason. If you put mutable objects in
slices, other threads can mutate them after reading, breaking the
immutability contract.

```python nocheck
# WRONG: mutable list in a "frozen" dataclass
@dataclass(slots=True, frozen=True)
class Plan:
    steps: list[str]  # list is mutable!

# Thread A reads the plan
plan = session[Plan].latest()
# Thread B mutates it through the shared reference
plan.steps.append("surprise")  # Visible to Thread A
```

**Fix:** Use tuples instead of lists. Use frozen dataclasses all the
way down.

```python nocheck
# Correct: immutable all the way down
@dataclass(slots=True, frozen=True)
class Plan:
    steps: tuple[str, ...]
```

### Recursive Dispatch

Dispatching an event from inside a reducer is technically possible
(RLock allows it) but dangerous.

```python nocheck
# WRONG: recursive dispatch
@reducer(on=StepCompleted)
def bad_reducer(self, event: StepCompleted) -> Replace["Plan"]:
    session.dispatch(CheckProgress())  # Recursive!
    return Replace((new_plan,))
```

**Problems:**

- The inner dispatch runs its reducers before the outer dispatch
  completes, so the outer reducer's return value overwrites whatever
  the inner dispatch changed.
- If the inner dispatch triggers the same reducer, you get unbounded
  recursion.
- State interleaving becomes unpredictable.

**Fix:** Return the new state from the reducer and let the caller
dispatch follow-up events after the first dispatch completes.

### Sharing Adapters Across Threads

Adapters may hold per-call state (HTTP connections, retry counters,
rate limiters). Sharing them across threads can cause subtle bugs.

```python nocheck
# WRONG: shared adapter
adapter = OpenAIAdapter(api_key="...")
threads = [
    Thread(target=run_agent, args=(adapter, session))
    for _ in range(4)
]
```

**Fix:** Create one adapter per thread or per worker.

## Summary

| Rule | Reason |
|------|--------|
| Share sessions and dispatchers freely | Protected by RLock |
| Don't share adapters across threads | May hold per-call state |
| Use frozen dataclasses with tuples | Preserves immutability contract |
| Keep reducers fast and pure | They run under the lock |
| One session per workflow | Isolation, debuggability, clean snapshots |
| Never dispatch from inside a reducer | Causes recursion and state interleaving |

## Next Steps

- [Sessions](sessions.md): Understand the state model
- [Lifecycle](lifecycle.md): Multi-worker setups with LoopGroup
- [Tools](tools.md): Where side effects belong
