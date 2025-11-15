# Thread Safety Specification

## Scope

This document inventories the components that currently assume single-threaded
usage and outlines the changes required to support multi-threaded adapters or
hosts. The focus areas were identified while reviewing the session subsystem and
the code review examples, which exercise the most realistic orchestration loop
in the repository. Asynchronous adapters are out of scope; the immediate goal is
to provide thread-based synchronization only.

## Current State Assessment

### Event Bus Implementations

- `InProcessEventBus` stores handlers in a mutable `dict[type, list]` without any
  locking. Concurrent calls to `subscribe` and `publish` can race, and a handler
  mutating the registry while the bus iterates over it will trigger
  `RuntimeError` or silently skip subscribers.
- The test helper `tests.helpers.events.NullEventBus` is effectively stateless
  and thread-safe today, but a unified interface should still guarantee that
  all bus implementations are safe to use from multiple threads.

### Session State Store

- `Session` mutates `_reducers` and `_state` dictionaries in response to
  subscriptions and event delivery with no synchronization. Concurrent publishes
  can interleave reducer mutations, drop updates, or observe partially-updated
  tuples.
- `clone`, `snapshot`, and `rollback` all read or replace the dictionaries
  without a consistent view of the data; a concurrent reducer can mutate state
  while the snapshot is being normalized.
- Reducer registration appends to shared lists, so two threads registering
  reducers for the same type can overwrite each other’s changes.

### Prompt Override Store

- `LocalPromptOverridesStore` caches the repository root in `_root` and writes
  override files after performing `exists()` checks. There is no locking around
  the cache mutation or the read-modify-write flow, so concurrent `seed` or
  `upsert` calls can step on each other, overwrite changes, or raise `FileNotFoundError`
  after one thread deletes the file another is about to write.

### Code Review Example Sessions

- The code reviewer example and `CodeReviewSession` both keep mutable state such as
  `_history` lists and override tags that are mutated from event handlers without
  locks. The event bus invokes those handlers synchronously on the publisher
  thread, so cross-thread publish calls would interleave access to these lists.
- The planning tools (`PlanningToolsSection`, `_PlanningToolSuite`) lean on the
  shared `Session` instance for coordination. They assume reducer registration is
  idempotent and will suffer from the same races described above.

### Serde and Tool Helpers

- Tool reducer helpers (`append`, `replace_latest`, etc.) are pure, but they rely
  on the caller to provide immutable tuples. Without a thread-safe session, the
  helpers will receive stale data or lose updates. No additional locking is
  currently required in these helpers once the caller provides atomic slice
  updates.

## Target Guarantees

1. Multiple threads must be able to register event handlers and publish events
   concurrently without corrupting the bus state or dropping deliveries.
1. Session state transitions must be atomic per reducer invocation. Reducers
   should observe a consistent snapshot of their slice, and readers should see
   completed updates.
1. File-backed prompt override operations must be safe when multiple threads
   seed or update overrides for the same prompt.
1. Example orchestration classes must not corrupt their history buffers or cause
   inconsistent console output when multiple threads publish tool events.

## Recommended Changes

### Event Bus

- Introduce a `threading.RLock` (or `Lock`) to guard `_handlers` mutations.
- Take a snapshot of the handler list under the lock before delivering events to
  avoid holding the lock while handlers run.
- Ensure `subscribe` is idempotent or document that duplicates are allowed when
  multiple threads register the same handler.
- Document that handlers are expected to be fast; no additional thread pool is
  required to offload delivery.
- Consider adding `unsubscribe` support while touching the concurrency model so
  sessions can detach safely.

### Session

- Protect `_reducers` and `_state` with a re-entrant lock. All read/modify/write
  operations (`register_reducer`, `_dispatch_data_event`, `clone`, `snapshot`,
  `rollback`, and selectors) should acquire the lock.
- Copy-on-write patterns should continue, but take the copy inside the critical
  section to prevent observing partially updated tuples.
- Ensure `_attach_to_bus` does not register handlers multiple times when called
  concurrently; dedupe subscriptions or guard with a flag.
- Audit reducer defaults to avoid race conditions when lazily importing
  `append`. Import once at module load or guard the import inside the lock.

### LocalPromptOverridesStore

- Guard `_root` lazy initialization with a lock so the git root discovery only
  runs once even when multiple threads resolve overrides simultaneously.
- Wrap file writes in atomic operations using temporary files (already in place)
  but add locking to serialize `exists`/`unlink`/`rename` flows per override
  path.
- Document that atomic file operations are sufficient; no additional
  process-level coordination is required for the override store.

### Code Review Sessions

- When printing to stdout from event handlers, serialize writes or route through
  the structured logger so output from concurrent publishers does not interleave.
- Verify that prompt override seeding runs once per session even if multiple
  threads construct review sessions.

### Testing Strategy

- Add unit tests that spin up threads publishing tool events concurrently to a
  shared `Session` and assert that all reducers observe the expected counts.
- Introduce stress tests for `LocalPromptOverridesStore.seed_if_necessary` that
  call it concurrently and validate the resulting file contents.

Implementing the changes above will let adapters safely share a single session
and event bus across worker threads, unblock background agents, and document the
remaining concurrency expectations for contributors.
