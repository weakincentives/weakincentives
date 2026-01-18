# Threading Patterns Review

This document reviews all threading usage in the codebase and proposes
alternative patterns that are easier to reason about and test.

## Executive Summary

The codebase uses threading extensively for:
1. **Worker thread pools** (LoopGroup runs multiple MainLoop instances)
2. **Background daemon threads** (Watchdog, HealthServer, mailbox reaper)
3. **State protection** (RLock/Lock on Session, Dispatcher, Heartbeat)
4. **Producer-consumer coordination** (Condition variable in InMemoryMailbox)
5. **Shutdown signaling** (threading.Event across all loop classes)

While the current patterns are correctly implemented, they introduce complexity:
- Careful lock ordering required to avoid deadlocks
- Copy-on-read patterns to prevent holding locks during I/O
- Difficult to test timing-sensitive code
- Hidden coupling through shared mutable state

## Current Threading Inventory

### Category 1: Worker Thread Pools

| Component | Location | Pattern | Purpose |
|-----------|----------|---------|---------|
| LoopGroup | `runtime/lifecycle.py:369` | ThreadPoolExecutor | Run multiple loops in parallel |

**Analysis**: LoopGroup creates one thread per loop. Each thread runs a
blocking `loop.run()` call that polls a mailbox. This is a reasonable pattern
for I/O-bound work, but:

- Thread count grows with loop count (not bounded by CPU)
- No backpressure mechanism across loops
- Testing requires actual threads (timing-dependent)

### Category 2: Background Daemon Threads

| Component | Location | Purpose |
|-----------|----------|---------|
| Watchdog | `runtime/watchdog.py:191` | Monitor heartbeats, terminate on stall |
| HealthServer | `runtime/watchdog.py:309` | HTTP server for k8s probes |
| InMemoryMailbox reaper | `runtime/mailbox/_in_memory.py:168` | Requeue expired messages |

**Analysis**: Daemon threads run independently and communicate via shared state
(Events, locks). This creates implicit coupling:

- Watchdog reads `Heartbeat._last_beat` while workers write it
- Reaper modifies `_pending` and `_invisible` queues while receive() waits
- HealthServer's readiness check calls `loop.running` across threads

### Category 3: State Protection Locks

| Component | Lock Type | Protected State |
|-----------|-----------|-----------------|
| Session | RLock | `_slices`, `_reducers`, `_children` |
| InProcessDispatcher | RLock | `_handlers` dict |
| Heartbeat | Lock | `_last_beat`, `_callbacks` |
| BudgetTracker | Lock | `_usage` dict |
| InMemoryMailbox | Lock + Condition | `_pending`, `_invisible` queues |
| PendingToolTracker | RLock | Pending execution snapshots |
| OverrideFilesystem | RLock (×3) | Override directory, per-file locks |
| LeaseExtender | Lock | `_msg`, `_heartbeat`, `_last_extension` |
| MainLoop | Lock | `_running` flag |
| ShutdownCoordinator | Lock | `_callbacks` list |

**Analysis**: Most locks protect mutable state accessed from multiple threads.
The patterns used are sound (copy-on-read, release before I/O), but:

- Each component manages its own synchronization
- Lock hierarchies are implicit, not documented
- RLock enables reentrant patterns that are hard to follow

### Category 4: Condition Variables

| Component | Location | Pattern |
|-----------|----------|---------|
| InMemoryMailbox | `runtime/mailbox/_in_memory.py:147` | Producer-consumer queue |

**Analysis**: Classic bounded buffer pattern. `send()` notifies waiters,
`receive()` waits with timeout. Correctly implemented but:

- Condition variable semantics are subtle
- Spurious wakeups require loop-based wait patterns
- Testing requires careful timing or mock clocks

### Category 5: Shutdown Coordination

| Component | Location | Mechanism |
|-----------|----------|-----------|
| ShutdownCoordinator | `runtime/lifecycle.py:160` | Event + callbacks |
| MainLoop | `runtime/main_loop.py:226` | Event + `wait_until()` |
| EvalLoop | `evals/_loop.py:131` | Event + `wait_until()` |
| DLQConsumer | `runtime/dlq.py:224` | Event + `wait_until()` |
| Watchdog | `runtime/watchdog.py:177` | Event + timeout wait |
| InMemoryMailbox | `runtime/mailbox/_in_memory.py:155` | Event + thread join |

**Analysis**: Shutdown is coordinated via `threading.Event`. This is a
reasonable pattern but requires:

- Checking `is_set()` at strategic points
- Timeout-based polling (not instant response)
- Careful ordering (signal before join)

---

## Proposed Alternative Patterns

### Pattern 1: Actor Model for Loop Coordination

**Current**: LoopGroup uses ThreadPoolExecutor with shared ShutdownCoordinator.

**Alternative**: Each loop becomes an actor with its own message queue. Control
messages (shutdown, health check) are sent explicitly rather than accessed via
shared state.

```python
@dataclass(frozen=True)
class ControlMessage:
    """Base class for control messages."""


@dataclass(frozen=True)
class ShutdownRequest(ControlMessage):
    timeout: float = 30.0


@dataclass(frozen=True)
class HealthCheckRequest(ControlMessage):
    ticket: Ticket[HealthStatus]


@dataclass(frozen=True)
class HealthStatus:
    loop_name: str
    running: bool
    last_heartbeat: float


class LoopActor:
    """Actor wrapper around a Runnable loop."""

    def __init__(self, loop: Runnable, name: str) -> None:
        self._loop = loop
        self._name = name
        self._control: Queue[ControlMessage] = Queue()

    def send(self, msg: ControlMessage) -> None:
        self._control.put_nowait(msg)

    def run(self) -> None:
        # Run loop with periodic control message checks
        while not self._shutdown_requested:
            self._process_control_messages()
            self._loop.run(max_iterations=1, ...)

    def _process_control_messages(self) -> None:
        while True:
            try:
                msg = self._control.get_nowait()
            except Empty:
                break
            self._handle_control(msg)
```

**Benefits**:
- No shared state between coordinator and loops
- Control flow explicit via messages
- Easy to test by sending messages directly
- No lock ordering concerns

### Pattern 2: Ticket-Based Request/Response

**Current**: Health checks read shared state across threads.

**Alternative**: Introduce tickets for request/response patterns. A ticket is a
single-use container for an asynchronous result.

```python
@dataclass(slots=True)
class Ticket[T]:
    """Single-use container for asynchronous results.

    Create a ticket, pass it to a worker, then await the result.
    Tickets are single-writer, single-reader.
    """
    _event: threading.Event = field(default_factory=threading.Event)
    _result: T | None = field(default=None)
    _error: Exception | None = field(default=None)

    def complete(self, result: T) -> None:
        """Complete the ticket with a result (writer side)."""
        if self._event.is_set():
            raise TicketAlreadyCompletedError()
        self._result = result
        self._event.set()

    def fail(self, error: Exception) -> None:
        """Complete the ticket with an error (writer side)."""
        if self._event.is_set():
            raise TicketAlreadyCompletedError()
        self._error = error
        self._event.set()

    def wait(self, timeout: float | None = None) -> T:
        """Wait for and return the result (reader side)."""
        if not self._event.wait(timeout=timeout):
            raise TicketTimeoutError()
        if self._error is not None:
            raise self._error
        return cast(T, self._result)

    def is_ready(self) -> bool:
        """Check if result is available without blocking."""
        return self._event.is_set()


# Usage example: Health check with ticket
def check_health(actors: Sequence[LoopActor]) -> list[HealthStatus]:
    tickets = []
    for actor in actors:
        ticket: Ticket[HealthStatus] = Ticket()
        actor.send(HealthCheckRequest(ticket=ticket))
        tickets.append(ticket)

    # Wait for all responses (with timeout)
    return [t.wait(timeout=5.0) for t in tickets]
```

**Benefits**:
- Clear ownership: one writer, one reader
- No shared mutable state beyond the ticket
- Easy to test: create ticket, call complete(), check result
- Timeouts are explicit per-request

### Pattern 3: Message Queue for Mailbox Operations

**Current**: InMemoryMailbox uses Lock + Condition for thread coordination.

**Alternative**: The existing mailbox abstraction already implements message
passing. Extend this pattern to internal coordination.

```python
class InMemoryMailbox[T, R]:
    """Mailbox using internal message passing for coordination."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._pending: deque[_InFlightMessage[T, R]] = deque()
        self._invisible: dict[str, _InFlightMessage[T, R]] = {}

        # Internal command queue replaces condition variable
        self._commands: Queue[_MailboxCommand] = Queue()
        self._results: dict[UUID, Ticket[Any]] = {}

    def send(self, body: T, *, reply_to: Mailbox[R, None] | None = None) -> str:
        ticket: Ticket[str] = Ticket()
        self._commands.put(_SendCommand(body=body, reply_to=reply_to, ticket=ticket))
        return ticket.wait()

    def receive(self, *, max_messages: int = 1, ...) -> Sequence[Message[T, R]]:
        ticket: Ticket[Sequence[Message[T, R]]] = Ticket()
        self._commands.put(_ReceiveCommand(max_messages=max_messages, ..., ticket=ticket))
        return ticket.wait(timeout=wait_time_seconds)

    def _run_loop(self) -> None:
        """Single-threaded command processor."""
        while not self._closed:
            try:
                cmd = self._commands.get(timeout=0.1)
            except Empty:
                self._reap_expired()
                continue
            self._handle_command(cmd)
```

**Benefits**:
- Single-threaded command processing (no locks needed)
- All state mutations happen in one place
- Reaper integrated into main loop (no daemon thread)
- Testing: send commands, check state

**Trade-off**: Adds latency (command queue hop). May not be appropriate for
high-throughput scenarios.

### Pattern 4: Structured Concurrency for Daemon Tasks

**Current**: Daemon threads for Watchdog, HealthServer, reaper run independently.

**Alternative**: Use structured concurrency where background tasks are tied to
a parent scope. When the scope exits, all tasks are cancelled and joined.

```python
@contextmanager
def daemon_scope() -> Iterator[DaemonScope]:
    """Context manager for structured daemon task management."""
    scope = DaemonScope()
    try:
        yield scope
    finally:
        scope.shutdown_all(timeout=5.0)


class DaemonScope:
    """Manages a set of daemon tasks with coordinated shutdown."""

    def __init__(self) -> None:
        self._daemons: list[tuple[threading.Thread, threading.Event]] = []

    def spawn(
        self,
        target: Callable[[], None],
        name: str,
        *,
        stop_event: threading.Event | None = None,
    ) -> threading.Event:
        """Spawn a daemon task that will be stopped on scope exit."""
        event = stop_event or threading.Event()
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._daemons.append((thread, event))
        return event

    def shutdown_all(self, timeout: float) -> None:
        """Signal all daemons to stop and wait for completion."""
        for _, event in self._daemons:
            event.set()
        deadline = time.monotonic() + timeout
        for thread, _ in self._daemons:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(timeout=remaining)


# Usage in LoopGroup
class LoopGroup:
    def run(self, ...) -> None:
        with daemon_scope() as scope:
            if self._watchdog_threshold is not None:
                scope.spawn(self._watchdog_loop, "watchdog")
            if self._health_port is not None:
                scope.spawn(self._health_server_loop, "health-server")

            # Run main loops...
        # All daemons automatically stopped on exit
```

**Benefits**:
- Clear lifetime: daemons tied to scope
- No orphaned threads on exception
- Easier to test: scope exit guarantees cleanup
- Explicit stop mechanism (no relying on daemon=True process exit)

### Pattern 5: Event Sourcing for Session State

**Current**: Session uses RLock to protect mutable state during dispatch.

**Alternative**: Session is already event-sourced (dispatch events, reducers
compute new state). Strengthen this by making dispatch single-threaded.

```python
class Session:
    """Session with single-threaded event processing."""

    def __init__(self, ...) -> None:
        # Event queue for all mutations
        self._event_queue: Queue[SupportsDataclass] = Queue()
        self._processor_thread = threading.Thread(
            target=self._process_events,
            daemon=True,
        )
        self._processor_thread.start()

    def dispatch(self, event: SupportsDataclass) -> Ticket[DispatchResult]:
        """Enqueue event for processing, return ticket for result."""
        ticket: Ticket[DispatchResult] = Ticket()
        self._event_queue.put((event, ticket))
        return ticket

    def _process_events(self) -> None:
        """Single-threaded event processor."""
        while not self._stopped:
            try:
                event, ticket = self._event_queue.get(timeout=0.1)
            except Empty:
                continue

            # No lock needed - single writer
            result = self._apply_event(event)
            ticket.complete(result)

    # Read operations can still be lock-free if slices are immutable
    def __getitem__(self, slice_type: type[S]) -> SliceAccessor[S]:
        # Returns immutable view - no lock needed
        return SliceAccessor(self, slice_type)
```

**Benefits**:
- No lock contention during reads
- All mutations serialized through queue
- Event ordering guaranteed
- Testing: enqueue events, check final state

**Trade-off**: Dispatch becomes async (returns ticket instead of result).
Existing code assumes synchronous dispatch.

---

## Recommendations

### Short-term (Low Risk)

1. **Adopt Ticket pattern for cross-thread requests**
   - Add `Ticket[T]` to `weakincentives.runtime`
   - Use for health checks, shutdown confirmation
   - No changes to existing lock patterns

2. **Document lock ordering**
   - Add explicit lock hierarchy to `specs/THREAD_SAFETY.md`
   - Annotate which locks may be held when calling external code

3. **Add structured concurrency for daemons**
   - Implement `DaemonScope` context manager
   - Use in LoopGroup for watchdog/health server lifecycle
   - Ensures cleanup on exception

### Medium-term (Moderate Refactoring)

4. **Actor pattern for LoopGroup coordination**
   - Replace ThreadPoolExecutor + ShutdownCoordinator
   - Each loop receives control messages
   - Easier to test shutdown sequences

5. **Single-threaded mailbox command processor**
   - Replace Lock + Condition in InMemoryMailbox
   - Internal command queue with one processor thread
   - Eliminates subtle condition variable bugs

### Long-term (API Changes)

6. **Async dispatch with tickets**
   - `session.dispatch()` returns `Ticket[DispatchResult]`
   - Enables single-threaded event processing
   - Breaking change for callers expecting sync dispatch

---

## Testing Improvements

With message passing patterns, tests become more deterministic:

```python
# Before: timing-dependent
def test_shutdown_during_processing():
    loop = create_loop()
    thread = Thread(target=loop.run)
    thread.start()
    time.sleep(0.1)  # Hope processing started
    loop.shutdown()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

# After: message-based
def test_shutdown_during_processing():
    actor = LoopActor(create_loop(), "test")
    actor.send(ProcessRequest(body=...))
    actor.send(ShutdownRequest(timeout=5.0))

    ticket: Ticket[ShutdownResult] = Ticket()
    actor.send(ShutdownConfirmRequest(ticket=ticket))

    result = ticket.wait(timeout=10.0)
    assert result.clean_shutdown
```

---

## Migration Path

1. **Add new primitives** (Ticket, DaemonScope) without changing existing code
2. **Use new patterns** in new components first
3. **Gradually migrate** existing components during refactoring
4. **Maintain compatibility** with existing sync APIs via wrapper methods

---

## Appendix: Threading Patterns by Risk Level

### Low Risk (Well-tested, simple patterns)
- `threading.Event` for shutdown signaling
- `Lock` for simple state protection
- Copy-on-read patterns (snapshot under lock, operate outside)

### Medium Risk (Correct but subtle)
- `RLock` with reentrant usage
- `Condition` for producer-consumer
- Daemon threads with shared state

### Higher Risk (Needs careful review)
- Multiple locks with implicit ordering
- Callbacks invoked under lock
- State accessed from multiple daemon threads

---

## Related Specifications

- `specs/THREAD_SAFETY.md` - Current thread safety guarantees
- `specs/MAILBOX.md` - Mailbox semantics and threading model
- `specs/LIFECYCLE.md` - LoopGroup and shutdown coordination
