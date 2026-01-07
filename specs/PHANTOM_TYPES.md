# Phantom Types for State Encoding Specification

## Purpose

Phantom types encode state information in the type system without runtime cost.
A `Session[Uninitialized]`, `Session[Running]`, and `Session[Stopped]` are the
same runtime object but represent different compile-time types. Functions
declare which states they accept (`def pause(s: Session[Running]) ->
Session[Paused]`), and passing a session in the wrong state becomes a type
error caught by pyright. This approach is lighter-weight than full runtime
state machines but still prevents "wrong state" bugs statically.

## Guiding Principles

- **Zero runtime cost**: Phantom type parameters exist only at type-checking
  time. No runtime isinstance checks, no state enums stored on objects, no
  overhead in production code paths.
- **Compile-time safety**: Invalid state transitions are type errors, not
  runtime exceptions. Developers see mistakes immediately in their editor.
- **Smart constructor discipline**: Phantom types alone don't enforce
  transition validity. Smart constructors (factory functions that control state
  creation) must gate all state transitions to prevent invalid phantom type
  construction.
- **Composable with existing patterns**: Integrates cleanly with DbC
  decorators, frozen dataclasses, and the session event system.
- **Explicit over implicit**: State requirements appear in function signatures,
  making valid call patterns self-documenting.

## Core Concept

### The Problem

Consider a `Connection` that must be opened before use and closed when done:

```python
class Connection:
    def __init__(self, host: str) -> None:
        self._host = host
        self._socket: Socket | None = None

    def open(self) -> None:
        self._socket = Socket(self._host)

    def send(self, data: bytes) -> None:
        self._socket.write(data)  # Crashes if not opened

    def close(self) -> None:
        self._socket.close()
        self._socket = None
```

Runtime checks can catch misuse, but errors surface late (in production, in
tests). The type system offers no help.

### The Phantom Type Solution

```python
from typing import Generic, TypeVar

# Phantom state types (never instantiated)
class Closed: ...
class Open: ...

S = TypeVar("S")

class Connection(Generic[S]):
    """Connection with phantom state parameter S."""

    def __init__(self, host: str) -> None:
        self._host = host
        self._socket: Socket | None = None

# Smart constructors control state transitions
def create_connection(host: str) -> Connection[Closed]:
    """Create a new closed connection."""
    return Connection(host)

def open_connection(conn: Connection[Closed]) -> Connection[Open]:
    """Open a closed connection."""
    conn._socket = Socket(conn._host)
    return conn  # type: ignore[return-value]  # Same object, new type

def send(conn: Connection[Open], data: bytes) -> None:
    """Send data on an open connection."""
    assert conn._socket is not None
    conn._socket.write(data)

def close_connection(conn: Connection[Open]) -> Connection[Closed]:
    """Close an open connection."""
    assert conn._socket is not None
    conn._socket.close()
    conn._socket = None
    return conn  # type: ignore[return-value]  # Same object, new type
```

Now the type checker enforces correct usage:

```python
conn = create_connection("localhost")  # Connection[Closed]
send(conn, b"hello")                   # ❌ Type error: expected Connection[Open]

conn = open_connection(conn)           # Connection[Open]
send(conn, b"hello")                   # ✅ Works

conn = close_connection(conn)          # Connection[Closed]
send(conn, b"goodbye")                 # ❌ Type error: expected Connection[Open]
```

## Implementation Patterns

### Pattern 1: Frozen Phantom Wrapper

For immutable state tracking, wrap the actual state in a frozen dataclass:

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

class Uninitialized: ...
class Ready: ...
class Complete: ...

S = TypeVar("S")

@dataclass(frozen=True, slots=True)
class Task(Generic[S]):
    """Task with phantom state parameter."""
    task_id: str
    payload: dict[str, object]

def create_task(task_id: str, payload: dict[str, object]) -> Task[Uninitialized]:
    return Task(task_id=task_id, payload=payload)

def initialize_task(task: Task[Uninitialized]) -> Task[Ready]:
    # Validation logic here
    return Task(task_id=task.task_id, payload=task.payload)  # New frozen instance

def complete_task(task: Task[Ready]) -> Task[Complete]:
    # Completion logic here
    return Task(task_id=task.task_id, payload=task.payload)
```

### Pattern 2: Protocol-Based Phantom States

Use protocols to define state capabilities:

```python
from typing import Protocol, TypeVar, Generic

class CanSend(Protocol):
    """Marker protocol for sendable state."""
    ...

class CanReceive(Protocol):
    """Marker protocol for receivable state."""
    ...

class Idle(CanSend, CanReceive): ...
class Sending: ...  # Cannot receive while sending
class Receiving: ...  # Cannot send while receiving

S = TypeVar("S")

class Channel(Generic[S]):
    """Bidirectional channel with state-dependent operations."""
    ...

def send[S: CanSend](channel: Channel[S], data: bytes) -> Channel[Sending]:
    ...

def receive[S: CanReceive](channel: Channel[S]) -> tuple[bytes, Channel[Receiving]]:
    ...
```

### Pattern 3: Multi-Dimensional Phantom Types

Track multiple independent state dimensions:

```python
from typing import Generic, TypeVar

# Authentication state
class Anonymous: ...
class Authenticated: ...

# Authorization state
class Unauthorized: ...
class Authorized: ...

AuthN = TypeVar("AuthN")
AuthZ = TypeVar("AuthZ")

@dataclass(frozen=True, slots=True)
class Request(Generic[AuthN, AuthZ]):
    """HTTP request with authentication and authorization state."""
    path: str
    headers: dict[str, str]

def authenticate(req: Request[Anonymous, AuthZ]) -> Request[Authenticated, AuthZ]:
    """Authenticate without changing authorization state."""
    ...

def authorize(req: Request[Authenticated, Unauthorized]) -> Request[Authenticated, Authorized]:
    """Authorize (requires authentication first)."""
    ...

def handle_protected(req: Request[Authenticated, Authorized]) -> Response:
    """Handle request (requires both auth states)."""
    ...
```

### Pattern 4: Linear Type Simulation

Phantom types can simulate linear types (use-exactly-once semantics):

```python
class Available: ...
class Consumed: ...

S = TypeVar("S")

@dataclass(frozen=True, slots=True)
class Token(Generic[S]):
    """Single-use token with phantom consumption state."""
    value: str

def create_token(value: str) -> Token[Available]:
    return Token(value=value)

def consume_token(token: Token[Available]) -> tuple[str, Token[Consumed]]:
    """Consume token, returning value and consumed marker."""
    return token.value, Token(value=token.value)

# Usage
token = create_token("secret")
value, consumed = consume_token(token)
# value2, _ = consume_token(token)  # ❌ Still Available type, but token is "used"
# value3, _ = consume_token(consumed)  # ❌ Type error: expected Token[Available]
```

**Note**: True linear types require the type system to track that `token` should
not be used after `consume_token`. Python's type system cannot enforce this, but
the pattern documents intent and catches reuse of consumed tokens.

## Smart Constructors

Smart constructors are the enforcement mechanism for phantom types. Without
them, phantom types are merely documentation.

### Rules for Smart Constructors

1. **All state creation goes through constructors**: Never expose `__init__`
   for direct phantom type construction.

2. **Constructors validate transitions**: Each constructor function encodes a
   valid state transition in its signature.

3. **Internal `# type: ignore` is acceptable**: Smart constructors may use
   `type: ignore[return-value]` when returning the same object with a new
   phantom type. This is the one place where bypassing the type checker is
   intentional and safe.

4. **Document state machine**: Provide a clear diagram or table of valid
   transitions.

### Example: Session Lifecycle

```python
from typing import Generic, TypeVar
from dataclasses import dataclass

# States
class New: ...
class Initialized: ...
class Running: ...
class Paused: ...
class Stopped: ...

S = TypeVar("S")

@dataclass(slots=True)
class Session(Generic[S]):
    """Session with phantom lifecycle state."""
    session_id: str
    _events: list[object]

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._events = []

# Smart constructors (the ONLY way to create/transition sessions)

def create_session(session_id: str) -> Session[New]:
    """Create a new session."""
    return Session(session_id)

def initialize(session: Session[New]) -> Session[Initialized]:
    """Initialize a new session."""
    session._events.append(("initialized",))
    return session  # type: ignore[return-value]

def start(session: Session[Initialized] | Session[Paused]) -> Session[Running]:
    """Start an initialized or paused session."""
    session._events.append(("started",))
    return session  # type: ignore[return-value]

def pause(session: Session[Running]) -> Session[Paused]:
    """Pause a running session."""
    session._events.append(("paused",))
    return session  # type: ignore[return-value]

def stop(session: Session[Running] | Session[Paused]) -> Session[Stopped]:
    """Stop a running or paused session."""
    session._events.append(("stopped",))
    return session  # type: ignore[return-value]

# State-specific operations

def dispatch(session: Session[Running], event: object) -> None:
    """Dispatch event to running session."""
    session._events.append(event)

def snapshot(session: Session[Running] | Session[Paused]) -> dict[str, object]:
    """Take snapshot of active session."""
    return {"session_id": session.session_id, "events": list(session._events)}
```

### State Transition Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Session State Machine                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌─────┐  initialize   ┌─────────────┐  start   ┌─────────┐  │
│    │ New │──────────────▶│ Initialized │─────────▶│ Running │  │
│    └─────┘               └─────────────┘          └────┬────┘  │
│                                                        │        │
│                               ┌────────────────────────┤        │
│                               │                        │        │
│                               ▼ pause                  │ stop   │
│                          ┌────────┐                    │        │
│                          │ Paused │───────┐            │        │
│                          └───┬────┘       │            │        │
│                              │            │ stop       │        │
│                        start │            │            │        │
│                              │            ▼            ▼        │
│                              │       ┌─────────┐               │
│                              └──────▶│ Stopped │◀──────────────┘│
│                                      └─────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

## Integration with WINK Patterns

### With Design-by-Contract

Phantom types complement DbC. Use DbC for runtime invariants that can't be
expressed in the type system:

```python
from weakincentives.dbc import require, ensure

@require(lambda session: len(session._events) < 10000, "event limit not exceeded")
def dispatch(session: Session[Running], event: object) -> None:
    """Dispatch event to running session.

    Type system ensures session is Running.
    DbC ensures event limit invariant.
    """
    session._events.append(event)
```

### With Frozen Dataclasses

Phantom types pair naturally with immutable data:

```python
from weakincentives.dataclasses import FrozenDataclass

class Draft: ...
class Published: ...
class Archived: ...

S = TypeVar("S")

@FrozenDataclass()
class Article(Generic[S]):
    article_id: str
    title: str
    content: str

def publish(article: Article[Draft]) -> Article[Published]:
    """Publishing creates a new frozen instance."""
    return Article(
        article_id=article.article_id,
        title=article.title,
        content=article.content,
    )
```

### With Session Events

Phantom types can encode session readiness for specific operations:

```python
class NoReducers: ...
class ReducersInstalled: ...

R = TypeVar("R")

@dataclass(slots=True)
class TypedSession(Generic[R]):
    """Session wrapper with reducer installation state."""
    inner: Session

def install_reducers(
    session: TypedSession[NoReducers],
    reducers: Sequence[Reducer],
) -> TypedSession[ReducersInstalled]:
    """Install reducers, enabling dispatch."""
    for reducer in reducers:
        session.inner.register(reducer)
    return session  # type: ignore[return-value]

def safe_dispatch(
    session: TypedSession[ReducersInstalled],
    event: object,
) -> None:
    """Dispatch only allowed after reducers installed."""
    session.inner.dispatch(event)
```

## Advanced Patterns

### Existential Phantom Types

Hide the phantom parameter when the specific state doesn't matter:

```python
from typing import Any

# Type alias for "any state"
AnySession = Session[Any]

def get_session_id(session: AnySession) -> str:
    """Works regardless of session state."""
    return session.session_id

def log_session(session: AnySession) -> None:
    """Log session info (state-agnostic)."""
    print(f"Session: {session.session_id}")
```

### Phantom Type Bounds

Constrain phantom parameters to valid states:

```python
from typing import TypeVar

# Define state hierarchy
class SessionState: ...
class ActiveState(SessionState): ...
class InactiveState(SessionState): ...

class Running(ActiveState): ...
class Paused(ActiveState): ...
class Stopped(InactiveState): ...

# Bound type variable to active states
ActiveS = TypeVar("ActiveS", bound=ActiveState)

def checkpoint(session: Session[ActiveS]) -> Snapshot:
    """Checkpoint any active session (Running or Paused)."""
    return Snapshot(session._events)
```

### Combining with Literal Types

Use `Literal` for finite, known states with runtime accessibility:

```python
from typing import Literal, Generic, TypeVar, get_args

State = Literal["new", "running", "stopped"]
S = TypeVar("S", bound=State)

@dataclass(frozen=True, slots=True)
class Worker(Generic[S]):
    worker_id: str
    state: S  # Runtime-accessible state

def create_worker(worker_id: str) -> Worker[Literal["new"]]:
    return Worker(worker_id=worker_id, state="new")

def start_worker(worker: Worker[Literal["new"]]) -> Worker[Literal["running"]]:
    return Worker(worker_id=worker.worker_id, state="running")
```

This hybrid approach provides both compile-time checking and runtime state
inspection when needed.

## Testing Strategy

### Type-Level Testing

Use `pyright` or `mypy` to verify type errors are caught:

```python
# tests/phantom_types/test_connection_types.py
"""Type-level tests for Connection phantom types.

Run with: pyright tests/phantom_types/
Expected: Type errors on marked lines
"""

def test_cannot_send_on_closed() -> None:
    conn = create_connection("localhost")
    send(conn, b"hello")  # type: ignore[arg-type]  # Expected error

def test_cannot_close_twice() -> None:
    conn = create_connection("localhost")
    conn = open_connection(conn)
    conn = close_connection(conn)
    close_connection(conn)  # type: ignore[arg-type]  # Expected error
```

### Runtime Behavior Tests

Test that smart constructors correctly transition state:

```python
def test_session_lifecycle():
    """Test valid state transitions."""
    session = create_session("test-1")
    session = initialize(session)
    session = start(session)

    dispatch(session, {"type": "test"})

    session = pause(session)
    snapshot_data = snapshot(session)

    session = start(session)
    session = stop(session)

    assert len(session._events) == 5

def test_snapshot_requires_active():
    """Verify snapshot works on active sessions."""
    session = create_session("test-2")
    session = initialize(session)
    session = start(session)

    # Should work on running
    data = snapshot(session)
    assert data["session_id"] == "test-2"

    session = pause(session)
    # Should work on paused
    data = snapshot(session)
    assert data["session_id"] == "test-2"
```

### Property-Based Testing

Use hypothesis to verify state machine properties:

```python
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

class SessionStateMachine(RuleBasedStateMachine):
    """Property-based state machine test."""

    def __init__(self):
        super().__init__()
        self.session = None
        self.state = "none"

    @rule()
    def create(self):
        if self.state == "none":
            self.session = create_session("test")
            self.state = "new"

    @rule()
    def do_initialize(self):
        if self.state == "new":
            self.session = initialize(self.session)
            self.state = "initialized"

    @rule()
    def do_start(self):
        if self.state in ("initialized", "paused"):
            self.session = start(self.session)
            self.state = "running"

    @rule()
    def do_pause(self):
        if self.state == "running":
            self.session = pause(self.session)
            self.state = "paused"

    @rule()
    def do_stop(self):
        if self.state in ("running", "paused"):
            self.session = stop(self.session)
            self.state = "stopped"

    @invariant()
    def events_only_grow(self):
        if self.session is not None:
            # Events list only grows (never shrinks)
            pass

TestSessionStateMachine = SessionStateMachine.TestCase
```

## Limitations

### No Runtime Enforcement

Phantom types are purely compile-time. A determined developer can bypass them:

```python
# Bypassing phantom types (DON'T DO THIS)
bad_session: Session[Running] = create_session("bad")  # type: ignore
dispatch(bad_session, event)  # Runtime error: session not actually running
```

**Mitigation**: Code review, linting for `type: ignore` comments, and DbC
runtime checks for critical invariants.

### No Transition Validity Enforcement

The type system doesn't prevent constructing invalid phantom types directly:

```python
# Nothing stops this at the type level
def bad_factory() -> Session[Running]:
    return Session("oops")  # type: ignore  # Lies about state
```

**Mitigation**: Smart constructors must be the only way to create typed
instances. Enforce via code review and module-level `__all__` exports.

### Increased API Surface

Each state transition requires a function, increasing API complexity:

```python
# Without phantom types: 1 class, 4 methods
# With phantom types: 1 class, 4 state types, 4+ factory functions
```

**Mitigation**: Reserve phantom types for critical state machines where
compile-time safety justifies the complexity.

### Type Inference Challenges

Complex phantom type hierarchies can confuse type inference:

```python
def process(session: Session[Running] | Session[Paused]) -> Session[Stopped]:
    # Type narrowing may not work as expected
    return stop(session)  # May need explicit annotation
```

**Mitigation**: Keep phantom type hierarchies shallow. Use explicit type
annotations when inference fails.

### No Support for Dependent Types

Python's type system cannot express relationships like "list length > 0":

```python
# Cannot express: NonEmptyList[T] where len > 0
# Must use runtime checks or separate NonEmpty type
```

**Mitigation**: Combine phantom types with DbC for complex constraints.

## When to Use Phantom Types

### Good Fit

- **Critical state machines**: Connection lifecycle, transaction states,
  authentication flows where wrong-state bugs cause security issues or data
  corruption.
- **Resource management**: Ensuring resources are acquired before use and
  released exactly once.
- **Protocol enforcement**: Multi-step protocols where operations must occur in
  order (handshake → authenticate → operate → close).
- **Capability-based security**: Encoding permissions in types to prevent
  unauthorized operations at compile time.

### Poor Fit

- **Simple boolean flags**: If state is just open/closed with one transition,
  phantom types add unnecessary complexity.
- **Frequently changing states**: If state changes on every operation, phantom
  types create excessive type annotation noise.
- **Runtime-determined states**: If valid transitions depend on runtime data
  the type checker cannot see, phantom types provide false confidence.
- **Prototyping**: When exploring a design, phantom types slow iteration. Add
  them once the state machine stabilizes.

## Migration Guide

### Adding Phantom Types to Existing Code

1. **Identify the state machine**: Document current states and valid
   transitions.

2. **Define phantom state types**: Create empty classes for each state.

3. **Add Generic parameter**: Make the class `Generic[S]` where `S` is the
   state type variable.

4. **Create smart constructors**: Write factory functions for each transition.
   Initially, these can wrap existing methods.

5. **Update call sites**: Change direct construction to smart constructor calls.
   The type checker will flag invalid transitions.

6. **Remove runtime state checks**: Once phantom types are in place, redundant
   `if self.state == ...` checks can often be removed.

### Example Migration

Before:

```python
class Worker:
    def __init__(self):
        self.state = "idle"

    def start(self):
        if self.state != "idle":
            raise ValueError("Cannot start non-idle worker")
        self.state = "running"

    def stop(self):
        if self.state != "running":
            raise ValueError("Cannot stop non-running worker")
        self.state = "stopped"
```

After:

```python
class Idle: ...
class Running: ...
class Stopped: ...

S = TypeVar("S")

class Worker(Generic[S]):
    pass

def create_worker() -> Worker[Idle]:
    return Worker()

def start_worker(w: Worker[Idle]) -> Worker[Running]:
    return w  # type: ignore[return-value]

def stop_worker(w: Worker[Running]) -> Worker[Stopped]:
    return w  # type: ignore[return-value]
```

## See Also

- [specs/SESSIONS.md](SESSIONS.md) - Session lifecycle and event system
- [specs/DBC.md](DBC.md) - Design-by-contract for runtime invariants
- [specs/DATACLASSES.md](DATACLASSES.md) - Frozen dataclass patterns
- [specs/LIFECYCLE.md](LIFECYCLE.md) - Loop lifecycle management
- [specs/FORMAL_VERIFICATION.md](FORMAL_VERIFICATION.md) - TLA+ for state
  machine verification
