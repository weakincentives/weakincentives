# Protocol Contracts Specification

## Purpose

This document specifies protocol contracts (session types) for `weakincentives`.
Protocol contracts verify that sequences of operations follow a declared
protocol, catching "wrong order" bugs at the earliest possible moment. These
bugs are notoriously difficult to test for because they require specific call
sequences that often slip through standard unit tests.

A protocol like `init → (process+ → checkpoint)* → finalize` declares that:

- `init` must be called first
- `process` must be called at least once between `init` and `checkpoint`
- The `process+ → checkpoint` sequence can repeat zero or more times
- `finalize` must come last

Any violation is caught statically (via type checking) or at the first wrong
call (via runtime enforcement). This is transformative for APIs with ordering
requirements.

## Guiding Principles

- **Complement existing DbC**: Protocol contracts extend the design-by-contract
  system. `@require` and `@ensure` validate individual calls; `@protocol`
  validates call sequences across an object's lifetime.
- **Declarative protocols**: Protocol definitions use a concise grammar that
  maps directly to finite state machines. No manual state tracking required.
- **Dual enforcement**: Static type checking catches violations at development
  time; runtime checking catches dynamic violations in tests and (optionally)
  production.
- **Zero-cost default**: Like DbC, runtime enforcement is disabled in
  production unless explicitly enabled. Protocols remain documentation and
  type-level contracts.
- **Clear diagnostics**: Violations report the expected state, actual state,
  attempted operation, and the protocol definition.

## Goals

- Provide a `@protocol` class decorator that defines valid operation sequences
  using a concise grammar.
- Support common patterns: sequences (`→`), alternatives (`|`), optionals
  (`?`), repetition (`*`, `+`), and grouping (`()`).
- Enable static verification via phantom types that encode protocol state.
- Provide runtime enforcement that activates in tests (via `dbc_active()`).
- Integrate with existing `weakincentives` patterns: sessions, resources,
  adapters.
- Detect violations at the earliest possible point with actionable diagnostics.

## Non-Goals

- Full session types as in academic literature (binary session types, linear
  types). The scope is practical protocol enforcement for Python APIs.
- Compile-time verification beyond what pyright/mypy can express. Static
  checking is best-effort via type narrowing.
- Distributed protocol verification. Protocols apply to single objects.
- Automatic protocol inference. Developers declare protocols explicitly.

## Protocol Grammar

Protocols are defined using a simple grammar:

```
protocol := sequence
sequence := term ('→' term)*
term     := factor quantifier?
factor   := name | '(' sequence ')' | alternatives
alternatives := factor ('|' factor)+
quantifier := '?' | '*' | '+'
name     := identifier
```

### Operators

| Operator | Meaning                       | Example                |
| -------- | ----------------------------- | ---------------------- |
| `→`      | Sequence (then)               | `init → run → close`   |
| `\|`     | Alternative (or)              | `commit \| rollback`   |
| `?`      | Optional (zero or one)        | `validate?`            |
| `*`      | Kleene star (zero or more)    | `process*`             |
| `+`      | Kleene plus (one or more)     | `process+`             |
| `()`     | Grouping                      | `(read \| write)*`     |

### Example Protocols

```python
# Database transaction
"begin → (execute)* → (commit | rollback)"

# Authentication flow
"challenge → response → verify"

# Batch processing pipeline
"init → (process+ → checkpoint)* → finalize"

# Multi-phase initialization
"configure → validate? → start → (run → pause?)* → stop"

# File handle
"open → (read | write | seek)* → close"

# Iterator protocol
"(next)* → close?"
```

## Core Types

### Protocol Definition

```python
from dataclasses import dataclass
from typing import FrozenSet

@dataclass(frozen=True, slots=True)
class ProtocolDef:
    """Compiled protocol definition."""

    name: str
    """Protocol identifier for diagnostics."""

    spec: str
    """Original protocol specification string."""

    states: FrozenSet[str]
    """All valid states in the protocol."""

    initial_state: str
    """Starting state (always 'START')."""

    final_states: FrozenSet[str]
    """Accepting states (protocol completed)."""

    transitions: Mapping[tuple[str, str], str]
    """State machine: (current_state, operation) → next_state."""

    @staticmethod
    def compile(spec: str, *, name: str = "Protocol") -> "ProtocolDef":
        """Parse protocol specification into state machine.

        Raises:
            ProtocolSyntaxError: Invalid protocol specification.
        """
        ...

    def accepts(self, operation: str, *, current_state: str) -> str | None:
        """Return next state if operation is valid, None otherwise."""
        ...

    def valid_operations(self, *, current_state: str) -> FrozenSet[str]:
        """Return operations valid from current state."""
        ...

    def is_final(self, state: str) -> bool:
        """Return True if state is an accepting state."""
        ...
```

### Protocol State Tracker

```python
@dataclass(slots=True)
class ProtocolState:
    """Tracks current state in a protocol."""

    protocol: ProtocolDef
    """The protocol definition."""

    current: str = "START"
    """Current state in the state machine."""

    history: tuple[str, ...] = ()
    """Sequence of operations performed."""

    def transition(self, operation: str) -> "ProtocolState":
        """Attempt transition; raises ProtocolViolationError if invalid."""
        ...

    def can_transition(self, operation: str) -> bool:
        """Return True if operation is valid from current state."""
        ...

    @property
    def is_complete(self) -> bool:
        """Return True if protocol is in a final state."""
        ...

    @property
    def valid_operations(self) -> FrozenSet[str]:
        """Return operations valid from current state."""
        ...
```

## The `@protocol` Decorator

### Basic Usage

```python
from weakincentives.dbc import protocol

@protocol("begin → (execute)* → (commit | rollback)")
class Transaction:
    """Database transaction with protocol enforcement."""

    def begin(self) -> None:
        """Start the transaction."""
        ...

    def execute(self, sql: str) -> Result:
        """Execute a SQL statement."""
        ...

    def commit(self) -> None:
        """Commit the transaction."""
        ...

    def rollback(self) -> None:
        """Roll back the transaction."""
        ...
```

### Decorator Implementation

```python
def protocol(
    spec: str,
    *,
    name: str | None = None,
    strict_completion: bool = True,
) -> Callable[[type[T]], type[T]]:
    """Enforce operation sequencing on a class.

    Args:
        spec: Protocol specification using the protocol grammar.
        name: Protocol name for diagnostics (defaults to class name).
        strict_completion: If True, raise on garbage collection of
            incomplete protocol. Set False for optional cleanup patterns.

    Returns:
        Class decorator that adds protocol enforcement.

    The decorator:
    1. Parses the protocol specification into a state machine.
    2. Wraps `__init__` to initialize protocol state.
    3. Wraps methods named in the protocol to validate transitions.
    4. Optionally wraps `__del__` to verify protocol completion.

    When `dbc_active()` is False, the decorator is a no-op (returns
    the class unchanged).
    """
    ...
```

### Method Wrapping

Methods named in the protocol are wrapped to:

1. Check if the operation is valid from the current state
2. If invalid, raise `ProtocolViolationError`
3. If valid, execute the method and transition state
4. Update the operation history

```python
# Pseudo-implementation of method wrapper
def wrapped_method(self, *args, **kwargs):
    if dbc_active():
        state = getattr(self, "__protocol_state__")
        if not state.can_transition(method_name):
            raise ProtocolViolationError(
                protocol=state.protocol,
                current_state=state.current,
                attempted=method_name,
                valid_operations=state.valid_operations,
                history=state.history,
            )
    result = original_method(self, *args, **kwargs)
    if dbc_active():
        new_state = state.transition(method_name)
        setattr(self, "__protocol_state__", new_state)
    return result
```

### Completion Checking

With `strict_completion=True` (default), the decorator adds a `__del__` wrapper
that verifies the protocol reached a final state:

```python
def __del__(self):
    if dbc_active():
        state = getattr(self, "__protocol_state__", None)
        if state is not None and not state.is_complete:
            warnings.warn(
                ProtocolIncompleteWarning(
                    protocol=state.protocol,
                    final_state=state.current,
                    history=state.history,
                )
            )
    if original_del is not None:
        original_del(self)
```

## Error Types

```python
class ProtocolError(WinkError, RuntimeError):
    """Base class for protocol contract errors."""


@dataclass(frozen=True, slots=True)
class ProtocolSyntaxError(ProtocolError, ValueError):
    """Invalid protocol specification syntax."""

    spec: str
    position: int
    message: str

    def __str__(self) -> str:
        pointer = " " * self.position + "^"
        return f"Protocol syntax error at position {self.position}:\n{self.spec}\n{pointer}\n{self.message}"


@dataclass(frozen=True, slots=True)
class ProtocolViolationError(ProtocolError):
    """Operation called in wrong protocol state."""

    protocol: ProtocolDef
    current_state: str
    attempted: str
    valid_operations: FrozenSet[str]
    history: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"Protocol '{self.protocol.name}' violation: "
            f"cannot call '{self.attempted}' in state '{self.current_state}'. "
            f"Valid operations: {sorted(self.valid_operations)}. "
            f"History: {' → '.join(self.history) or '(none)'}"
        )


class ProtocolIncompleteWarning(UserWarning):
    """Protocol did not reach a final state before object destruction."""

    protocol: ProtocolDef
    final_state: str
    history: tuple[str, ...]
```

## Static Type Checking

For enhanced static verification, protocol-decorated classes can use phantom
types to encode state:

### Phantom State Types

```python
from typing import Generic, TypeVar

# State phantom types
class Uninitialized: ...
class Ready: ...
class Running: ...
class Closed: ...

S = TypeVar("S")

class Connection(Generic[S]):
    """Connection with phantom state type."""

    @staticmethod
    def create() -> "Connection[Uninitialized]":
        ...

    def connect(self: "Connection[Uninitialized]") -> "Connection[Ready]":
        ...

    def execute(self: "Connection[Ready]", sql: str) -> "Connection[Ready]":
        ...

    def close(self: "Connection[Ready]") -> "Connection[Closed]":
        ...
```

### Protocol-Aware Type Narrowing

The `@protocol` decorator can generate overload signatures that encode state
transitions:

```python
from typing import overload

@protocol("connect → (execute)* → close")
class Connection:
    @overload
    def connect(self: "Connection[Uninitialized]") -> "Connection[Ready]": ...
    @overload
    def connect(self: "Connection[Ready]") -> Never: ...
    @overload
    def connect(self: "Connection[Closed]") -> Never: ...

    def connect(self) -> "Connection[Ready]":
        # Implementation
        ...
```

Type checkers will flag violations:

```python
conn = Connection.create()  # Connection[Uninitialized]
conn.execute("SELECT 1")    # Type error: execute requires Connection[Ready]
conn.connect()              # OK: Connection[Ready]
conn.execute("SELECT 1")    # OK
conn.close()                # OK: Connection[Closed]
conn.execute("SELECT 1")    # Type error: execute requires Connection[Ready]
```

## Runtime Behavior

### Enforcement Modes

Protocol enforcement follows the same pattern as DbC:

```python
# Production: decorators are no-ops
assert not dbc_active()
tx = Transaction()
tx.execute("SELECT 1")  # No error (protocol not enforced)
tx.commit()             # Works fine

# Tests: decorators enforce protocols
with dbc_enabled():
    tx = Transaction()
    tx.execute("SELECT 1")  # ProtocolViolationError: must call begin first
```

### Thread Safety

Protocol state is stored per-instance in `__protocol_state__`. Access is not
thread-safe by default. For thread-safe protocols:

```python
@protocol("acquire → (use)* → release", thread_safe=True)
class Lock:
    """Thread-safe protocol enforcement."""
    ...
```

With `thread_safe=True`, state transitions use a per-instance lock.

## Integration with Existing DbC

Protocol contracts compose with other DbC decorators:

```python
@protocol("begin → (execute)* → (commit | rollback)")
@invariant(lambda self: self._connection is not None or self._state == "closed")
class Transaction:

    @require(lambda self: not self._readonly, "Cannot execute in readonly mode")
    @ensure(lambda self, result: result.row_count >= 0)
    def execute(self, sql: str) -> Result:
        ...

    @require(lambda self: self._pending_count > 0, "Nothing to commit")
    def commit(self) -> None:
        ...
```

Enforcement order:

1. Protocol state check (can this operation be called?)
2. Preconditions (`@require`)
3. Method execution
4. Postconditions (`@ensure`)
5. Protocol state transition
6. Invariants (`@invariant`)

## Use Cases

### Database Transactions

```python
@protocol("begin → (execute)* → (commit | rollback)")
class Transaction:
    """Prevents data corruption from uncommitted transactions."""

    def begin(self) -> None:
        self._conn.execute("BEGIN")

    def execute(self, sql: str, params: tuple = ()) -> Cursor:
        return self._conn.execute(sql, params)

    def commit(self) -> None:
        self._conn.execute("COMMIT")

    def rollback(self) -> None:
        self._conn.execute("ROLLBACK")


# Usage
tx = Transaction(conn)
tx.begin()
tx.execute("INSERT INTO users VALUES (?)", ("alice",))
tx.commit()  # Correct sequence

# Violation caught:
tx = Transaction(conn)
tx.execute("INSERT ...")  # ProtocolViolationError: must begin first
```

### Authentication Flows

```python
@protocol("challenge → response → verify")
class AuthFlow:
    """Prevents security vulnerabilities from skipped verification."""

    def challenge(self) -> bytes:
        """Generate and store challenge."""
        self._challenge = secrets.token_bytes(32)
        return self._challenge

    def response(self, client_response: bytes) -> None:
        """Store client response for verification."""
        self._response = client_response

    def verify(self) -> bool:
        """Verify response matches expected value."""
        expected = hmac.digest(self._secret, self._challenge, "sha256")
        return hmac.compare_digest(expected, self._response)


# Violation caught:
auth = AuthFlow(secret)
auth.verify()  # ProtocolViolationError: must call challenge, then response
```

### Batch Processing Pipelines

```python
@protocol("init → (process+ → checkpoint)* → finalize")
class BatchProcessor:
    """Ensures checkpoints follow processing and finalization happens."""

    def init(self, config: Config) -> None:
        """Initialize processing state."""
        self._buffer = []
        self._checkpoint_id = 0

    def process(self, item: Item) -> None:
        """Process a single item."""
        result = transform(item)
        self._buffer.append(result)

    def checkpoint(self) -> int:
        """Persist buffered results."""
        save_batch(self._buffer, self._checkpoint_id)
        self._checkpoint_id += 1
        self._buffer.clear()
        return self._checkpoint_id

    def finalize(self) -> Summary:
        """Complete processing and return summary."""
        if self._buffer:
            self.checkpoint()
        return Summary(checkpoints=self._checkpoint_id)


# Violation caught:
proc = BatchProcessor()
proc.init(config)
proc.checkpoint()  # ProtocolViolationError: must process at least once
```

### Multi-Phase Initialization

```python
@protocol("configure → validate? → start → (run → pause?)* → stop")
class Service:
    """Prevents running before configuration or double-start."""

    def configure(self, config: ServiceConfig) -> None:
        self._config = config

    def validate(self) -> list[str]:
        """Optional validation step."""
        return self._config.validate()

    def start(self) -> None:
        self._running = True
        self._server.start()

    def run(self) -> None:
        """Process pending requests."""
        self._server.process_batch()

    def pause(self) -> None:
        """Optional pause between run cycles."""
        time.sleep(self._config.pause_interval)

    def stop(self) -> None:
        self._running = False
        self._server.stop()
```

### Resource Lifecycle (Context Managers)

```python
@protocol("__enter__ → (read | write)* → __exit__")
class ManagedFile:
    """File handle with protocol-enforced lifecycle."""

    def __enter__(self) -> "ManagedFile":
        self._handle = open(self._path, self._mode)
        return self

    def read(self, size: int = -1) -> bytes:
        return self._handle.read(size)

    def write(self, data: bytes) -> int:
        return self._handle.write(data)

    def __exit__(self, *args) -> None:
        self._handle.close()


# Context manager ensures protocol completion
with ManagedFile("data.bin", "rb") as f:
    data = f.read()
# __exit__ called automatically
```

## Session Integration

Protocol contracts integrate with WINK sessions for agent-scoped validation:

```python
from weakincentives.runtime import Session
from weakincentives.dbc import protocol, session_scoped

@protocol("plan → (execute → evaluate)* → report")
@session_scoped  # Protocol state stored in session, not instance
class AgentWorkflow:
    """Agent workflow with session-tracked protocol state."""

    def __init__(self, session: Session):
        self._session = session

    def plan(self, objective: str) -> Plan:
        ...

    def execute(self, step: PlanStep) -> ExecutionResult:
        ...

    def evaluate(self, result: ExecutionResult) -> Evaluation:
        ...

    def report(self) -> FinalReport:
        ...
```

With `@session_scoped`, protocol state persists across session snapshots and
can be restored on rollback.

## Implementation Sketch

### Protocol Parser

```python
# src/weakincentives/dbc/protocol_parser.py

from dataclasses import dataclass
from enum import Enum, auto

class TokenType(Enum):
    NAME = auto()
    ARROW = auto()       # →
    PIPE = auto()        # |
    QUESTION = auto()    # ?
    STAR = auto()        # *
    PLUS = auto()        # +
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    EOF = auto()


@dataclass(frozen=True, slots=True)
class Token:
    type: TokenType
    value: str
    position: int


def tokenize(spec: str) -> Iterator[Token]:
    """Tokenize protocol specification."""
    ...


def parse(tokens: Iterator[Token]) -> ProtocolAST:
    """Parse token stream into AST."""
    ...


def compile_to_nfa(ast: ProtocolAST) -> NFA:
    """Compile AST to NFA using Thompson's construction."""
    ...


def nfa_to_dfa(nfa: NFA) -> DFA:
    """Convert NFA to DFA using subset construction."""
    ...


def minimize_dfa(dfa: DFA) -> ProtocolDef:
    """Minimize DFA and return ProtocolDef."""
    ...
```

### Decorator Implementation

```python
# src/weakincentives/dbc/__init__.py (additions)

def protocol(
    spec: str,
    *,
    name: str | None = None,
    strict_completion: bool = True,
    thread_safe: bool = False,
) -> Callable[[type[T]], type[T]]:
    """Protocol contract decorator."""

    protocol_def = ProtocolDef.compile(spec, name=name or "Protocol")

    def decorator(cls: type[T]) -> type[T]:
        if not dbc_active():
            return cls

        # Extract operation names from protocol
        operations = protocol_def.states - {"START", "ACCEPT"}

        # Wrap __init__ to initialize state
        original_init = cls.__init__

        @wraps(original_init)
        def init_wrapper(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            state = ProtocolState(protocol=protocol_def)
            object.__setattr__(self, "__protocol_state__", state)
            if thread_safe:
                object.__setattr__(self, "__protocol_lock__", threading.RLock())

        cls.__init__ = init_wrapper

        # Wrap each operation method
        for op_name in operations:
            if not hasattr(cls, op_name):
                continue
            original_method = getattr(cls, op_name)
            wrapped = _wrap_protocol_method(
                original_method, op_name, thread_safe
            )
            setattr(cls, op_name, wrapped)

        # Add completion check in __del__ if strict
        if strict_completion:
            cls.__del__ = _make_del_wrapper(cls, protocol_def)

        return cls

    return decorator
```

## Testing Strategy

### Unit Tests

```python
def test_protocol_parsing():
    """Verify protocol grammar parsing."""
    proto = ProtocolDef.compile("a → b* → (c | d)")
    assert "a" in proto.valid_operations(current_state="START")
    assert proto.accepts("a", current_state="START") is not None


def test_protocol_enforcement():
    """Verify runtime enforcement."""
    @protocol("init → run → stop")
    class Worker:
        def init(self): pass
        def run(self): pass
        def stop(self): pass

    with dbc_enabled():
        w = Worker()
        with pytest.raises(ProtocolViolationError):
            w.run()  # Must init first

        w.init()
        w.run()
        w.stop()  # OK


def test_protocol_alternatives():
    """Verify alternative paths."""
    @protocol("begin → (commit | rollback)")
    class Tx:
        def begin(self): pass
        def commit(self): pass
        def rollback(self): pass

    with dbc_enabled():
        tx = Tx()
        tx.begin()
        tx.commit()  # OK

        tx2 = Tx()
        tx2.begin()
        tx2.rollback()  # Also OK


def test_protocol_repetition():
    """Verify Kleene operators."""
    @protocol("open → read* → close")
    class File:
        def open(self): pass
        def read(self): pass
        def close(self): pass

    with dbc_enabled():
        f = File()
        f.open()
        f.close()  # OK: read* allows zero reads

        f2 = File()
        f2.open()
        f2.read()
        f2.read()
        f2.read()
        f2.close()  # OK: multiple reads


def test_protocol_completion_warning():
    """Verify incomplete protocol warning."""
    @protocol("start → finish", strict_completion=True)
    class Task:
        def start(self): pass
        def finish(self): pass

    with dbc_enabled():
        with pytest.warns(ProtocolIncompleteWarning):
            t = Task()
            t.start()
            del t  # Never finished


def test_protocol_disabled_in_production():
    """Verify no-op behavior when DbC inactive."""
    @protocol("a → b")
    class P:
        def a(self): pass
        def b(self): pass

    with dbc_enabled(False):
        p = P()
        p.b()  # No error when DbC disabled
        p.a()
```

### Integration Tests

```python
def test_protocol_with_invariants():
    """Verify protocol + invariant composition."""
    @protocol("lock → (read | write)* → unlock")
    @invariant(lambda self: self._count >= 0)
    class Counter:
        def __init__(self):
            self._count = 0
            self._locked = False

        def lock(self):
            self._locked = True

        def read(self) -> int:
            return self._count

        def write(self, value: int):
            self._count = value

        def unlock(self):
            self._locked = False

    with dbc_enabled():
        c = Counter()
        c.lock()
        c.write(5)
        assert c.read() == 5
        c.unlock()


def test_protocol_with_session():
    """Verify session-scoped protocol state."""
    @protocol("plan → execute → complete")
    @session_scoped
    class Workflow:
        ...

    session = Session(bus=InProcessDispatcher())
    w = Workflow(session=session)

    with dbc_enabled():
        w.plan()
        snapshot = session.snapshot()

        w.execute()
        session.restore(snapshot)  # Roll back to after plan

        w.execute()  # OK: protocol state also rolled back
        w.complete()
```

## Limitations

- **Single-object scope**: Protocols apply to one object. Cross-object
  protocols require explicit coordination.
- **No async support**: Protocol enforcement is synchronous. Async methods
  work but concurrent calls may race.
- **Dynamic methods**: Protocols only track methods that exist at decoration
  time. Dynamically added methods are not tracked.
- **Inheritance**: Subclasses can extend protocols but not override them
  safely. Use composition for protocol variation.
- **Performance**: Each protocol method incurs state lookup and transition
  cost when DbC is active. Keep protocols coarse-grained for performance-
  critical code.

## Future Considerations

- **Protocol composition**: Combine multiple protocols on one class.
- **Parametric protocols**: Protocols with parameters (e.g., `read(n)` with
  constraints on `n`).
- **Temporal properties**: Beyond sequences, express timing constraints.
- **Protocol visualization**: Generate state machine diagrams from protocols.
- **IDE integration**: Language server support for protocol-aware completions.
