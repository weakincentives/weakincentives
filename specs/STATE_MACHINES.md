# State Machine Transition Enforcement Specification

## Purpose

This document specifies a state machine enforcement framework for
`weakincentives`. The framework enables declaring valid states and transitions
explicitly, then enforcing them automatically at runtime. State machine bugs—
calling methods in invalid states, illegal transitions, forgotten state
updates—are among the most insidious because they often manifest as rare,
hard-to-reproduce failures in production. This specification aims to convert
these runtime mysteries into immediate failures with clear error messages.

The framework is particularly high-impact for:

- **Resources with lifecycles** (connections, sessions, transactions)
- **Protocols with ordering requirements** (handshakes, multi-phase operations)
- **Objects that progress through phases** (builders, workflows, pipelines)

Common bugs caught:

- "Started a transaction twice"
- "Used connection after close"
- "Skipped required initialization step"
- "Called `send()` before `connect()`"
- "Finalized builder that was already built"

## Guiding Principles

- **Declarative over imperative**: State machines are declared via decorators
  and metadata, not scattered `if/else` checks throughout code.
- **Fail fast with clarity**: Invalid transitions raise immediately with
  context-rich error messages including current state, attempted transition,
  and valid alternatives.
- **Zero-cost production default**: Like the DbC framework, enforcement
  activates in tests by default. Production code pays no overhead unless
  explicitly opted in.
- **Composable with DbC**: State machine decorators compose naturally with
  `@require`, `@ensure`, and `@invariant`.
- **Static analysis friendly**: The declarative structure enables tooling to
  analyze call graphs and prove invalid transitions are unreachable.
- **Internal-facing**: These are contributor tools for catching bugs early,
  not public API contracts.

## Goals

- Provide decorators for declaring state machines on classes: `@state_machine`,
  `@transition`, `@in_state`, `@enters`, `@exits`.
- Support both enum-based and string-based state definitions.
- Enable compile-time analysis via extractable transition graphs.
- Integrate with the existing DbC activation mechanism (`dbc_active()`).
- Provide clear diagnostics naming the class, method, current state, attempted
  transition, and valid transitions.
- Support hierarchical/nested states for complex lifecycles.
- Enable transition guards (conditional transitions based on runtime values).

## Non-Goals

- Full statechart/UML state machine semantics (orthogonal regions, history
  states, etc.). The framework targets simple finite state machines.
- Automatic state persistence or serialization. State storage is the class's
  responsibility.
- Distributed state machine coordination. This is single-object, single-process.
- Public API stability. Like DbC, this is an internal tool.

## Core Concepts

### State

A state is a named condition an object can be in. States are typically defined
as an `Enum` for type safety, though string literals are supported for simpler
cases.

```python
from enum import Enum, auto

class ConnectionState(Enum):
    INITIAL = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    CLOSING = auto()
    CLOSED = auto()
```

### Transition

A transition is a valid state change triggered by a method call. Transitions
have:

- **Source state(s)**: The state(s) the object must be in before the method
- **Target state**: The state the object enters after the method completes
- **Guard** (optional): A predicate that must be true for the transition
- **Action**: The method body itself

### State Machine

A state machine is a class decorated with `@state_machine` that declares:

- The state variable (attribute name holding current state)
- The set of valid states
- The initial state
- Optionally, the state type (enum class)

## Decorator API

### `@state_machine`

Class decorator that registers state machine metadata and wraps methods for
enforcement.

```python
from weakincentives.dbc import state_machine, transition
from enum import Enum, auto

class ConnectionState(Enum):
    INITIAL = auto()
    CONNECTED = auto()
    CLOSED = auto()

@state_machine(
    state_var="_state",              # Attribute name holding state
    states=ConnectionState,          # Enum class or tuple of valid states
    initial=ConnectionState.INITIAL, # Initial state (set in __init__ wrapper)
)
class Connection:
    def __init__(self, host: str):
        self.host = host
        # _state automatically initialized to ConnectionState.INITIAL

    @transition(from_=ConnectionState.INITIAL, to=ConnectionState.CONNECTED)
    def connect(self) -> None:
        # ... establish connection ...
        pass

    @transition(from_=ConnectionState.CONNECTED, to=ConnectionState.CLOSED)
    def close(self) -> None:
        # ... close connection ...
        pass
```

**Parameters:**

- `state_var: str` - Name of the instance attribute holding current state
- `states: type[Enum] | tuple[str, ...]` - Valid states (enum class or strings)
- `initial: Enum | str` - Initial state, set after `__init__` completes
- `strict: bool = True` - If `True`, all public methods must declare transitions
  or be marked `@state_independent`

### `@transition`

Method decorator declaring a state transition.

```python
@transition(
    from_=ConnectionState.INITIAL,           # Source state(s)
    to=ConnectionState.CONNECTED,            # Target state
    guard=lambda self: self.host is not None, # Optional guard predicate
)
def connect(self) -> None:
    ...
```

**Parameters:**

- `from_: State | tuple[State, ...]` - Valid source state(s). Use `...`
  (Ellipsis) for "any state".
- `to: State` - Target state after successful completion.
- `guard: Callable[[Self], bool] | None` - Optional predicate; transition only
  valid when guard returns `True`.
- `on_error: State | None` - State to enter if method raises. Defaults to
  staying in source state.

**Behavior:**

1. Before method execution: verify current state is in `from_` and guard passes
2. Execute method body
3. After successful completion: set state to `to`
4. On exception: set state to `on_error` if specified, else leave unchanged

### `@in_state`

Method decorator requiring specific state(s) without causing a transition.
Useful for query methods or operations that don't change state.

```python
@in_state(ConnectionState.CONNECTED)
def send(self, data: bytes) -> int:
    """Send data. Only valid when connected."""
    ...

@in_state(ConnectionState.CONNECTED, ConnectionState.CLOSING)
def pending_bytes(self) -> int:
    """Check pending bytes. Valid when connected or closing."""
    ...
```

**Parameters:**

- `*states: State` - One or more valid states for this method

### `@enters`

Shorthand for transitions that can occur from any state.

```python
@enters(ConnectionState.CLOSED)  # Equivalent to @transition(from_=..., to=CLOSED)
def force_close(self) -> None:
    """Emergency close from any state."""
    ...
```

### `@exits`

Shorthand for transitions from a specific state to any valid next state.
Requires explicit state assignment in method body.

```python
@exits(ConnectionState.CONNECTING)
def on_connect_result(self, success: bool) -> None:
    """Handle connection result. Must explicitly set next state."""
    if success:
        self._state = ConnectionState.CONNECTED
    else:
        self._state = ConnectionState.INITIAL
```

### `@state_independent`

Marks a method as not participating in state machine enforcement. Required
when `strict=True` for methods that genuinely don't depend on state.

```python
@state_independent
def get_host(self) -> str:
    """Return configured host. Valid in any state."""
    return self.host
```

## State Variable Access

### Reading State

The current state is always readable via the state variable:

```python
conn = Connection("localhost")
assert conn._state == ConnectionState.INITIAL

conn.connect()
assert conn._state == ConnectionState.CONNECTED
```

### Writing State

Direct state assignment is blocked by default when enforcement is active.
All state changes must go through declared transitions.

```python
conn._state = ConnectionState.CLOSED  # Raises IllegalStateAssignmentError
```

To allow internal state management (e.g., in `@exits` handlers), use
`@allow_state_assignment`:

```python
@exits(ConnectionState.CONNECTING)
@allow_state_assignment
def handle_result(self, success: bool) -> None:
    self._state = ConnectionState.CONNECTED if success else ConnectionState.INITIAL
```

## Error Types

### `StateError`

Base class for all state machine errors.

```python
class StateError(WinkError, RuntimeError):
    """Base class for state machine enforcement errors."""
    pass
```

### `InvalidStateError`

Raised when a method is called in an invalid state.

```python
@dataclass(frozen=True)
class InvalidStateError(StateError):
    """Method called in invalid state."""
    cls: type
    method: str
    current_state: object
    valid_states: tuple[object, ...]

    def __str__(self) -> str:
        valid = ", ".join(str(s) for s in self.valid_states)
        return (
            f"{self.cls.__name__}.{self.method}() requires state in "
            f"[{valid}], but current state is {self.current_state}"
        )
```

### `InvalidTransitionError`

Raised when a transition is not allowed.

```python
@dataclass(frozen=True)
class InvalidTransitionError(StateError):
    """Transition not allowed from current state."""
    cls: type
    method: str
    current_state: object
    target_state: object
    valid_transitions: tuple[tuple[object, object], ...]

    def __str__(self) -> str:
        return (
            f"{self.cls.__name__}.{self.method}() cannot transition from "
            f"{self.current_state} to {self.target_state}. "
            f"Valid transitions from {self.current_state}: "
            f"{self._format_transitions()}"
        )
```

### `TransitionGuardError`

Raised when a transition guard fails.

```python
@dataclass(frozen=True)
class TransitionGuardError(StateError):
    """Transition guard predicate returned False."""
    cls: type
    method: str
    current_state: object
    target_state: object
    guard: Callable[..., bool]
```

### `IllegalStateAssignmentError`

Raised on direct state variable assignment when enforcement is active.

```python
@dataclass(frozen=True)
class IllegalStateAssignmentError(StateError):
    """Direct state assignment outside allowed context."""
    cls: type
    attempted_state: object
```

### `IncompleteStateMachineError`

Raised during class decoration if `strict=True` and a public method lacks
state annotations.

```python
@dataclass(frozen=True)
class IncompleteStateMachineError(StateError, TypeError):
    """Public method missing state machine annotation."""
    cls: type
    method: str
```

## Runtime Behavior

### Activation

State machine enforcement shares the DbC activation mechanism:

```python
from weakincentives.dbc import dbc_active, dbc_enabled

# Enforcement active in tests via pytest plugin
# Manual activation:
with dbc_enabled():
    conn.connect()  # Enforcement active
```

When `dbc_active()` returns `False`, all decorators are no-ops except for
the initial state assignment in `__init__`.

### Enforcement Sequence

For `@transition(from_=A, to=B, guard=g)`:

```
1. [dbc_active?] → No → call method directly
                   ↓ Yes
2. [state == A?] → No → raise InvalidStateError
                   ↓ Yes
3. [guard(self)?] → No → raise TransitionGuardError
                   ↓ Yes (or no guard)
4. Execute method body
5. [exception?] → Yes → set state to on_error (if specified)
                 ↓ No
6. Set state to B
7. Return result
```

### Thread Safety

State machine enforcement is **not thread-safe** by default. The state variable
is a regular attribute subject to race conditions. For thread-safe state
machines, the class must implement its own synchronization:

```python
@state_machine(state_var="_state", states=State, initial=State.INITIAL)
class ThreadSafeResource:
    def __init__(self):
        self._lock = threading.Lock()

    @transition(from_=State.INITIAL, to=State.READY)
    def initialize(self) -> None:
        with self._lock:
            # ... initialization ...
            pass
```

## Transition Graph Extraction

The `@state_machine` decorator attaches metadata enabling static analysis:

```python
from weakincentives.dbc import extract_state_machine

@state_machine(state_var="_state", states=ConnectionState, initial=ConnectionState.INITIAL)
class Connection:
    @transition(from_=ConnectionState.INITIAL, to=ConnectionState.CONNECTED)
    def connect(self) -> None: ...

    @transition(from_=ConnectionState.CONNECTED, to=ConnectionState.CLOSED)
    def close(self) -> None: ...

# Extract transition graph
sm = extract_state_machine(Connection)
print(sm.states)       # frozenset({INITIAL, CONNECTED, CLOSED})
print(sm.initial)      # ConnectionState.INITIAL
print(sm.transitions)  # {(INITIAL, CONNECTED, 'connect'), (CONNECTED, CLOSED, 'close')}

# Validate reachability
unreachable = sm.unreachable_states()  # States with no path from initial
dead_ends = sm.terminal_states()       # States with no outgoing transitions

# Export for visualization
dot = sm.to_graphviz()  # DOT format for Graphviz
mermaid = sm.to_mermaid()  # Mermaid diagram syntax
```

### StateMachineSpec

The extracted specification:

```python
@dataclass(frozen=True)
class StateMachineSpec:
    """Extracted state machine specification."""
    cls: type
    state_var: str
    states: frozenset[object]
    initial: object
    transitions: frozenset[TransitionSpec]

    def unreachable_states(self) -> frozenset[object]:
        """Return states not reachable from initial state."""
        ...

    def terminal_states(self) -> frozenset[object]:
        """Return states with no outgoing transitions."""
        ...

    def valid_transitions_from(self, state: object) -> frozenset[TransitionSpec]:
        """Return valid transitions from given state."""
        ...

    def to_graphviz(self) -> str:
        """Export as Graphviz DOT format."""
        ...

    def to_mermaid(self) -> str:
        """Export as Mermaid state diagram."""
        ...


@dataclass(frozen=True)
class TransitionSpec:
    """Specification of a single transition."""
    from_states: frozenset[object]
    to_state: object
    method: str
    has_guard: bool
```

## Static Analysis Integration

### Type Narrowing

The framework provides type stubs enabling static analyzers to narrow types
based on state:

```python
# With proper type stubs, analyzers can track:
conn = Connection("localhost")
# conn._state is ConnectionState.INITIAL

conn.connect()
# conn._state is ConnectionState.CONNECTED

conn.send(b"data")  # Valid: in_state(CONNECTED)
conn.connect()      # Type error: invalid from CONNECTED
```

### Call Graph Analysis

Tools can analyze the transition graph to prove properties:

```python
# Example: verify all states are reachable
sm = extract_state_machine(Connection)
assert not sm.unreachable_states(), "Unreachable states detected"

# Example: verify no unintended terminal states
expected_terminals = {ConnectionState.CLOSED}
assert sm.terminal_states() == expected_terminals
```

### Integration with mypy/pyright

A mypy plugin can leverage transition metadata:

```python
# mypy.ini
[mypy]
plugins = weakincentives.dbc.mypy_plugin

# Enables warnings like:
# warning: Connection.send() only valid in state CONNECTED,
#          but may be called in state INITIAL
```

## Composition with DbC

State machine decorators compose with existing DbC decorators:

```python
from weakincentives.dbc import require, ensure, invariant, state_machine, transition

@invariant(lambda self: self._retries >= 0)
@state_machine(state_var="_state", states=State, initial=State.INITIAL)
class RetryingConnection:
    def __init__(self):
        self._retries = 0

    @require(lambda self, host: host is not None)
    @transition(from_=State.INITIAL, to=State.CONNECTING)
    def connect(self, host: str) -> None:
        ...

    @ensure(lambda self, result: result >= 0)
    @in_state(State.CONNECTED)
    def send(self, data: bytes) -> int:
        ...
```

Order of evaluation when stacked:

1. `@require` preconditions
2. `@in_state` / `@transition` state check
3. Method body
4. `@transition` state update
5. `@ensure` postconditions
6. `@invariant` checks (before and after, per invariant spec)

## Hierarchical States

For complex lifecycles, states can be nested:

```python
class TransactionState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    ACTIVE_READONLY = auto()   # Sub-state of ACTIVE
    ACTIVE_READWRITE = auto()  # Sub-state of ACTIVE
    COMMITTED = auto()
    ROLLED_BACK = auto()

@state_machine(
    state_var="_state",
    states=TransactionState,
    initial=TransactionState.IDLE,
    hierarchy={
        TransactionState.ACTIVE: {
            TransactionState.ACTIVE_READONLY,
            TransactionState.ACTIVE_READWRITE,
        },
    },
)
class Transaction:
    @transition(from_=TransactionState.IDLE, to=TransactionState.ACTIVE_READONLY)
    def begin(self, readonly: bool = True) -> None:
        ...

    @in_state(TransactionState.ACTIVE)  # Matches both sub-states
    def execute(self, sql: str) -> Result:
        ...

    @transition(from_=TransactionState.ACTIVE, to=TransactionState.COMMITTED)
    def commit(self) -> None:
        ...
```

The `hierarchy` parameter defines parent-child relationships. `@in_state` and
`@transition(from_=...)` match both the specified state and any sub-states.

## Example Use Cases

### Connection Pool

```python
class PoolState(Enum):
    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    DRAINING = auto()
    STOPPED = auto()

@state_machine(state_var="_state", states=PoolState, initial=PoolState.CREATED)
class ConnectionPool:
    @transition(from_=PoolState.CREATED, to=PoolState.STARTING)
    def start(self) -> None:
        """Begin pool initialization."""
        self._spawn_workers()

    @transition(from_=PoolState.STARTING, to=PoolState.RUNNING)
    def on_ready(self) -> None:
        """Called when all workers are ready."""
        pass

    @in_state(PoolState.RUNNING)
    def acquire(self) -> Connection:
        """Acquire a connection from the pool."""
        return self._get_available()

    @in_state(PoolState.RUNNING, PoolState.DRAINING)
    def release(self, conn: Connection) -> None:
        """Return a connection to the pool."""
        self._return_connection(conn)

    @transition(from_=PoolState.RUNNING, to=PoolState.DRAINING)
    def drain(self) -> None:
        """Stop accepting new requests, wait for in-flight."""
        self._stop_accepting()

    @transition(from_=PoolState.DRAINING, to=PoolState.STOPPED)
    def stop(self) -> None:
        """Final shutdown after draining."""
        self._shutdown_workers()
```

### Multi-Phase Builder

```python
class BuilderPhase(Enum):
    CONFIGURING = auto()
    VALIDATING = auto()
    BUILDING = auto()
    COMPLETE = auto()

@state_machine(state_var="_phase", states=BuilderPhase, initial=BuilderPhase.CONFIGURING)
class PipelineBuilder:
    @in_state(BuilderPhase.CONFIGURING)
    def add_stage(self, stage: Stage) -> Self:
        """Add a pipeline stage. Only during configuration."""
        self._stages.append(stage)
        return self

    @transition(from_=BuilderPhase.CONFIGURING, to=BuilderPhase.VALIDATING)
    def validate(self) -> Self:
        """Transition to validation phase."""
        return self

    @transition(
        from_=BuilderPhase.VALIDATING,
        to=BuilderPhase.BUILDING,
        guard=lambda self: len(self._stages) > 0,
    )
    def build(self) -> Pipeline:
        """Build the pipeline. Requires at least one stage."""
        return Pipeline(self._stages)
```

### Session with Transactions

```python
class SessionState(Enum):
    CREATED = auto()
    ACTIVE = auto()
    IN_TRANSACTION = auto()
    CLOSED = auto()

@state_machine(state_var="_state", states=SessionState, initial=SessionState.CREATED)
class DatabaseSession:
    @transition(from_=SessionState.CREATED, to=SessionState.ACTIVE)
    def open(self) -> None:
        """Open the database session."""
        self._conn = self._pool.acquire()

    @transition(from_=SessionState.ACTIVE, to=SessionState.IN_TRANSACTION)
    def begin_transaction(self) -> None:
        """Start a transaction. Fails if already in transaction."""
        self._conn.execute("BEGIN")

    @in_state(SessionState.ACTIVE, SessionState.IN_TRANSACTION)
    def execute(self, sql: str) -> Result:
        """Execute SQL. Valid whether in transaction or not."""
        return self._conn.execute(sql)

    @transition(from_=SessionState.IN_TRANSACTION, to=SessionState.ACTIVE)
    def commit(self) -> None:
        """Commit current transaction."""
        self._conn.execute("COMMIT")

    @transition(from_=SessionState.IN_TRANSACTION, to=SessionState.ACTIVE)
    def rollback(self) -> None:
        """Roll back current transaction."""
        self._conn.execute("ROLLBACK")

    @transition(from_=(SessionState.ACTIVE, SessionState.IN_TRANSACTION), to=SessionState.CLOSED)
    def close(self) -> None:
        """Close session. Auto-rollback if in transaction."""
        if self._state == SessionState.IN_TRANSACTION:
            self._conn.execute("ROLLBACK")
        self._pool.release(self._conn)
```

## Implementation Sketch

### Module Structure

```
weakincentives/dbc/
├── __init__.py           # Existing exports + new state machine exports
├── _state_machine.py     # @state_machine decorator implementation
├── _transition.py        # @transition and related decorators
├── _extraction.py        # StateMachineSpec extraction
└── _errors.py           # Error types (StateError hierarchy)
```

### Key Implementation Details

1. `@state_machine` attaches `__state_machine_spec__` to the class containing
   the full specification.

2. Methods decorated with `@transition` etc. gain `__transition_spec__`
   attributes.

3. The class decorator collects all method specs and validates completeness
   when `strict=True`.

4. State variable assignment is intercepted via `__setattr__` override when
   enforcement is active.

5. All enforcement checks are gated by `dbc_active()` for zero production cost.

### Prototype Implementation

```python
def state_machine(
    *,
    state_var: str,
    states: type[Enum] | tuple[str, ...],
    initial: object,
    strict: bool = True,
    hierarchy: Mapping[object, set[object]] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator that enables state machine enforcement on a class."""

    def decorator(cls: type[T]) -> type[T]:
        # Validate states
        if isinstance(states, type) and issubclass(states, Enum):
            valid_states = frozenset(states)
        else:
            valid_states = frozenset(states)

        if initial not in valid_states:
            raise ValueError(f"Initial state {initial} not in valid states")

        # Collect transition specs from decorated methods
        transitions: list[TransitionSpec] = []
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if hasattr(method, "__transition_spec__"):
                transitions.append(method.__transition_spec__)

        # Validate strict mode
        if strict:
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if name.startswith("_"):
                    continue
                if not hasattr(method, "__transition_spec__") and \
                   not hasattr(method, "__state_independent__"):
                    raise IncompleteStateMachineError(cls, name)

        # Attach spec
        spec = StateMachineSpec(
            cls=cls,
            state_var=state_var,
            states=valid_states,
            initial=initial,
            transitions=frozenset(transitions),
        )
        cls.__state_machine_spec__ = spec

        # Wrap __init__ to set initial state
        original_init = cls.__init__

        @wraps(original_init)
        def init_wrapper(self: T, *args: object, **kwargs: object) -> None:
            original_init(self, *args, **kwargs)
            object.__setattr__(self, state_var, initial)

        cls.__init__ = init_wrapper

        # Wrap __setattr__ to guard state variable
        original_setattr = cls.__setattr__ if hasattr(cls, "__setattr__") else object.__setattr__

        def setattr_wrapper(self: T, name: str, value: object) -> None:
            if name == state_var and dbc_active():
                # Check if we're in an allowed context
                if not getattr(self, "__allow_state_assignment__", False):
                    raise IllegalStateAssignmentError(cls, value)
            original_setattr(self, name, value)

        cls.__setattr__ = setattr_wrapper

        return cls

    return decorator
```

## Testing Strategy

### Unit Tests

```python
def test_valid_transition():
    """Transition succeeds when in valid source state."""
    with dbc_enabled():
        conn = Connection("localhost")
        assert conn._state == ConnectionState.INITIAL

        conn.connect()
        assert conn._state == ConnectionState.CONNECTED


def test_invalid_state_raises():
    """Method in wrong state raises InvalidStateError."""
    with dbc_enabled():
        conn = Connection("localhost")

        with pytest.raises(InvalidStateError) as exc:
            conn.close()  # Can't close before connect

        assert exc.value.current_state == ConnectionState.INITIAL
        assert ConnectionState.CONNECTED in exc.value.valid_states


def test_transition_guard():
    """Guard predicate blocks transition when False."""
    with dbc_enabled():
        builder = PipelineBuilder()
        builder.validate()

        with pytest.raises(TransitionGuardError):
            builder.build()  # No stages added, guard fails


def test_strict_mode_validation():
    """Strict mode requires all public methods to declare state."""
    with pytest.raises(IncompleteStateMachineError):
        @state_machine(state_var="_s", states=State, initial=State.A, strict=True)
        class Incomplete:
            def missing_annotation(self):
                pass


def test_no_enforcement_when_inactive():
    """Decorators are no-ops when dbc_active() is False."""
    # Default: enforcement off
    conn = Connection("localhost")
    conn.close()  # Would fail if enforcement active
    assert conn._state == ConnectionState.CLOSED
```

### Graph Analysis Tests

```python
def test_unreachable_states():
    """Detect states with no path from initial."""
    @state_machine(state_var="_s", states=State, initial=State.A)
    class HasUnreachable:
        @transition(from_=State.A, to=State.B)
        def a_to_b(self): pass
        # State.C has no incoming transitions

    sm = extract_state_machine(HasUnreachable)
    assert State.C in sm.unreachable_states()


def test_mermaid_export():
    """Export transition graph as Mermaid diagram."""
    sm = extract_state_machine(Connection)
    mermaid = sm.to_mermaid()

    assert "stateDiagram-v2" in mermaid
    assert "INITIAL --> CONNECTED: connect" in mermaid
    assert "CONNECTED --> CLOSED: close" in mermaid
```

### Integration Tests

```python
def test_composition_with_invariant():
    """State machine composes with @invariant."""
    @invariant(lambda self: self._count >= 0)
    @state_machine(state_var="_state", states=State, initial=State.READY)
    class Counter:
        def __init__(self):
            self._count = 0

        @transition(from_=State.READY, to=State.COUNTING)
        def start(self):
            pass

        @in_state(State.COUNTING)
        def increment(self):
            self._count += 1

    with dbc_enabled():
        c = Counter()
        c.start()
        c.increment()
        assert c._count == 1


def test_hierarchy_matching():
    """Parent state matches child states in @in_state."""
    with dbc_enabled():
        tx = Transaction()
        tx.begin(readonly=True)
        assert tx._state == TransactionState.ACTIVE_READONLY

        # @in_state(ACTIVE) should match ACTIVE_READONLY
        tx.execute("SELECT 1")  # No error
```

## Acceptance Criteria

1. **Basic transitions**: `@transition` enforces from/to states correctly
2. **Invalid state detection**: Clear errors for wrong-state method calls
3. **Guard predicates**: Transitions gated by runtime conditions
4. **Strict mode**: Forces complete state annotations
5. **Hierarchical states**: Parent state matches sub-states
6. **Graph extraction**: `extract_state_machine()` returns valid spec
7. **Mermaid/Graphviz export**: Diagrams render correctly
8. **DbC composition**: Works with `@require`, `@ensure`, `@invariant`
9. **Zero-cost default**: No overhead when `dbc_active()` is False
10. **Clear diagnostics**: Error messages include state context

## Limitations

- **Single-threaded assumption**: No built-in synchronization
- **No persistence**: State not automatically serialized
- **No async support**: Decorators assume synchronous methods
- **Limited hierarchy**: No orthogonal regions or history states
- **Python-only analysis**: Static checks require mypy plugin, not standalone

## Future Considerations

- **Async transitions**: `@async_transition` for async method support
- **Timed transitions**: Auto-transition after timeout (requires scheduler)
- **State entry/exit hooks**: `@on_enter(State.X)`, `@on_exit(State.X)`
- **Formal verification**: Export to TLA+ for model checking
- **Runtime visualization**: Live state diagram in debug UI
