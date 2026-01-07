# State Machine Transition Enforcement Specification

## Purpose

This document specifies state machine enforcement decorators for `weakincentives`.
The framework enables declaring valid states and transitions explicitly, then
enforcing them automatically during tests. This catches bugs like "used resource
after close," "called run() while already running," or "skipped required
initialization" with clear error messages.

## Target Use Cases

Based on codebase analysis, the primary candidates are:

| Class | Current Tracking | Bugs to Catch |
|-------|------------------|---------------|
| `MainLoop` | `_running` bool | Double-run, use-after-shutdown |
| `ScopedResourceContext` | Implicit | Use-after-close, get-before-start |
| `RedisMailbox` | `_closed` bool | Send/receive after close |
| `EvalLoop` | `_running` bool | Same as MainLoop |

## Guiding Principles

- **Zero-cost default**: Decorators are no-ops unless `dbc_active()` is True
- **Declarative**: States and transitions declared via decorators, not scattered
  `if` checks
- **Fail fast**: Invalid transitions raise immediately with actionable messages
- **Composable**: Works with existing `@require`, `@ensure`, `@invariant`

## Decorator API

### `@state_machine`

Class decorator declaring state variable and valid states.

```python
from enum import Enum, auto
from weakincentives.dbc import state_machine, transition, in_state

class LoopState(Enum):
    IDLE = auto()
    RUNNING = auto()
    STOPPED = auto()

@state_machine(state_var="_state", states=LoopState, initial=LoopState.IDLE)
class MainLoop:
    ...
```

**Parameters:**

- `state_var: str` - Attribute name holding current state
- `states: type[Enum]` - Enum class defining valid states
- `initial: Enum` - Initial state (set after `__init__`)

### `@transition`

Method decorator declaring a state transition.

```python
@transition(from_=LoopState.IDLE, to=LoopState.RUNNING)
def run(self) -> None:
    ...

@transition(from_=LoopState.RUNNING, to=LoopState.STOPPED)
def shutdown(self) -> None:
    ...
```

**Parameters:**

- `from_: State | tuple[State, ...]` - Valid source state(s)
- `to: State` - Target state after method completes

### `@in_state`

Method decorator requiring specific state(s) without transition.

```python
@in_state(LoopState.RUNNING)
def execute(self, request: Request) -> Response:
    """Only valid while running."""
    ...
```

### `@enters`

Shorthand for transitions from any state.

```python
@enters(ContextState.CLOSED)
def close(self) -> None:
    """Can close from any state."""
    ...
```

## Concrete Examples

### MainLoop / EvalLoop

```python
class LoopState(Enum):
    IDLE = auto()
    RUNNING = auto()
    STOPPED = auto()

@state_machine(state_var="_state", states=LoopState, initial=LoopState.IDLE)
class MainLoop(ABC):
    @transition(from_=LoopState.IDLE, to=LoopState.RUNNING)
    def run(self, *, max_iterations: int | None = None, ...) -> None:
        # Current: manually sets _running = True at start, False at end
        ...

    @transition(from_=LoopState.RUNNING, to=LoopState.STOPPED)
    def shutdown(self, *, timeout: float = 30.0) -> bool:
        # Current: sets _shutdown_event, waits for _running = False
        ...

    @in_state(LoopState.IDLE, LoopState.RUNNING)
    def execute(self, request: UserRequestT, ...) -> tuple[PromptResponse, Session]:
        """Direct execution - valid when idle or running."""
        ...
```

**Bugs caught:**
- `loop.run()` called twice → `InvalidStateError: run() requires IDLE, got RUNNING`
- `loop.execute()` after shutdown → `InvalidStateError: execute() requires [IDLE, RUNNING], got STOPPED`

### ScopedResourceContext

```python
class ContextState(Enum):
    CREATED = auto()
    STARTED = auto()
    CLOSED = auto()

@state_machine(state_var="_state", states=ContextState, initial=ContextState.CREATED)
class ScopedResourceContext:
    @transition(from_=ContextState.CREATED, to=ContextState.STARTED)
    def start(self) -> None:
        """Initialize and instantiate eager singletons."""
        ...

    @in_state(ContextState.STARTED)
    def get(self, protocol: type[T]) -> T:
        """Resolve resource - only valid after start()."""
        ...

    @in_state(ContextState.STARTED)
    def tool_scope(self) -> Iterator[ResourceResolver]:
        """Enter tool scope - only valid after start()."""
        ...

    @enters(ContextState.CLOSED)
    def close(self) -> None:
        """Dispose resources - can close from any state."""
        ...
```

**Bugs caught:**
- `ctx.get(Service)` before `start()` → `InvalidStateError: get() requires STARTED, got CREATED`
- `ctx.start()` after `close()` → `InvalidStateError: start() requires CREATED, got CLOSED`

### RedisMailbox

```python
class MailboxState(Enum):
    OPEN = auto()
    CLOSED = auto()

@state_machine(state_var="_state", states=MailboxState, initial=MailboxState.OPEN)
class RedisMailbox:
    @in_state(MailboxState.OPEN)
    def send(self, body: T, *, reply_to: str | None = None) -> str:
        ...

    @in_state(MailboxState.OPEN)
    def receive(self, ...) -> Sequence[Message[T, R]]:
        ...

    @enters(MailboxState.CLOSED)
    def close(self) -> None:
        ...
```

**Bugs caught:**
- `mailbox.send(msg)` after `close()` → `InvalidStateError: send() requires OPEN, got CLOSED`

## Error Types

```python
class StateError(WinkError, RuntimeError):
    """Base class for state machine errors."""

@dataclass(frozen=True)
class InvalidStateError(StateError):
    """Method called in invalid state."""
    cls: type
    method: str
    current_state: object
    valid_states: tuple[object, ...]

    def __str__(self) -> str:
        valid = ", ".join(s.name for s in self.valid_states)
        return (
            f"{self.cls.__name__}.{self.method}() requires state in "
            f"[{valid}], but current state is {self.current_state.name}"
        )
```

## Runtime Behavior

### Activation

Uses existing DbC mechanism:

```python
from weakincentives.dbc import dbc_active, dbc_enabled

# Active in tests via pytest plugin
# Manual activation:
with dbc_enabled():
    loop.run()  # Enforcement active
```

When `dbc_active()` is False, decorators are no-ops.

### Enforcement Sequence

For `@transition(from_=A, to=B)`:

1. Check `dbc_active()` → if False, call method directly
2. Check current state is `A` → if not, raise `InvalidStateError`
3. Execute method body
4. Set state to `B`
5. Return result

## Graph Extraction

Extract transition graph for visualization/analysis:

```python
from weakincentives.dbc import extract_state_machine

sm = extract_state_machine(MainLoop)
print(sm.to_mermaid())
```

Output:
```
stateDiagram-v2
    [*] --> IDLE
    IDLE --> RUNNING: run()
    RUNNING --> STOPPED: shutdown()
```

## Implementation

### Module Location

```
weakincentives/dbc/
├── __init__.py          # Add state_machine, transition, in_state, enters
├── _state_machine.py    # Implementation
└── _errors.py           # StateError, InvalidStateError
```

### Core Implementation

```python
def state_machine(
    *,
    state_var: str,
    states: type[Enum],
    initial: Enum,
) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        # Wrap __init__ to set initial state
        original_init = cls.__init__

        @wraps(original_init)
        def init_wrapper(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            object.__setattr__(self, state_var, initial)

        cls.__init__ = init_wrapper
        cls.__state_machine_spec__ = StateMachineSpec(cls, state_var, states, initial)
        return cls

    return decorator


def transition(*, from_: Enum | tuple[Enum, ...], to: Enum):
    from_states = (from_,) if isinstance(from_, Enum) else from_

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if dbc_active():
                spec = type(self).__state_machine_spec__
                current = getattr(self, spec.state_var)
                if current not in from_states:
                    raise InvalidStateError(type(self), method.__name__, current, from_states)
            result = method(self, *args, **kwargs)
            if dbc_active():
                object.__setattr__(self, spec.state_var, to)
            return result
        return wrapper
    return decorator
```

## Testing

```python
def test_mainloop_double_run():
    with dbc_enabled():
        loop = TestLoop(...)
        thread = Thread(target=loop.run)
        thread.start()
        time.sleep(0.1)

        with pytest.raises(InvalidStateError) as exc:
            loop.run()  # Already running

        assert exc.value.current_state == LoopState.RUNNING
        assert LoopState.IDLE in exc.value.valid_states

        loop.shutdown()
        thread.join()


def test_resource_context_get_before_start():
    with dbc_enabled():
        ctx = ScopedResourceContext(registry=registry)

        with pytest.raises(InvalidStateError) as exc:
            ctx.get(Service)

        assert "get() requires state in [STARTED]" in str(exc.value)
        assert "current state is CREATED" in str(exc.value)
```

## Limitations

- **Single-threaded assumption**: No built-in synchronization
- **No guards**: Conditional transitions not supported (use `@require` instead)
- **No on_error**: Exception handling doesn't change state

## Future Considerations

- **Transition guards**: `@transition(..., guard=lambda self: self.ready)`
- **Error states**: `@transition(..., on_error=State.FAILED)`
- **Async support**: `@async_transition` for async methods
