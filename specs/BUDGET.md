# Budget Specification

## Goal

Introduce a `Budget` abstraction that wraps `Deadline` and token limits into a
single enforceable constraint. Enforcement occurs after every tool call and when
prompt evaluation completes. Budgets apply across subagent invocations regardless
of isolation level.

## API

### Budget

Immutable value object in `weakincentives.budget`:

```python
@dataclass(slots=True, frozen=True)
class Budget:
    """Resource envelope combining time and token limits."""

    deadline: Deadline | None = None
    max_total_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
```

At least one limit must be set. Token limits must be positive when provided.

### BudgetTracker

Thread-safe accumulator for token consumption:

```python
@dataclass
class BudgetTracker:
    """Accumulates TokenUsage against a Budget."""

    budget: Budget
    _consumed: TokenUsage = field(default_factory=lambda: TokenUsage())
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, usage: TokenUsage) -> None:
        """Add usage to running totals (thread-safe)."""

    def check(self) -> None:
        """Raise BudgetExceededError if any limit is breached."""
```

### BudgetExceededError

```python
@dataclass(slots=True, frozen=True)
class BudgetExceededError(RuntimeError):
    budget: Budget
    consumed: TokenUsage
    exceeded_dimension: str  # "deadline", "total_tokens", etc.
```

Converted to `PromptEvaluationError` with `phase="budget"` by the runtime.

## Adapter Integration

Add optional `budget` argument to `evaluate()`:

```python
def evaluate(
    ...,
    budget: Budget | None = None,
) -> PromptResponse[OutputT]:
```

When `budget` is provided, the adapter creates a `BudgetTracker` and enforces
limits at checkpoints. Omitting `budget` preserves current behavior.

## Enforcement Checkpoints

1. **After every provider response** – Record usage, then check limits
2. **After every tool call** – Check limits before continuing
3. **On evaluation completion** – Final check before returning

```python
# Conversation loop
response = provider.call(...)
budget_tracker.record(token_usage_from_payload(response))
budget_tracker.check()

for tool_call in tool_calls:
    execute_tool(...)
    budget_tracker.check()
```

## Subagent Propagation

The `BudgetTracker` is **always shared** with children, regardless of isolation:

| Isolation Level | Session/Bus | BudgetTracker |
|-----------------|-------------|---------------|
| NO_ISOLATION    | Shared      | Shared        |
| FULL_ISOLATION  | Cloned      | Shared        |

This ensures parallel subagents contribute to the parent's global limits.

## Parallel Token Accounting

When subagents run in parallel via `ThreadPoolExecutor`:

```
Parent evaluate()
├── Provider call → record(usage_1)
├── dispatch_subagents (3 children)
│   ├── Child A (thread 1) → record(usage_a)  ─┐
│   ├── Child B (thread 2) → record(usage_b)  ─┼─ Same BudgetTracker
│   └── Child C (thread 3) → record(usage_c)  ─┘
│   └── (join)
├── check() → enforce limits
└── continue...
```

**Thread safety**: `BudgetTracker.record()` uses a `threading.Lock` to serialize
updates. The lock is held only during the atomic addition:

```python
def record(self, usage: TokenUsage) -> None:
    with self._lock:
        self._consumed = TokenUsage(
            input_tokens=(self._consumed.input_tokens or 0) + (usage.input_tokens or 0),
            output_tokens=(self._consumed.output_tokens or 0) + (usage.output_tokens or 0),
            cached_tokens=(self._consumed.cached_tokens or 0) + (usage.cached_tokens or 0),
        )
```

**Key invariant**: Even with `FULL_ISOLATION`, children share the parent's
tracker. Session and event bus isolation is orthogonal to budget enforcement.

## Caveats

- No mid-request cancellation; limits checked after responses arrive
- Token counts depend on provider accuracy
- Deadline checks require synchronized UTC clocks

## Testing

- Verify `BudgetTracker.record()` is correct under concurrent calls
- Test parallel subagents that collectively exceed the budget
- Confirm `FULL_ISOLATION` still shares the tracker
- Check enforcement triggers at correct checkpoints
