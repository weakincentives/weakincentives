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

Thread-safe tracker for token consumption across conversations:

```python
@dataclass
class BudgetTracker:
    """Tracks cumulative TokenUsage per conversation against a Budget."""

    budget: Budget
    _per_conversation: dict[str, TokenUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_cumulative(self, conversation_id: str, usage: TokenUsage) -> None:
        """Record cumulative usage for a conversation (replaces previous)."""

    @property
    def consumed(self) -> TokenUsage:
        """Sum usage across all conversations."""

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

1. **After every provider response** – Record cumulative usage, then check limits
2. **After every tool call** – Check limits before continuing
3. **On evaluation completion** – Final check before returning

```python
# Conversation loop (each evaluate() has a unique conversation_id)
response = provider.call(...)
budget_tracker.record_cumulative(conversation_id, token_usage_from_payload(response))
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

## Token Accounting

`token_usage_from_payload` returns **cumulative** values within a single
provider conversation. The tracker must account for this:

### Single Evaluation

Within one `evaluate()` call, provider responses report running totals. The
tracker replaces (not adds) per-conversation usage:

```python
def record_cumulative(self, conversation_id: str, usage: TokenUsage) -> None:
    """Replace the cumulative total for a conversation."""
    with self._lock:
        self._per_conversation[conversation_id] = usage
```

### Parallel Subagents

Each subagent runs its own provider conversation with independent cumulative
counters. The tracker sums final totals across conversations:

```
Parent evaluate() [conv_0]
├── Provider call → cumulative=100 (conv_0)
├── Provider call → cumulative=250 (conv_0, replaces 100)
├── dispatch_subagents
│   ├── Child A [conv_1] → final cumulative=500
│   ├── Child B [conv_2] → final cumulative=300
│   └── Child C [conv_3] → final cumulative=400
│   └── (join)
├── check() → total = 250 + 500 + 300 + 400 = 1450
└── Provider call → cumulative=400 (conv_0, replaces 250)
    check() → total = 400 + 500 + 300 + 400 = 1600
```

### BudgetTracker Implementation

```python
@dataclass
class BudgetTracker:
    budget: Budget
    _per_conversation: dict[str, TokenUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_cumulative(self, conversation_id: str, usage: TokenUsage) -> None:
        """Record cumulative usage for a conversation (replaces previous)."""
        with self._lock:
            self._per_conversation[conversation_id] = usage

    @property
    def consumed(self) -> TokenUsage:
        """Sum across all conversations."""
        with self._lock:
            return TokenUsage(
                input_tokens=sum(u.input_tokens or 0 for u in self._per_conversation.values()),
                output_tokens=sum(u.output_tokens or 0 for u in self._per_conversation.values()),
                cached_tokens=sum(u.cached_tokens or 0 for u in self._per_conversation.values()),
            )

    def check(self) -> None:
        """Raise BudgetExceededError if any limit is breached."""
```

**Key invariant**: Even with `FULL_ISOLATION`, children share the parent's
tracker. Session and event bus isolation is orthogonal to budget enforcement.

## Caveats

- No mid-request cancellation; limits checked after responses arrive
- Token counts depend on provider accuracy
- Deadline checks require synchronized UTC clocks

## Testing

- Verify cumulative replacement per conversation is correct
- Verify `consumed` sums across conversations correctly under concurrent calls
- Test parallel subagents that collectively exceed the budget
- Confirm `FULL_ISOLATION` still shares the tracker
- Check enforcement triggers at correct checkpoints
