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

Thread-safe tracker for token consumption across evaluations:

```python
@dataclass
class BudgetTracker:
    """Tracks cumulative TokenUsage per evaluation against a Budget."""

    budget: Budget
    _per_evaluation: dict[str, TokenUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_cumulative(self, evaluation_id: str, usage: TokenUsage) -> None:
        """Record cumulative usage for an evaluation (replaces previous)."""

    @property
    def consumed(self) -> TokenUsage:
        """Sum usage across all evaluations."""

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
1. **After every tool call** – Check limits before continuing
1. **On evaluation completion** – Final check before returning

```python
# Evaluation loop (each evaluate() has a unique evaluation_id)
response = provider.call(...)
budget_tracker.record_cumulative(evaluation_id, token_usage_from_payload(response))
budget_tracker.check()

for tool_call in tool_calls:
    execute_tool(...)
    budget_tracker.check()
```

## Subagent Propagation

The `BudgetTracker` is **always shared** with children, regardless of isolation:

| Isolation Level | Session/Bus | BudgetTracker |
|-----------------|-------------|---------------|
| NO_ISOLATION | Shared | Shared |
| FULL_ISOLATION | Cloned | Shared |

This ensures parallel subagents contribute to the parent's global limits.

## Token Accounting

`token_usage_from_payload` returns **cumulative** values within a single
provider conversation. The tracker must account for this:

### Single Evaluation

Within one `evaluate()` call, provider responses report running totals. The
tracker replaces (not adds) per-evaluation usage:

```python
def record_cumulative(self, evaluation_id: str, usage: TokenUsage) -> None:
    """Replace the cumulative total for an evaluation."""
    with self._lock:
        self._per_evaluation[evaluation_id] = usage
```

### Parallel Subagents

Each subagent runs its own provider evaluation with independent cumulative
counters. The tracker sums final totals across evaluations:

```
Parent evaluate() [eval_0]
├── Provider call → cumulative=100 (eval_0)
├── Provider call → cumulative=250 (eval_0, replaces 100)
├── dispatch_subagents
│   ├── Child A [eval_1] → final cumulative=500
│   ├── Child B [eval_2] → final cumulative=300
│   └── Child C [eval_3] → final cumulative=400
│   └── (join)
├── check() → total = 250 + 500 + 300 + 400 = 1450
└── Provider call → cumulative=400 (eval_0, replaces 250)
    check() → total = 400 + 500 + 300 + 400 = 1600
```

### BudgetTracker Implementation

```python
@dataclass
class BudgetTracker:
    budget: Budget
    _per_evaluation: dict[str, TokenUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_cumulative(self, evaluation_id: str, usage: TokenUsage) -> None:
        """Record cumulative usage for an evaluation (replaces previous)."""
        with self._lock:
            self._per_evaluation[evaluation_id] = usage

    @property
    def consumed(self) -> TokenUsage:
        """Sum across all evaluations."""
        with self._lock:
            return TokenUsage(
                input_tokens=sum(u.input_tokens or 0 for u in self._per_evaluation.values()),
                output_tokens=sum(u.output_tokens or 0 for u in self._per_evaluation.values()),
                cached_tokens=sum(u.cached_tokens or 0 for u in self._per_evaluation.values()),
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

- Verify cumulative replacement per evaluation is correct
- Verify `consumed` sums across evaluations correctly under concurrent calls
- Test parallel subagents that collectively exceed the budget
- Confirm `FULL_ISOLATION` still shares the tracker
- Check enforcement triggers at correct checkpoints
