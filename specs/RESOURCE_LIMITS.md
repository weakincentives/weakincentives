# Resource Limits Specification

This specification covers deadline and budget enforcement for prompt evaluations.
Deadlines bound wall-clock time while budgets extend deadlines with token limits.

## Deadlines

### Goal

Allow orchestration hosts to enforce a wall-clock deadline for an entire
`evaluate()` run, ensuring tool handlers and delegated subagents stop before the
window closes. The deadline is provided by the caller and applies uniformly to
all nested work spawned during that evaluation.

### Rationale

- Protect hosts from runaway evaluations and tool handlers that would otherwise
  overrun their resource budgets.
- Standardize how providers, tools, and subagents receive deadline data so
  integrations can make consistent decisions about when to abort.
- Preserve compatibility with existing prompts by keeping the deadline optional
  while still enforcing validation when a value is present.

### Scope

- Applies to every provider adapter (`openai`, `litellm`, and shared orchestration
  helpers) plus all tool handler entry points executed through
  `ToolContext.deadline`.
- Covers deadline validation, propagation into rendered prompts, enforcement
  before provider requests, and translation of tool-level deadline errors into
  prompt failures.
- Excludes any out-of-band cancellation of in-flight provider calls; enforcement
  happens at discrete checkpoints in the orchestrator and inside tool handlers.

### Deadline API

Define `Deadline` as an immutable value object in `weakincentives.deadlines`
with a timezone-aware `expires_at: datetime` attribute. Its constructor MUST
reject naive datetimes, timestamps in the past, or values that fall within
the current second relative to `datetime.now(datetime.UTC)`.

```python
from weakincentives.deadlines import Deadline

def evaluate(
    self,
    prompt: Prompt[OutputT],
    *params: SupportsDataclass,
    parse_output: bool = True,
    *,
    bus: EventBus,
    session: SessionProtocol,
    deadline: Deadline | None = None,
) -> PromptResponse[OutputT]:
    ...
```

- Callers may omit `deadline`; adapters treat `None` as "no deadline" to remain
  backwards compatible with existing integrations.
- Adapters validate the argument eagerly and raise `PromptEvaluationError`
  with `phase="preflight"` when the `Deadline` instance fails validation or
  does not extend past the current time.
- `DeadlineExceededError(RuntimeError)` in `weakincentives.tools.errors` is
  raised by tools when they cannot finish before the cutoff.

### Deadline Propagation

1. **Rendered Prompt Metadata** - After validation the orchestrator stores the
   deadline on the rendered prompt instance
   (`RenderedPrompt.deadline: Deadline | None`). All child prompts created during
   the run inherit this value unless a stricter deadline is provided explicitly.
1. **Tool Context** - `ToolContext` exposes the active `RenderedPrompt`, so
   handlers read the deadline from `context.rendered_prompt.deadline`.

### Enforcement Rules

- **Before Provider Calls** - Adapters compare `datetime.now(datetime.UTC)` to
  `deadline.expires_at` immediately before sending a request. If the deadline has
  passed they skip the provider call and raise `PromptEvaluationError` with
  `phase="deadline"`.
- **Tool Execution** - Tool handlers MUST raise `DeadlineExceededError` if the
  remaining time is insufficient. The runtime converts it into a
  `PromptEvaluationError` with `phase="deadline"`.
- **Subagents** - Delegation helpers reuse the deadline supplied by the parent
  context. Child prompts run with the earlier of the parent deadline and any
  explicit child override.
- **Polling & Retries** - Any retry loops must re-check the deadline before each
  iteration to avoid overshooting the wall-clock budget.

### Caveats

- Deadline checks happen at well-defined checkpoints but do not interrupt
  in-flight SDK requests.
- Tool handlers must cooperate by consulting `context.deadline` or raising
  `DeadlineExceededError`.
- Deadlines rely on synchronized UTC clocks between callers and the runtime.

## Budgets

### Goal

Introduce a `Budget` abstraction that wraps `Deadline` and token limits into a
single enforceable constraint. Enforcement occurs after every tool call and when
prompt evaluation completes. Budgets apply across subagent invocations regardless
of isolation level.

### Budget API

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

### Adapter Integration

Add optional `budget` argument to `evaluate()`:

```python
def evaluate(
    ...,
    budget: Budget | None = None,
) -> PromptResponse[OutputT]:
```

When `budget` is provided, the adapter creates a `BudgetTracker` and enforces
limits at checkpoints. Omitting `budget` preserves current behavior.

### Enforcement Checkpoints

1. **After every provider response** - Record cumulative usage, then check limits
1. **After every tool call** - Check limits before continuing
1. **On evaluation completion** - Final check before returning

### Subagent Propagation

The `BudgetTracker` is **always shared** with children, regardless of isolation:

| Isolation Level | Session/Bus | BudgetTracker |
|-----------------|-------------|---------------|
| NO_ISOLATION | Shared | Shared |
| FULL_ISOLATION | Cloned | Shared |

This ensures parallel subagents contribute to the parent's global limits.

### Token Accounting

Within one `evaluate()` call, provider responses report running totals. The
tracker replaces (not adds) per-evaluation usage. Each subagent runs its own
provider evaluation with independent cumulative counters. The tracker sums
final totals across evaluations.

## Observability

- Include the `Deadline.expires_at` timestamp in the `provider_payload` envelope
  when raising deadline errors.
- Emit structured log lines summarizing time remaining when a prompt finishes.

## Testing Strategy

- Unit tests cover validation failures (`tzinfo` missing, past deadlines, etc.).
- Adapter integration tests simulate deadlines and assert tool handlers receive
  the value via `ToolContext`.
- Verify cumulative replacement per evaluation is correct.
- Verify `consumed` sums across evaluations correctly under concurrent calls.
- Test parallel subagents that collectively exceed the budget.
- Confirm `FULL_ISOLATION` still shares the tracker.
