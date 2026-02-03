# Nested Adapter Execution Specification

## Purpose

Define how adapters execute within feedback providers and task completion checkers,
enabling LLM-as-judge patterns for progress evaluation and completion verification.
This specification addresses session hierarchy, resource sharing, budget inheritance,
and trace correlation for nested adapter invocations.

**Status:** Draft specification. Implementation pending.

## Motivation

Agents benefit from LLM-based evaluation at two critical junctures:

1. **Ongoing feedback**: Periodically assess trajectory quality and inject guidance
2. **Completion verification**: Validate task completion before allowing stop

Both require invoking an adapter within the parent execution context. Current
infrastructure provides partial support (e.g., `TaskCompletionContext.adapter`)
but lacks clear semantics for:

- Session state visibility and isolation
- Resource context nesting
- Budget and deadline inheritance
- Event correlation and tracing
- Error handling and recovery

This specification fills those gaps.

## Principles

### Child Execution Isolation

Nested adapter calls create isolated execution contexts. The child:

- Cannot mutate parent session state directly
- Has its own resource lifecycle
- Produces events scoped to its own trace
- Fails independently (parent decides how to handle)

### Parent Context Visibility

Despite isolation, children need parent context for informed evaluation:

- Read-only access to parent session state
- Inherited deadline constraints
- Shared budget tracking (optional)
- Correlated trace identifiers

### Minimal Footprint

Nested evaluations should be lightweight:

- Smaller prompts focused on evaluation
- Fewer tools (often none)
- Tighter budgets and deadlines
- No cascading nested calls

## Core Abstractions

### NestedExecutionContext

Context for invoking an adapter within feedback/completion checking:

```python
@dataclass(slots=True, frozen=True)
class NestedExecutionContext:
    """Context for nested adapter invocation.

    Provides controlled access to parent execution state while maintaining
    isolation for the nested call.

    Attributes:
        parent_session: Read-only view of parent session state.
        parent_run_context: Trace context from parent for correlation.
        deadline: Deadline for nested execution (derived from parent).
        budget: Optional budget allocation for nested call.
        resource_overrides: Resources to inject into nested prompt context.
    """

    parent_session: SessionProtocol
    parent_run_context: RunContext | None = None
    deadline: Deadline | None = None
    budget: Budget | None = None
    resource_overrides: Mapping[type, Any] = field(default_factory=dict)
```

### NestedEvaluator

Helper for executing nested adapter calls:

```python
class NestedEvaluator:
    """Execute nested adapter calls with proper isolation.

    Manages session creation, resource setup, trace correlation, and
    result extraction for nested LLM evaluations.
    """

    def __init__(
        self,
        adapter: ProviderAdapter,
        *,
        prompt: Prompt[OutputT],
        context: NestedExecutionContext,
    ) -> None: ...

    def evaluate(self) -> NestedEvaluationResult[OutputT]:
        """Execute the nested evaluation.

        Creates an isolated child session, sets up resource context,
        invokes the adapter, and returns results with any errors captured.
        """
        ...
```

### NestedEvaluationResult

Result of a nested adapter invocation:

```python
@dataclass(slots=True, frozen=True)
class NestedEvaluationResult(Generic[OutputT]):
    """Result of a nested adapter evaluation.

    Attributes:
        success: Whether evaluation completed without error.
        output: Parsed output if successful, None otherwise.
        error: Error details if evaluation failed.
        usage: Token usage for budget tracking.
        child_run_context: Trace context for correlation.
    """

    success: bool
    output: OutputT | None = None
    error: NestedEvaluationError | None = None
    usage: TokenUsage | None = None
    child_run_context: RunContext | None = None
```

## Session Hierarchy

### Child Session Creation

Nested evaluations create child sessions linked to the parent:

```python
child_session = Session(
    parent=parent_session,
    session_id=uuid4(),
    dispatcher=InProcessDispatcher(),
)
```

The parent link enables:

- Bottom-up traversal via `iter_sessions_bottom_up()`
- Trace correlation through session hierarchy
- Future: cascading snapshots

### State Visibility

| Operation | Parent → Child | Child → Parent |
|-----------|----------------|----------------|
| Read state | ✓ Via explicit query | ✗ Blocked |
| Write state | ✗ Blocked | ✗ Blocked |
| Dispatch events | ✗ (uses own dispatcher) | ✗ Blocked |

The child can read parent state by querying the parent session directly,
but all mutations occur in the child's isolated slices.

### State Injection Pattern

To provide parent state to the nested prompt, use template variables:

```python
# In feedback provider
parent_plan = context.session[Plan].latest()
parent_tool_calls = context.recent_tool_calls(10)

nested_prompt = EvaluationPrompt(
    plan_summary=render_plan_summary(parent_plan),
    recent_actions=render_tool_calls(parent_tool_calls),
)
```

This makes state transfer explicit and auditable.

## Resource Context

### Nesting Semantics

Each nested evaluation gets its own resource context:

```
Parent Prompt Resources (active)
  └─ Nested Evaluation
       └─ Child Prompt Resources (new context)
            ├─ Inherited: clock, filesystem (read-only)
            ├─ Overridden: via resource_overrides
            └─ Fresh: TOOL_CALL scoped resources
```

### Resource Inheritance Rules

| Resource Type | Inheritance | Rationale |
|---------------|-------------|-----------|
| Clock protocols | Inherit | Consistent time view |
| Filesystem | Read-only view | Evaluation can inspect |
| Budget tracker | Share (optional) | Unified accounting |
| HTTP clients | Fresh | Isolation |
| Custom resources | Explicit override | Case-by-case |

### Configuration

```python
context = NestedExecutionContext(
    parent_session=session,
    resource_overrides={
        Filesystem: parent_fs.read_only_view(),
        CustomResource: evaluation_specific_instance,
    },
)
```

## Budget and Deadline

### Deadline Inheritance

Nested calls must complete before the parent deadline:

```python
def derive_nested_deadline(
    parent_deadline: Deadline | None,
    max_nested_duration: timedelta = timedelta(seconds=30),
) -> Deadline | None:
    """Derive deadline for nested execution.

    Returns the earlier of:
    - parent_deadline (if set)
    - now + max_nested_duration

    This ensures nested calls don't block the parent indefinitely.
    """
    now = datetime.now(UTC)
    nested_limit = Deadline(now + max_nested_duration)

    if parent_deadline is None:
        return nested_limit

    return min(parent_deadline, nested_limit, key=lambda d: d.at)
```

### Budget Allocation

Two strategies for budget management:

**Shared Budget** (default for completion checking):
```python
# Child draws from parent's remaining budget
context = NestedExecutionContext(
    budget=None,  # Uses parent's budget_tracker
    ...
)
evaluator.evaluate(budget_tracker=parent_budget_tracker)
```

**Isolated Budget** (default for feedback):
```python
# Child has independent allocation
context = NestedExecutionContext(
    budget=Budget(max_total_tokens=1000),
    ...
)
```

### Recommendations

| Use Case | Budget Strategy | Rationale |
|----------|-----------------|-----------|
| Task completion | Shared | Completion check is part of task |
| Periodic feedback | Isolated | Don't starve main execution |
| Critical verification | Shared | Must complete for valid output |

## Trace Correlation

### Parent-Child Correlation

Nested executions create child traces linked to the parent:

```python
def derive_child_run_context(
    parent: RunContext | None,
    child_session_id: UUID,
) -> RunContext:
    """Create RunContext for nested execution.

    Preserves parent trace_id for correlation while generating
    fresh run_id and span_id for the child execution.
    """
    return RunContext(
        run_id=uuid4(),
        request_id=parent.request_id if parent else uuid4(),
        session_id=child_session_id,
        attempt=1,
        worker_id=parent.worker_id if parent else "nested",
        trace_id=parent.trace_id if parent else None,
        span_id=generate_span_id(),  # Fresh span under same trace
        parent_span_id=parent.span_id if parent else None,
    )
```

### Event Correlation

Events from nested execution include:

- `run_context.trace_id` matching parent (same distributed trace)
- `run_context.parent_span_id` pointing to parent span
- `run_context.run_id` unique to nested execution

This enables:

```
Parent Trace (trace_id=abc123)
├─ Span: parent_execution (span_id=def456)
│   ├─ ToolInvoked: edit_file
│   ├─ ToolInvoked: run_tests
│   └─ Span: completion_check (span_id=ghi789, parent=def456)
│       └─ PromptExecuted: verify_completion
└─ PromptExecuted: main_output
```

## Error Handling

### Error Categories

| Category | Nested Behavior | Parent Response |
|----------|-----------------|-----------------|
| `DeadlineExceededError` | Propagates | Handle gracefully |
| `BudgetExceededError` | Propagates | Handle gracefully |
| `ThrottleError` | Captured | Retry or skip |
| `OutputParseError` | Captured | Use fallback |
| `PromptEvaluationError` | Captured | Log and continue |

### Recovery Patterns

**For Feedback Providers:**
```python
def provide(self, *, context: FeedbackContext) -> Feedback:
    try:
        result = self._evaluator.evaluate()
        if result.success:
            return self._build_feedback(result.output)
    except DeadlineExceededError:
        pass  # Skip feedback, don't block parent
    except Exception as e:
        logger.warning("feedback.nested_evaluation_failed", error=str(e))

    return self._fallback_feedback()
```

**For Task Completion:**
```python
def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
    try:
        result = self._evaluator.evaluate()
        if result.success:
            return self._interpret_result(result.output)
    except (DeadlineExceededError, BudgetExceededError):
        # Budget/time exhausted - allow completion
        return TaskCompletionResult.ok("Verification skipped (resource limit)")
    except Exception as e:
        logger.error("completion.verification_failed", error=str(e))
        # Fail-open: allow completion but log
        return TaskCompletionResult.ok("Verification failed (allowing completion)")
```

### Fail-Open vs Fail-Closed

| Scenario | Default | Rationale |
|----------|---------|-----------|
| Feedback evaluation fails | Fail-open (skip) | Feedback is advisory |
| Completion check fails | Fail-open (allow) | Don't block on verification bugs |
| Critical verification | Configurable | Business logic decides |

## Integration Patterns

### Feedback Provider with Nested Adapter

```python
@dataclass(frozen=True)
class LLMTrajectoryFeedback:
    """Feedback provider using LLM to evaluate agent trajectory."""

    adapter: ProviderAdapter
    evaluation_prompt: Prompt[TrajectoryAssessment]

    @property
    def name(self) -> str:
        return "LLMTrajectoryFeedback"

    def should_run(self, *, context: FeedbackContext) -> bool:
        return context.tool_call_count >= 5

    def provide(self, *, context: FeedbackContext) -> Feedback:
        nested_ctx = NestedExecutionContext(
            parent_session=context.session,
            deadline=derive_nested_deadline(context.deadline),
            budget=Budget(max_total_tokens=500),
        )

        # Inject parent state into evaluation prompt
        prompt = self.evaluation_prompt.with_variables(
            recent_actions=self._render_recent_actions(context),
            current_plan=self._render_plan(context),
        )

        evaluator = NestedEvaluator(
            self.adapter,
            prompt=prompt,
            context=nested_ctx,
        )

        result = evaluator.evaluate()

        if result.success and result.output:
            return self._assessment_to_feedback(result.output)

        return Feedback(
            provider_name=self.name,
            summary="Trajectory assessment unavailable",
            severity="info",
        )
```

### Task Completion with LLM Verification

```python
class LLMCompletionVerifier:
    """Verify task completion using LLM judgment."""

    def __init__(
        self,
        verification_prompt: Prompt[CompletionVerdict],
        *,
        share_budget: bool = True,
    ) -> None:
        self._prompt = verification_prompt
        self._share_budget = share_budget

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        if context.adapter is None:
            return TaskCompletionResult.ok("No adapter for verification")

        nested_ctx = NestedExecutionContext(
            parent_session=context.session,
            deadline=derive_nested_deadline(None, timedelta(seconds=15)),
            budget=None if self._share_budget else Budget(max_total_tokens=500),
            resource_overrides={
                Filesystem: context.filesystem,
            } if context.filesystem else {},
        )

        prompt = self._prompt.with_variables(
            task_description=self._get_task_description(context),
            tentative_output=context.tentative_output,
            stop_reason=context.stop_reason,
        )

        evaluator = NestedEvaluator(
            context.adapter,
            prompt=prompt,
            context=nested_ctx,
        )

        result = evaluator.evaluate()

        if result.success and result.output:
            verdict = result.output
            if verdict.complete:
                return TaskCompletionResult.ok(verdict.explanation)
            return TaskCompletionResult.incomplete(verdict.explanation)

        # Fail-open on evaluation failure
        return TaskCompletionResult.ok("Verification inconclusive")
```

## Extending FeedbackContext

To support nested adapter usage, extend `FeedbackContext`:

```python
@dataclass(slots=True, frozen=True)
class FeedbackContext:
    session: SessionProtocol
    prompt: PromptProtocol[Any]
    deadline: Deadline | None = None
    adapter: ProviderAdapter | None = None  # NEW: Optional adapter for LLM feedback
    run_context: RunContext | None = None   # NEW: For trace correlation

    def create_nested_context(
        self,
        *,
        budget: Budget | None = None,
        max_duration: timedelta = timedelta(seconds=30),
    ) -> NestedExecutionContext:
        """Create context for nested adapter invocation.

        Convenience method for feedback providers that need to invoke
        the adapter for LLM-based evaluation.
        """
        return NestedExecutionContext(
            parent_session=self.session,
            parent_run_context=self.run_context,
            deadline=derive_nested_deadline(self.deadline, max_duration),
            budget=budget,
        )
```

## Configuration

### Adapter-Level Configuration

```python
@dataclass(frozen=True)
class NestedExecutionConfig:
    """Configuration for nested adapter execution.

    Attributes:
        max_nested_duration: Maximum time for any nested call.
        default_nested_budget: Default budget when not shared.
        allow_nested_tools: Whether nested prompts can use tools.
        max_nesting_depth: Prevent infinite nesting (default: 1).
    """

    max_nested_duration: timedelta = timedelta(seconds=30)
    default_nested_budget: Budget | None = Budget(max_total_tokens=1000)
    allow_nested_tools: bool = False
    max_nesting_depth: int = 1
```

### Prompt-Level Configuration

```python
template = PromptTemplate(
    ...,
    feedback_providers=(
        FeedbackProviderConfig(
            provider=LLMTrajectoryFeedback(
                adapter=evaluation_adapter,
                evaluation_prompt=trajectory_assessment_prompt,
            ),
            trigger=FeedbackTrigger(every_n_calls=10),
        ),
    ),
    nested_execution_config=NestedExecutionConfig(
        max_nested_duration=timedelta(seconds=15),
    ),
)
```

## Anti-Patterns

### Recursive Nesting

**Don't** allow nested prompts to themselves use nested adapters:

```python
# BAD: Nested prompt triggers its own feedback provider with nesting
nested_prompt = PromptTemplate(
    ...,
    feedback_providers=(
        FeedbackProviderConfig(
            provider=LLMTrajectoryFeedback(...),  # This would nest again!
            ...
        ),
    ),
)
```

**Do** enforce `max_nesting_depth` and use tool-less evaluation prompts.

### Shared Mutable State

**Don't** pass mutable resources to nested context:

```python
# BAD: Nested execution could mutate parent's filesystem
context = NestedExecutionContext(
    resource_overrides={Filesystem: parent_fs},  # Mutable!
)
```

**Do** pass read-only views or fresh instances.

### Blocking Parent Indefinitely

**Don't** allow nested calls to consume parent's entire budget/deadline:

```python
# BAD: No limit on nested execution
context = NestedExecutionContext(
    deadline=parent_deadline,  # Could run until parent deadline
    budget=None,              # Could consume all remaining budget
)
```

**Do** derive constrained deadlines and allocate limited budgets.

## Implementation Roadmap

### Phase 1: Core Infrastructure

1. Implement `NestedExecutionContext` dataclass
2. Implement `NestedEvaluator` with session isolation
3. Add `derive_nested_deadline()` and `derive_child_run_context()`
4. Extend `RunContext` with `parent_span_id`

### Phase 2: Integration

1. Add `adapter` and `run_context` to `FeedbackContext`
2. Add `create_nested_context()` convenience method
3. Update adapter hooks to pass run context to feedback
4. Document integration patterns

### Phase 3: Built-in Providers

1. Implement `LLMTrajectoryFeedback` reference provider
2. Implement `LLMCompletionVerifier` reference checker
3. Create reusable evaluation prompt templates

### Phase 4: Observability

1. Add nested execution metrics
2. Structured logging for nested calls
3. Trace visualization support

## Testing Considerations

### Unit Tests

- Session isolation: verify child cannot mutate parent
- Resource scoping: verify lifecycle independence
- Budget enforcement: verify limits respected
- Deadline propagation: verify constraints applied

### Integration Tests

- Round-trip nested evaluation
- Error recovery scenarios
- Budget exhaustion mid-nested-call
- Trace correlation verification

### Property Tests

- Nesting depth enforcement
- Resource cleanup on failure
- State isolation under concurrent execution

## Related Specifications

- `specs/ADAPTERS.md` - Adapter protocol and lifecycle
- `specs/SESSIONS.md` - Session hierarchy and state management
- `specs/FEEDBACK_PROVIDERS.md` - Feedback provider protocol
- `specs/TASK_COMPLETION.md` - Task completion checking
- `specs/RUN_CONTEXT.md` - Trace correlation
- `specs/RESOURCE_REGISTRY.md` - Resource lifecycle

## Open Questions

1. **Budget accounting granularity**: Should nested usage be tracked separately
   or merged into parent's usage? Current recommendation: separate tracking with
   optional consolidation.

2. **Snapshot semantics**: Should child session automatically snapshot parent
   state on creation, or require explicit copying? Current recommendation:
   explicit copying via template variables.

3. **Tool availability**: Should nested evaluations ever have tool access, or
   always be pure evaluation? Current recommendation: default no tools, but
   configurable for advanced use cases.

4. **Async support**: Should `NestedEvaluator` support async execution for
   non-blocking feedback? Current recommendation: sync-first, async later.
