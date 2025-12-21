# State Management Abstractions Proposal

This document proposes new abstractions to enhance the weakincentives state
management system while preserving its core strengths: immutability, type
safety, and deterministic execution.

## Executive Summary

The current Session-based architecture provides excellent foundations with
copy-on-write semantics, typed slices, and transactional tool execution.
However, there are opportunities to address:

1. **Unbounded state growth** in long-running agents
2. **Cross-cutting concerns** that span multiple reducers
3. **Computed/derived state** that depends on multiple slices
4. **External side-effect coordination** for reliable workflows
5. **Complex workflow orchestration** with state machine semantics

## Proposal 1: Computed Slices (Selectors)

### Problem

Agents often need derived state that combines data from multiple slices. Currently,
this requires manual computation at each access point, leading to:

- Duplicated logic across handlers
- Potential inconsistencies if computation differs
- No caching of expensive derivations

### Solution

Introduce `ComputedSlice` that automatically derives state from source slices:

```python
from weakincentives.runtime import Session, ComputedSlice, computed

@dataclass(frozen=True)
class TaskSummary:
    total_steps: int
    completed_steps: int
    progress_pct: float
    blocking_issues: tuple[str, ...]

@computed
def task_summary(plan: Plan, issues: tuple[Issue, ...]) -> TaskSummary:
    """Derives summary from Plan and Issue slices."""
    completed = sum(1 for s in plan.steps if s.done)
    blockers = tuple(i.title for i in issues if i.severity == "blocker")
    return TaskSummary(
        total_steps=len(plan.steps),
        completed_steps=completed,
        progress_pct=completed / max(len(plan.steps), 1) * 100,
        blocking_issues=blockers,
    )

# Installation
session.install_computed(task_summary, sources=(Plan, Issue))

# Usage - automatically recomputes when sources change
summary = session[TaskSummary].latest()  # Same accessor pattern
```

### Design Notes

- **Lazy evaluation**: Only recomputes when sources change and value is accessed
- **Memoization**: Caches result until any source slice mutates
- **Type inference**: Derives input types from function signature
- **Read-only**: Cannot register reducers on computed slices
- **Deterministic**: Same inputs always produce same output (pure function)

### Implementation Sketch

```python
@dataclass(slots=True)
class ComputedSliceDescriptor(Generic[T]):
    compute_fn: Callable[..., T]
    sources: tuple[type, ...]
    _cached: T | None = field(default=None, init=False)
    _source_versions: tuple[int, ...] = field(default=(), init=False)

    def get(self, session: Session) -> T:
        current_versions = tuple(
            session._get_slice_version(s) for s in self.sources
        )
        if current_versions != self._source_versions:
            args = [session[s].latest() for s in self.sources]
            self._cached = self.compute_fn(*args)
            self._source_versions = current_versions
        return self._cached
```

---

## Proposal 2: Slice Windows and Eviction Policies

### Problem

Long-running agents accumulate unbounded state in LOG slices. Memory grows
indefinitely, and there's no mechanism for automatic cleanup.

### Solution

Introduce configurable retention policies for slices:

```python
from weakincentives.runtime import SliceWindow, EvictionPolicy

# Retain only the last N items
session[ToolInvoked].configure(
    window=SliceWindow.count(max_items=1000),
    eviction=EvictionPolicy.FIFO,
)

# Retain items from the last N seconds
session[Message].configure(
    window=SliceWindow.time(max_age_seconds=3600),
    eviction=EvictionPolicy.OLDEST_FIRST,
)

# Retain items matching a predicate (keep errors, evict successes)
session[ToolInvoked].configure(
    window=SliceWindow.predicate(
        keep=lambda t: not t.result.success,
        max_items=500,
    ),
)

# Composite: time + count limits
session[Event].configure(
    window=SliceWindow.composite(
        SliceWindow.count(max_items=10000),
        SliceWindow.time(max_age_seconds=7200),
    ),
)
```

### Eviction Strategies

```python
class EvictionPolicy(Enum):
    FIFO = "fifo"              # Remove oldest first
    LIFO = "lifo"              # Remove newest first (rare)
    PRIORITY = "priority"      # Use priority_fn to score items
    SAMPLING = "sampling"      # Keep statistically representative sample
```

### Design Notes

- **Non-breaking**: Default window is unbounded (current behavior)
- **Eviction triggers**: After each reducer execution, not during
- **Snapshot interaction**: Snapshots capture current window state; evicted items cannot be restored
- **Observer notification**: Eviction triggers observer with `(old_slice, new_slice)`

---

## Proposal 3: Middleware Chain

### Problem

Cross-cutting concerns like logging, metrics, validation, and rate limiting
require modification to every reducer. This violates DRY and makes auditing
difficult.

### Solution

Introduce a middleware chain that wraps reducer execution:

```python
from weakincentives.runtime import Middleware, ReducerContext

class LoggingMiddleware(Middleware):
    def __call__(
        self,
        ctx: ReducerContext,
        next_fn: Callable[[ReducerContext], T],
    ) -> T:
        start = time.perf_counter()
        result = next_fn(ctx)
        elapsed = time.perf_counter() - start
        logger.debug(
            "Reducer %s processed %s in %.3fms",
            ctx.reducer_name,
            ctx.event_type.__name__,
            elapsed * 1000,
        )
        return result

class ValidationMiddleware(Middleware):
    def __call__(
        self,
        ctx: ReducerContext,
        next_fn: Callable[[ReducerContext], T],
    ) -> T:
        result = next_fn(ctx)
        if hasattr(result, "__post_validate__"):
            result.__post_validate__()
        return result

class MetricsMiddleware(Middleware):
    def __init__(self, metrics: MetricsClient):
        self.metrics = metrics

    def __call__(
        self,
        ctx: ReducerContext,
        next_fn: Callable[[ReducerContext], T],
    ) -> T:
        with self.metrics.timer(f"reducer.{ctx.slice_type.__name__}"):
            return next_fn(ctx)

# Installation
session.use(
    LoggingMiddleware(),
    ValidationMiddleware(),
    MetricsMiddleware(metrics_client),
)
```

### ReducerContext

```python
@dataclass(frozen=True)
class ReducerContext(Generic[E, S]):
    event: E
    event_type: type[E]
    slice_type: type[S]
    previous_state: tuple[S, ...]
    reducer_name: str
    session: Session
    metadata: Mapping[str, Any]
```

### Design Notes

- **Composable**: Middleware chains in order of `use()` calls
- **Per-slice override**: `session[Plan].use(...)` for slice-specific middleware
- **Short-circuit**: Middleware can skip `next_fn()` to prevent reducer execution
- **Error handling**: Middleware can catch and transform exceptions

---

## Proposal 4: Effect System

### Problem

Reducers must be pure, but agents need side effects: API calls, file I/O,
notifications. Currently, side effects are scattered in tool handlers with
no coordination or retry logic.

### Solution

Introduce a declarative Effect system inspired by Redux-Saga:

```python
from weakincentives.runtime import Effect, effect, retry, timeout

@dataclass(frozen=True)
class SendNotification(Effect):
    channel: str
    message: str
    retry_policy: RetryPolicy = RetryPolicy.exponential(max_attempts=3)

@dataclass(frozen=True)
class FetchUrl(Effect[bytes]):
    url: str
    timeout_seconds: float = 30.0

@dataclass(frozen=True)
class WriteFile(Effect[None]):
    path: str
    content: bytes

# Effect handlers registered separately
@effect_handler(SendNotification)
async def handle_notification(eff: SendNotification, ctx: EffectContext) -> None:
    await ctx.http.post(f"/notify/{eff.channel}", json={"text": eff.message})

@effect_handler(FetchUrl)
async def handle_fetch(eff: FetchUrl, ctx: EffectContext) -> bytes:
    async with timeout(eff.timeout_seconds):
        resp = await ctx.http.get(eff.url)
        return await resp.read()

# Reducers yield effects instead of executing them
@reducer(on=TaskCompleted)
def on_complete(self, event: TaskCompleted) -> tuple[Plan, tuple[Effect, ...]]:
    new_plan = replace(self, status="done")
    effects = (
        SendNotification(channel="alerts", message=f"Task {event.task_id} done"),
        WriteFile(path=f"/logs/{event.task_id}.json", content=event.summary),
    )
    return new_plan, effects
```

### Effect Execution Pipeline

```
Reducer returns (new_state, effects)
           ↓
Session applies new_state
           ↓
EffectRunner.execute(effects)
           ↓
For each effect:
    - Look up handler
    - Apply retry policy
    - Execute with context
    - On failure: optionally dispatch compensation event
```

### Effect Composition

```python
from weakincentives.runtime import all_of, sequence, race

# Parallel execution
combined = all_of(
    FetchUrl("https://api.example.com/a"),
    FetchUrl("https://api.example.com/b"),
)

# Sequential execution
workflow = sequence(
    FetchUrl("https://api.example.com/token"),
    lambda token: FetchUrl(f"https://api.example.com/data?t={token}"),
)

# Race - first to complete wins
fastest = race(
    FetchUrl("https://primary.example.com/data"),
    FetchUrl("https://backup.example.com/data"),
)
```

### Design Notes

- **Testability**: Effects are data; tests can inspect without execution
- **Retry policies**: Exponential backoff, jitter, max attempts
- **Compensation**: Failed effects can trigger rollback events
- **Async-ready**: Effect handlers are async, reducers remain sync

---

## Proposal 5: State Machines

### Problem

Complex agent workflows have distinct phases (planning, executing, reviewing).
Encoding these as ad-hoc conditionals leads to:

- Implicit state transitions scattered across handlers
- No validation of allowed transitions
- Difficult to visualize or debug workflow

### Solution

First-class state machine support for workflow orchestration:

```python
from weakincentives.runtime import StateMachine, state, transition

class AgentPhase(StateMachine):
    # States
    idle = state(initial=True)
    planning = state()
    executing = state()
    reviewing = state()
    complete = state(terminal=True)
    failed = state(terminal=True)

    # Transitions with guards and actions
    @transition(from_=idle, to=planning)
    def start_planning(self, event: StartTask) -> None:
        """Triggered when task is assigned."""
        pass

    @transition(from_=planning, to=executing)
    def begin_execution(self, event: PlanApproved) -> None:
        """Requires plan approval before execution."""
        pass

    @transition(from_=executing, to=reviewing, guard=lambda e: e.all_steps_done)
    def request_review(self, event: ExecutionComplete) -> None:
        """Only transition if all steps completed."""
        pass

    @transition(from_=reviewing, to=complete)
    def approve(self, event: ReviewApproved) -> None:
        pass

    @transition(from_=reviewing, to=executing)
    def request_changes(self, event: ChangesRequested) -> None:
        """Loop back for revisions."""
        pass

    @transition(from_=(planning, executing, reviewing), to=failed)
    def abort(self, event: TaskAborted) -> None:
        """Can abort from any active phase."""
        pass

# Installation and usage
session.install(AgentPhase)
phase = session[AgentPhase].latest()
assert phase.current == AgentPhase.idle

# Transitions are events
session.broadcast(StartTask(task_id="123"))
assert session[AgentPhase].latest().current == AgentPhase.planning

# Invalid transitions raise
session.broadcast(ReviewApproved())  # InvalidTransition: planning → complete not allowed
```

### State Machine Features

```python
# Hierarchical states
class DetailedPhase(StateMachine):
    executing = state()
    executing_setup = state(parent=executing)
    executing_main = state(parent=executing)
    executing_cleanup = state(parent=executing)

# History states (remember sub-state when re-entering)
reviewing = state(history=True)

# Parallel regions
class ParallelWorkflow(StateMachine):
    region_a = region(states=[state_a1, state_a2])
    region_b = region(states=[state_b1, state_b2])

# Entry/exit actions
@state
def executing(self):
    @on_enter
    def setup(self) -> tuple[Effect, ...]:
        return (LogEvent("Entering execution phase"),)

    @on_exit
    def cleanup(self) -> tuple[Effect, ...]:
        return (FlushBuffers(),)
```

### Visualization

```python
# Generate Mermaid diagram
diagram = session[AgentPhase].to_mermaid()
print(diagram)
# stateDiagram-v2
#     [*] --> idle
#     idle --> planning: start_planning
#     planning --> executing: begin_execution
#     executing --> reviewing: request_review [all_steps_done]
#     reviewing --> complete: approve
#     reviewing --> executing: request_changes
#     planning --> failed: abort
#     executing --> failed: abort
#     reviewing --> failed: abort
#     complete --> [*]
#     failed --> [*]
```

---

## Proposal 6: Saga Pattern for Distributed Coordination

### Problem

Multi-step workflows involving external services need coordinated rollback.
If step 3 fails, steps 1 and 2 may need compensation. Current tool transaction
only handles in-memory state.

### Solution

Saga orchestrator for long-running, compensatable workflows:

```python
from weakincentives.runtime import Saga, saga_step, compensation

class DeploymentSaga(Saga):
    @saga_step(order=1)
    async def create_resources(self, ctx: SagaContext) -> ResourceIds:
        return await ctx.cloud.create_vm(self.config)

    @compensation(for_step=create_resources)
    async def destroy_resources(self, ctx: SagaContext, resources: ResourceIds) -> None:
        await ctx.cloud.delete_vm(resources.vm_id)

    @saga_step(order=2)
    async def configure_networking(self, ctx: SagaContext) -> NetworkConfig:
        resources = ctx.get_result(self.create_resources)
        return await ctx.cloud.setup_network(resources.vm_id)

    @compensation(for_step=configure_networking)
    async def teardown_networking(self, ctx: SagaContext, config: NetworkConfig) -> None:
        await ctx.cloud.delete_network(config.network_id)

    @saga_step(order=3)
    async def deploy_application(self, ctx: SagaContext) -> DeploymentResult:
        network = ctx.get_result(self.configure_networking)
        return await ctx.deploy(network.endpoint)

    @compensation(for_step=deploy_application)
    async def undeploy_application(self, ctx: SagaContext, result: DeploymentResult) -> None:
        await ctx.undeploy(result.deployment_id)

# Execution with automatic compensation on failure
async def run_deployment(session: Session, config: DeployConfig) -> None:
    saga = DeploymentSaga(config=config)
    try:
        result = await session.execute_saga(saga)
    except SagaFailed as e:
        # Compensations already executed in reverse order
        logger.error("Deployment failed at step %s: %s", e.failed_step, e.cause)
        raise
```

### Saga Execution Semantics

1. Execute steps in `order` sequence
2. Store each step's result in saga context
3. On step failure:
   - Execute compensations in reverse order (3 → 2 → 1)
   - Each compensation receives original step's result
   - Compensation failures are logged but don't stop rollback
4. On success: persist saga completion record

### Design Notes

- **Idempotent compensations**: Compensations should be safe to retry
- **Timeout per step**: Each step can have its own deadline
- **Checkpoint persistence**: Saga state persisted for crash recovery
- **Observability**: Saga execution emits events for tracing

---

## Proposal 7: Selective Snapshot and Restore

### Problem

Current snapshots are all-or-nothing. Sometimes you need to:

- Restore only specific slices
- Exclude certain slices from snapshot
- Merge partial snapshots

### Solution

Granular snapshot control:

```python
# Snapshot specific slices only
partial = session.snapshot(
    include={Plan, Config},
    exclude=None,
    tag="plan-checkpoint",
)

# Restore specific slices, preserve others
session.restore(
    partial,
    slices={Plan},  # Only restore Plan, keep current Config
)

# Merge snapshots (later values win)
merged = Snapshot.merge(
    snapshot_a,  # Has Plan
    snapshot_b,  # Has Config
)
session.restore(merged)

# Diff snapshots
diff = Snapshot.diff(before, after)
# SnapshotDiff(
#     added={Config: (Config(...),)},
#     removed={},
#     modified={Plan: (Plan(steps=()), Plan(steps=(Step(...),)))},
# )
```

### Snapshot Queries

```python
# Find snapshots by criteria
snapshots = session.find_snapshots(
    tag_pattern="tool:*",
    created_after=datetime(2024, 1, 1),
    contains_slice=Plan,
)

# Snapshot lineage (for debugging)
lineage = session.snapshot_lineage(current_snapshot)
# [snapshot_1, snapshot_2, ..., current_snapshot]
```

---

## Proposal 8: Typed Event Channels

### Problem

The current event bus is untyped—any event can be published to any handler.
This leads to runtime errors when handlers receive unexpected events.

### Solution

Typed channels that enforce compile-time event type checking:

```python
from weakincentives.runtime import Channel, channel

# Define typed channels
plan_events: Channel[PlanEvent] = channel("plan")
tool_events: Channel[ToolEvent] = channel("tools")
system_events: Channel[SystemEvent] = channel("system")

# Publish to specific channel
plan_events.publish(PlanUpdated(plan_id="123"))

# Subscribe with type safety
@plan_events.subscribe
def on_plan_event(event: PlanEvent) -> None:  # Type-checked!
    match event:
        case PlanUpdated(plan_id):
            ...
        case PlanCompleted(plan_id):
            ...

# Reducers declare their channel
@reducer(on=PlanUpdated, channel=plan_events)
def handle_plan_update(self, event: PlanUpdated) -> Plan:
    ...
```

### Channel Features

```python
# Filtered subscription
@plan_events.subscribe(filter=lambda e: e.priority == "high")
def on_high_priority(event: PlanEvent) -> None:
    ...

# Channel composition
all_events = plan_events | tool_events  # Union channel

# Replay capability
plan_events.replay(from_sequence=100)  # Re-emit events from sequence number
```

---

## Summary of Proposals

| Proposal | Problem Solved | Complexity | Priority |
|----------|---------------|------------|----------|
| Computed Slices | Derived state computation | Medium | High |
| Slice Windows | Unbounded memory growth | Low | High |
| Middleware | Cross-cutting concerns | Medium | Medium |
| Effect System | Side-effect coordination | High | Medium |
| State Machines | Workflow phase management | Medium | Medium |
| Saga Pattern | Distributed compensation | High | Low |
| Selective Snapshot | Granular state restore | Low | Medium |
| Typed Channels | Event type safety | Medium | Low |

## Implementation Roadmap

### Phase 1: Foundation (Low Risk, High Value)

1. **Slice Windows** - Simple addition to SliceAccessor
2. **Computed Slices** - New descriptor type, no breaking changes
3. **Selective Snapshot** - Extend existing Snapshot class

### Phase 2: Orchestration (Medium Risk, High Value)

4. **Middleware Chain** - Wrap reducer dispatch
5. **State Machines** - New slice type with transition validation

### Phase 3: Advanced Patterns (Higher Risk, Specialized Value)

6. **Effect System** - Requires async consideration
7. **Saga Pattern** - Depends on Effect System
8. **Typed Channels** - May require EventBus redesign

## Backward Compatibility

All proposals are additive. Existing code continues to work:

- Default window is unbounded
- No middleware by default
- Existing reducers unchanged
- Snapshot API extends, not replaces

## Open Questions

1. **Async reducers**: Should we support `async def` reducers for I/O-bound computations?
2. **Distributed sessions**: Should sessions support multi-process coordination?
3. **Time-travel debugging**: Should we add replay/step-through debugging tools?
4. **Schema versioning**: How to handle slice schema changes across versions?

---

## Concrete Refactoring Opportunities

This section identifies specific locations in the codebase that would benefit
from these abstractions, with before/after code examples.

### 1. Workspace Digest Index (Computed Slice)

**Location:** `src/weakincentives/contrib/tools/digests.py:52-82`

The current implementation filters through ALL workspace digests on every
lookup, creating O(n) complexity for what should be O(1).

**Before:**

```python
# digests.py:70-82 - O(n) scan on every lookup
def latest_workspace_digest(
    session: SessionProtocol,
    section_key: str,
) -> WorkspaceDigest | None:
    normalized_key = _normalized_key(section_key)
    entries = session[WorkspaceDigest].all()  # Get ALL entries
    for digest in reversed(entries):          # Scan backward
        if getattr(digest, "section_key", None) == normalized_key:
            return digest
    return None

# digests.py:52-57 - O(n) filter on every write
def set_workspace_digest(...) -> WorkspaceDigest:
    existing = tuple(
        digest
        for digest in session[WorkspaceDigest].all()
        if getattr(digest, "section_key", None) != normalized_key
    )
    session[WorkspaceDigest].seed((*existing, entry))
```

**After (with Computed Slice):**

```python
from weakincentives.runtime import computed

@computed
def workspace_digest_index(
    digests: tuple[WorkspaceDigest, ...]
) -> Mapping[str, WorkspaceDigest]:
    """Derives a section_key -> latest_digest index from WorkspaceDigest slice."""
    index: dict[str, WorkspaceDigest] = {}
    for digest in digests:
        # Later entries overwrite earlier ones (latest wins)
        index[digest.section_key] = digest
    return MappingProxyType(index)

# Installation (once during section init)
session.install_computed(
    workspace_digest_index,
    sources=(WorkspaceDigest,),
)

# Usage - O(1) lookup
def latest_workspace_digest(
    session: SessionProtocol,
    section_key: str,
) -> WorkspaceDigest | None:
    normalized_key = _normalized_key(section_key)
    index = session[WorkspaceDigestIndex].latest()
    return index.get(normalized_key)

# Write simplifies to append-only (index auto-updates)
def set_workspace_digest(...) -> WorkspaceDigest:
    entry = WorkspaceDigest(section_key=normalized_key, body=body.strip())
    session[WorkspaceDigest].append(entry)
    return entry
```

**Impact:** Reduces lookup from O(n) to O(1), eliminates filtering logic.

---

### 2. Bounded Event Ledgers (Slice Windows)

**Location:** `src/weakincentives/runtime/session/session.py:207-210`

The built-in LOG slices grow unbounded. Long-running agents accumulate
massive event histories.

**Before:**

```python
# session.py:207-210 - Unbounded growth
self._slice_policies: dict[SessionSliceType, SlicePolicy] = {
    _PROMPT_RENDERED_TYPE: SlicePolicy.LOG,
    _PROMPT_EXECUTED_TYPE: SlicePolicy.LOG,
    _TOOL_INVOKED_TYPE: SlicePolicy.LOG,
}
# No retention limit - grows forever
```

**After (with Slice Windows):**

```python
from weakincentives.runtime import SliceWindow, EvictionPolicy

# During session initialization
session[ToolInvoked].configure(
    window=SliceWindow.count(max_items=1000),
    eviction=EvictionPolicy.FIFO,
)
session[PromptExecuted].configure(
    window=SliceWindow.count(max_items=500),
    eviction=EvictionPolicy.FIFO,
)
session[PromptRendered].configure(
    window=SliceWindow.count(max_items=500),
    eviction=EvictionPolicy.FIFO,
)

# Or time-based for recent history
session[ToolInvoked].configure(
    window=SliceWindow.time(max_age_seconds=3600),  # Keep last hour
)

# Or composite for important events
session[ToolInvoked].configure(
    window=SliceWindow.composite(
        SliceWindow.predicate(
            keep=lambda t: not t.result.success,  # Always keep failures
            max_items=100,
        ),
        SliceWindow.count(max_items=900),  # Plus recent 900
    ),
)
```

**Impact:** Prevents OOM in long-running agents, keeps relevant history.

---

### 3. Plan Workflow State Machine

**Location:** `src/weakincentives/contrib/tools/planning.py:529-640`

The planning tools have implicit state machine semantics with scattered
validation. Every handler checks preconditions manually.

**Before:**

```python
# planning.py:529-605 - Scattered precondition checks
def setup_plan(self, params: SetupPlan, *, context: ToolContext) -> ToolResult[Plan]:
    ensure_context_uses_session(context=context, session=self._section.session)
    del context
    objective = _normalize_text(params.objective, "objective")  # Validation
    initial_steps = _normalize_step_titles(params.initial_steps)  # Validation
    # ... handler logic

def add_step(self, params: AddStep, *, context: ToolContext) -> ToolResult[Plan]:
    ensure_context_uses_session(context=context, session=self._section.session)
    del context
    session = self._section.session
    plan = _require_plan(session)    # Precondition: plan must exist
    _ensure_active(plan)              # Precondition: must be active
    normalized_steps = _normalize_step_titles(params.steps)  # Validation
    if not normalized_steps:
        raise ToolValidationError("Provide at least one step to add.")
    # ... handler logic

def update_step(self, params: UpdateStep, *, context: ToolContext) -> ToolResult[Plan]:
    ensure_context_uses_session(context=context, session=self._section.session)
    del context
    session = self._section.session
    plan = _require_plan(session)    # Same precondition
    _ensure_active(plan)              # Same precondition
    _ensure_step_exists(plan, params.step_id)  # Additional check
    # ... handler logic

# Helper functions scattered at bottom of file
def _require_plan(session: Session) -> Plan: ...
def _ensure_active(plan: Plan) -> None: ...
def _ensure_step_exists(plan: Plan, step_id: int) -> None: ...
```

**After (with State Machine):**

```python
from weakincentives.runtime import StateMachine, state, transition, guard

class PlanWorkflow(StateMachine):
    """Explicit state machine for plan lifecycle."""

    # States
    uninitialized = state(initial=True)
    active = state()
    completed = state(terminal=True)

    # Transitions with guards
    @transition(from_=uninitialized, to=active)
    def setup(self, event: SetupPlan) -> None:
        """Initialize a new plan."""
        pass

    @transition(from_=active, to=active)
    @guard(lambda self, e: len(e.steps) > 0, "Must provide at least one step")
    def add_step(self, event: AddStep) -> None:
        """Add steps to active plan."""
        pass

    @transition(from_=active, to=active)
    @guard(lambda self, e: self._step_exists(e.step_id), "Step must exist")
    def update_step(self, event: UpdateStep) -> None:
        """Update a step in active plan."""
        pass

    @transition(from_=active, to=completed)
    @guard(lambda self, e: self._all_steps_done(), "All steps must be done")
    def complete(self, event: CompletePlan) -> None:
        """Mark plan as completed."""
        pass

    def _step_exists(self, step_id: int) -> bool:
        plan = self.context.session[Plan].latest()
        return any(s.step_id == step_id for s in plan.steps)

    def _all_steps_done(self) -> bool:
        plan = self.context.session[Plan].latest()
        return all(s.status == "done" for s in plan.steps)

# Installation
session.install(PlanWorkflow)

# Handlers become simple - state machine validates preconditions
def add_step(self, params: AddStep, *, context: ToolContext) -> ToolResult[Plan]:
    # State machine validates: plan exists, is active, steps non-empty
    session.broadcast(params)  # Transition happens here
    return ToolResult(message="Steps added.", value=session[Plan].latest())

# Introspection for agents
workflow = session[PlanWorkflow].latest()
workflow.can_transition(AddStep(steps=("x",)))  # True/False
workflow.available_transitions()  # [AddStep, UpdateStep, CompletePlan]
```

**Impact:** Eliminates ~60 lines of scattered validation, makes state
transitions explicit and introspectable.

---

### 4. Validation Middleware

**Location:** `src/weakincentives/contrib/tools/planning.py:608-640`,
`src/weakincentives/contrib/tools/podman.py:491-560`

Validation patterns are duplicated across tool handlers.

**Before:**

```python
# planning.py:608-620 - Repeated in every handler
def _normalize_text(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ToolValidationError(f"{field_name.title()} must not be empty.")
    if len(stripped) > _MAX_TITLE_LENGTH:
        raise ToolValidationError(f"...must be <= {_MAX_TITLE_LENGTH} characters.")
    return stripped

# podman.py:491-510 - Similar pattern
def _ensure_ascii(value: str, *, field: str) -> str:
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        raise ToolValidationError(f"{field} must be ASCII.")

def _validate_command(cmd: list[str]) -> list[str]:
    if not cmd:
        raise ToolValidationError("command must contain at least one entry.")
    # ... more validation
```

**After (with Validation Middleware):**

```python
from weakincentives.runtime import Middleware, ReducerContext
from weakincentives.validation import (
    Validator, non_empty, max_length, ascii_only, compose
)

# Reusable validators
title_validator = compose(
    non_empty("title"),
    max_length(500, "title"),
)

command_validator = compose(
    non_empty("command"),
    each(ascii_only("command entry")),
)

class ValidationMiddleware(Middleware):
    """Apply registered validators before reducer execution."""

    def __init__(self, validators: Mapping[type, Validator]) -> None:
        self.validators = validators

    def __call__(
        self,
        ctx: ReducerContext,
        next_fn: Callable[[ReducerContext], T],
    ) -> T:
        validator = self.validators.get(ctx.event_type)
        if validator:
            validator(ctx.event)  # Raises ToolValidationError if invalid
        return next_fn(ctx)

# Registration
session.use(
    ValidationMiddleware({
        SetupPlan: compose(
            field("objective", title_validator),
            field("initial_steps", each(title_validator)),
        ),
        AddStep: field("steps", compose(non_empty(), each(title_validator))),
        UpdateStep: field("title", optional(title_validator)),
    })
)

# Handlers become pure business logic
def setup_plan(self, params: SetupPlan, *, context: ToolContext) -> ToolResult[Plan]:
    # Validation already done by middleware
    session.broadcast(params)
    return ToolResult(message="Plan created.", value=session[Plan].latest())
```

**Impact:** Centralizes validation, reduces handler boilerplate by ~50%.

---

### 5. Notification Retention (Slice Windows + Computed)

**Location:** `src/weakincentives/adapters/claude_agent_sdk/adapter.py:245`

Notifications use `append_all` with no retention limit.

**Before:**

```python
# adapter.py:245 - Unbounded notifications
session[Notification].register(Notification, append_all)

# Usage - accumulates forever
session.broadcast(Notification(source=..., payload=...))
```

**After:**

```python
from weakincentives.runtime import SliceWindow, EvictionPolicy

# Bounded retention with priority for errors
session[Notification].configure(
    window=SliceWindow.composite(
        # Keep all errors indefinitely (up to 100)
        SliceWindow.predicate(
            keep=lambda n: n.source == NotificationSource.ERROR,
            max_items=100,
        ),
        # Keep last 500 other notifications
        SliceWindow.count(max_items=500),
    ),
    eviction=EvictionPolicy.FIFO,
)

# Computed slice for quick error access
@computed
def notification_errors(
    notifications: tuple[Notification, ...]
) -> tuple[Notification, ...]:
    return tuple(n for n in notifications if n.source == NotificationSource.ERROR)

session.install_computed(notification_errors, sources=(Notification,))

# Usage
errors = session[NotificationErrors].latest()  # O(1) cached
```

---

## Priority Ranking for Implementation

Based on the codebase analysis, here's the recommended implementation order:

| Priority | Abstraction | Effort | Impact | Refactoring Target |
|----------|-------------|--------|--------|-------------------|
| **1** | Slice Windows | Low | High | LOG slices, Notification |
| **2** | Computed Slices | Medium | High | WorkspaceDigest index |
| **3** | Validation Middleware | Medium | Medium | Planning, Podman, VFS tools |
| **4** | State Machines | Medium | Medium | Plan workflow |
| **5** | Selective Snapshot | Low | Medium | Tool transactions |
| **6** | Effect System | High | Medium | Logging, external calls |
| **7** | Typed Channels | Medium | Low | Future-proofing |
| **8** | Saga Pattern | High | Low | Specialized use cases |

### Why This Order?

1. **Slice Windows** solves an immediate problem (memory growth) with minimal
   API surface. Non-breaking addition.

2. **Computed Slices** addresses a clear performance pattern (O(n) → O(1))
   visible in digests.py. The memoization pattern is well-understood.

3. **Validation Middleware** reduces boilerplate significantly across all
   tool handlers. Familiar concept from web frameworks.

4. **State Machines** makes implicit workflow logic explicit. Moderate
   complexity but high documentation value.

5-8. Lower priority as they address less common patterns or require more
     design work.

---



### Computed Slices

```python
class ComputedSlice(Protocol[T]):
    def latest(self) -> T: ...
    def invalidate(self) -> None: ...
    @property
    def is_stale(self) -> bool: ...

def computed(fn: Callable[..., T]) -> ComputedSliceDescriptor[T]: ...
```

### Slice Windows

```python
@dataclass(frozen=True)
class SliceWindow:
    @staticmethod
    def count(max_items: int) -> SliceWindow: ...
    @staticmethod
    def time(max_age_seconds: float) -> SliceWindow: ...
    @staticmethod
    def predicate(keep: Callable[[T], bool], max_items: int) -> SliceWindow: ...
    @staticmethod
    def composite(*windows: SliceWindow) -> SliceWindow: ...

class EvictionPolicy(Enum):
    FIFO = "fifo"
    LIFO = "lifo"
    PRIORITY = "priority"
    SAMPLING = "sampling"
```

### Middleware

```python
class Middleware(Protocol):
    def __call__(
        self,
        ctx: ReducerContext[E, S],
        next_fn: Callable[[ReducerContext[E, S]], S],
    ) -> S: ...

class Session:
    def use(self, *middleware: Middleware) -> None: ...
```

### State Machines

```python
class StateMachine:
    @property
    def current(self) -> State: ...
    def can_transition(self, event: Event) -> bool: ...
    def to_mermaid(self) -> str: ...

def state(
    initial: bool = False,
    terminal: bool = False,
    parent: State | None = None,
    history: bool = False,
) -> State: ...

def transition(
    from_: State | tuple[State, ...],
    to: State,
    guard: Callable[[Event], bool] | None = None,
) -> Callable[[Callable], Callable]: ...
```
