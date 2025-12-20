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

## Appendix: API Reference Sketches

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
