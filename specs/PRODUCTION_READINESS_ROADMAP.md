# Production Readiness Roadmap

This document assesses WINK's production readiness for building reliable
background agents and outlines a roadmap for addressing gaps.

## Executive Summary

WINK provides a **solid foundation** for production agent systems. The library
excels at:

- **Determinism and reproducibility** through immutable event ledgers
- **Reliability** via at-least-once delivery, transactions, and graceful
  shutdown
- **Correctness** with formal verification, design-by-contract, and 100% test
  coverage
- **Operability** through health endpoints, watchdogs, and structured logging

Key gaps exist in **observability at scale** (metrics, tracing),
**adaptive resource management**, and **enterprise features** (multi-tenancy,
audit trails).

---

## Current Production Capabilities

### What's Production-Ready

| Capability | Implementation | Confidence |
|------------|----------------|------------|
| Session State Management | Redux-style immutable event ledgers | High |
| Message Delivery | At-least-once via visibility timeout | High |
| Graceful Shutdown | Signal coordination, in-flight completion | High |
| Health Monitoring | Kubernetes probes, watchdog termination | High |
| Rate Limiting | Exponential backoff, provider signal handling | High |
| Deadline Enforcement | Wall-clock deadlines with checkpoints | High |
| Budget Tracking | Token limits across input/output/total | High |
| Transaction Rollback | Snapshot/restore on tool failure | High |
| Resource Lifecycle | Scoped DI with cleanup ordering | High |
| Persistence | JSONL slices, Redis mailbox | High |
| Formal Verification | TLA+ specs, property-based tests | High |
| Thread Safety | RLock patterns, copy-on-write | High |

### Architecture Strengths

**1. Deterministic Replay**

Sessions maintain an immutable event ledger. Any session can be replayed from
its event history, enabling:

- Post-hoc debugging of production failures
- Deterministic testing with recorded sessions
- A/B testing of reducer changes against historical data

**2. Provider Isolation**

Adapters abstract provider differences behind a unified interface. Rate limits,
token accounting, and structured output work identically across OpenAI,
LiteLLM, and Claude Agent SDK.

**3. Transactional Tool Execution**

The `tool_transaction` context manager creates composite snapshots before tool
execution and automatically restores on failure:

```python
with tool_transaction(session, resources) as snapshot:
    result = execute_tool(...)
    # If exception raised, session and resources are restored
```

**4. Defense in Depth**

Multiple layers prevent runaway agents:

- Deadlines enforce wall-clock time limits
- Budgets enforce token consumption limits
- Watchdog terminates stuck workers
- Task completion checkers verify objectives before stopping

**5. Formal Correctness**

Critical components (RedisMailbox) have embedded TLA+ specifications with
verified invariants:

- Message state exclusivity
- Receipt handle freshness
- No message loss
- Delivery count monotonicity

---

## Gap Analysis

### P0: Critical for Production

#### Metrics Collection

**Gap**: No integration with metrics systems (Prometheus, StatsD, CloudWatch).

**Impact**: Cannot monitor agent health, performance, or costs at scale.
Operators have no visibility into:

- Request latency distributions
- Token consumption rates
- Error rates by error type
- Queue depths and processing rates

**Recommendation**: Add a `MetricsCollector` protocol with implementations for
common backends:

```python
class MetricsCollector(Protocol):
    def counter(self, name: str, value: int = 1, tags: Tags = None) -> None: ...
    def gauge(self, name: str, value: float, tags: Tags = None) -> None: ...
    def histogram(self, name: str, value: float, tags: Tags = None) -> None: ...
    def timer(self, name: str) -> ContextManager[None]: ...
```

Key metrics to emit:

| Metric | Type | Description |
|--------|------|-------------|
| `wink.prompt.duration_seconds` | Histogram | Prompt evaluation latency |
| `wink.prompt.tokens.input` | Counter | Input tokens consumed |
| `wink.prompt.tokens.output` | Counter | Output tokens generated |
| `wink.tool.invocations` | Counter | Tool calls by tool name |
| `wink.tool.duration_seconds` | Histogram | Tool execution latency |
| `wink.mailbox.depth` | Gauge | Messages waiting |
| `wink.mailbox.age_seconds` | Gauge | Oldest message age |
| `wink.errors` | Counter | Errors by type |
| `wink.session.active` | Gauge | Active sessions |

#### Distributed Tracing

**Gap**: No trace context propagation or span emission.

**Impact**: Cannot trace requests across service boundaries or understand
end-to-end latency breakdown.

**Recommendation**: Add OpenTelemetry integration:

```python
class TracingAdapter:
    """Wraps a ProviderAdapter to emit spans."""

    def evaluate(self, prompt, *, session, ...) -> ProviderResponse:
        with tracer.start_as_current_span("wink.prompt.evaluate") as span:
            span.set_attribute("prompt.namespace", prompt.ns)
            span.set_attribute("prompt.key", prompt.key)
            # ... delegate to wrapped adapter
```

Trace context should flow through:

- Mailbox messages (propagate trace ID in message metadata)
- Tool executions (child spans)
- Provider calls (child spans with model info)

#### Dead Letter Queue

**Gap**: Poison messages retry indefinitely until budget exhaustion.

**Impact**: A malformed request can consume resources repeatedly before
failing, and operators have no way to inspect or replay failed messages.

**Recommendation**: Add DLQ support to mailbox protocol:

```python
class Mailbox(Protocol[T, R]):
    def send_to_dlq(
        self,
        message: Message[T, R],
        reason: str,
        *,
        max_retries: int = 3,
    ) -> None:
        """Move message to dead letter queue after max retries."""
```

MainLoop should check `delivery_count` and route to DLQ when threshold
exceeded.

---

### P1: Important for Scale

#### Circuit Breaker

**Gap**: ThrottlePolicy retries until exhaustion but doesn't prevent cascade
failures.

**Impact**: When a provider is degraded, all workers continue hammering it,
potentially making the situation worse.

**Recommendation**: Add circuit breaker with three states:

- **Closed**: Normal operation, track failure rate
- **Open**: Fail fast without calling provider
- **Half-Open**: Allow probe requests to test recovery

```python
@dataclass
class CircuitBreakerPolicy:
    failure_threshold: int = 5  # Failures to open
    success_threshold: int = 3  # Successes to close
    open_duration: timedelta = timedelta(seconds=60)

class CircuitBreaker:
    def call(self, fn: Callable[[], T]) -> T:
        if self.state == State.OPEN:
            if self._should_probe():
                self.state = State.HALF_OPEN
            else:
                raise CircuitOpenError(...)
        # ...
```

#### Adaptive Budgets

**Gap**: Budgets are static per-prompt configuration.

**Impact**: No ability to learn from historical patterns or adjust based on
task complexity.

**Recommendation**: Add `BudgetEstimator` protocol:

```python
class BudgetEstimator(Protocol):
    def estimate(
        self,
        prompt: PromptProtocol,
        *,
        context: EstimationContext,
    ) -> Budget:
        """Estimate budget based on prompt characteristics."""
```

Implementations could:

- Use historical p95 for similar prompts
- Adjust based on input complexity (code size, task description length)
- Apply safety margins based on confidence

#### Streaming Support

**Gap**: All operations are eager; no streaming or backpressure handling.

**Impact**: Large outputs must be buffered entirely in memory before
processing.

**Recommendation**: Add streaming protocol:

```python
class StreamingAdapter(Protocol):
    async def evaluate_streaming(
        self,
        prompt: PromptProtocol,
        *,
        session: Session,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response chunks as they arrive."""
```

StreamChunk types:

- `TextDelta`: Partial text content
- `ToolCallStart`: Tool invocation beginning
- `ToolCallDelta`: Tool call argument chunks
- `ToolCallEnd`: Tool call complete
- `Usage`: Token counts

---

### P2: Enterprise Features

#### Multi-Tenancy

**Gap**: Resource scopes are per-session, not per-tenant.

**Impact**: No isolation between tenants sharing infrastructure. Cannot
enforce per-tenant quotas or routing.

**Recommendation**: Add tenant context:

```python
@dataclass
class TenantContext:
    tenant_id: str
    quotas: TenantQuotas
    routing: TenantRouting

class TenantAwareMailbox(Mailbox[T, R]):
    def receive(self, ...) -> Sequence[Message[T, R]]:
        # Filter by tenant, enforce quotas
```

#### Audit Trail

**Gap**: Events are session-scoped; no cross-session audit log.

**Impact**: Cannot reconstruct what happened across sessions for compliance
or debugging.

**Recommendation**: Add immutable audit log:

```python
class AuditLogger(Protocol):
    def log(self, entry: AuditEntry) -> None:
        """Append entry to immutable audit log."""

@dataclass
class AuditEntry:
    timestamp: datetime
    session_id: UUID
    event_type: str
    actor: str  # user, system, agent
    details: Mapping[str, Any]
    # Optional: cryptographic chaining for tamper evidence
```

#### Cost Optimization

**Gap**: No model selection or batch API support.

**Impact**: Users cannot optimize cost/quality tradeoffs or leverage cheaper
batch processing for non-urgent work.

**Recommendation**:

1. Add model routing:

```python
class ModelRouter(Protocol):
    def select_model(
        self,
        prompt: PromptProtocol,
        *,
        constraints: ModelConstraints,
    ) -> str:
        """Select model based on task requirements."""
```

2. Add batch API support for non-interactive workloads:

```python
class BatchAdapter(Protocol):
    def submit_batch(
        self,
        prompts: Sequence[PromptProtocol],
    ) -> BatchJob:
        """Submit prompts for batch processing."""

    def poll_batch(self, job: BatchJob) -> BatchResult:
        """Check batch job status and retrieve results."""
```

---

### P3: Nice to Have

#### Real-Time Dashboard

**Gap**: No built-in operational visibility beyond health endpoints.

**Impact**: Operators must build custom dashboards from scratch.

**Recommendation**: Extend `wink debug` with real-time views:

- Active sessions and their states
- Message queue depths
- Recent errors and warnings
- Resource utilization

#### Chaos Engineering Support

**Gap**: Fault injection exists in tests but isn't exposed for production
validation.

**Impact**: Cannot validate resilience in staging environments.

**Recommendation**: Add runtime fault injection:

```python
@dataclass
class ChaosPolicy:
    provider_failure_rate: float = 0.0
    provider_latency_ms: int = 0
    mailbox_failure_rate: float = 0.0
```

---

## Implementation Roadmap

### Phase 1: Observability Foundation (P0)

**Goal**: Enable production monitoring and debugging.

1. **Metrics Protocol**
   - Define `MetricsCollector` protocol
   - Add null implementation (default, no-op)
   - Add Prometheus implementation in contrib
   - Instrument adapters, mailbox, session

2. **Tracing Protocol**
   - Define `Tracer` protocol
   - Add null implementation (default)
   - Add OpenTelemetry implementation in contrib
   - Propagate trace context through mailbox

3. **Dead Letter Queue**
   - Extend mailbox protocol with DLQ methods
   - Add delivery count checking to MainLoop
   - Implement DLQ for Redis mailbox
   - Add DLQ inspection CLI command

**Deliverables**:
- `specs/OBSERVABILITY.md`
- `src/weakincentives/observability/` module
- `src/weakincentives/contrib/observability/` implementations

### Phase 2: Resilience Patterns (P1)

**Goal**: Handle degraded conditions gracefully.

1. **Circuit Breaker**
   - Add circuit breaker to adapters
   - Integrate with metrics for state visibility
   - Add health check integration (ready=false when open)

2. **Adaptive Budgets**
   - Define `BudgetEstimator` protocol
   - Add historical statistics collector
   - Implement percentile-based estimator

3. **Streaming Foundation**
   - Define streaming protocol
   - Add streaming support to OpenAI adapter
   - Add backpressure handling

**Deliverables**:
- `specs/CIRCUIT_BREAKER.md`
- `specs/ADAPTIVE_BUDGETS.md`
- Streaming adapter implementations

### Phase 3: Enterprise Features (P2)

**Goal**: Support enterprise deployment patterns.

1. **Multi-Tenancy**
   - Add tenant context to session
   - Implement tenant-aware mailbox routing
   - Add per-tenant quota enforcement

2. **Audit Trail**
   - Define audit protocol
   - Add SQL-based implementation
   - Integrate with session events

3. **Cost Optimization**
   - Add model routing protocol
   - Add batch API support for OpenAI

**Deliverables**:
- `specs/MULTI_TENANCY.md`
- `specs/AUDIT.md`
- Enterprise-focused implementations

### Phase 4: Operational Excellence (P3)

**Goal**: Enhance day-to-day operations.

1. **Enhanced Debug UI**
   - Real-time session views
   - Queue depth visualization
   - Error aggregation

2. **Chaos Engineering**
   - Runtime fault injection
   - Resilience validation tooling

**Deliverables**:
- Enhanced `wink debug` command
- Chaos testing utilities

---

## Current State Summary

```
Production Readiness Score: 7/10

Excellent (9-10/10):
  - Session management and determinism
  - Message delivery guarantees
  - Graceful shutdown
  - Formal verification

Good (7-8/10):
  - Rate limiting and throttling
  - Health monitoring
  - Structured logging

Needs Work (5-6/10):
  - Metrics and monitoring
  - Distributed tracing
  - Dead letter handling

Missing (0-4/10):
  - Circuit breakers
  - Adaptive budgets
  - Multi-tenancy
  - Audit trails
```

**Bottom Line**: WINK is production-ready for **single-tenant deployments at
moderate scale** with external monitoring. For high-volume or enterprise
deployments, invest in Phase 1 (observability) before going live.

---

## Appendix: Existing Abstraction Value

### Why Redux-Style Sessions Matter

Traditional agent frameworks store state in mutable objects, making debugging
difficult and replay impossible. WINK's immutable event ledger provides:

1. **Time Travel Debugging**: Restore any historical state
2. **Deterministic Tests**: Replay recorded sessions with expected outcomes
3. **Safe Concurrency**: No races on shared mutable state
4. **Audit Trail**: Every state change is an explicit event

### Why Formal Verification Matters

The RedisMailbox TLA+ spec proves critical invariants:

- Messages are never lost (INV-5)
- Duplicate processing is prevented (INV-3)
- State transitions are atomic (INV-1)

These guarantees hold under concurrent access and network partitions, which
unit tests cannot verify.

### Why Design-by-Contract Matters

DbC decorators catch contract violations during testing:

```python
@require(lambda budget: budget.total > 0, "budget must be positive")
@ensure(lambda result: result.tokens_used <= result.budget.total)
def evaluate(prompt, *, budget: Budget) -> ProviderResponse:
    ...
```

Violations surface as clear assertion errors with full context, not mysterious
downstream failures.

### Why Scoped Resources Matter

The resource registry ensures:

- Connections are reused within scope (SINGLETON)
- Per-request state is isolated (TOOL_CALL)
- Cleanup happens in reverse order
- Cycles are detected at registration time

This prevents resource leaks and connection exhaustion under load.

---

## References

| Spec | Relevance |
|------|-----------|
| `specs/HEALTH.md` | Current health monitoring |
| `specs/LIFECYCLE.md` | Graceful shutdown |
| `specs/MAILBOX.md` | Message delivery |
| `specs/SESSIONS.md` | State management |
| `specs/ADAPTERS.md` | Provider integration |
| `specs/VERIFICATION.md` | Formal correctness |
| `specs/RESOURCE_REGISTRY.md` | Dependency injection |
| `specs/TASK_COMPLETION.md` | Objective verification |
