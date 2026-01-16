# Metrics Specification

## Purpose

In-memory metrics collection for observability. Captures latency, throughput,
error rates, and queue health across adapters, tools, and mailbox.

**Use metrics for:** Latency analysis, token tracking, failure rates, queue health.

**Not for:** Billing (use session token budgets), alerting (export to external systems).

## Design Principles

- **In-memory storage**: Metrics accumulate in collector without Session persistence.
  Avoids session bloat—metrics are observability, not application state.
- **Resource-scoped lifecycle**: Bound via `ResourceRegistry` with `Scope.SINGLETON`,
  surviving across requests within a worker process.
- **Optional persistence**: `weakincentives.debug` provides `dump_metrics()` for
  post-hoc analysis.
- **Compact histograms**: Fixed exponential buckets bound memory regardless of volume.
- **RunContext correlation**: Integrates with distributed tracing.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ResourceRegistry                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              MetricsCollector (Scope.SINGLETON)                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │ │
│  │  │ Counter[T]  │  │Histogram[T] │  │  Gauge[T]   │                 │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │ │
│  │                                                                     │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │ │
│  │  │ AdapterMetrics   │  │  ToolMetrics     │  │ MailboxMetrics   │  │ │
│  │  │ - latency hist   │  │ - latency hist   │  │ - lag hist       │  │ │
│  │  │ - token counter  │  │ - failure count  │  │ - retry dist     │  │ │
│  │  │ - error count    │  │ - call count     │  │ - nack count     │  │ │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
         ▲                        ▲                        ▲
         │                        │                        │
    ┌────┴────┐              ┌────┴────┐              ┌────┴────┐
    │ Adapter │              │  Tool   │              │ Mailbox │
    │  Layer  │              │ Executor│              │  Layer  │
    └─────────┘              └─────────┘              └─────────┘
                                  │
                                  ▼ (optional)
                      ┌───────────────────────┐
                      │  weakincentives.debug │
                      │  dump_metrics(path)   │
                      └───────────────────────┘
```

## Metric Primitives

### Counter

Monotonically increasing value. Supports increment by arbitrary positive delta.

```python
from dataclasses import dataclass, field
from weakincentives.dataclasses import FrozenDataclass

@FrozenDataclass()
class Counter:
    """Monotonically increasing counter with labeled dimensions."""

    name: str
    value: int = 0
    labels: tuple[tuple[str, str], ...] = ()

    def inc(self, delta: int = 1) -> "Counter":
        """Return new Counter with incremented value."""
        if delta < 0:
            raise ValueError("Counter delta must be non-negative")
        return Counter(
            name=self.name,
            value=self.value + delta,
            labels=self.labels,
        )
```

### Histogram

Captures value distributions using fixed exponential buckets. Stores counts per
bucket rather than raw values for bounded memory.

```python
import math
from typing import ClassVar

@FrozenDataclass()
class Histogram:
    """Fixed-bucket histogram for latency and size distributions."""

    name: str
    # Bucket boundaries in milliseconds (exponential: 1, 2, 4, 8, ... 65536)
    LATENCY_BUCKETS: ClassVar[tuple[int, ...]] = (
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536,
    )
    # Bucket counts (one extra for +Inf)
    bucket_counts: tuple[int, ...] = field(
        default_factory=lambda: (0,) * 18
    )
    total_count: int = 0
    total_sum: int = 0  # Sum of all observed values
    labels: tuple[tuple[str, str], ...] = ()

    def observe(self, value: int) -> "Histogram":
        """Return new Histogram with observed value in appropriate bucket."""
        counts = list(self.bucket_counts)
        for i, boundary in enumerate(self.LATENCY_BUCKETS):
            if value <= boundary:
                counts[i] += 1
                break
        else:
            counts[-1] += 1  # +Inf bucket
        return Histogram(
            name=self.name,
            bucket_counts=tuple(counts),
            total_count=self.total_count + 1,
            total_sum=self.total_sum + value,
            labels=self.labels,
        )

    def percentile(self, p: float) -> int | None:
        """Estimate percentile from bucket distribution."""
        if self.total_count == 0:
            return None
        # Clamp to at least 1 to avoid returning empty buckets
        target = max(1, math.ceil(self.total_count * p))
        cumulative = 0
        for i, count in enumerate(self.bucket_counts):
            cumulative += count
            if cumulative >= target:
                if i < len(self.LATENCY_BUCKETS):
                    return self.LATENCY_BUCKETS[i]
                return self.LATENCY_BUCKETS[-1]  # +Inf approximation
        return None

    @property
    def mean(self) -> float | None:
        """Mean of observed values."""
        if self.total_count == 0:
            return None
        return self.total_sum / self.total_count
```

### Gauge

Point-in-time value that can increase or decrease. Used for queue depths and
concurrent operation counts.

```python
@FrozenDataclass()
class Gauge:
    """Point-in-time gauge value."""

    name: str
    value: int = 0
    labels: tuple[tuple[str, str], ...] = ()

    def set(self, value: int) -> "Gauge":
        """Return new Gauge with updated value."""
        return Gauge(name=self.name, value=value, labels=self.labels)

    def inc(self, delta: int = 1) -> "Gauge":
        """Return new Gauge with incremented value."""
        return Gauge(name=self.name, value=self.value + delta, labels=self.labels)

    def dec(self, delta: int = 1) -> "Gauge":
        """Return new Gauge with decremented value."""
        return Gauge(name=self.name, value=self.value - delta, labels=self.labels)
```

## Core Metric Types

### AdapterMetrics

Captures prompt execution performance per adapter.

```python
from datetime import datetime, UTC
from uuid import UUID, uuid4
from weakincentives.adapters._names import AdapterName

@FrozenDataclass()
class AdapterMetrics:
    """Aggregated adapter metrics with compact histogram storage."""

    adapter: AdapterName
    # Latency histograms by phase
    render_latency: Histogram = field(
        default_factory=lambda: Histogram(name="adapter.render_latency_ms")
    )
    call_latency: Histogram = field(
        default_factory=lambda: Histogram(name="adapter.call_latency_ms")
    )
    parse_latency: Histogram = field(
        default_factory=lambda: Histogram(name="adapter.parse_latency_ms")
    )
    tool_latency: Histogram = field(
        default_factory=lambda: Histogram(name="adapter.tool_latency_ms")
    )
    total_latency: Histogram = field(
        default_factory=lambda: Histogram(name="adapter.total_latency_ms")
    )
    # Token counters
    input_tokens: Counter = field(
        default_factory=lambda: Counter(name="adapter.input_tokens")
    )
    output_tokens: Counter = field(
        default_factory=lambda: Counter(name="adapter.output_tokens")
    )
    cached_tokens: Counter = field(
        default_factory=lambda: Counter(name="adapter.cached_tokens")
    )
    # Status counters
    request_count: Counter = field(
        default_factory=lambda: Counter(name="adapter.requests")
    )
    error_count: Counter = field(
        default_factory=lambda: Counter(name="adapter.errors")
    )
    throttle_count: Counter = field(
        default_factory=lambda: Counter(name="adapter.throttles")
    )
    # Timestamp for rollup tracking
    period_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### ToolMetrics

Captures tool execution performance and failure rates.

```python
@FrozenDataclass()
class ToolMetrics:
    """Per-tool execution metrics."""

    tool_name: str
    # Latency
    latency: Histogram = field(
        default_factory=lambda: Histogram(name="tool.latency_ms")
    )
    # Counters
    call_count: Counter = field(
        default_factory=lambda: Counter(name="tool.calls")
    )
    success_count: Counter = field(
        default_factory=lambda: Counter(name="tool.successes")
    )
    failure_count: Counter = field(
        default_factory=lambda: Counter(name="tool.failures")
    )
    # Error breakdown by code
    error_codes: tuple[tuple[str, int], ...] = ()
    # Timestamp
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def failure_rate(self) -> float | None:
        """Calculate failure rate as failures / total calls."""
        total = self.call_count.value
        if total == 0:
            return None
        return self.failure_count.value / total
```

### MailboxMetrics

Captures message queue health and delivery patterns.

```python
@FrozenDataclass()
class MailboxMetrics:
    """Message queue health metrics."""

    queue_name: str
    # Queue lag (age of oldest unprocessed message in ms)
    queue_lag: Histogram = field(
        default_factory=lambda: Histogram(name="mailbox.queue_lag_ms")
    )
    # Delivery count distribution (how many times messages are delivered)
    delivery_count_dist: tuple[int, ...] = field(
        default_factory=lambda: (0,) * 10  # Buckets: 1, 2, 3, ..., 10+
    )
    # Counters
    messages_received: Counter = field(
        default_factory=lambda: Counter(name="mailbox.received")
    )
    messages_acked: Counter = field(
        default_factory=lambda: Counter(name="mailbox.acked")
    )
    messages_nacked: Counter = field(
        default_factory=lambda: Counter(name="mailbox.nacked")
    )
    messages_expired: Counter = field(
        default_factory=lambda: Counter(name="mailbox.expired")
    )
    messages_dead_lettered: Counter = field(
        default_factory=lambda: Counter(name="mailbox.dead_lettered")
    )
    # Retry tracking
    total_retries: Counter = field(
        default_factory=lambda: Counter(name="mailbox.retries")
    )
    # Current queue depth (gauge semantics but stored as snapshot)
    queue_depth: int = 0
    oldest_message_age_ms: int | None = None
    # Timestamp
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def observe_delivery(self, delivery_count: int) -> "MailboxMetrics":
        """Record delivery count for a message."""
        dist = list(self.delivery_count_dist)
        bucket = min(delivery_count - 1, 9)  # 1-indexed, cap at 10+
        if bucket >= 0:
            dist[bucket] += 1
        retries = delivery_count - 1 if delivery_count > 1 else 0
        return MailboxMetrics(
            queue_name=self.queue_name,
            queue_lag=self.queue_lag,
            delivery_count_dist=tuple(dist),
            messages_received=self.messages_received.inc(),
            messages_acked=self.messages_acked,
            messages_nacked=self.messages_nacked,
            messages_expired=self.messages_expired,
            total_retries=self.total_retries.inc(retries),
            queue_depth=self.queue_depth,
            oldest_message_age_ms=self.oldest_message_age_ms,
            last_updated=datetime.now(UTC),
        )
```

## MetricsCollector Protocol

Central interface for recording metrics. Wired into Session resources.

```python
from typing import Protocol
from weakincentives.runtime.events import TokenUsage

class MetricsCollector(Protocol):
    """Protocol for metrics collection."""

    def record_adapter_call(
        self,
        adapter: AdapterName,
        *,
        render_ms: int,
        call_ms: int,
        parse_ms: int,
        tool_ms: int,
        usage: TokenUsage | None = None,
        error: str | None = None,
        throttled: bool = False,
    ) -> None:
        """Record a complete adapter call with phase timing."""
        ...

    def record_tool_call(
        self,
        tool_name: str,
        *,
        latency_ms: int,
        success: bool,
        error_code: str | None = None,
    ) -> None:
        """Record a tool execution."""
        ...

    def record_message_received(
        self,
        queue_name: str,
        *,
        delivery_count: int,
        age_ms: int,
    ) -> None:
        """Record message receipt with queue lag."""
        ...

    def record_message_ack(self, queue_name: str) -> None:
        """Record successful message acknowledgment."""
        ...

    def record_message_nack(self, queue_name: str) -> None:
        """Record message negative acknowledgment (will retry)."""
        ...

    def record_message_dead_lettered(self, queue_name: str) -> None:
        """Record message sent to dead letter queue."""
        ...

    def record_queue_depth(self, queue_name: str, depth: int) -> None:
        """Record current queue depth snapshot."""
        ...

    def get_adapter_metrics(self, adapter: AdapterName) -> AdapterMetrics | None:
        """Retrieve metrics for a specific adapter."""
        ...

    def get_tool_metrics(self, tool_name: str) -> ToolMetrics | None:
        """Retrieve metrics for a specific tool."""
        ...

    def get_mailbox_metrics(self, queue_name: str) -> MailboxMetrics | None:
        """Retrieve metrics for a specific queue."""
        ...

    def snapshot(self) -> "MetricsSnapshot":
        """Return immutable snapshot of all metrics."""
        ...
```

## Collector Implementation

### MetricsSnapshot

Immutable snapshot of all collected metrics.

```python
@FrozenDataclass()
class MetricsSnapshot:
    """Complete metrics snapshot for export or inspection."""

    adapters: tuple[AdapterMetrics, ...] = ()
    tools: tuple[ToolMetrics, ...] = ()
    mailboxes: tuple[MailboxMetrics, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    worker_id: str | None = None

    def adapter(self, name: AdapterName) -> AdapterMetrics | None:
        """Look up adapter metrics by name."""
        for m in self.adapters:
            if m.adapter == name:
                return m
        return None

    def tool(self, name: str) -> ToolMetrics | None:
        """Look up tool metrics by name."""
        for m in self.tools:
            if m.tool_name == name:
                return m
        return None

    def mailbox(self, name: str) -> MailboxMetrics | None:
        """Look up mailbox metrics by name."""
        for m in self.mailboxes:
            if m.queue_name == name:
                return m
        return None

    def total_tokens(self) -> tuple[int, int, int]:
        """Sum of (input, output, cached) tokens across all adapters."""
        input_t = sum(m.input_tokens.value for m in self.adapters)
        output_t = sum(m.output_tokens.value for m in self.adapters)
        cached_t = sum(m.cached_tokens.value for m in self.adapters)
        return (input_t, output_t, cached_t)
```

### InMemoryMetricsCollector

Default implementation storing metrics in memory. No automatic persistence—
metrics live for the lifetime of the collector (typically worker process).

```python
from dataclasses import replace

class InMemoryMetricsCollector:
    """In-memory metrics collector with no automatic persistence."""

    def __init__(self, worker_id: str | None = None) -> None:
        self._worker_id = worker_id
        self._adapters: dict[AdapterName, AdapterMetrics] = {}
        self._tools: dict[str, ToolMetrics] = {}
        self._mailboxes: dict[str, MailboxMetrics] = {}

    def record_adapter_call(
        self,
        adapter: AdapterName,
        *,
        render_ms: int,
        call_ms: int,
        parse_ms: int,
        tool_ms: int,
        usage: TokenUsage | None = None,
        error: str | None = None,
        throttled: bool = False,
    ) -> None:
        metrics = self._adapters.get(adapter) or AdapterMetrics(adapter=adapter)
        total_ms = render_ms + call_ms + parse_ms + tool_ms

        metrics = replace(
            metrics,
            render_latency=metrics.render_latency.observe(render_ms),
            call_latency=metrics.call_latency.observe(call_ms),
            parse_latency=metrics.parse_latency.observe(parse_ms),
            tool_latency=metrics.tool_latency.observe(tool_ms),
            total_latency=metrics.total_latency.observe(total_ms),
            request_count=metrics.request_count.inc(),
            last_updated=datetime.now(UTC),
        )

        if usage:
            metrics = replace(
                metrics,
                input_tokens=metrics.input_tokens.inc(usage.input_tokens or 0),
                output_tokens=metrics.output_tokens.inc(usage.output_tokens or 0),
                cached_tokens=metrics.cached_tokens.inc(usage.cached_tokens or 0),
            )

        if error:
            metrics = replace(metrics, error_count=metrics.error_count.inc())

        if throttled:
            metrics = replace(metrics, throttle_count=metrics.throttle_count.inc())

        self._adapters[adapter] = metrics

    def record_tool_call(
        self,
        tool_name: str,
        *,
        latency_ms: int,
        success: bool,
        error_code: str | None = None,
    ) -> None:
        metrics = self._tools.get(tool_name) or ToolMetrics(tool_name=tool_name)

        metrics = replace(
            metrics,
            latency=metrics.latency.observe(latency_ms),
            call_count=metrics.call_count.inc(),
            last_updated=datetime.now(UTC),
        )

        if success:
            metrics = replace(metrics, success_count=metrics.success_count.inc())
        else:
            metrics = replace(metrics, failure_count=metrics.failure_count.inc())
            if error_code:
                codes = dict(metrics.error_codes)
                codes[error_code] = codes.get(error_code, 0) + 1
                metrics = replace(metrics, error_codes=tuple(codes.items()))

        self._tools[tool_name] = metrics

    def record_message_received(
        self,
        queue_name: str,
        *,
        delivery_count: int,
        age_ms: int,
    ) -> None:
        metrics = self._mailboxes.get(queue_name) or MailboxMetrics(
            queue_name=queue_name
        )
        metrics = metrics.observe_delivery(delivery_count)
        metrics = replace(
            metrics,
            queue_lag=metrics.queue_lag.observe(age_ms),
            oldest_message_age_ms=max(
                metrics.oldest_message_age_ms or 0, age_ms
            ),
        )
        self._mailboxes[queue_name] = metrics

    def record_message_ack(self, queue_name: str) -> None:
        metrics = self._mailboxes.get(queue_name) or MailboxMetrics(
            queue_name=queue_name
        )
        metrics = replace(
            metrics,
            messages_acked=metrics.messages_acked.inc(),
            last_updated=datetime.now(UTC),
        )
        self._mailboxes[queue_name] = metrics

    def record_message_nack(self, queue_name: str) -> None:
        metrics = self._mailboxes.get(queue_name) or MailboxMetrics(
            queue_name=queue_name
        )
        metrics = replace(
            metrics,
            messages_nacked=metrics.messages_nacked.inc(),
            last_updated=datetime.now(UTC),
        )
        self._mailboxes[queue_name] = metrics

    def record_message_dead_lettered(self, queue_name: str) -> None:
        metrics = self._mailboxes.get(queue_name) or MailboxMetrics(
            queue_name=queue_name
        )
        metrics = replace(
            metrics,
            messages_dead_lettered=metrics.messages_dead_lettered.inc(),
            last_updated=datetime.now(UTC),
        )
        self._mailboxes[queue_name] = metrics

    def record_queue_depth(self, queue_name: str, depth: int) -> None:
        metrics = self._mailboxes.get(queue_name) or MailboxMetrics(
            queue_name=queue_name
        )
        metrics = replace(
            metrics,
            queue_depth=depth,
            last_updated=datetime.now(UTC),
        )
        self._mailboxes[queue_name] = metrics

    def snapshot(self) -> MetricsSnapshot:
        return MetricsSnapshot(
            adapters=tuple(self._adapters.values()),
            tools=tuple(self._tools.values()),
            mailboxes=tuple(self._mailboxes.values()),
            worker_id=self._worker_id,
        )

    def reset(self) -> None:
        """Clear all collected metrics."""
        self._adapters.clear()
        self._tools.clear()
        self._mailboxes.clear()

    # Query methods delegate to snapshot
    def get_adapter_metrics(self, adapter: AdapterName) -> AdapterMetrics | None:
        return self.snapshot().adapter(adapter)

    def get_tool_metrics(self, tool_name: str) -> ToolMetrics | None:
        return self.snapshot().tool(tool_name)

    def get_mailbox_metrics(self, queue_name: str) -> MailboxMetrics | None:
        return self.snapshot().mailbox(queue_name)
```

### Resource Binding

Register MetricsCollector in ResourceRegistry for automatic injection. The
collector is `SINGLETON` scoped, surviving across requests within a worker.

```python
from weakincentives.resources import Binding, Scope
import os

# In resource configuration
metrics_binding = Binding(
    MetricsCollector,
    lambda r: InMemoryMetricsCollector(worker_id=os.getenv("WORKER_ID")),
    scope=Scope.SINGLETON,  # One collector per worker process
)
```

## Integration Points

### Adapter Layer

Wire into `InnerLoop` phases. Each phase records timing.

```python
# In adapters/inner_loop.py

class InnerLoop:
    def __init__(
        self,
        *,
        metrics: MetricsCollector | None = None,
        # ... other params
    ) -> None:
        self._metrics = metrics

    async def execute(self, prompt: Prompt[T]) -> T:
        render_start = time.monotonic_ns()
        rendered = self._render(prompt)
        render_ms = (time.monotonic_ns() - render_start) // 1_000_000

        call_start = time.monotonic_ns()
        response = await self._call(rendered)
        call_ms = (time.monotonic_ns() - call_start) // 1_000_000

        parse_start = time.monotonic_ns()
        parsed = self._parse(response)
        parse_ms = (time.monotonic_ns() - parse_start) // 1_000_000

        tool_start = time.monotonic_ns()
        result = await self._execute_tools(parsed)
        tool_ms = (time.monotonic_ns() - tool_start) // 1_000_000

        if self._metrics:
            self._metrics.record_adapter_call(
                self._adapter.name,
                render_ms=render_ms,
                call_ms=call_ms,
                parse_ms=parse_ms,
                tool_ms=tool_ms,
                usage=response.usage,
                error=response.error,
                throttled=response.throttled,
            )

        return result
```

### Tool Executor

Wrap tool execution with timing.

```python
# In adapters/tool_executor.py

class ToolExecutor:
    def __init__(
        self,
        *,
        metrics: MetricsCollector | None = None,
        # ... other params
    ) -> None:
        self._metrics = metrics

    async def execute(
        self,
        tool: Tool,
        params: Any,
        context: ToolContext,
    ) -> ToolResult:
        start = time.monotonic_ns()
        try:
            result = await tool.handler(params, context=context)
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000

            if self._metrics:
                self._metrics.record_tool_call(
                    tool.name,
                    latency_ms=elapsed_ms,
                    success=result.is_ok,
                    error_code=result.error_code if not result.is_ok else None,
                )

            return result
        except Exception as e:
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            if self._metrics:
                self._metrics.record_tool_call(
                    tool.name,
                    latency_ms=elapsed_ms,
                    success=False,
                    error_code=type(e).__name__,
                )
            raise
```

### Mailbox Layer

Record delivery metadata on message receipt.

```python
# In runtime/mailbox/

class MailboxReceiver:
    def __init__(
        self,
        *,
        metrics: MetricsCollector | None = None,
        # ... other params
    ) -> None:
        self._metrics = metrics

    async def receive(self) -> Message[T, R] | None:
        message = await self._backend.receive()
        if message and self._metrics:
            age_ms = int(
                (datetime.now(UTC) - message.enqueued_at).total_seconds() * 1000
            )
            self._metrics.record_message_received(
                self._queue_name,
                delivery_count=message.delivery_count,
                age_ms=age_ms,
            )
        return message

    async def ack(self, message: Message[T, R]) -> None:
        await self._backend.ack(message)
        if self._metrics:
            self._metrics.record_message_ack(self._queue_name)

    async def nack(self, message: Message[T, R]) -> None:
        await self._backend.nack(message)
        if self._metrics:
            self._metrics.record_message_nack(self._queue_name)
```

### MainLoop Integration

Periodic queue depth sampling.

```python
# In runtime/main_loop.py

class MainLoop:
    async def _sample_queue_metrics(self) -> None:
        """Periodically sample queue depth."""
        if self._metrics and self._mailbox:
            depth = await self._mailbox.approximate_count()
            self._metrics.record_queue_depth(self._mailbox.name, depth)
```

## Compact Storage Format

Metrics are stored in memory using fixed-size structures. Histograms use
exponential buckets rather than storing raw values, ensuring bounded memory
regardless of request volume.

### Memory Budget

| Component | Size Estimate |
|-----------|---------------|
| Histogram (18 buckets) | ~200 bytes |
| Counter | ~50 bytes |
| AdapterMetrics | ~1.5 KB |
| ToolMetrics | ~500 bytes |
| MailboxMetrics | ~800 bytes |
| Typical session (5 adapters, 20 tools, 3 queues) | ~25 KB |

### Serialization

Metrics serialize using standard `weakincentives.serde` patterns:

```python
from weakincentives.serde import to_dict, from_dict

# Serialize for persistence
data = to_dict(metrics_snapshot)

# Restore from storage
snapshot = from_dict(MetricsSnapshot, data)
```

## Query Patterns

### Collector Access

```python
# Get metrics from collector (via ResourceRegistry)
metrics: MetricsCollector = registry.get(MetricsCollector)
snapshot = metrics.snapshot()

# Check adapter performance
openai = snapshot.adapter(AdapterName("openai"))
if openai:
    p99 = openai.call_latency.percentile(0.99)
    print(f"OpenAI p99 latency: {p99}ms")
    print(f"Total tokens: {openai.input_tokens.value + openai.output_tokens.value}")
```

### Aggregated Queries

```python
# Total tokens across all adapters
input_t, output_t, cached_t = snapshot.total_tokens()

# Tools with high failure rates
failing_tools = [
    t for t in snapshot.tools
    if (rate := t.failure_rate()) and rate > 0.1
]

# Queues with high retry rates
high_retry_queues = [
    m for m in snapshot.mailboxes
    if m.total_retries.value > m.messages_received.value * 0.2
]
```

## Export Patterns

### Prometheus Format

```python
def to_prometheus(snapshot: MetricsSnapshot) -> str:
    """Export metrics in Prometheus text format."""
    lines = []

    for adapter in snapshot.adapters:
        name = adapter.adapter.value
        # Latency histograms
        for i, count in enumerate(adapter.call_latency.bucket_counts):
            if i < len(Histogram.LATENCY_BUCKETS):
                le = Histogram.LATENCY_BUCKETS[i]
            else:
                le = "+Inf"
            lines.append(
                f'wink_adapter_call_latency_ms_bucket{{adapter="{name}",le="{le}"}} {count}'
            )
        lines.append(
            f'wink_adapter_call_latency_ms_count{{adapter="{name}"}} {adapter.call_latency.total_count}'
        )
        lines.append(
            f'wink_adapter_call_latency_ms_sum{{adapter="{name}"}} {adapter.call_latency.total_sum}'
        )
        # Counters
        lines.append(
            f'wink_adapter_input_tokens_total{{adapter="{name}"}} {adapter.input_tokens.value}'
        )
        lines.append(
            f'wink_adapter_output_tokens_total{{adapter="{name}"}} {adapter.output_tokens.value}'
        )

    return "\n".join(lines)
```

### JSON Export

```python
from weakincentives.serde import to_dict
import json

def to_json(snapshot: MetricsSnapshot) -> str:
    """Export metrics as JSON."""
    return json.dumps(to_dict(snapshot), default=str)
```

## Debug Persistence

The `weakincentives.debug` module provides utilities to dump metrics to disk
for post-hoc analysis. This is opt-in—metrics are not persisted by default.

```python
from weakincentives.debug import dump_metrics

# Dump current metrics to a file
metrics: MetricsCollector = registry.get(MetricsCollector)
dump_metrics(metrics.snapshot(), path="/tmp/metrics.json")

# Or dump to the debug archive directory
from weakincentives.debug import archive_metrics
archive_metrics(metrics.snapshot())  # Writes to .weakincentives/debug/metrics/
```

### Periodic Dumping

For long-running workers, dump metrics periodically:

```python
import asyncio
from weakincentives.debug import archive_metrics

async def periodic_metrics_dump(
    metrics: MetricsCollector,
    interval_seconds: float = 300.0,  # 5 minutes
) -> None:
    """Dump metrics snapshot every interval."""
    while True:
        await asyncio.sleep(interval_seconds)
        archive_metrics(metrics.snapshot())
        metrics.reset()  # Optional: reset after dump
```

### Debug Archive Structure

Metrics dumps are stored alongside other debug artifacts:

```
.weakincentives/
└── debug/
    ├── metrics/
    │   ├── 2024-01-15T10:30:00Z_worker-1.json
    │   └── 2024-01-15T10:35:00Z_worker-1.json
    ├── sessions/
    └── logs/
```

## Health Integration

Metrics integrate with the health system for stuck worker detection.

```python
# In runtime/health.py

class HealthEndpoint:
    def __init__(self, metrics: MetricsCollector) -> None:
        self._metrics = metrics

    def liveness(self) -> dict:
        """Kubernetes liveness probe."""
        return {"status": "ok"}

    def readiness(self) -> dict:
        """Kubernetes readiness probe with metrics summary."""
        snapshot = self._metrics.snapshot()
        return {
            "status": "ok",
            "metrics": {
                "total_requests": sum(
                    m.request_count.value for m in snapshot.adapters
                ),
                "total_errors": sum(
                    m.error_count.value for m in snapshot.adapters
                ),
                "queue_depths": {
                    m.queue_name: m.queue_depth for m in snapshot.mailboxes
                },
            },
        }
```

## RunContext Integration

Metrics correlate with `RunContext` for distributed tracing. The collector
accepts optional context for request-level correlation.

```python
from weakincentives.runtime import RunContext

class MetricsCollector(Protocol):
    def record_adapter_call(
        self,
        adapter: AdapterName,
        *,
        render_ms: int,
        call_ms: int,
        parse_ms: int,
        tool_ms: int,
        usage: TokenUsage | None = None,
        error: str | None = None,
        throttled: bool = False,
        run_context: RunContext | None = None,  # For correlation
    ) -> None:
        ...
```

When `RunContext` is provided, implementations can:

- Tag metrics with `trace_id` and `span_id` for distributed tracing
- Use `worker_id` for per-worker aggregation
- Track `attempt` (delivery count) for retry analysis
- Correlate with `request_id` for end-to-end request tracking

```python
# Example: MainLoop passes RunContext to metrics
async def _process_request(self, request: MainLoopRequest) -> MainLoopResult:
    run_context = RunContext(
        request_id=request.request_id,
        attempt=message.delivery_count,
        worker_id=self._worker_id,
        trace_id=request.trace_id,
    )

    # ... execute prompt ...

    if self._metrics:
        self._metrics.record_adapter_call(
            adapter=self._adapter.name,
            render_ms=render_ms,
            call_ms=call_ms,
            parse_ms=parse_ms,
            tool_ms=tool_ms,
            usage=response.usage,
            run_context=run_context,
        )
```

## Experiment Labels

For A/B testing with `Experiment`, metrics can be tagged by experiment name
to enable comparison across variants.

```python
from weakincentives.evals import Experiment

@FrozenDataclass()
class AdapterMetrics:
    adapter: AdapterName
    experiment: str | None = None  # Optional experiment tag
    # ... other fields
```

When processing requests with experiments:

```python
# MainLoop with experiment
async def _process_request(self, request: MainLoopRequest) -> MainLoopResult:
    experiment_name = request.experiment.name if request.experiment else None

    # Metrics tagged by experiment for A/B comparison
    if self._metrics:
        self._metrics.record_adapter_call(
            adapter=self._adapter.name,
            # ... timing fields ...
            experiment=experiment_name,
        )
```

Query metrics by experiment:

```python
# Compare latency across experiment variants
baseline = [m for m in snapshot.adapters if m.experiment == "baseline"]
variant = [m for m in snapshot.adapters if m.experiment == "v2-concise"]

baseline_p99 = baseline[0].call_latency.percentile(0.99) if baseline else None
variant_p99 = variant[0].call_latency.percentile(0.99) if variant else None
```

## Limitations

- **No distributed aggregation**: Metrics are worker-local. For cross-worker
  aggregation, export to external systems (Prometheus, DataDog).
- **Lost on process restart**: In-memory storage means metrics don't survive
  worker restarts. Use debug persistence or external export for durability.
- **Histogram bucket boundaries fixed**: Exponential buckets suit latency well
  but may lose precision for other distributions.
- **No cardinality limits**: High-cardinality labels (e.g., user IDs) will
  consume unbounded memory. Use only bounded label sets.

## Related Specifications

- `specs/ADAPTERS.md` - Adapter lifecycle and token usage
- `specs/MAILBOX.md` - Message delivery metadata
- `specs/DLQ.md` - Dead letter queue handling and poison message isolation
- `specs/TOOLS.md` - Tool execution patterns
- `specs/HEALTH.md` - Health endpoints and watchdog integration
- `specs/RESOURCE_REGISTRY.md` - Dependency injection and collector binding
- `specs/RUN_CONTEXT.md` - Execution metadata and distributed tracing
- `specs/EXPERIMENTS.md` - A/B testing and experiment configuration
- `specs/DEBUGGING.md` - Debug persistence utilities
