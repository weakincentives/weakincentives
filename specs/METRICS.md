# Metrics Specification

## Purpose

In-memory metrics collection for observability. Captures latency, throughput,
error rates, and queue health across adapters, tools, and mailbox layers.

**Use metrics for:** Latency analysis, token tracking, failure rates, queue health.

**Not for:** Billing (use session token budgets), alerting (export to StatsD/Prometheus).

## Design Principles

- **In-memory storage**: Metrics accumulate in collector without Session persistence.
  Avoids session bloat—metrics are observability, not application state.
- **Resource-scoped lifecycle**: Bound via `ResourceRegistry` with `Scope.SINGLETON`,
  surviving across requests within a worker process.
- **Pluggable sinks**: Optional export to StatsD, Prometheus, or debug files.
- **Compact histograms**: Fixed exponential buckets bound memory regardless of volume.
- **RunContext correlation**: Integrates with distributed tracing.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ResourceRegistry                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           MetricsCollector (Scope.SINGLETON)               │  │
│  │                                                            │  │
│  │  AdapterMetrics    ToolMetrics       MailboxMetrics        │  │
│  │  - latency hist    - latency hist    - queue lag hist      │  │
│  │  - token counts    - failure rate    - delivery dist       │  │
│  │  - error count     - call count      - nack/dlq counts     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
    Adapter Layer       Tool Executor        Mailbox Layer
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         StatsdSink      DebugSink      PrometheusSink
```

## Metric Primitives

### Counter

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Metric name |
| `value` | `int` | Current count (monotonic) |
| `labels` | `tuple[tuple[str, str], ...]` | Dimension labels |

Methods: `inc(delta=1)` returns new Counter with incremented value.

### Histogram

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Metric name |
| `bucket_counts` | `tuple[int, ...]` | Counts per bucket (18 buckets) |
| `total_count` | `int` | Total observations |
| `total_sum` | `int` | Sum of all values |
| `labels` | `tuple[tuple[str, str], ...]` | Dimension labels |

Buckets: Exponential boundaries `(1, 2, 4, 8, ... 65536, +Inf)` milliseconds.

Methods:

- `observe(value)` → new Histogram with value in appropriate bucket
- `percentile(p)` → estimated percentile from distribution
- `mean` → average of observed values

### Gauge

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Metric name |
| `value` | `int` | Current value (can increase/decrease) |
| `labels` | `tuple[tuple[str, str], ...]` | Dimension labels |

Methods: `set(value)`, `inc(delta=1)`, `dec(delta=1)`.

## Core Metric Types

### AdapterMetrics

| Field | Type | Description |
|-------|------|-------------|
| `adapter` | `AdapterName` | Provider identifier |
| `render_latency` | `Histogram` | Prompt rendering time |
| `call_latency` | `Histogram` | LLM API call time |
| `parse_latency` | `Histogram` | Response parsing time |
| `tool_latency` | `Histogram` | Tool execution time |
| `total_latency` | `Histogram` | End-to-end request time |
| `input_tokens` | `Counter` | Total input tokens |
| `output_tokens` | `Counter` | Total output tokens |
| `cached_tokens` | `Counter` | Tokens served from cache |
| `request_count` | `Counter` | Total requests |
| `error_count` | `Counter` | Failed requests |
| `throttle_count` | `Counter` | Rate-limited requests |
| `experiment` | `str \| None` | Experiment tag for A/B testing |

### ToolMetrics

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Tool identifier |
| `latency` | `Histogram` | Execution time |
| `call_count` | `Counter` | Total invocations |
| `success_count` | `Counter` | Successful calls |
| `failure_count` | `Counter` | Failed calls |
| `error_codes` | `tuple[tuple[str, int], ...]` | Error breakdown |

Method: `failure_rate()` → `failures / total` or `None` if no calls.

### MailboxMetrics

| Field | Type | Description |
|-------|------|-------------|
| `queue_name` | `str` | Queue identifier |
| `queue_lag` | `Histogram` | Age of messages at receive time |
| `delivery_count_dist` | `tuple[int, ...]` | Distribution of delivery attempts (1-10+) |
| `messages_received` | `Counter` | Total received |
| `messages_acked` | `Counter` | Successfully processed |
| `messages_nacked` | `Counter` | Returned to queue |
| `messages_expired` | `Counter` | TTL exceeded |
| `messages_dead_lettered` | `Counter` | Sent to DLQ |
| `total_retries` | `Counter` | Sum of retry attempts |
| `queue_depth` | `int` | Current queue size |
| `oldest_message_age_ms` | `int \| None` | Age of oldest message |

## MetricsCollector Protocol

```python
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
        run_context: RunContext | None = None,
        experiment: str | None = None,
    ) -> None: ...

    def record_tool_call(
        self,
        tool_name: str,
        *,
        latency_ms: int,
        success: bool,
        error_code: str | None = None,
    ) -> None: ...

    def record_message_received(
        self, queue_name: str, *, delivery_count: int, age_ms: int
    ) -> None: ...

    def record_message_ack(self, queue_name: str) -> None: ...
    def record_message_nack(self, queue_name: str) -> None: ...
    def record_message_dead_lettered(self, queue_name: str) -> None: ...
    def record_queue_depth(self, queue_name: str, depth: int) -> None: ...

    def snapshot(self) -> MetricsSnapshot: ...
    def reset(self) -> None: ...
```

## Resource Binding

```python
from weakincentives.resources import Binding, Scope

metrics_binding = Binding(
    MetricsCollector,
    lambda r: InMemoryMetricsCollector(
        worker_id=os.getenv("WORKER_ID"),
        sinks=[StatsdSink(host="localhost", port=8125)] if STATSD_ENABLED else [],
    ),
    scope=Scope.SINGLETON,
)
```

## StatsD Integration

Export metrics to any sink implementing the StatsD protocol (DataDog, Telegraf,
StatsD, Graphite).

### Sink Protocol

```python
class MetricsSink(Protocol):
    """Protocol for external metrics export."""

    def emit_counter(self, name: str, value: int, tags: dict[str, str]) -> None: ...
    def emit_histogram(self, name: str, value: int, tags: dict[str, str]) -> None: ...
    def emit_gauge(self, name: str, value: int, tags: dict[str, str]) -> None: ...
    def flush(self) -> None: ...
```

### StatsdSink Implementation

```python
class StatsdSink:
    """StatsD protocol sink for metrics export."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8125,
        prefix: str = "wink",
        sample_rate: float = 1.0,
    ) -> None:
        self._client = StatsdClient(host, port)
        self._prefix = prefix
        self._sample_rate = sample_rate

    def emit_counter(self, name: str, value: int, tags: dict[str, str]) -> None:
        self._client.incr(
            f"{self._prefix}.{name}",
            value,
            tags=tags,
            sample_rate=self._sample_rate,
        )

    def emit_histogram(self, name: str, value: int, tags: dict[str, str]) -> None:
        # StatsD timing (interpreted as histogram by DataDog/Telegraf)
        self._client.timing(
            f"{self._prefix}.{name}",
            value,
            tags=tags,
            sample_rate=self._sample_rate,
        )

    def emit_gauge(self, name: str, value: int, tags: dict[str, str]) -> None:
        self._client.gauge(f"{self._prefix}.{name}", value, tags=tags)

    def flush(self) -> None:
        self._client.flush()
```

### Metric Names (StatsD)

| Metric | StatsD Name | Type |
|--------|-------------|------|
| Adapter call latency | `wink.adapter.call_latency_ms` | timing |
| Adapter tokens | `wink.adapter.tokens` | counter |
| Adapter errors | `wink.adapter.errors` | counter |
| Tool latency | `wink.tool.latency_ms` | timing |
| Tool failures | `wink.tool.failures` | counter |
| Queue lag | `wink.mailbox.lag_ms` | timing |
| Queue depth | `wink.mailbox.depth` | gauge |
| Messages dead-lettered | `wink.mailbox.dead_lettered` | counter |

### Tags

All metrics include dimensional tags:

| Tag | Source | Example |
|-----|--------|---------|
| `adapter` | AdapterName | `openai`, `litellm` |
| `tool` | Tool name | `read_file`, `execute` |
| `queue` | Mailbox name | `requests`, `replies` |
| `worker` | RunContext.worker_id | `worker-1` |
| `experiment` | Experiment.name | `baseline`, `v2-concise` |

### Collector with Sinks

```python
class InMemoryMetricsCollector:
    def __init__(
        self,
        worker_id: str | None = None,
        sinks: Sequence[MetricsSink] = (),
    ) -> None:
        self._worker_id = worker_id
        self._sinks = list(sinks)
        # ... internal state

    def record_adapter_call(self, adapter: AdapterName, ...) -> None:
        # Update in-memory metrics
        # ...

        # Emit to sinks
        tags = {"adapter": adapter.value, "worker": self._worker_id or "unknown"}
        if experiment:
            tags["experiment"] = experiment
        for sink in self._sinks:
            sink.emit_histogram("adapter.call_latency_ms", call_ms, tags)
            sink.emit_counter("adapter.requests", 1, tags)
            if usage:
                sink.emit_counter("adapter.input_tokens", usage.input_tokens or 0, tags)
                sink.emit_counter("adapter.output_tokens", usage.output_tokens or 0, tags)
```

## Integration Points

### Adapter Layer

Metrics recorded in `InnerLoop` after each phase completes:

```python
# In adapters/inner_loop.py - conceptual flow
render_ms = time_phase(self._render)
call_ms = time_phase(self._call)
parse_ms = time_phase(self._parse)
tool_ms = time_phase(self._execute_tools)

metrics.record_adapter_call(
    adapter=self._adapter.name,
    render_ms=render_ms, call_ms=call_ms,
    parse_ms=parse_ms, tool_ms=tool_ms,
    usage=response.usage,
    run_context=run_context,
)
```

### Tool Executor

```python
# In adapters/tool_executor.py - conceptual flow
start = time.monotonic_ns()
result = await tool.handler(params, context=context)
latency_ms = (time.monotonic_ns() - start) // 1_000_000

metrics.record_tool_call(
    tool.name,
    latency_ms=latency_ms,
    success=result.is_ok,
    error_code=result.error_code if not result.is_ok else None,
)
```

### Mailbox Layer

```python
# On message receive
age_ms = (now() - message.enqueued_at).total_seconds() * 1000
metrics.record_message_received(queue_name, delivery_count=message.delivery_count, age_ms=age_ms)

# On ack/nack/dlq
metrics.record_message_ack(queue_name)
metrics.record_message_nack(queue_name)
metrics.record_message_dead_lettered(queue_name)
```

### MainLoop Queue Sampling

```python
# Periodic queue depth sampling
async def _sample_queue_metrics(self) -> None:
    depth = await self._mailbox.approximate_count()
    metrics.record_queue_depth(self._mailbox.name, depth)
```

## Debug Persistence

The `weakincentives.debug` module provides utilities to dump metrics to disk.

```python
from weakincentives.debug import dump_metrics, archive_metrics

# Dump to specific path
dump_metrics(metrics.snapshot(), path="/tmp/metrics.json")

# Dump to debug archive (.weakincentives/debug/metrics/)
archive_metrics(metrics.snapshot())
```

### Archive Structure

```
.weakincentives/
└── debug/
    ├── metrics/
    │   ├── 2024-01-15T10:30:00Z_worker-1.json
    │   └── 2024-01-15T10:35:00Z_worker-1.json
    ├── sessions/
    └── logs/
```

## Memory Budget

| Component | Size |
|-----------|------|
| Histogram (18 buckets) | ~200 bytes |
| Counter | ~50 bytes |
| AdapterMetrics | ~1.5 KB |
| ToolMetrics | ~500 bytes |
| MailboxMetrics | ~800 bytes |
| Typical worker (5 adapters, 20 tools, 3 queues) | ~25 KB |

## Experiment Comparison

Tag metrics by experiment name for A/B analysis:

```python
# During request processing
metrics.record_adapter_call(
    adapter=adapter.name,
    ...,
    experiment=request.experiment.name if request.experiment else None,
)

# Query by experiment
baseline = [m for m in snapshot.adapters if m.experiment == "baseline"]
variant = [m for m in snapshot.adapters if m.experiment == "v2-concise"]
```

## Limitations

- **Worker-local**: No cross-worker aggregation. Export to external systems for fleet views.
- **Lost on restart**: In-memory storage. Use StatsD sink or debug persistence for durability.
- **Fixed buckets**: Exponential histogram boundaries optimized for latency, not all distributions.
- **No cardinality limits**: Unbounded labels will consume memory. Use bounded label sets.

## Related Specifications

- `specs/ADAPTERS.md` - Adapter lifecycle, token usage tracking
- `specs/MAILBOX.md` - Message delivery metadata, queue semantics
- `specs/DLQ.md` - Dead letter queue handling
- `specs/TOOLS.md` - Tool execution patterns
- `specs/HEALTH.md` - Health endpoints, watchdog integration
- `specs/RESOURCE_REGISTRY.md` - Collector binding, singleton scope
- `specs/RUN_CONTEXT.md` - Request correlation, distributed tracing
- `specs/EXPERIMENTS.md` - A/B testing configuration
- `specs/DEBUGGING.md` - Debug persistence utilities
