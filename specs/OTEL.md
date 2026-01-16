# OpenTelemetry Integration Specification

## Purpose

OpenTelemetry (OTEL) integration enables distributed tracing, metrics export, and
log correlation for WINK agents in production environments. The integration bridges
WINK's existing observability primitives (`RunContext`, `MetricsCollector`, structured
logging) with the OpenTelemetry SDK for export to backends like Jaeger, Tempo,
Prometheus, and Datadog.

**Implementation:** `src/weakincentives/contrib/otel/`

## Design Principles

- **Optional dependency**: OpenTelemetry SDK is not required for core functionality
- **Non-invasive**: Integrates via existing hooks (`RunContext`, `MetricsSink`, logging)
- **Context propagation**: W3C Trace Context flows through `RunContext.trace_id`/`span_id`
- **Resource-bound lifecycle**: OTEL providers bind via `ResourceRegistry` for clean shutdown
- **Semantic conventions**: Follows OpenTelemetry semantic conventions for LLM systems

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Application Layer                              │
│  MainLoop → Adapter → InnerLoop → ToolExecutor                       │
│      │          │          │           │                              │
│      ▼          ▼          ▼           ▼                              │
│  RunContext (trace_id, span_id) flows through all components         │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ OtelTracer   │    │ OtelMetricsSink  │    │ OtelLogHandler   │
│ (spans)      │    │ (OTLP export)    │    │ (log correlation)│
└──────────────┘    └──────────────────┘    └──────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    OpenTelemetry SDK
                    (TracerProvider, MeterProvider, LoggerProvider)
                              │
                              ▼
                    OTLP Exporter → Backend
                    (Jaeger, Tempo, Prometheus, etc.)
```

## Trace Propagation

### RunContext Integration

`RunContext` carries trace context through the execution flow. OTEL integration
extracts and injects trace IDs at system boundaries.

```python
from opentelemetry import trace
from opentelemetry.trace import SpanContext, TraceFlags
from weakincentives.runtime import RunContext

def run_context_from_otel() -> RunContext:
    """Create RunContext from current OTEL span context."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    return RunContext(
        trace_id=format(ctx.trace_id, "032x") if ctx.is_valid else None,
        span_id=format(ctx.span_id, "016x") if ctx.is_valid else None,
    )

def otel_context_from_run_context(run_context: RunContext) -> SpanContext | None:
    """Restore OTEL SpanContext from RunContext."""
    if not run_context.trace_id or not run_context.span_id:
        return None
    return SpanContext(
        trace_id=int(run_context.trace_id, 16),
        span_id=int(run_context.span_id, 16),
        is_remote=True,
        trace_flags=TraceFlags.SAMPLED,
    )
```

### W3C Trace Context Headers

For HTTP-based mailbox transports, extract and inject trace context via headers:

```python
from opentelemetry.propagate import extract, inject

# Incoming request - extract context
carrier = {"traceparent": request.headers.get("traceparent")}
context = extract(carrier)

# Outgoing request - inject context
carrier = {}
inject(carrier)
# carrier now contains {"traceparent": "00-{trace_id}-{span_id}-01"}
```

## Span Instrumentation

### Span Hierarchy

```
request.process                    # MainLoop._handle_message
├── prompt.prepare                 # MainLoop.prepare()
├── prompt.evaluate                # Adapter.evaluate()
│   ├── prompt.render              # Prompt.render()
│   ├── llm.call                   # Provider API call
│   │   └── llm.call.retry         # Retry attempts (if any)
│   ├── tool.execute               # ToolExecutor (per tool)
│   │   └── tool.{name}            # Individual tool span
│   └── response.parse             # Output parsing
└── prompt.finalize                # MainLoop.finalize()
```

### OtelTracer Protocol

```python
from typing import Protocol, Iterator
from contextlib import contextmanager
from weakincentives.runtime import RunContext

class OtelTracer(Protocol):
    """Protocol for OpenTelemetry tracing integration."""

    @contextmanager
    def span(
        self,
        name: str,
        *,
        run_context: RunContext | None = None,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> Iterator[SpanHandle]:
        """Create a span with optional parent from RunContext."""
        ...

    def record_exception(self, exception: BaseException) -> None:
        """Record exception on current span."""
        ...

    def set_attribute(self, key: str, value: str | int | float | bool) -> None:
        """Set attribute on current span."""
        ...


class SpanHandle(Protocol):
    """Handle to an active span for attribute updates."""

    def set_attribute(self, key: str, value: str | int | float | bool) -> None: ...
    def set_status(self, status: SpanStatus) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...
    def to_run_context(self) -> RunContext: ...
```

### DefaultOtelTracer Implementation

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

class DefaultOtelTracer:
    """OpenTelemetry tracer implementation."""

    def __init__(self, service_name: str = "wink-agent") -> None:
        self._tracer = trace.get_tracer(service_name)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        run_context: RunContext | None = None,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> Iterator[SpanHandle]:
        # Restore parent context if provided
        parent_context = None
        if run_context:
            parent_span_ctx = otel_context_from_run_context(run_context)
            if parent_span_ctx:
                parent_context = trace.set_span_in_context(
                    trace.NonRecordingSpan(parent_span_ctx)
                )

        with self._tracer.start_as_current_span(
            name,
            context=parent_context,
            attributes=attributes or {},
        ) as span:
            yield DefaultSpanHandle(span)

    def record_exception(self, exception: BaseException) -> None:
        span = trace.get_current_span()
        span.record_exception(exception)
        span.set_status(Status(StatusCode.ERROR, str(exception)))

    def set_attribute(self, key: str, value: str | int | float | bool) -> None:
        trace.get_current_span().set_attribute(key, value)
```

## Span Attributes

### Semantic Conventions

Follow OpenTelemetry semantic conventions with LLM-specific extensions:

| Attribute | Type | Description |
|-----------|------|-------------|
| `wink.run_id` | string | Unique execution identifier |
| `wink.request_id` | string | Logical request correlation |
| `wink.session_id` | string | Session identifier |
| `wink.worker_id` | string | Worker processing request |
| `wink.attempt` | int | Delivery attempt number |

### LLM Span Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.system` | string | Provider name (openai, litellm, anthropic) |
| `gen_ai.request.model` | string | Model identifier |
| `gen_ai.request.temperature` | float | Sampling temperature |
| `gen_ai.request.max_tokens` | int | Maximum output tokens |
| `gen_ai.response.model` | string | Actual model used |
| `gen_ai.usage.input_tokens` | int | Input token count |
| `gen_ai.usage.output_tokens` | int | Output token count |
| `gen_ai.usage.total_tokens` | int | Total token count |

### Tool Span Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `wink.tool.name` | string | Tool identifier |
| `wink.tool.success` | bool | Execution outcome |
| `wink.tool.error_code` | string | Error code if failed |

### Prompt Span Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `wink.prompt.namespace` | string | Prompt namespace |
| `wink.prompt.key` | string | Prompt key |
| `wink.prompt.name` | string | Full prompt name |
| `wink.prompt.tool_count` | int | Number of tools available |

## Metrics Export

### OtelMetricsSink

Bridge `MetricsCollector` to OpenTelemetry Metrics API:

```python
from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram

class OtelMetricsSink:
    """Export WINK metrics via OpenTelemetry Metrics API."""

    def __init__(self, service_name: str = "wink-agent") -> None:
        self._meter = metrics.get_meter(service_name)
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}

    def emit_counter(self, name: str, value: int, tags: dict[str, str]) -> None:
        counter = self._get_or_create_counter(name)
        counter.add(value, attributes=tags)

    def emit_histogram(self, name: str, value: int, tags: dict[str, str]) -> None:
        histogram = self._get_or_create_histogram(name)
        histogram.record(value, attributes=tags)

    def emit_gauge(self, name: str, value: int, tags: dict[str, str]) -> None:
        # OpenTelemetry gauges use observable callbacks
        # For simplicity, emit as histogram observation
        histogram = self._get_or_create_histogram(f"{name}.gauge")
        histogram.record(value, attributes=tags)

    def flush(self) -> None:
        # OTLP exporter handles batching; no-op here
        pass

    def _get_or_create_counter(self, name: str) -> Counter:
        if name not in self._counters:
            self._counters[name] = self._meter.create_counter(
                f"wink.{name}",
                description=f"WINK counter: {name}",
            )
        return self._counters[name]

    def _get_or_create_histogram(self, name: str) -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = self._meter.create_histogram(
                f"wink.{name}",
                description=f"WINK histogram: {name}",
                unit="ms",
            )
        return self._histograms[name]
```

### Metric Names (OTLP)

| WINK Metric | OTLP Name | Unit |
|-------------|-----------|------|
| Adapter call latency | `wink.adapter.call_latency` | ms |
| Adapter tokens | `wink.adapter.tokens` | count |
| Adapter errors | `wink.adapter.errors` | count |
| Tool latency | `wink.tool.latency` | ms |
| Tool failures | `wink.tool.failures` | count |
| Queue lag | `wink.mailbox.lag` | ms |
| Queue depth | `wink.mailbox.depth.gauge` | count |
| Messages dead-lettered | `wink.mailbox.dead_lettered` | count |

## Log Correlation

### OtelLogHandler

Inject trace context into structured logs for correlation:

```python
import logging
from opentelemetry import trace

class OtelLogHandler(logging.Handler):
    """Inject OTEL trace context into log records."""

    def emit(self, record: logging.LogRecord) -> None:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            record.trace_id = format(ctx.trace_id, "032x")
            record.span_id = format(ctx.span_id, "016x")
            record.trace_flags = ctx.trace_flags
```

### Log Record Enhancement

With OTEL correlation, log records include:

```json
{
  "timestamp": "2024-01-15T10:30:00+00:00",
  "level": "INFO",
  "logger": "weakincentives.adapters.openai",
  "event": "tool.execution.complete",
  "message": "Tool execution completed",
  "context": {"tool_name": "read_file", "success": true},
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7"
}
```

## Resource Attributes

### Service Resource

Configure service-level resource attributes:

```python
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION

resource = Resource.create({
    SERVICE_NAME: "my-agent",
    SERVICE_VERSION: "1.0.0",
    "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    "wink.version": weakincentives.__version__,
})
```

### Worker Resource

Per-worker attributes bound at startup:

| Attribute | Source | Description |
|-----------|--------|-------------|
| `service.instance.id` | `WORKER_ID` env | Worker identifier |
| `host.name` | `socket.gethostname()` | Host name |
| `process.pid` | `os.getpid()` | Process ID |

## Resource Registry Integration

### OtelProvider

Bind OTEL components via `ResourceRegistry` for lifecycle management:

```python
from weakincentives.resources import Binding, ResourceRegistry, Scope

def create_otel_bindings(
    service_name: str = "wink-agent",
    otlp_endpoint: str | None = None,
) -> tuple[Binding, ...]:
    """Create resource bindings for OTEL integration."""
    return (
        Binding(
            OtelTracer,
            lambda r: _create_tracer(service_name, otlp_endpoint),
            scope=Scope.SINGLETON,
        ),
        Binding(
            OtelMetricsSink,
            lambda r: OtelMetricsSink(service_name),
            scope=Scope.SINGLETON,
        ),
    )

def _create_tracer(service_name: str, endpoint: str | None) -> DefaultOtelTracer:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=_create_resource(service_name))
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return DefaultOtelTracer(service_name)
```

### Registry Setup

```python
from weakincentives.resources import ResourceRegistry

otel_bindings = create_otel_bindings(
    service_name="code-review-agent",
    otlp_endpoint="http://localhost:4317",
)

registry = ResourceRegistry.of(
    *otel_bindings,
    # ... other bindings
)

# MetricsCollector with OTEL sink
metrics_binding = Binding(
    MetricsCollector,
    lambda r: InMemoryMetricsCollector(
        worker_id=os.getenv("WORKER_ID"),
        sinks=[r.get(OtelMetricsSink)],
    ),
    scope=Scope.SINGLETON,
)
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_SERVICE_NAME` | `wink-agent` | Service name for traces/metrics |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | None | OTLP collector endpoint |
| `OTEL_EXPORTER_OTLP_HEADERS` | None | Auth headers (comma-separated) |
| `OTEL_TRACES_SAMPLER` | `parentbased_always_on` | Sampling strategy |
| `OTEL_TRACES_SAMPLER_ARG` | None | Sampler argument (e.g., ratio) |
| `OTEL_METRICS_EXPORTER` | `otlp` | Metrics exporter type |
| `OTEL_LOGS_EXPORTER` | `otlp` | Logs exporter type |

### Programmatic Configuration

```python
from weakincentives.contrib.otel import configure_otel

configure_otel(
    service_name="my-agent",
    otlp_endpoint="http://collector:4317",
    sampling_ratio=0.1,  # Sample 10% of traces
    enable_metrics=True,
    enable_logs=True,
)
```

## Instrumentation Points

### MainLoop Instrumentation

```python
class InstrumentedMainLoop(MainLoop[RequestT, OutputT]):
    """MainLoop with OTEL instrumentation."""

    def __init__(self, ..., tracer: OtelTracer | None = None) -> None:
        super().__init__(...)
        self._tracer = tracer

    def _handle_message(self, msg: Message[MainLoopRequest[RequestT]]) -> None:
        if not self._tracer:
            return super()._handle_message(msg)

        run_context = self._build_run_context(msg)
        with self._tracer.span(
            "request.process",
            run_context=run_context,
            attributes={
                "wink.request_id": str(run_context.request_id),
                "wink.attempt": run_context.attempt,
            },
        ) as span:
            # Update run_context with new span IDs
            run_context = span.to_run_context()
            try:
                result = self._execute(msg.payload.request, run_context=run_context)
                span.set_attribute("wink.success", True)
            except Exception as e:
                span.record_exception(e)
                span.set_status(SpanStatus.ERROR)
                raise
```

### Adapter Instrumentation

```python
# In InnerLoop._call_provider()
with tracer.span(
    "llm.call",
    run_context=run_context,
    attributes={
        "gen_ai.system": self._adapter_name,
        "gen_ai.request.model": self._model,
        "gen_ai.request.temperature": self._config.temperature,
    },
) as span:
    response = self._client.chat.completions.create(...)
    span.set_attribute("gen_ai.response.model", response.model)
    span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)
```

### Tool Instrumentation

```python
# In ToolExecutor._execute_tool()
with tracer.span(
    f"tool.{tool.name}",
    run_context=context.run_context,
    attributes={"wink.tool.name": tool.name},
) as span:
    result = tool.handler(params, context=context)
    span.set_attribute("wink.tool.success", result.is_ok)
    if not result.is_ok:
        span.set_attribute("wink.tool.error_code", result.error_code or "unknown")
```

## Sampling Strategies

### Parent-Based Sampling

Default strategy respects upstream sampling decisions:

```python
from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio

sampler = ParentBasedTraceIdRatio(0.1)  # 10% of root spans
```

### Adaptive Sampling

For high-volume production, consider adaptive sampling based on error rate:

```python
class ErrorBiasedSampler:
    """Sample all errors, ratio-sample successes."""

    def __init__(self, success_ratio: float = 0.01) -> None:
        self._success_ratio = success_ratio

    def should_sample(self, ...) -> SamplingResult:
        # Always sample errors (determined post-execution via span processor)
        # Ratio-sample normal requests
        ...
```

## Shutdown and Cleanup

### Graceful Shutdown

OTEL providers implement `Closeable` for proper cleanup:

```python
class OtelProviderWrapper:
    """Wrapper ensuring graceful OTEL shutdown."""

    def __init__(self, provider: TracerProvider | MeterProvider) -> None:
        self._provider = provider

    def close(self) -> None:
        """Flush pending spans/metrics and shutdown."""
        if hasattr(self._provider, "force_flush"):
            self._provider.force_flush(timeout_millis=5000)
        if hasattr(self._provider, "shutdown"):
            self._provider.shutdown()
```

### LoopGroup Integration

```python
from weakincentives.runtime import LoopGroup

# OTEL cleanup happens via resource context close
group = LoopGroup(
    loops=[main_loop],
    health_port=8080,
)
# On SIGTERM: LoopGroup closes resource contexts → OTEL providers flush and shutdown
```

## Testing

### Mock Tracer

```python
class MockOtelTracer:
    """In-memory tracer for testing."""

    def __init__(self) -> None:
        self.spans: list[MockSpan] = []

    @contextmanager
    def span(self, name: str, **kwargs) -> Iterator[MockSpanHandle]:
        span = MockSpan(name=name, attributes=kwargs.get("attributes", {}))
        self.spans.append(span)
        yield MockSpanHandle(span)

    def assert_span_exists(self, name: str) -> MockSpan:
        for span in self.spans:
            if span.name == name:
                return span
        raise AssertionError(f"Span '{name}' not found")
```

### Testing Patterns

```python
def test_tool_creates_span():
    tracer = MockOtelTracer()
    executor = ToolExecutor(tracer=tracer)

    executor.execute(read_file_tool, {"path": "/tmp/test"}, context=ctx)

    span = tracer.assert_span_exists("tool.read_file")
    assert span.attributes["wink.tool.name"] == "read_file"
    assert span.attributes["wink.tool.success"] is True
```

## Limitations

- **Synchronous only**: Async span management requires additional coordination
- **No automatic instrumentation**: Explicit instrumentation at defined points
- **Memory overhead**: BatchSpanProcessor buffers pending spans
- **Network dependency**: OTLP export failures may cause backpressure
- **Sampling post-decision**: Error-biased sampling requires span processor hooks

## Related Specifications

- `specs/RUN_CONTEXT.md` - Execution metadata and trace context propagation
- `specs/METRICS.md` - In-memory metrics and sink protocol
- `specs/DEBUGGING.md` - Debug capture and log collector
- `specs/LOGGING.md` - Structured logging conventions
- `specs/RESOURCE_REGISTRY.md` - Dependency injection and lifecycle
- `specs/ADAPTERS.md` - Provider adapter instrumentation points
- `specs/MAIN_LOOP.md` - Request processing lifecycle
- `specs/LIFECYCLE.md` - Graceful shutdown coordination
