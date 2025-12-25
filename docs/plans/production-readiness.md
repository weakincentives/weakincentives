# Production Readiness Plan

## Context

This plan prioritizes production readiness features for a **data science analyst
agent** that:

- Receives analysis briefs via a Mailbox (Redis/SQS)
- Completes data analysis on behalf of users
- Requires observability, debugging, and graceful lifecycle management

## Priority Framework

| Priority | Criteria |
| -------- | --------------------------------------------------------- |
| **P0** | Blocks production deployment; no workarounds |
| **P1** | Critical for operations; manual workarounds possible |
| **P2** | Important for scale/reliability; can defer initially |
| **P3** | Nice to have; can build incrementally |

______________________________________________________________________

## P0: Production Blockers

### 1. Graceful Shutdown & Signal Handling

**Problem**: MainLoop.run() has no shutdown mechanism. SIGTERM/SIGINT kill the
process immediately, potentially losing in-flight work.

**Scope**: ~300 LOC

**Deliverables**:

```python
# New shutdown protocol
class MainLoop[UserRequestT, OutputT](ABC):
    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
        signals: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT),
    ) -> None:
        """Run with graceful shutdown on signal."""
        ...

    def shutdown(self, *, timeout: float = 30.0) -> None:
        """Initiate graceful shutdown.

        - Sets shutdown flag to stop polling
        - Waits for in-flight message to complete (up to timeout)
        - Extends visibility timeout if needed during drain
        """
        ...

    @contextmanager
    def running(self, **kwargs) -> Iterator[Self]:
        """Context manager for lifecycle management."""
        ...
```

**Implementation Steps**:

1. Add `_shutdown_event: threading.Event` to MainLoop
1. Register signal handlers in `run()` that set the event
1. Check event in main loop before each `receive()` call
1. Add `_current_message` tracking for visibility extension
1. Implement `shutdown()` method with configurable drain timeout
1. Add context manager support for clean resource management
1. Update `_handle_message` to support cancellation check

**Spec Updates**: `specs/MAIN_LOOP.md` - Add "Lifecycle Management" section

______________________________________________________________________

### 2. Session & Filesystem Archival to S3

**Problem**: Sessions exist only in-memory or local JSONL. Cannot debug
production issues without capturing state at evaluation completion.

**Scope**: ~500 LOC (new module)

**Deliverables**:

```python
# New archival protocol
from weakincentives.contrib.archival import S3Archiver, ArchivalConfig

@dataclass(frozen=True, slots=True)
class ArchivalConfig:
    bucket: str
    prefix: str = "sessions/"
    include_filesystem: bool = True
    compression: Literal["gzip", "zstd", "none"] = "gzip"

class S3Archiver:
    """Archive session snapshots and filesystem state to S3."""

    def archive(
        self,
        session: Session,
        filesystem: Filesystem | None = None,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> str:
        """Archive to S3, returns S3 URI."""
        ...

    def restore(self, uri: str) -> tuple[Snapshot, Filesystem | None]:
        """Restore from S3 URI."""
        ...
```

**Archive Structure**:

```
s3://bucket/prefix/{session_id}/{timestamp}/
├── snapshot.json.gz       # Session snapshot (Snapshot.to_json())
├── filesystem.tar.gz      # VFS contents (if enabled)
└── metadata.json          # Request context, user_id, tags
```

**Integration with MainLoop**:

```python
class AnalystLoop(MainLoop[AnalysisBrief, AnalysisResult]):
    def __init__(self, *, archiver: S3Archiver, **kwargs):
        super().__init__(**kwargs)
        self._archiver = archiver

    def finalize(self, prompt: Prompt[AnalysisResult], session: Session) -> None:
        # Archive on every completion for debugging
        filesystem = session.get_resource(Filesystem)
        uri = self._archiver.archive(
            session,
            filesystem,
            metadata={"prompt_key": prompt.key},
        )
        logger.info("Archived session", session_id=session.session_id, uri=uri)
```

**Implementation Steps**:

1. Create `src/weakincentives/contrib/archival/` module
1. Implement `ArchivalConfig` dataclass with validation
1. Implement `S3Archiver` with boto3 (optional dependency)
1. Add filesystem serialization (tar.gz of VFS contents)
1. Add compression support (gzip default, zstd optional)
1. Implement `restore()` for debugging workflows
1. Add async variant `archive_async()` for non-blocking uploads

**New Spec**: `specs/ARCHIVAL.md`

______________________________________________________________________

### 3. User Identity Context

**Problem**: No structured way to track which user initiated a request. Tool
handlers lack user context for audit logging and personalization.

**Scope**: ~200 LOC

**Deliverables**:

```python
# User identity model
@dataclass(frozen=True, slots=True)
class UserIdentity:
    """Structured user identity for request attribution."""

    user_id: str
    """Unique identifier (e.g., UUID, email, external ID)."""

    tenant_id: str | None = None
    """Multi-tenant isolation key."""

    roles: tuple[str, ...] = ()
    """Authorization roles (e.g., "analyst", "admin")."""

    attributes: Mapping[str, str] = field(default_factory=dict)
    """Additional context (e.g., department, team)."""

# Extended MainLoopRequest
@FrozenDataclass()
class MainLoopRequest[UserRequestT]:
    request: UserRequestT
    user: UserIdentity | None = None  # NEW
    budget: Budget | None = None
    deadline: Deadline | None = None
    resources: ResourceRegistry | None = None
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

# Tool context enhancement
class ToolContext:
    @property
    def user(self) -> UserIdentity | None:
        """User identity for the current request."""
        ...
```

**Integration Pattern**:

```python
# In tool handler
def run_analysis(params: AnalysisParams, *, context: ToolContext) -> ToolResult:
    user = context.user
    if user is None:
        return ToolResult(success=False, message="User identity required")

    logger.info(
        "Running analysis",
        user_id=user.user_id,
        tenant_id=user.tenant_id,
    )
    # ... tool logic
```

**Implementation Steps**:

1. Create `UserIdentity` dataclass in `src/weakincentives/runtime/identity.py`
1. Add `user` field to `MainLoopRequest`
1. Propagate identity through Session (as tag or dedicated field)
1. Expose via `ToolContext.user` property
1. Add to archival metadata automatically
1. Update OpenAI adapter to pass `user.user_id` as `user` parameter

**Spec Updates**: `specs/MAIN_LOOP.md`, `specs/TOOLS.md`

______________________________________________________________________

## P1: Critical for Operations

### 4. LangSmith Integration

**Problem**: Comprehensive spec exists (`specs/LANGSMITH.md`, 965 lines) but
implementation is 0% complete. No observability for production debugging.

**Scope**: ~800 LOC

**Deliverables** (per spec):

```python
from weakincentives.contrib.langsmith import configure_wink

# Auto-instrumentation
configure_wink(
    project="analyst-agent",
    tracing_enabled=True,
    hub_enabled=True,
)

# All evaluations now traced to LangSmith
response = adapter.evaluate(prompt, session=session)
```

**Key Components**:

1. `LangSmithTelemetryHandler` - Event bus subscriber, async upload queue
1. `LangSmithPromptOverridesStore` - Hub-backed prompt management
1. `configure_wink()` - One-call instrumentation function
1. Trace context propagation via `contextvars`

**Implementation Steps**:

1. Review `specs/LANGSMITH.md` thoroughly
1. Implement `LangSmithTelemetryHandler` with batched async uploads
1. Implement `LangSmithPromptOverridesStore` for Hub integration
1. Implement `configure_wink()` auto-instrumentation
1. Add deduplication for Claude Agent SDK native integration
1. Add graceful degradation (don't block on LangSmith failures)
1. Integration tests with mock LangSmith client

**Spec**: Already complete at `specs/LANGSMITH.md`

______________________________________________________________________

### 5. SQS Mailbox Implementation

**Problem**: Redis Mailbox works for single-region deployments. AWS-native
deployments need SQS for durability, dead-letter queues, and managed scaling.

**Scope**: ~400 LOC

**Deliverables**:

```python
from weakincentives.contrib.mailbox import SQSMailbox

requests = SQSMailbox[MainLoopRequest[AnalysisBrief]](
    queue_url="https://sqs.us-east-1.amazonaws.com/123456789/analyst-requests",
    dead_letter_queue_url="https://sqs.../analyst-requests-dlq",
    max_receive_count=3,  # Move to DLQ after 3 failures
)
```

**Implementation Steps**:

1. Create `src/weakincentives/contrib/mailbox/sqs.py`
1. Implement SQS send/receive/ack/nack with boto3
1. Handle MessageAttributes for serde type hints
1. Implement long polling with SQS wait semantics
1. Add dead-letter queue configuration
1. Add message deduplication support (FIFO queues)
1. Tests with moto mock SQS

**Spec**: Already covered in `specs/MAILBOX.md`

______________________________________________________________________

## P2: Scale & Reliability

### 6. AWS Bedrock Adapter

**Problem**: Cannot use managed AWS Bedrock service for Claude models. Required
for AWS-native deployments with VPC isolation requirements.

**Scope**: ~350 LOC

**Deliverables**:

```python
from weakincentives.adapters.bedrock import BedrockAdapter

adapter = BedrockAdapter(
    model_id="anthropic.claude-sonnet-4-5-20250929",
    region_name="us-east-1",
)
response = adapter.evaluate(prompt, session=session)
```

**Implementation Steps**:

1. Create `src/weakincentives/adapters/bedrock.py`
1. Implement Bedrock Converse API integration
1. Map structured output to Bedrock tool use patterns
1. Handle Bedrock-specific throttling (ThrottlingException)
1. Add cost tracking based on Bedrock pricing
1. Support cross-region inference profiles
1. Tests with moto mock Bedrock

**Spec Updates**: `specs/ADAPTERS.md`, `specs/CLAUDE_AGENT_SDK.md`

______________________________________________________________________

### 7. Metrics & Health Checks

**Problem**: No metrics export for monitoring infrastructure. Cannot integrate
with Prometheus/Grafana or AWS CloudWatch.

**Scope**: ~300 LOC

**Deliverables**:

```python
from weakincentives.contrib.metrics import PrometheusMetrics

metrics = PrometheusMetrics(namespace="analyst")

# Attach to MainLoop
loop = AnalystLoop(..., metrics=metrics)

# Metrics exported:
# - analyst_requests_total{status="success|error"}
# - analyst_request_duration_seconds
# - analyst_tokens_total{type="input|output"}
# - analyst_queue_depth
```

**Implementation Steps**:

1. Create `src/weakincentives/contrib/metrics/` module
1. Define `MetricsCollector` protocol
1. Implement `PrometheusMetrics` with prometheus-client
1. Implement `CloudWatchMetrics` with boto3
1. Instrument MainLoop with request/error/latency metrics
1. Add token usage tracking from adapter responses
1. Add health check endpoint (`/health`, `/ready`)

**New Spec**: `specs/METRICS.md`

______________________________________________________________________

### 8. Configurable Retry & Dead-Letter Handling

**Problem**: Basic backoff exists but no configurable retry policies or
dead-letter routing for poison messages.

**Scope**: ~250 LOC

**Deliverables**:

```python
@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 300.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: tuple[type[Exception], ...] = (
        RateLimitError,
        TimeoutError,
    )

class MainLoop:
    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | None = None,
        dead_letter: Mailbox[FailedMessage] | None = None,
        ...
    ):
        ...
```

**Implementation Steps**:

1. Create `RetryPolicy` dataclass with configurable parameters
1. Add `dead_letter` mailbox parameter to MainLoop
1. Classify errors as retryable vs. permanent
1. Route permanent failures to dead-letter queue
1. Include failure context (exception, stack trace, attempt count)
1. Add retry metrics

**Spec Updates**: `specs/MAIN_LOOP.md`

______________________________________________________________________

## P3: Nice to Have

### 9. Configuration Management

**Problem**: Hardcoded values require code changes for environment differences.

**Scope**: ~200 LOC

**Deliverables**:

```python
from weakincentives.config import WinkConfig

config = WinkConfig.from_env()  # Reads WINK_* environment variables
# or
config = WinkConfig.from_file("config.toml")
```

**Implementation Steps**:

1. Define configuration schema as dataclass
1. Environment variable loading with prefixes
1. TOML/YAML file support
1. Validation on load
1. Secrets manager integration (AWS Secrets Manager)

______________________________________________________________________

### 10. EvalLoop Co-location

**Problem**: Running evaluation loops (for prompt optimization) alongside
MainLoop in the same process requires coordinated shutdown.

**Scope**: ~150 LOC

**Deliverables**:

```python
from weakincentives.runtime import ProcessGroup

group = ProcessGroup()
group.add(main_loop)
group.add(eval_loop)

# Single shutdown signal stops all loops gracefully
group.run()
```

This builds on top of the graceful shutdown work from P0.

______________________________________________________________________

## Implementation Roadmap

### Phase 1: Production Minimum (P0)

| Item | Estimate | Dependencies |
| ----------------------------- | -------- | ------------ |
| Graceful Shutdown | ~300 LOC | None |
| Session Archival to S3 | ~500 LOC | boto3 |
| User Identity Context | ~200 LOC | None |

**Total**: ~1,000 LOC

### Phase 2: Observability (P1)

| Item | Estimate | Dependencies |
| ----------------------------- | -------- | ---------------- |
| LangSmith Integration | ~800 LOC | langsmith SDK |
| SQS Mailbox | ~400 LOC | boto3 |

**Total**: ~1,200 LOC

### Phase 3: AWS Native (P2)

| Item | Estimate | Dependencies |
| ----------------------------- | -------- | ---------------- |
| Bedrock Adapter | ~350 LOC | boto3 |
| Metrics & Health Checks | ~300 LOC | prometheus-client|
| Retry & Dead-Letter | ~250 LOC | None |

**Total**: ~900 LOC

### Phase 4: Polish (P3)

| Item | Estimate | Dependencies |
| ----------------------------- | -------- | ------------ |
| Configuration Management | ~200 LOC | tomli |
| EvalLoop Co-location | ~150 LOC | Phase 1 |

**Total**: ~350 LOC

______________________________________________________________________

## Dependencies to Add

```toml
# pyproject.toml [project.optional-dependencies]

archival = ["boto3>=1.34"]
langsmith = ["langsmith>=0.1"]
sqs = ["boto3>=1.34"]
bedrock = ["boto3>=1.34"]
metrics = ["prometheus-client>=0.20"]

# Combined production bundle
production = [
    "weakincentives[archival,langsmith,sqs,metrics]",
]

aws = [
    "weakincentives[archival,sqs,bedrock]",
]
```

______________________________________________________________________

## Testing Strategy

Each feature requires:

1. **Unit tests** with mocks (moto for AWS services, responses for HTTP)
1. **Integration tests** with real services (optional, CI-gated)
1. **Property tests** for serialization round-trips

**Coverage gate**: Maintain 100% coverage requirement

**Mutation testing**: Add archival and metrics modules to mutation testing scope

______________________________________________________________________

## Open Questions

1. **Archival format**: Should we support Parquet for analytics use cases?
1. **Multi-region**: Is cross-region session recovery needed?
1. **Tenant isolation**: Hard isolation (separate queues) or soft (tags)?
1. **Bedrock streaming**: Support streaming responses for long analyses?

______________________________________________________________________

## Appendix: Current State Summary

| Component | Status | Notes |
| -------------------- | ------ | ---------------------------------------- |
| MainLoop | 80% | Works, needs shutdown |
| Session Snapshots | 100% | `Snapshot.to_json()` / `from_json()` |
| External Storage | 0% | No S3/DB backends |
| User Identity | 10% | Basic `user` field in OpenAI adapter |
| LangSmith | 0% | Spec complete, no implementation |
| Redis Mailbox | 100% | Production-ready with Lua scripts |
| SQS Mailbox | 0% | Spec'd, not implemented |
| OpenAI Adapter | 100% | Full structured output support |
| LiteLLM Adapter | 100% | 100+ providers |
| Claude Agent SDK | 100% | MCP tool bridging |
| Bedrock Adapter | 0% | Not started |
| Metrics | 0% | Logging only |
