# Worker Deployment Production Readiness Roadmap

Target: Deploy WINK agents as Docker containers in Kubernetes, pulling tasks from
Redis, with Datadog/LangSmith observability and S3 state persistence.

## Current State Summary

| Component | Status | Notes |
|-----------|--------|-------|
| **MainLoop** | ✅ Complete | Mailbox-based worker pattern ready |
| **RedisMailbox** | ✅ Complete | Cluster-aware, Lua atomics, visibility timeout |
| **InMemoryMailbox** | ✅ Complete | Thread-safe, for testing |
| **SQSMailbox** | ❌ Spec only | `specs/MAILBOX.md:568-588` |
| **Session Snapshots** | ✅ Complete | JSON serialization, restore |
| **Filesystem Protocol** | ✅ Complete | InMemory, Host, Podman backends |
| **CompositeSnapshot** | ✅ Complete | Session + FS combined snapshots |
| **ExecutionState** | ✅ Complete | Transactional tool execution |
| **LangSmith** | ❌ Spec only | `specs/LANGSMITH.md` (965 lines) |
| **Datadog/OTEL** | ❌ None | Not started |
| **Bedrock Adapter** | ❌ None | Not started |
| **S3 Persistence** | ❌ None | Not started |
| **Docker/K8s** | ❌ None | Library only |

---

## Phase 1: Observability Foundation

**Goal**: Full visibility into agent execution for debugging and monitoring.

### 1.1 LangSmith Integration (Spec Exists)

Location: `src/weakincentives/contrib/langsmith/`

```
contrib/langsmith/
├── __init__.py           # configure_wink() entry point
├── _handler.py           # LangSmithTelemetryHandler
├── _hub.py               # LangSmithPromptOverridesStore
└── _deduplication.py     # Claude SDK trace deduplication
```

Implement per `specs/LANGSMITH.md`:

- [ ] **Auto-instrumentation** via `configure_wink()` patching `InProcessDispatcher`
- [ ] **Telemetry handler** subscribing to `PromptRendered`, `ToolInvoked`, `PromptExecuted`
- [ ] **Async upload** with background thread and graceful flush on exit
- [ ] **Hub integration** for pull/push prompt overrides
- [ ] **Claude SDK deduplication** to avoid duplicate traces

Key events to trace:
```python
# Maps to LangSmith run types
PromptRendered   → run_type="chain", name=prompt.key
ToolInvoked      → run_type="tool", name=tool.name
PromptExecuted   → updates parent chain run with output/error
TokenUsage       → run.usage.total_tokens
```

### 1.2 Datadog Integration (New)

Location: `src/weakincentives/contrib/datadog/`

```
contrib/datadog/
├── __init__.py           # configure_datadog()
├── _handler.py           # DatadogTelemetryHandler (StatsD + APM)
├── _metrics.py           # Metric definitions
└── _traces.py            # DDTrace span integration
```

Features:
- [ ] **Metrics via DogStatsD**:
  - `wink.prompt.duration` (histogram)
  - `wink.prompt.tokens.input` / `wink.prompt.tokens.output` (count)
  - `wink.tool.invocations` (count, tagged by tool name)
  - `wink.tool.duration` (histogram)
  - `wink.mailbox.messages.received` / `wink.mailbox.messages.acked`
  - `wink.worker.active` (gauge)

- [ ] **APM Traces via ddtrace**:
  - Span per prompt evaluation
  - Child spans for tool invocations
  - Correlation with LangSmith trace IDs

- [ ] **Service check**: `wink.worker.heartbeat`

### 1.3 Structured Logging

Location: `src/weakincentives/runtime/logging/`

- [ ] **JSON log formatter** for container stdout
- [ ] **Correlation IDs**: `request_id`, `session_id`, `trace_id`
- [ ] **Redaction** for sensitive fields (PII, API keys)
- [ ] **Log levels**: Map event severity to appropriate levels

---

## Phase 2: State Persistence

**Goal**: Durable snapshots for long-running evaluations and crash recovery.

### 2.1 Snapshot Store Protocol

Location: `src/weakincentives/runtime/snapshot_store/`

```python
class SnapshotStore(Protocol):
    """Durable storage for CompositeSnapshot."""

    def save(
        self,
        snapshot: CompositeSnapshot,
        *,
        key: str | None = None,
    ) -> str:
        """Persist snapshot, return storage key."""
        ...

    def load(self, key: str) -> CompositeSnapshot:
        """Retrieve snapshot by key."""
        ...

    def list(
        self,
        *,
        session_id: UUID | None = None,
        prefix: str | None = None,
        limit: int = 100,
    ) -> Sequence[SnapshotMetadata]:
        """List available snapshots."""
        ...

    def delete(self, key: str) -> None:
        """Remove snapshot."""
        ...
```

### 2.2 S3 Snapshot Store

Location: `src/weakincentives/contrib/snapshot_store/_s3.py`

```python
@dataclass(frozen=True, slots=True)
class S3SnapshotStoreConfig:
    bucket: str
    prefix: str = "snapshots/"
    region: str | None = None
    endpoint_url: str | None = None  # LocalStack/MinIO
    compression: Literal["none", "gzip", "zstd"] = "zstd"
```

Features:
- [ ] **Key format**: `{prefix}{session_id}/{snapshot_id}.json.zst`
- [ ] **Compression**: zstd for ~80% size reduction
- [ ] **Metadata**: S3 object metadata with tags, created_at
- [ ] **Lifecycle**: Integrate with S3 lifecycle policies for retention
- [ ] **Multipart upload**: For large snapshots (>5MB)

### 2.3 MainLoop Persistence Integration

Extend `MainLoop` for automatic snapshot persistence:

```python
class MainLoop[UserRequestT, OutputT](ABC):
    def __init__(
        self,
        *,
        snapshot_store: SnapshotStore | None = None,
        snapshot_on_complete: bool = True,
        snapshot_on_error: bool = True,
    ) -> None:
        ...
```

- [ ] **Auto-save** on successful completion
- [ ] **Auto-save** on error (for debugging)
- [ ] **Resume** from stored snapshot

---

## Phase 3: AWS Bedrock Adapter

**Goal**: Native AWS Bedrock support for enterprise deployments.

### 3.1 Bedrock Adapter

Location: `src/weakincentives/adapters/bedrock/`

```
adapters/bedrock/
├── __init__.py
├── adapter.py            # BedrockAdapter
├── _config.py            # BedrockClientConfig, BedrockModelConfig
├── _formatting.py        # Prompt → Bedrock message format
└── _parsing.py           # Response parsing
```

```python
@dataclass(frozen=True, slots=True)
class BedrockClientConfig:
    region_name: str | None = None
    endpoint_url: str | None = None
    profile_name: str | None = None
    # Uses boto3 credential chain by default

@dataclass(frozen=True, slots=True)
class BedrockModelConfig:
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: tuple[str, ...] | None = None

class BedrockAdapter(ProviderAdapter[OutputT]):
    def __init__(
        self,
        model_id: str,  # e.g., "anthropic.claude-3-sonnet-20240229-v1:0"
        client_config: BedrockClientConfig | None = None,
        model_config: BedrockModelConfig | None = None,
    ) -> None:
        ...
```

Features:
- [ ] **Converse API** for multi-turn with tool use
- [ ] **Streaming** via `converse_stream`
- [ ] **Guardrails** integration (optional)
- [ ] **Cross-region inference** profile support
- [ ] **Throttle handling** with exponential backoff

Model support:
- Claude 3.x (Sonnet, Haiku, Opus)
- Claude 3.5 Sonnet
- Amazon Titan
- Meta Llama 3.x

---

## Phase 4: Container & Orchestration

**Goal**: Production-ready Docker images and Kubernetes manifests.

### 4.1 Docker Image

Location: `docker/`

```dockerfile
# docker/Dockerfile
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (production only)
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ src/

# Default entrypoint for workers
ENTRYPOINT ["uv", "run", "python", "-m"]
CMD ["weakincentives.cli.worker"]
```

```yaml
# docker/docker-compose.yml
services:
  worker:
    build: .
    environment:
      - REDIS_URL=redis://redis:6379
      - AWS_REGION=us-east-1
      - DD_AGENT_HOST=datadog
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
    depends_on:
      - redis
      - localstack

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  localstack:
    image: localstack/localstack
    ports:
      - "4566:4566"
    environment:
      - SERVICES=s3,sqs
```

### 4.2 Kubernetes Manifests

Location: `k8s/`

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wink-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wink-worker
  template:
    metadata:
      labels:
        app: wink-worker
      annotations:
        ad.datadoghq.com/wink-worker.logs: '[{"source":"python","service":"wink-worker"}]'
    spec:
      containers:
        - name: worker
          image: wink-worker:latest
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
          env:
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: wink-secrets
                  key: redis-url
            - name: AWS_REGION
              value: "us-east-1"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
```

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: wink-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: wink-worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: External
      external:
        metric:
          name: redis_mailbox_pending_messages
        target:
          type: AverageValue
          averageValue: "10"
```

### 4.3 Worker CLI

Location: `src/weakincentives/cli/worker.py`

```python
@click.command()
@click.option("--mailbox", required=True, help="Mailbox connection URL")
@click.option("--concurrency", default=1, help="Concurrent workers")
@click.option("--max-iterations", default=0, help="0 = unlimited")
@click.option("--health-port", default=8080, help="Health check port")
def worker(mailbox: str, concurrency: int, max_iterations: int, health_port: int):
    """Run WINK worker daemon."""
    ...
```

Features:
- [ ] **Health endpoints**: `/health`, `/ready`, `/metrics`
- [ ] **Graceful shutdown**: SIGTERM handling
- [ ] **Concurrency**: Thread pool or asyncio workers
- [ ] **Metrics export**: Prometheus format for scraping

---

## Phase 5: Production Hardening

### 5.1 Error Handling & Retry

- [ ] **Dead letter queue**: Failed messages after N retries
- [ ] **Circuit breaker**: For provider API failures
- [ ] **Idempotency**: Request deduplication via `request_id`

### 5.2 Security

- [ ] **Secrets management**: AWS Secrets Manager / K8s secrets
- [ ] **IAM roles**: Fine-grained S3/Bedrock permissions
- [ ] **Network policies**: Restrict egress to required services
- [ ] **Pod security**: Non-root, read-only filesystem

### 5.3 Testing

- [ ] **Integration tests**: Full stack with LocalStack
- [ ] **Load tests**: Worker throughput benchmarks
- [ ] **Chaos tests**: Pod failures, Redis failover
- [ ] **E2E tests**: Request → Response validation

---

## Implementation Priority

| Priority | Phase | Component | Effort | Dependencies |
|----------|-------|-----------|--------|--------------|
| **P0** | 1.1 | LangSmith integration | 2 weeks | None (spec exists) |
| **P0** | 2.1-2.2 | S3 Snapshot Store | 1 week | None |
| **P0** | 4.1 | Docker image | 2 days | None |
| **P1** | 1.2 | Datadog integration | 1 week | Phase 1.1 (shared patterns) |
| **P1** | 3.1 | Bedrock adapter | 2 weeks | None |
| **P1** | 4.2 | K8s manifests | 3 days | Phase 4.1 |
| **P2** | 2.3 | MainLoop persistence | 3 days | Phase 2.2 |
| **P2** | 4.3 | Worker CLI | 1 week | Phase 4.1 |
| **P2** | 5.1 | Error handling | 1 week | Phase 4.3 |
| **P3** | 5.2 | Security hardening | 1 week | Phase 4.2 |
| **P3** | 5.3 | Testing suite | 2 weeks | All above |

---

## Gaps Not Covered

These are explicitly **out of scope** for this roadmap:

1. **SQSMailbox** - Redis preferred for K8s deployments; add later if needed
2. **Multi-region** - Single region initially; DR patterns later
3. **GPU workloads** - CPU-only workers; GPU scheduling is orthogonal
4. **Custom metrics aggregation** - Use Datadog/Prometheus directly
5. **Web dashboard** - Use LangSmith UI + Datadog dashboards

---

## Quick Start for Contributors

```bash
# 1. Set up development environment
uv sync && ./install-hooks.sh

# 2. Review relevant specs
cat specs/LANGSMITH.md    # For Phase 1.1
cat specs/MAILBOX.md      # For mailbox semantics
cat specs/ADAPTERS.md     # For Phase 3.1

# 3. Run tests
make check

# 4. Start implementing
# Begin with Phase 1.1 (LangSmith) as it has the most detailed spec
```
