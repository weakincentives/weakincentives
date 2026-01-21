# Production Deployment

*Related guides: [Lifecycle](lifecycle.md), [Claude Agent SDK](claude-agent-sdk.md),
[Orchestration](orchestration.md)*

This guide covers deploying WINK agents to production environments. It assumes
familiarity with containerization, orchestration platforms, and cloud
infrastructure.

## Deployment Checklist

Before deploying, verify:

- [ ] `make check` passes (mandatory)
- [ ] Budgets and deadlines configured (prevent runaway costs)
- [ ] Health endpoints enabled (Kubernetes probes)
- [ ] Secrets management configured (no hardcoded credentials)
- [ ] Network isolation appropriate for workload
- [ ] Logging configured for your observability stack
- [ ] Debug bundles enabled for post-mortem analysis

## Container Image

### Dockerfile

```dockerfile
FROM python:3.12-slim

# Install system dependencies for sandboxing
RUN apt-get update && apt-get install -y --no-install-recommends \
    bubblewrap \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for Claude Agent SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

# Copy application code
COPY src/ src/
COPY skills/ skills/

# Create non-root user
RUN useradd -m -u 1000 agent
USER agent

# Health check endpoint
EXPOSE 8080

CMD ["python", "-m", "your_agent.main"]
```

### Multi-Stage Build (Smaller Image)

```dockerfile
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

FROM python:3.12-slim

# Runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    bubblewrap nodejs npm \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY skills/ skills/

RUN useradd -m -u 1000 agent
USER agent

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080

CMD ["python", "-m", "your_agent.main"]
```

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wink-agent
  labels:
    app: wink-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wink-agent
  template:
    metadata:
      labels:
        app: wink-agent
    spec:
      containers:
        - name: agent
          image: your-registry/wink-agent:latest
          ports:
            - containerPort: 8080
              name: health
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
          env:
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wink-secrets
                  key: anthropic-api-key
            - name: WEAKINCENTIVES_LOG_LEVEL
              value: "INFO"
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 30
            timeoutSeconds: 5
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 5
          volumeMounts:
            - name: workspace
              mountPath: /workspace
            - name: debug-bundles
              mountPath: /debug
      volumes:
        - name: workspace
          emptyDir:
            sizeLimit: 1Gi
        - name: debug-bundles
          persistentVolumeClaim:
            claimName: debug-bundles-pvc
```

### Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: wink-secrets
type: Opaque
stringData:
  anthropic-api-key: "sk-ant-..."
```

For production, use a secrets manager (AWS Secrets Manager, HashiCorp Vault,
etc.) instead of Kubernetes secrets directly.

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: wink-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: wink-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

## AWS Deployment

### ECS Task Definition

```json
{
  "family": "wink-agent",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/winkAgentRole",
  "containerDefinitions": [
    {
      "name": "agent",
      "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/wink-agent:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health/live || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      },
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:wink/api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/wink-agent",
          "awslogs-region": "REGION",
          "awslogs-stream-prefix": "agent"
        }
      }
    }
  ]
}
```

### Using AWS Bedrock

For Bedrock deployments, configure the task role with Bedrock permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.*"
    }
  ]
}
```

Configure the adapter:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
    get_default_model,
)
import os

# Set environment for Bedrock
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
os.environ["AWS_REGION"] = "us-east-1"

adapter = ClaudeAgentSDKAdapter(
    model=get_default_model(),  # Returns Bedrock model ID when USE_BEDROCK=1
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig.for_bedrock(),
    ),
)
```

## Configuration Management

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | Required (unless Bedrock) |
| `CLAUDE_CODE_USE_BEDROCK` | Use AWS Bedrock | `0` |
| `AWS_REGION` | AWS region for Bedrock | Required if Bedrock |
| `WEAKINCENTIVES_LOG_LEVEL` | Log verbosity | `INFO` |
| `WEAKINCENTIVES_LOG_JSON` | JSON log format | `true` in containers |

### Application Configuration

Use environment-based configuration with validation:

```python nocheck
from dataclasses import dataclass
from os import environ


@dataclass(frozen=True)
class AgentConfig:
    """Production configuration with validation."""

    model: str
    max_turns: int
    max_budget_usd: float
    health_port: int
    watchdog_threshold: float
    debug_bundle_path: str

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            model=environ.get("AGENT_MODEL", "claude-sonnet-4-5-20250929"),
            max_turns=int(environ.get("AGENT_MAX_TURNS", "25")),
            max_budget_usd=float(environ.get("AGENT_MAX_BUDGET_USD", "5.0")),
            health_port=int(environ.get("HEALTH_PORT", "8080")),
            watchdog_threshold=float(environ.get("WATCHDOG_THRESHOLD", "720")),
            debug_bundle_path=environ.get("DEBUG_BUNDLE_PATH", "/debug"),
        )
```

## Observability

### Structured Logging

Configure JSON logging for log aggregation:

```python nocheck
from weakincentives.runtime import configure_logging

configure_logging(
    level="INFO",
    json_mode=True,  # Machine-parseable logs
)
```

Logs include structured `event` and `context` fields. Route by event pattern:

| Event Pattern | Description |
|---------------|-------------|
| `prompt.render.*` | Prompt lifecycle |
| `tool.execution.*` | Tool calls |
| `adapter.evaluate.*` | Model interactions |
| `hook.*` | SDK hooks |

### Metrics

Export key metrics to your monitoring system:

```python nocheck
from weakincentives.runtime import InProcessDispatcher, PromptExecuted, ToolInvoked


def setup_metrics(dispatcher: InProcessDispatcher) -> None:
    def on_prompt_executed(event: PromptExecuted) -> None:
        # Export to Prometheus, Datadog, etc.
        metrics.histogram("prompt.tokens.input", event.usage.input_tokens)
        metrics.histogram("prompt.tokens.output", event.usage.output_tokens)
        metrics.counter("prompt.executions.total").inc()

    def on_tool_invoked(event: ToolInvoked) -> None:
        metrics.counter("tool.invocations.total", tags={"tool": event.name}).inc()
        if event.result.is_error:
            metrics.counter("tool.errors.total", tags={"tool": event.name}).inc()

    dispatcher.subscribe(PromptExecuted, on_prompt_executed)
    dispatcher.subscribe(ToolInvoked, on_tool_invoked)
```

### Debug Bundles in Production

Enable debug bundles for post-mortem analysis:

```python nocheck
from weakincentives.debug import BundleConfig
from weakincentives.runtime import MainLoopConfig

config = MainLoopConfig(
    debug_bundle=BundleConfig(
        target="/debug/bundles/",
        capture_filesystem=True,  # Include workspace snapshots
    ),
)
```

Mount a persistent volume for debug bundles and configure retention policies.

## Security Hardening

### Network Isolation

For code review workloads, disable network access:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import (
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)

isolation = IsolationConfig(
    network_policy=NetworkPolicy.no_network(),
    sandbox=SandboxConfig(
        enabled=True,
        allow_unsandboxed_commands=False,
    ),
)
```

### Workspace Boundaries

Restrict file access with `allowed_host_roots`:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentWorkspaceSection,
    HostMount,
)

workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(
        HostMount(
            host_path="/workspace/repos/project",
            mount_path="code",
            exclude_glob=(".git/*", "*.env", "*.key", "*.pem"),
            max_bytes=10_000_000,
        ),
    ),
    allowed_host_roots=("/workspace/repos",),  # Security boundary
)
```

### Resource Limits

Always set limits to prevent runaway agents:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKClientConfig

config = ClaudeAgentSDKClientConfig(
    max_turns=25,           # Prevent infinite loops
    max_budget_usd=5.0,     # Hard cost cap
    stop_on_structured_output=True,
)
```

### Secrets Hygiene

- Never log API keys or credentials
- Use secrets managers, not environment files
- Rotate keys regularly
- Use separate keys for dev/staging/prod
- Exclude sensitive files from workspace mounts (`.env`, `*.key`, `credentials.*`)

## Scaling Considerations

### Stateless Design

WINK agents are stateless by default—session state lives in memory during
execution. For scaling:

- Run multiple replicas behind a load balancer
- Use message queues (SQS, RabbitMQ) for work distribution
- Store debug bundles on shared storage (S3, EFS)

### Queue-Based Architecture

```python nocheck
from weakincentives.contrib.mailbox import SQSMailbox
from weakincentives.runtime import LoopGroup, MainLoop

mailbox = SQSMailbox(
    queue_url="https://sqs.region.amazonaws.com/account/queue",
    visibility_timeout=1800,  # 30 minutes for long tasks
    wait_time_seconds=20,
)

loop = MyMainLoop(adapter=adapter, mailbox=mailbox)
group = LoopGroup(
    loops=[loop],
    health_port=8080,
    watchdog_threshold=720.0,
)
group.run()
```

### Timeout Calibration

See [Lifecycle](lifecycle.md) for timeout calibration formulas. Key
relationships:

```
visibility_timeout > watchdog_threshold + max_processing_time
watchdog_threshold > wait_time_seconds + max_processing_time
```

## Rollout Strategy

### Canary Deployment

1. Deploy new version to a small percentage of replicas
1. Monitor error rates, latency, and cost metrics
1. Gradually increase traffic if metrics are healthy
1. Roll back immediately if issues detected

### Feature Flags

Use prompt overrides for gradual rollout of prompt changes:

```python nocheck
from weakincentives.prompt import Prompt
from weakincentives.prompt.overrides import LocalPromptOverridesStore

prompt = Prompt(
    template,
    overrides_store=LocalPromptOverridesStore(base_path="/config/overrides"),
    overrides_tag="v2-prompts" if feature_flag_enabled else None,
)
```

This lets you A/B test prompt changes without code deploys.

## Troubleshooting Production Issues

**Agent stuck/not responding:**

1. Check `/health/ready` endpoint
1. Review watchdog logs for stall detection
1. Check debug bundles for last known state
1. Verify network connectivity to API

**High error rates:**

1. Check `tool.errors.total` metrics
1. Review structured logs for error patterns
1. Examine debug bundles for failing requests
1. Verify API key validity and rate limits

**Cost spikes:**

1. Check `prompt.tokens.*` metrics
1. Review debug bundles for large prompts
1. Verify budget limits are set
1. Check for infinite loops in tool usage

**Slow responses:**

1. Check API latency metrics
1. Review tool execution timing in debug bundles
1. Consider enabling extended thinking for complex tasks
1. Verify resource limits aren't too restrictive

## Next Steps

- [Lifecycle](lifecycle.md): Health checks, watchdogs, and shutdown
- [Claude Agent SDK](claude-agent-sdk.md): Production adapter configuration
- [Debugging](debugging.md): Debug bundles and the debug UI
- [Evaluation](evaluation.md): Test agents before deploying
