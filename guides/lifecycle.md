# Lifecycle Management

*Canonical spec: [specs/HEALTH.md](../specs/HEALTH.md)*

When running agents in production—especially in containerized environments like
Kubernetes—you need coordinated shutdown, health monitoring, and watchdog
protection. WINK provides lifecycle primitives that integrate with MainLoop and
EvalLoop.

## LoopGroup: Running Multiple Loops

`LoopGroup` runs multiple loops in separate threads with coordinated shutdown
and optional health endpoints:

```python
from weakincentives.runtime import LoopGroup

# Run MainLoop and EvalLoop together
group = LoopGroup(loops=[main_loop, eval_loop])
group.run()  # Blocks until SIGTERM/SIGINT
```

For Kubernetes deployments, enable health endpoints and watchdog monitoring:

```python
group = LoopGroup(
    loops=[main_loop],
    health_port=8080,           # Exposes /health/live and /health/ready
    watchdog_threshold=720.0,   # Terminate if worker stalls for 12 minutes
)
group.run()
```

**Key features:**

- **Health endpoints**: `/health/live` (liveness) and `/health/ready`
  (readiness) for Kubernetes probes
- **Watchdog monitoring**: Detects stuck workers and terminates the process via
  SIGKILL when heartbeats stall
- **Coordinated shutdown**: SIGTERM/SIGINT triggers graceful shutdown of all
  loops

## ShutdownCoordinator: Manual Signal Handling

For finer control, use `ShutdownCoordinator` directly:

```python
from weakincentives.runtime import ShutdownCoordinator

coordinator = ShutdownCoordinator.install()
coordinator.register(loop.shutdown)
loop.run()
```

The coordinator installs signal handlers for SIGTERM and SIGINT. When a signal
arrives, all registered callbacks are invoked in registration order.

## The Runnable Protocol

Both `MainLoop` and `EvalLoop` implement the `Runnable` protocol:

```python
from typing import Protocol
from weakincentives.runtime import Heartbeat


class Runnable(Protocol):
    """Protocol for loops managed by LoopGroup."""

    def run(self, *, max_iterations: int | None = None) -> None: ...
    def shutdown(self, *, timeout: float = 30.0) -> bool: ...

    @property
    def running(self) -> bool: ...

    @property
    def heartbeat(self) -> Heartbeat | None: ...
```

This enables `LoopGroup` to manage any compliant loop implementation.

## The Heartbeat Mechanism

Workers prove liveness by calling `heartbeat.beat()` at regular intervals. The
watchdog tracks the last heartbeat time per loop and flags loops that exceed the
stall threshold.

```python
from weakincentives.runtime import Heartbeat

hb = Heartbeat()

# In your processing loop:
hb.beat()  # Record a heartbeat
print(hb.elapsed())  # Seconds since last heartbeat
```

**When heartbeats occur in MainLoop:**

1. After successfully receiving messages from the mailbox (proves the loop isn't
   stuck in `receive()`)
1. After processing each message (proves processing completes)

This means both idle loops (waiting on empty queues) and busy loops (processing
messages) demonstrate liveness.

## Health and Watchdog Configuration

The watchdog monitors heartbeats from loops and terminates the process if any
loop stalls beyond the threshold. This prevents "stuck worker" scenarios where a
loop hangs indefinitely.

```python
group = LoopGroup(
    loops=[main_loop, eval_loop],
    health_port=8080,           # Health endpoint port
    health_host="0.0.0.0",      # Bind to all interfaces
    watchdog_threshold=720.0,   # 12 minutes (calibrated for 10-min prompts)
    watchdog_interval=60.0,     # Check every minute
)
```

**Timeout calibration formulas:**

The stall threshold must be greater than:

- Maximum expected message processing time (e.g., 10 minutes = 600s)
- Mailbox long poll duration (`wait_time_seconds` = 30s)
- Sum of the above plus safety margin

```
visibility_timeout > watchdog_threshold + max_processing_time
    1800s > 720s + 600s ✓

watchdog_threshold > wait_time_seconds + max_processing_time
    720s > 30s + 600s ✓

watchdog_interval < watchdog_threshold / 3
    60s < 240s ✓
```

| Parameter | Default | Rationale |
| -------------------- | -------------- | ----------------------------------- |
| `wait_time_seconds` | 30s | Maximum long poll duration |
| `watchdog_threshold` | 720s (12 min) | > 30s + 600s with 90s margin |
| `watchdog_interval` | 60s | < 720s / 3, checks ~12x per threshold |
| `visibility_timeout` | 1800s (30 min) | > 720s + 600s with margin for retries |

## Why the Watchdog Uses SIGKILL

The watchdog uses `SIGKILL` rather than `SIGTERM` because:

1. **Stuck threads cannot respond**: If a thread is deadlocked or in an infinite
   loop, it cannot process signals or check shutdown flags.
1. **Graceful shutdown already failed**: The stall indicates the cooperative
   shutdown mechanism is ineffective for this failure mode.
1. **Container restart is the recovery path**: Kubernetes and Docker will
   restart the container. The visibility timeout ensures in-flight messages are
   redelivered to healthy workers.
1. **Prevent resource exhaustion**: A stuck process continues consuming CPU,
   memory, and connections. Immediate termination releases resources.

## Layered Defense with Readiness

When both health endpoints and watchdog are enabled, you get layered defense:

1. **Early warning**: Readiness probe fails when heartbeats are stale, removing
   the pod from service endpoints (traffic stops flowing)
1. **Hard stop**: Watchdog terminates the process if heartbeats remain stale

This allows Kubernetes to stop routing traffic before the watchdog terminates.
The readiness check incorporates heartbeat freshness—if any loop's heartbeat
exceeds the threshold, `/health/ready` returns 503.

## Graceful Shutdown

When SIGTERM arrives, LoopGroup:

1. Signals all loops to stop accepting new work
1. Waits for in-flight requests to complete (up to timeout)
1. Calls `shutdown()` on each loop
1. Exits cleanly

This ensures you don't drop work in progress and gives long-running evaluations
a chance to complete.

## Health Endpoints

When `health_port` is configured, LoopGroup exposes:

- `GET /health/live` — Returns 200 if the process is alive
- `GET /health/ready` — Returns 200 if all loops are ready to accept work

Use these with Kubernetes probes:

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
```

## Next Steps

- [Orchestration](orchestration.md): Learn about MainLoop
- [Evaluation](evaluation.md): Learn about EvalLoop
- [Debugging](debugging.md): Monitor running agents
