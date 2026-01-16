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

**Timeout calibration:**

- `watchdog_threshold` should exceed your maximum expected prompt evaluation
  time
- `visibility_timeout` (in `run()`) should exceed `watchdog_threshold` to
  prevent message redelivery during long evaluations

## Graceful Shutdown

When SIGTERM arrives, LoopGroup:

1. Signals all loops to stop accepting new work
2. Waits for in-flight requests to complete (up to timeout)
3. Calls `shutdown()` on each loop
4. Exits cleanly

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
