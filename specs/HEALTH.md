# Health and Monitoring Specification

## Purpose

Enable container orchestrators to manage `LoopGroup` workers through health
probes and automatic termination of stuck workers. Two complementary mechanisms:

1. **Health endpoints**: HTTP for Kubernetes liveness/readiness probes
2. **Watchdog**: Internal monitor terminating unresponsive workers

**Implementation:**
- `src/weakincentives/runtime/watchdog.py` - Heartbeat, Watchdog, HealthServer
- `src/weakincentives/runtime/lifecycle.py` - LoopGroup

## Guiding Principles

- **External visibility**: Orchestrators query state without internal coupling
- **Self-healing**: Watchdog terminates stuck processes proactively
- **Fail fast**: Stuck worker = unrecoverable; terminate immediately
- **Heartbeat-based**: Workers prove liveness via timestamps; no bidirectional protocol
- **Observable**: Structured logs before termination for diagnosis

## Health Endpoints

### GET /health/live

Returns 200 if process responsive (liveness probe). Failed check restarts container.

### GET /health/ready

Returns 200 if all loops healthy, 503 otherwise (readiness probe). Failed check
removes pod from endpoints without restart.

### HealthServer

Minimal HTTP server using Python stdlib. Runs in daemon thread.

| Method | Description |
|--------|-------------|
| `start()` | Start in daemon thread |
| `stop()` | Graceful shutdown |
| `address` | `(host, port)` if running |

## Watchdog

### Heartbeat

Workers signal liveness via `heartbeat.beat()`. Supports observer pattern for
callbacks (lease extension, metrics).

| Method | Description |
|--------|-------------|
| `beat()` | Record heartbeat, invoke callbacks |
| `elapsed()` | Seconds since last heartbeat |
| `add_callback(fn)` | Register callback for beats |
| `remove_callback(fn)` | Unregister callback |

### Watchdog

Monitors heartbeats in daemon thread. Terminates via SIGKILL when threshold exceeded.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stall_threshold` | 720.0s | Seconds without heartbeat before termination |
| `check_interval` | 60.0s | Seconds between checks |

### Why SIGKILL

- Stuck threads cannot respond to SIGTERM
- Graceful shutdown already failed
- Container restart is recovery path
- Prevents resource exhaustion

## MainLoop Integration

MainLoop beats heartbeat at two points:
1. After receiving messages from mailbox
2. After processing each message

This proves both idle and busy loops are live.

## LoopGroup Integration

LoopGroup manages health endpoints and watchdog:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `health_port` | `None` | Port for health endpoints (disabled if None) |
| `watchdog_threshold` | 720.0s | Seconds without heartbeat before termination |
| `watchdog_interval` | 60.0s | Seconds between watchdog checks |

### Readiness and Heartbeats

When both enabled, `/health/ready` incorporates heartbeat freshness:
1. **Early warning**: Readiness fails when heartbeats stale
2. **Hard stop**: Watchdog terminates if heartbeats remain stale

## Default Timeout Calibration

Defaults calibrated for 10-minute prompt evaluations:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `wait_time_seconds` | 30s | Maximum long poll duration |
| `watchdog_threshold` | 720s (12 min) | > 30s + 600s with margin |
| `watchdog_interval` | 60s | < 720s / 3 |
| `visibility_timeout` | 1800s (30 min) | > 720s + 600s with margin |

### Threshold Relationships

```
visibility_timeout > watchdog_threshold + max_processing_time
watchdog_threshold > wait_time_seconds + max_processing_time
watchdog_interval < watchdog_threshold / 3
```

## Kubernetes Configuration

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 2
```

## Usage Examples

### Default (Recommended)

```python
group = LoopGroup(loops=[main_loop], health_port=8080)
group.run()
```

### Watchdog Only

```python
group = LoopGroup(loops=[main_loop])  # No health_port
group.run()
```

### Disable Watchdog

```python
group = LoopGroup(loops=[main_loop], health_port=8080, watchdog_threshold=None)
```

### Shorter Deadlines (2-Minute Evaluations)

```python
group = LoopGroup(
    loops=[main_loop],
    health_port=8080,
    watchdog_threshold=180.0,  # 3 minutes
    watchdog_interval=30.0,
)
```

### Longer Deadlines (30-Minute Batch Jobs)

```python
group = LoopGroup(
    loops=[batch_processor],
    watchdog_threshold=2100.0,  # 35 minutes
    watchdog_interval=120.0,
)
```

## Observability

### Logging

Watchdog emits CRITICAL logs before termination:
```
CRITICAL: Watchdog: main-loop stalled for 65.2s (threshold: 60.0s)
CRITICAL: Watchdog: terminating process due to stalled workers
```

### Metrics (Future)

- `wink_health_ready` - Gauge: 1 if ready
- `wink_heartbeat_age_seconds{loop="..."}` - Current age
- `wink_watchdog_stalls_total{loop="..."}` - Stall counter

## Limitations

- **No partial recovery**: Terminates entire process; cannot restart individual loops
- **Coarse granularity**: Per-loop heartbeats, not per-message
- **False positives**: Long external calls may trigger termination
- **Single process**: Health/watchdog per-process; multi-process needs separate monitoring
- **Daemon threads**: Will not prevent process exit
