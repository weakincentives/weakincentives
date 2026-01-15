# Health and Monitoring Specification

Health probes and automatic termination of stuck workers for container orchestrators.

**Source:** `src/weakincentives/runtime/health.py`

## Components

### Health Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `GET /health/live` | Liveness probe | 200 always |
| `GET /health/ready` | Readiness probe | 200 if healthy, 503 if unhealthy |

Use for Kubernetes `livenessProbe` (restart on failure) and `readinessProbe`
(remove from endpoints).

### Heartbeat

**Definition:** `runtime/heartbeat.py:Heartbeat`

Workers call `beat()` at regular intervals. Watchdog checks `elapsed()`.

```python
hb = Heartbeat()
hb.beat()          # Record heartbeat
hb.elapsed()       # Seconds since last beat
hb.add_callback(fn)  # Observer pattern for lease extension, metrics
```

### Watchdog

**Definition:** `runtime/health.py:Watchdog`

Daemon thread monitoring heartbeats. Terminates process via SIGKILL when
stall threshold exceeded.

```python
watchdog = Watchdog(heartbeats, stall_threshold=720.0, check_interval=60.0)
watchdog.start()
```

**Why SIGKILL:** Stuck threads cannot respond to SIGTERM. Container restart
is the recovery path; visibility timeout handles message redelivery.

## LoopGroup Integration

```python
group = LoopGroup(
    loops=[main_loop, eval_loop],
    health_port=8080,           # Enable health endpoints
    watchdog_threshold=720.0,   # 12 minutes (default)
    watchdog_interval=60.0,     # Check frequency
)
group.run()
```

## MainLoop Integration

Beats heartbeat:
1. After receiving messages from mailbox
2. After processing each message

## Timeout Calibration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `wait_time_seconds` | 30s | Long poll duration |
| `watchdog_threshold` | 720s | > long_poll + max_processing |
| `visibility_timeout` | 1800s | > threshold + processing + margin |

```
visibility_timeout > watchdog_threshold + max_processing_time
watchdog_threshold > wait_time_seconds + max_processing_time
watchdog_interval < watchdog_threshold / 3
```

## Kubernetes Configuration

```yaml
livenessProbe:
  httpGet: { path: /health/live, port: 8080 }
  periodSeconds: 10
  failureThreshold: 3
readinessProbe:
  httpGet: { path: /health/ready, port: 8080 }
  periodSeconds: 5
  failureThreshold: 2
```

## Limitations

- No partial recovery (kills entire process)
- Coarse granularity (per-loop, not per-message)
- False positives from long external calls
- Single process (multi-process needs separate monitoring)
