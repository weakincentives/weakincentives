# Lifecycle Specification

## Purpose

Graceful shutdown coordination for `AgentLoop` and `EvalLoop` instances.
Core at `src/weakincentives/runtime/lifecycle.py`.

## Principles

- **Cooperative shutdown**: Loops poll shutdown flag; in-flight work completes
- **Signal-driven**: SIGTERM/SIGINT trigger shutdown
- **Composable**: `LoopGroup` coordinates multiple loops
- **Timeout-bounded**: Configurable wait for in-flight work

## Core Abstractions

### Runnable Protocol

Both `AgentLoop` and `EvalLoop` implement:

| Method | Description |
| --- | --- |
| `run(max_iterations, visibility_timeout, wait_time_seconds)` | Process messages until stopped |
| `shutdown(timeout)` | Request graceful shutdown, wait for completion |
| `running` property | True if processing messages |
| `heartbeat` property | Heartbeat tracker for watchdog monitoring (or None) |
| Context manager | Exit triggers shutdown |

### ShutdownCoordinator

Singleton managing signal handlers at `src/weakincentives/runtime/lifecycle.py`:

| Method | Description |
| --- | --- |
| `install(signals)` | Install handlers, return singleton |
| `get()` | Return installed coordinator or None |
| `reset()` | Reset singleton (for testing) |
| `register(callback)` | Add shutdown callback |
| `unregister(callback)` | Remove callback |
| `trigger()` | Manually trigger shutdown |
| `triggered` property | True if shutdown triggered |

Thread-safe: `_callbacks` protected by lock, `_triggered` is `threading.Event`.

### LoopGroup

Coordinates multiple loops at `src/weakincentives/runtime/lifecycle.py`:

| Method | Description |
| --- | --- |
| `run(install_signals, visibility_timeout, wait_time_seconds)` | Run all loops in threads |
| `shutdown(timeout)` | Stop all loops gracefully |

Each loop runs in dedicated thread via `ThreadPoolExecutor`.

| Feature | Description |
| --- | --- |
| Health endpoints | Optional HTTP `/health/live` and `/health/ready` for Kubernetes probes |
| Watchdog | Detect stuck workers via heartbeat monitoring |
| Coordinated shutdown | Signal-driven termination of all loops |

## Prompt Cleanup Phase

After execution completes, `AgentLoop` calls `prompt.cleanup()` to release
resources held by sections (e.g., temporary directories). In bundled execution,
cleanup is deferred until after bundle artifacts are captured. A
`prompt_cleaned_up` guard flag prevents double-cleanup in error paths.

See `PROMPTS.md` (Cleanup section) and `AGENT_LOOP.md` (Prompt Cleanup section)
for details.

## AgentLoop Lifecycle

### Run Loop

1. Set `_running = True`, clear `_shutdown_event`
1. Poll messages until max_iterations or shutdown
1. Check `_shutdown_event` before each message
1. On shutdown during batch: nack remaining messages
1. Set `_running = False` on exit

### Shutdown Method

Sets `_shutdown_event`, waits via `wait_until(lambda: not self.running, timeout)`.
Returns True if stopped cleanly, False on timeout.

## Usage Patterns

- **Single loop**: Install `ShutdownCoordinator`, register `loop.shutdown` as a
  callback, call `loop.run()`.
- **Multiple loops**: `LoopGroup(loops=[...]).run()` — blocks until SIGTERM/SIGINT.
- **With health/watchdog**: add `health_port=8080` and `watchdog_threshold=720.0`
  to `LoopGroup`.
- **Context manager**: `LoopGroup` supports `with` syntax; shutdown triggers on
  `__exit__`.

## Message Recovery

| Scenario | Recovery |
| --- | --- |
| In-flight when shutdown | Completes, acknowledged |
| Received but not started | Nacked immediately, redelivered |
| Never received | Picked up by other worker |
| Visibility expired | Redelivered by reaper |

## Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `shutdown_timeout` | 30s | Max wait for in-flight work |
| `visibility_timeout` | 300s | Mailbox invisibility period |
| `wait_time_seconds` | 20s | Long poll duration |
| `health_port` | None | Port for health endpoints |
| `watchdog_threshold` | 720s | Seconds without heartbeat before termination |

**Relationship:** `visibility_timeout` > `shutdown_timeout` + max processing time.

### Signals

- `SIGTERM`: Kubernetes, Docker, systemd, manual kill
- `SIGINT`: Ctrl+C, IDE stop

## Limitations

- **Single process**: Multi-process needs external coordination
- **No mid-message cancellation**: Use deadlines for time bounds
- **Python GIL**: Thread-based parallelism limited
- **Signal handler restrictions**: Handlers kept minimal
