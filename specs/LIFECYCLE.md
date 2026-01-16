# Lifecycle Specification

## Purpose

Graceful shutdown coordination for `MainLoop` and `EvalLoop` instances.
Core at `src/weakincentives/runtime/lifecycle.py`.

## Principles

- **Cooperative shutdown**: Loops poll shutdown flag; in-flight work completes
- **Signal-driven**: SIGTERM/SIGINT trigger shutdown
- **Composable**: `LoopGroup` coordinates multiple loops
- **Timeout-bounded**: Configurable wait for in-flight work

## Core Abstractions

### Runnable Protocol

Both `MainLoop` and `EvalLoop` implement:

| Method | Description |
| --- | --- |
| `run(max_iterations, visibility_timeout, wait_time_seconds)` | Process messages until stopped |
| `shutdown(timeout)` | Request graceful shutdown, wait for completion |
| `running` property | True if processing messages |
| Context manager | Exit triggers shutdown |

### ShutdownCoordinator

Singleton managing signal handlers at `src/weakincentives/runtime/lifecycle.py`:

| Method | Description |
| --- | --- |
| `install(signals)` | Install handlers, return singleton |
| `get()` | Return installed coordinator or None |
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

## MainLoop Changes

### Lifecycle Attributes

- `_shutdown_event: threading.Event`
- `_running: bool`
- `_lock: threading.Lock`

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

### Single Loop

```python
coordinator = ShutdownCoordinator.install()
coordinator.register(loop.shutdown)
loop.run()
```

### Multiple Loops

```python
group = LoopGroup(loops=[main_loop, eval_loop])
group.run()  # Blocks until SIGTERM/SIGINT
```

### Context Manager

```python
with LoopGroup(loops=[main_loop, eval_loop]) as group:
    thread = threading.Thread(target=group.run)
    thread.start()
# Shutdown triggered on exit
```

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

**Relationship:** `visibility_timeout` > `shutdown_timeout` + max processing time.

### Signals

- `SIGTERM`: Kubernetes, Docker, systemd, manual kill
- `SIGINT`: Ctrl+C, IDE stop

## Limitations

- **Single process**: Multi-process needs external coordination
- **No mid-message cancellation**: Use deadlines for time bounds
- **Python GIL**: Thread-based parallelism limited
- **Signal handler restrictions**: Handlers kept minimal

## Future Considerations

- Checkpoint support for long evaluations
- Health probes for Kubernetes
- Drain mode (stop accepting, complete in-flight)
- Priority shutdown (critical loops wait longer)
