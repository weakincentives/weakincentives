# Health Endpoint Specification

## Purpose

Expose HTTP health endpoints from `LoopGroup` for Kubernetes liveness and
readiness probes. Uses Python stdlib exclusively (`http.server`) to avoid
external dependencies.

## Guiding Principles

- **Stdlib-only**: No external dependencies. Use `http.server` from the
  standard library.
- **Kubernetes-native**: Follow Kubernetes probe conventions with `/health/live`
  and `/health/ready` endpoints.
- **Non-blocking**: Health server runs in a dedicated daemon thread, never
  blocks the main loops.
- **Minimal overhead**: Lightweight responses, no database queries or expensive
  checks.
- **Graceful integration**: Health server lifecycle managed by `LoopGroup`,
  shuts down cleanly with the group.

## Core Abstractions

### HealthStatus

Enum representing the possible health states:

```python
from enum import Enum

class HealthStatus(Enum):
    """Health check result status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
```

### HealthCheck Protocol

Interface for pluggable health checks:

```python
from typing import Protocol

class HealthCheck(Protocol):
    """Protocol for health check implementations."""

    @property
    def name(self) -> str:
        """Identifier for this health check."""
        ...

    def check(self) -> HealthStatus:
        """Perform the health check.

        Returns:
            HealthStatus indicating current health.

        Note:
            Implementations must be thread-safe and non-blocking.
            Expensive checks should cache results.
        """
        ...
```

### HealthServer

HTTP server exposing health endpoints:

```python
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable

class HealthRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health endpoints."""

    # Set by HealthServer before serving
    liveness_check: Callable[[], HealthStatus]
    readiness_check: Callable[[], HealthStatus]

    def do_GET(self) -> None:
        """Handle GET requests to health endpoints."""
        if self.path == "/health/live":
            self._respond_health(self.liveness_check())
        elif self.path == "/health/ready":
            self._respond_health(self.readiness_check())
        elif self.path == "/health":
            # Combined endpoint returns both checks
            self._respond_combined()
        else:
            self.send_error(404, "Not Found")

    def _respond_health(self, status: HealthStatus) -> None:
        """Send health check response."""
        http_status = 200 if status == HealthStatus.HEALTHY else 503
        body = json.dumps({"status": status.value}).encode("utf-8")

        self.send_response(http_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_combined(self) -> None:
        """Send combined health check response."""
        live = self.liveness_check()
        ready = self.readiness_check()

        # Overall status: healthy only if both pass
        overall = HealthStatus.HEALTHY
        if live != HealthStatus.HEALTHY or ready != HealthStatus.HEALTHY:
            overall = HealthStatus.UNHEALTHY

        http_status = 200 if overall == HealthStatus.HEALTHY else 503
        body = json.dumps({
            "status": overall.value,
            "checks": {
                "liveness": live.value,
                "readiness": ready.value,
            },
        }).encode("utf-8")

        self.send_response(http_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default logging to stderr."""
        pass  # Health checks are high-frequency; don't spam logs


class HealthServer:
    """HTTP server for health endpoints.

    Runs in a daemon thread, automatically stopping when the main
    process exits. Integrates with LoopGroup for coordinated shutdown.

    Example:
        server = HealthServer(port=8080)
        server.start()
        # ... run loops ...
        server.stop()

    Example with custom checks:
        def my_readiness() -> HealthStatus:
            return HealthStatus.HEALTHY if db.connected else HealthStatus.UNHEALTHY

        server = HealthServer(port=8080, readiness_check=my_readiness)
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        liveness_check: Callable[[], HealthStatus] | None = None,
        readiness_check: Callable[[], HealthStatus] | None = None,
    ) -> None:
        """Initialize the health server.

        Args:
            host: Address to bind to. Defaults to all interfaces.
            port: Port to listen on. Defaults to 8080.
            liveness_check: Custom liveness check. Defaults to always healthy.
            readiness_check: Custom readiness check. Defaults to always healthy.
        """
        self.host = host
        self.port = port
        self._liveness_check = liveness_check or (lambda: HealthStatus.HEALTHY)
        self._readiness_check = readiness_check or (lambda: HealthStatus.HEALTHY)

        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the health server in a background thread.

        Safe to call multiple times; subsequent calls are no-ops.
        Blocks until the server is ready to accept connections.
        """
        with self._lock:
            if self._server is not None:
                return

            # Create handler class with bound checks
            handler = type(
                "BoundHealthHandler",
                (HealthRequestHandler,),
                {
                    "liveness_check": staticmethod(self._liveness_check),
                    "readiness_check": staticmethod(self._readiness_check),
                },
            )

            self._server = HTTPServer((self.host, self.port), handler)

            self._thread = threading.Thread(
                target=self._serve,
                name="health-server",
                daemon=True,
            )
            self._thread.start()

            # Wait for server to be ready
            self._started.wait(timeout=5.0)

    def _serve(self) -> None:
        """Server thread entry point."""
        assert self._server is not None
        self._started.set()
        self._server.serve_forever()

    def stop(self, *, timeout: float = 5.0) -> bool:
        """Stop the health server.

        Args:
            timeout: Maximum seconds to wait for shutdown.

        Returns:
            True if stopped cleanly, False if timeout expired.
        """
        with self._lock:
            if self._server is None:
                return True

            self._server.shutdown()

            if self._thread is not None:
                self._thread.join(timeout=timeout)
                if self._thread.is_alive():
                    return False

            self._server = None
            self._thread = None
            return True

    @property
    def running(self) -> bool:
        """True if the health server is running."""
        with self._lock:
            return self._server is not None

    @property
    def address(self) -> tuple[str, int] | None:
        """Return (host, port) if running, None otherwise."""
        with self._lock:
            if self._server is None:
                return None
            return self._server.server_address
```

## LoopGroup Integration

### Updated LoopGroup

`LoopGroup` gains optional health server management:

```python
class LoopGroup:
    """Coordinates lifecycle of multiple loops with optional health endpoints."""

    def __init__(
        self,
        loops: Sequence[Runnable],
        *,
        shutdown_timeout: float = 30.0,
        health_port: int | None = None,
        health_host: str = "0.0.0.0",
    ) -> None:
        """Initialize the loop group.

        Args:
            loops: Sequence of Runnable loops to manage.
            shutdown_timeout: Max seconds to wait per loop during shutdown.
            health_port: Port for health endpoints. None disables health server.
            health_host: Address for health server to bind to.
        """
        self.loops = loops
        self.shutdown_timeout = shutdown_timeout
        self._health_port = health_port
        self._health_host = health_host

        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future[None]] = []
        self._health_server: HealthServer | None = None

    def _loops_liveness(self) -> HealthStatus:
        """Liveness check: process is alive and responsive."""
        # Liveness just confirms the process can respond
        return HealthStatus.HEALTHY

    def _loops_readiness(self) -> HealthStatus:
        """Readiness check: all loops are running and accepting work."""
        if not self.loops:
            return HealthStatus.HEALTHY

        all_running = all(loop.running for loop in self.loops)
        return HealthStatus.HEALTHY if all_running else HealthStatus.UNHEALTHY

    def run(
        self,
        *,
        install_signals: bool = True,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        """Run all loops until shutdown.

        If health_port was specified, starts the health server before
        running loops. Health server stops during shutdown.
        """
        # Start health server if configured
        if self._health_port is not None:
            self._health_server = HealthServer(
                host=self._health_host,
                port=self._health_port,
                liveness_check=self._loops_liveness,
                readiness_check=self._loops_readiness,
            )
            self._health_server.start()

        if install_signals:
            coordinator = ShutdownCoordinator.install()
            coordinator.register(self._trigger_shutdown)

        self._executor = ThreadPoolExecutor(
            max_workers=len(self.loops),
            thread_name_prefix="loop-worker",
        )

        try:
            for loop in self.loops:
                future = self._executor.submit(
                    loop.run,
                    visibility_timeout=visibility_timeout,
                    wait_time_seconds=wait_time_seconds,
                )
                self._futures.append(future)

            for future in self._futures:
                future.result()

        finally:
            self._executor.shutdown(wait=True)

            # Stop health server
            if self._health_server is not None:
                self._health_server.stop()
                self._health_server = None

    @property
    def health_server(self) -> HealthServer | None:
        """Return the health server if running, None otherwise."""
        return self._health_server
```

### ShutdownCoordinator Health Integration

`ShutdownCoordinator` can optionally run a standalone health server:

```python
class ShutdownCoordinator:
    """Coordinates graceful shutdown with optional health endpoints."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[], None]] = []
        self._callbacks_lock = threading.Lock()
        self._triggered = threading.Event()
        self._health_server: HealthServer | None = None

    @classmethod
    def install(
        cls,
        *,
        signals: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT),
        health_port: int | None = None,
        health_host: str = "0.0.0.0",
    ) -> ShutdownCoordinator:
        """Install signal handlers and optionally start health server.

        Args:
            signals: Signals to handle.
            health_port: Port for health endpoints. None disables.
            health_host: Address to bind health server.

        Returns:
            The singleton ShutdownCoordinator instance.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                for sig in signals:
                    signal.signal(sig, cls._instance._handle_signal)

                if health_port is not None:
                    cls._instance._health_server = HealthServer(
                        host=health_host,
                        port=health_port,
                        liveness_check=cls._instance._coordinator_liveness,
                        readiness_check=cls._instance._coordinator_readiness,
                    )
                    cls._instance._health_server.start()

            return cls._instance

    def _coordinator_liveness(self) -> HealthStatus:
        """Liveness: process responsive, not yet triggered."""
        return HealthStatus.HEALTHY

    def _coordinator_readiness(self) -> HealthStatus:
        """Readiness: shutdown not yet triggered."""
        if self._triggered.is_set():
            return HealthStatus.UNHEALTHY
        return HealthStatus.HEALTHY

    @classmethod
    def reset(cls) -> None:
        """Clear singleton and stop health server (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                if cls._instance._health_server is not None:
                    cls._instance._health_server.stop()
            cls._instance = None
```

## Endpoints

### GET /health/live

Liveness probe. Returns 200 if the process is alive and responsive.

**Response (healthy):**

```json
HTTP/1.1 200 OK
Content-Type: application/json

{"status": "healthy"}
```

**Response (unhealthy):**

```json
HTTP/1.1 503 Service Unavailable
Content-Type: application/json

{"status": "unhealthy"}
```

### GET /health/ready

Readiness probe. Returns 200 if all loops are running and ready to accept work.

**Response (healthy):**

```json
HTTP/1.1 200 OK
Content-Type: application/json

{"status": "healthy"}
```

**Response (unhealthy - during shutdown or startup):**

```json
HTTP/1.1 503 Service Unavailable
Content-Type: application/json

{"status": "unhealthy"}
```

### GET /health

Combined endpoint returning both checks.

**Response (healthy):**

```json
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "healthy",
  "checks": {
    "liveness": "healthy",
    "readiness": "healthy"
  }
}
```

**Response (degraded):**

```json
HTTP/1.1 503 Service Unavailable
Content-Type: application/json

{
  "status": "unhealthy",
  "checks": {
    "liveness": "healthy",
    "readiness": "unhealthy"
  }
}
```

## Usage Patterns

### Basic LoopGroup with Health

```python
from weakincentives.runtime import LoopGroup

group = LoopGroup(
    loops=[main_loop, eval_loop],
    health_port=8080,
)
group.run()  # Health available at http://localhost:8080/health
```

### Custom Health Checks

```python
from weakincentives.runtime import LoopGroup, HealthServer, HealthStatus

def check_database() -> HealthStatus:
    try:
        db.ping()
        return HealthStatus.HEALTHY
    except Exception:
        return HealthStatus.UNHEALTHY

# Create server with custom checks
server = HealthServer(
    port=8080,
    readiness_check=check_database,
)
server.start()

group = LoopGroup(loops=[main_loop])
try:
    group.run()
finally:
    server.stop()
```

### Standalone ShutdownCoordinator with Health

```python
from weakincentives.runtime import ShutdownCoordinator

coordinator = ShutdownCoordinator.install(health_port=8080)
coordinator.register(loop.shutdown)

loop.run()
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wink-agent
spec:
  template:
    spec:
      containers:
        - name: agent
          image: myregistry/wink-agent:latest
          ports:
            - containerPort: 8080
              name: health
          livenessProbe:
            httpGet:
              path: /health/live
              port: health
            initialDelaySeconds: 5
            periodSeconds: 10
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health/ready
              port: health
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 3
```

## Thread Safety

### HealthServer

- `_server` and `_thread` protected by `_lock`
- `_started` is a `threading.Event` (inherently thread-safe)
- Request handlers run in the server thread, isolated from main loops
- `start()` and `stop()` can be called from any thread

### Handler Thread Safety

- `liveness_check` and `readiness_check` callables must be thread-safe
- Default checks only read `threading.Event` or iterate loop references
- Custom checks are responsible for their own thread safety

### LoopGroup Integration

- Health server starts before loop threads
- Health server stops after loop threads complete
- `_loops_readiness()` iterates `self.loops` (immutable sequence reference)
- `loop.running` property is thread-safe per Runnable protocol

## Configuration

### Default Ports

| Component | Default Port | Environment Variable |
|-----------|--------------|---------------------|
| LoopGroup | None (disabled) | `WINK_HEALTH_PORT` |
| ShutdownCoordinator | None (disabled) | `WINK_HEALTH_PORT` |

### Recommended Settings

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `health_port` | 8080 | Standard HTTP port for containers |
| `health_host` | "0.0.0.0" | Accept connections from all interfaces |
| Kubernetes `initialDelaySeconds` | 5 | Time for loops to start |
| Kubernetes `periodSeconds` | 10 (live), 5 (ready) | Check frequency |
| Kubernetes `failureThreshold` | 3 | Failures before unhealthy |

## Testing

### Unit Tests

```python
def test_health_server_starts_and_stops():
    server = HealthServer(port=0)  # Port 0 = OS assigns
    server.start()

    assert server.running
    host, port = server.address

    # Make request
    import urllib.request
    response = urllib.request.urlopen(f"http://{host}:{port}/health/live")
    assert response.status == 200

    server.stop()
    assert not server.running


def test_readiness_check_called():
    called = threading.Event()

    def my_check() -> HealthStatus:
        called.set()
        return HealthStatus.HEALTHY

    server = HealthServer(port=0, readiness_check=my_check)
    server.start()

    host, port = server.address
    urllib.request.urlopen(f"http://{host}:{port}/health/ready")

    assert called.is_set()
    server.stop()


def test_unhealthy_returns_503():
    server = HealthServer(
        port=0,
        readiness_check=lambda: HealthStatus.UNHEALTHY,
    )
    server.start()

    host, port = server.address

    try:
        urllib.request.urlopen(f"http://{host}:{port}/health/ready")
        assert False, "Expected 503"
    except urllib.error.HTTPError as e:
        assert e.code == 503

    server.stop()


def test_loop_group_health_integration():
    loop = MockRunnable()
    group = LoopGroup(loops=[loop], health_port=0)

    # Run in thread
    thread = threading.Thread(target=group.run)
    thread.start()

    time.sleep(0.1)  # Let health server start

    assert group.health_server is not None
    host, port = group.health_server.address

    response = urllib.request.urlopen(f"http://{host}:{port}/health")
    assert response.status == 200

    group.shutdown()
    thread.join()

    assert group.health_server is None
```

### Integration Tests

```python
def test_kubernetes_probe_simulation():
    """Simulate Kubernetes probe behavior."""
    group = LoopGroup(loops=[main_loop, eval_loop], health_port=0)

    thread = threading.Thread(target=group.run)
    thread.start()

    time.sleep(0.5)  # Startup delay

    host, port = group.health_server.address

    # Liveness should pass immediately
    live = urllib.request.urlopen(f"http://{host}:{port}/health/live")
    assert live.status == 200

    # Readiness should pass once loops are running
    ready = urllib.request.urlopen(f"http://{host}:{port}/health/ready")
    assert ready.status == 200

    # Trigger shutdown
    group.shutdown()

    # Readiness should fail during shutdown (loops stopping)
    # Note: This tests the transition period

    thread.join()
```

## Limitations

- **No HTTPS**: stdlib `http.server` doesn't support TLS. Use a sidecar proxy
  (nginx, envoy) or service mesh for TLS termination.
- **No authentication**: Health endpoints are unauthenticated. Restrict access
  via network policies or firewall rules.
- **Single server**: One health server per LoopGroup/ShutdownCoordinator. For
  complex deployments, use a dedicated health aggregator.
- **Port conflicts**: Ensure `health_port` doesn't conflict with other services.
  Use port 0 in tests for OS-assigned ports.
- **No metrics**: Health endpoints don't expose Prometheus metrics. Use a
  separate metrics endpoint if needed.

## Future Considerations

- **Detailed checks**: Expose individual loop status in `/health` response
- **Custom endpoints**: Allow registering additional health checks by name
- **Graceful drain**: Return 503 on `/health/ready` during graceful shutdown
  period to stop new traffic before termination
- **Startup probe**: Add `/health/startup` for slow-starting applications
- **Metrics integration**: Expose health metrics for observability platforms
