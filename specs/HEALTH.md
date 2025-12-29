# Health Endpoint Specification

## Purpose

Expose HTTP health endpoints from `LoopGroup` for Kubernetes probes. Uses
Python stdlib exclusively (`http.server`).

## Endpoints

### GET /health/live

Returns 200 if the process is responsive.

```json
{"status": "healthy"}
```

### GET /health/ready

Returns 200 if all loops are running, 503 otherwise.

```json
{"status": "healthy"}
```

## Implementation

### HealthServer

```python
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable

class HealthServer:
    """Minimal HTTP server for health probes."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        readiness_check: Callable[[], bool] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._readiness_check = readiness_check or (lambda: True)
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start health server in a daemon thread."""
        if self._server is not None:
            return

        readiness = self._readiness_check

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/health/live":
                    self._send(200, {"status": "healthy"})
                elif self.path == "/health/ready":
                    ok = readiness()
                    self._send(200 if ok else 503, {"status": "healthy" if ok else "unhealthy"})
                else:
                    self.send_error(404)

            def _send(self, code: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args) -> None:
                pass  # Suppress logging

        self._server = HTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the health server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            self._thread = None

    @property
    def address(self) -> tuple[str, int] | None:
        """Return (host, port) if running."""
        return self._server.server_address if self._server else None
```

### LoopGroup Integration

```python
class LoopGroup:
    def __init__(
        self,
        loops: Sequence[Runnable],
        *,
        shutdown_timeout: float = 30.0,
        health_port: int | None = None,
    ) -> None:
        self.loops = loops
        self.shutdown_timeout = shutdown_timeout
        self._health_port = health_port
        self._health_server: HealthServer | None = None
        # ... existing attributes

    def run(self, ...) -> None:
        # Start health server if configured
        if self._health_port is not None:
            self._health_server = HealthServer(
                port=self._health_port,
                readiness_check=lambda: all(loop.running for loop in self.loops),
            )
            self._health_server.start()

        try:
            # ... existing run logic
        finally:
            if self._health_server is not None:
                self._health_server.stop()
```

## Usage

```python
group = LoopGroup(loops=[main_loop], health_port=8080)
group.run()
```

## Kubernetes

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
```

## Testing

```python
def test_health_server():
    server = HealthServer(port=0)  # OS assigns port
    server.start()

    host, port = server.address
    resp = urllib.request.urlopen(f"http://{host}:{port}/health/live")
    assert resp.status == 200

    server.stop()
```
