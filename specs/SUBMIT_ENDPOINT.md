# Submit Endpoint Specification

## Purpose

`SubmitEndpoint` provides an HTTP interface for submitting `MainLoopRequest`
messages to a worker pool without requiring clients to understand Mailbox
internals. Enables load balancer placement in front of workers for horizontal
scaling and simplified client integration.

**Promise:** Request durably persisted in mailbox. Nothing more.

**Implementation:** `src/weakincentives/runtime/submit_endpoint.py` (proposed)

## Principles

- **HTTP-native**: Standard REST semantics; clients use any HTTP library
- **Mailbox-agnostic**: Clients submit requests; endpoint handles queue routing
- **Optional auth**: Basic authentication for multi-tenant and untrusted networks
- **Fire-and-forget**: Confirm durable persistence; no result tracking
- **Load-balancer friendly**: Stateless design; any instance can accept requests
- **Minimal surface**: Single endpoint; complexity lives in MainLoop workers

## Core Components

### SubmitEndpoint

HTTP server accepting MainLoop requests and routing to configured mailbox.

| Method | Description |
|--------|-------------|
| `start()` | Start HTTP server in daemon thread |
| `stop()` | Graceful shutdown with drain period |
| `address` | `(host, port)` tuple if running |

### SubmitEndpointConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | `str` | `"0.0.0.0"` | Bind address |
| `port` | `int` | `8081` | Listen port |
| `auth` | `BasicAuthConfig \| None` | `None` | Optional basic auth |
| `request_body_limit` | `int` | `1048576` | Max request body (1MB) |

### BasicAuthConfig

| Field | Type | Description |
|-------|------|-------------|
| `username` | `str` | Required username |
| `password` | `str` | Required password (use secrets in production) |
| `realm` | `str` | HTTP realm for 401 response (default: "weakincentives") |

### SubmitRequest

Client-facing request schema (JSON):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request` | `object` | Yes | Application-specific request payload |
| `budget` | `BudgetSpec` | No | Override default budget |
| `deadline` | `string` | No | ISO 8601 deadline |
| `request_id` | `string` | No | Client-provided ID (generated if omitted) |
| `resources` | `object` | No | Resource overrides |
| `experiment` | `string` | No | Experiment identifier |

### SubmitResponse

Response schema (JSON):

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `string` | Request identifier for correlation |
| `status` | `string` | `"accepted"` on success |
| `error` | `string \| null` | Error message if submission failed |

## HTTP API

### POST /submit

Submit a MainLoopRequest to the worker pool.

**Request:**

```http
POST /submit HTTP/1.1
Host: submit.example.com
Content-Type: application/json
Authorization: Basic dXNlcjpwYXNz

{
  "request": {
    "task": "analyze",
    "input": "..."
  },
  "deadline": "2024-01-15T12:00:00Z"
}
```

**Response (success):**

```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{
  "request_id": "req_abc123",
  "status": "accepted"
}
```

202 confirms the request is durably persisted in the mailbox. The client can
trust the request will be processed (subject to mailbox delivery guarantees).

### Error Responses

| Status | Condition |
|--------|-----------|
| 400 | Invalid JSON or missing required fields |
| 401 | Missing or invalid Authorization header |
| 413 | Request body exceeds limit |
| 429 | Rate limit exceeded (if configured) |
| 503 | Mailbox unavailable or queue full |

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│ Load        │────▶│  Submit     │
│             │     │ Balancer    │     │  Endpoint   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Mailbox   │
                                        │  (Redis)    │
                                        └──────┬──────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
             ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
             │  MainLoop   │            │  MainLoop   │            │  MainLoop   │
             │  Worker 1   │            │  Worker 2   │            │  Worker N   │
             └─────────────┘            └─────────────┘            └─────────────┘
```

### Request Flow

1. Client POSTs to any SubmitEndpoint instance via load balancer
2. SubmitEndpoint validates request and auth
3. SubmitEndpoint constructs `MainLoopRequest` and sends to mailbox
4. Mailbox confirms durable persistence
5. Return 202 Accepted with request_id

The endpoint returns only after the mailbox confirms persistence. If the
mailbox write fails, the client receives an error and can retry.

## Authentication

### Basic Auth

Standard HTTP Basic Authentication (RFC 7617).

```python
config = SubmitEndpointConfig(
    auth=BasicAuthConfig(
        username="service",
        password=os.environ["SUBMIT_PASSWORD"],
    )
)
```

**Request:**

```http
Authorization: Basic c2VydmljZTpzZWNyZXQ=
```

### Security Considerations

- **HTTPS required**: Basic auth transmits credentials base64-encoded (not encrypted)
- **Secrets management**: Use environment variables or secrets manager for passwords
- **Rate limiting**: Configure per-client rate limits to prevent abuse
- **Request validation**: Validate and sanitize all input fields
- **Audit logging**: Log all submissions with client identity (not credentials)

### Future: Token Auth

For multi-tenant scenarios, token-based auth with scopes:

```python
auth=TokenAuthConfig(
    issuer="https://auth.example.com",
    audience="weakincentives",
    scopes=["submit:write"],
)
```

## Integration with LoopGroup

SubmitEndpoint runs alongside LoopGroup workers:

```python
# Worker process - runs MainLoop and SubmitEndpoint
submit = SubmitEndpoint(
    requests=requests_mailbox,
    config=SubmitEndpointConfig(port=8081, auth=auth_config),
)

group = LoopGroup(
    loops=[main_loop],
    health_port=8080,
)

submit.start()
try:
    group.run()
finally:
    submit.stop()
```

### Deployment Options

**Option A: Co-located (simple)**

Each worker runs both SubmitEndpoint and MainLoop. Load balancer routes to any
instance.

```
Worker Pod:
├── MainLoop (processes requests)
└── SubmitEndpoint :8081 (accepts submissions)
```

**Option B: Separated (scalable)**

Dedicated SubmitEndpoint instances, separate from workers. Enables independent
scaling of submission and processing capacity.

```
Submit Pod:                    Worker Pod:
└── SubmitEndpoint :8081       └── MainLoop
         │                              │
         └──────── Mailbox ─────────────┘
```

## Usage Examples

### Minimal Setup

```python
from weakincentives.runtime import SubmitEndpoint, SubmitEndpointConfig

endpoint = SubmitEndpoint(
    requests=requests_mailbox,
    config=SubmitEndpointConfig(port=8081),
)
endpoint.start()
```

### With Authentication

```python
from weakincentives.runtime import BasicAuthConfig

endpoint = SubmitEndpoint(
    requests=requests_mailbox,
    config=SubmitEndpointConfig(
        port=8081,
        auth=BasicAuthConfig(
            username="api",
            password=os.environ["API_PASSWORD"],
        ),
    ),
)
```

### Client Usage (Python)

```python
import httpx

response = httpx.post(
    "https://submit.example.com/submit",
    auth=("api", "secret"),
    json={
        "request": {"task": "analyze", "input": data},
        "deadline": "2024-01-15T12:00:00Z",
    },
)
response.raise_for_status()  # 202 = persisted
result = response.json()
print(f"Request ID: {result['request_id']}")
```

### Client Usage (curl)

```bash
curl -X POST https://submit.example.com/submit \
  -u api:secret \
  -H "Content-Type: application/json" \
  -d '{"request": {"task": "analyze", "input": "..."}}'
```

## Request ID

### Generation

If client omits `request_id`, endpoint generates UUID v4:

```python
request_id = request.get("request_id") or str(uuid4())
```

### Correlation

Request ID propagates through the system for debugging:

1. `SubmitResponse.request_id` - Returned to client
2. `MainLoopRequest.request_id` - In mailbox message
3. `MainLoopResult.request_id` - In worker response
4. Structured logs - For tracing

Clients needing results must implement their own collection mechanism (e.g.,
configure MainLoop workers to write results to a database or separate queue).

## Error Handling

| Error | HTTP Status | Action |
|-------|-------------|--------|
| Invalid JSON | 400 | Return error message |
| Missing `request` field | 400 | Return field-level error |
| Auth failure | 401 | Return WWW-Authenticate header |
| Body too large | 413 | Return limit in error |
| Mailbox full | 503 | Return Retry-After header |
| Mailbox unavailable | 503 | Return Retry-After header |
| Serialization failure | 400 | Return field-level errors |

## Observability

### Structured Logging

```python
logger.info(
    "request_submitted",
    request_id=request_id,
    client=client_id,  # From auth
)
```

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `wink_submit_requests_total` | Counter | Total submissions |
| `wink_submit_latency_seconds` | Histogram | Submission latency |
| `wink_submit_errors_total` | Counter | Submission errors by type |

### Health Integration

SubmitEndpoint exposes health via existing HealthServer:

- `/health/live` - Endpoint thread alive
- `/health/ready` - Mailbox connection healthy

## Limitations

- **Fire-and-forget only**: No result tracking; clients collect results separately
- **No request modification**: Once submitted, requests cannot be cancelled
- **No batching**: One request per HTTP call; batch at client level
- **Basic auth only**: Token auth planned but not implemented
- **Single mailbox**: One SubmitEndpoint routes to one request mailbox

## Related Specifications

- `specs/MAIN_LOOP.md` - Request processing
- `specs/MAILBOX.md` - Queue protocol
- `specs/HEALTH.md` - Health endpoints
- `specs/LIFECYCLE.md` - LoopGroup coordination
