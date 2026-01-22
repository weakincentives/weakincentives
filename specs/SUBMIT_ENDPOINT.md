# Submit Endpoint Specification

## Purpose

HTTP endpoint for submitting `MainLoopRequest` messages to a mailbox without
requiring clients to understand Mailbox internals. Enables load balancer
placement in front of worker pools.

**Promise:** Request durably persisted in mailbox. Nothing more.

**Implementation:** `src/weakincentives/runtime/submit_endpoint.py` (proposed)

## HTTP API

### POST /submit

Submit a MainLoopRequest. Returns 202 after mailbox confirms persistence.

**Request:**

```http
POST /submit HTTP/1.1
Content-Type: application/json
Authorization: Basic dXNlcjpwYXNz

{
  "request": {...},
  "budget": {"max_iterations": 100},
  "deadline": "2024-01-15T12:00:00Z",
  "request_id": "client-provided-id",
  "experiment": "exp-001"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request` | `object` | Yes | Application-specific payload |
| `budget` | `BudgetSpec` | No | Override default budget |
| `deadline` | `string` | No | ISO 8601 deadline |
| `request_id` | `string` | No | Client ID (generated if omitted) |
| `experiment` | `string` | No | Experiment identifier |

**Response:**

```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{"request_id": "req_abc123", "status": "accepted"}
```

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `string` | Request identifier for correlation |
| `status` | `string` | `"accepted"` on success |
| `error` | `string \| null` | Error message on failure |

### Error Responses

| Status | Condition |
|--------|-----------|
| 400 | Invalid JSON or missing `request` field |
| 401 | Missing or invalid Authorization header |
| 413 | Request body exceeds `request_body_limit` |
| 503 | Mailbox unavailable or full |

## Configuration

```python
SubmitEndpoint(
    requests=mailbox,
    config=SubmitEndpointConfig(
        host="0.0.0.0",           # Bind address
        port=8081,                # Listen port
        auth=BasicAuthConfig(...),# Optional
        request_body_limit=1048576,
    ),
)
```

### BasicAuthConfig

Standard HTTP Basic Authentication (RFC 7617). HTTPS required in production.

```python
BasicAuthConfig(
    username="service",
    password=os.environ["SUBMIT_PASSWORD"],
    realm="weakincentives",  # For WWW-Authenticate header
)
```

## Limitations

- **Fire-and-forget**: No result tracking; clients collect results separately
- **No cancellation**: Once submitted, requests cannot be cancelled
- **No batching**: One request per HTTP call
- **Single mailbox**: One endpoint routes to one mailbox

## Related Specifications

- `specs/MAIN_LOOP.md` - Request processing
- `specs/MAILBOX.md` - Queue protocol and delivery guarantees
