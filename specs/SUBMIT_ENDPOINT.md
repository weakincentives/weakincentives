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
  "reply_to": "client-results-abc123",
  "experiment": {
    "name": "v2-concise-prompts",
    "overrides_tag": "v2",
    "flags": {"verbose_logging": true}
  },
  "debug_bundle": {
    "target": "/var/bundles",
    "max_file_size": 10000000,
    "compression": "deflate"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request` | `object` | Yes | Application-specific payload |
| `budget` | `BudgetSpec` | No | Override default budget |
| `deadline` | `string` | No | ISO 8601 deadline |
| `request_id` | `string` | No | Client ID (generated if omitted) |
| `reply_to` | `string` | No | Mailbox identifier for result delivery |
| `experiment` | `Experiment` | No | Experiment config (see below) |
| `debug_bundle` | `BundleConfig` | No | Debug bundle config (see below) |

#### Experiment

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | Yes | Unique experiment identifier |
| `overrides_tag` | `string` | No | Prompt overrides tag (default: "latest") |
| `flags` | `object` | No | Feature flags (string keys, primitive values) |
| `owner` | `string` | No | Owner identifier |
| `description` | `string` | No | Human-readable description |

#### BundleConfig

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `target` | `string` | No | Output directory for bundles (null disables) |
| `max_file_size` | `int` | No | Skip files larger than this (default: 10MB) |
| `max_total_size` | `int` | No | Max filesystem capture size (default: 50MB) |
| `compression` | `string` | No | Zip compression: "deflate", "stored", "bzip2", "lzma" |

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

## Reply Delivery

When `reply_to` is provided, MainLoop workers send `MainLoopResult` to the
specified mailbox after processing. The identifier is resolved via
`MailboxResolver` configured on the worker.

**Client responsibilities:**

1. Create a mailbox accessible to workers (e.g., `RedisMailbox`)
2. Provide mailbox name as `reply_to`
3. Poll the mailbox for results, keyed by `request_id`

**Without `reply_to`:** Fire-and-forget. Results must be collected via
side-channel (e.g., workers write to database).

**Note:** This requires workers and clients share a `MailboxResolver` that can
resolve the `reply_to` identifier. See `specs/MAILBOX.md` for resolver patterns.

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

- **No idempotency**: Duplicate `request_id` submissions create duplicate messages
- **No cancellation**: Once submitted, requests cannot be cancelled
- **No batching**: One request per HTTP call
- **Single mailbox**: One endpoint routes to one request mailbox
- **Resolver required**: `reply_to` requires shared `MailboxResolver` between endpoint and workers

## Future Work

- **Idempotency**: Deduplicate by `request_id` with configurable window

## Related Specifications

- `specs/MAIN_LOOP.md` - Request processing
- `specs/MAILBOX.md` - Queue protocol, delivery guarantees, and resolver patterns
- `specs/DEBUG_BUNDLE.md` - Debug bundle capture
