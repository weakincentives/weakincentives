# Evals Specification

## Purpose

Minimal evaluation framework built on MainLoop. MainLoop handles orchestration;
this spec adds datasets and scoring.

**Implementation:** `src/weakincentives/evals/`

## Guiding Principles

- **EvalLoop communicates via Mailbox** - Decoupled from MainLoop via message passing
- **Evaluators are functions** - Pure `(output, expected) -> Score`
- **Datasets are immutable** - Frozen dataclass with typed samples

## Core Types

### Sample

Single evaluation case: `id`, `input`, `expected`. Generic parameters enable
typed datasets.

**Implementation:** `src/weakincentives/evals/_types.py`

### Dataset

Immutable collection of samples with JSONL loading support.

| Method | Description |
|--------|-------------|
| `__len__()` | Sample count |
| `__iter__()` | Iterate samples |
| `__getitem__(i)` | Get by index |
| `load(path, input_type, expected_type)` | Load from JSONL |

**JSONL format:**

```jsonl
{"id": "1", "input": "What is 2+2?", "expected": "4"}
```

### Score

| Field | Type | Description |
|-------|------|-------------|
| `value` | `float` | 0.0-1.0 normalized |
| `passed` | `bool` | Binary pass/fail |
| `reason` | `str` | Explanation |

### Evaluator Types

| Type | Signature |
|------|-----------|
| `Evaluator` | `(output, expected) -> Score` |
| `SessionEvaluator` | `(output, expected, SessionView) -> Score` |

## Built-in Evaluators

**Implementation:** `src/weakincentives/evals/_evaluators.py`

| Evaluator | Purpose |
|-----------|---------|
| `exact_match` | Strict equality |
| `contains` | Substring presence |
| `all_of(*evals)` | All must pass (score=mean) |
| `any_of(*evals)` | One must pass (score=max) |

## Session-Aware Evaluators

Enable behavioral assertions on tool usage, token budgets, custom state.

**Implementation:** `src/weakincentives/evals/_session_evaluators.py`

### SessionView Protocol

Read-only session access: `session_id`, `__getitem__(slice_type) -> SliceView[T]`

### Built-in Session Evaluators

| Evaluator | Purpose |
|-----------|---------|
| `tool_called(name)` | Assert tool invoked |
| `tool_not_called(name)` | Assert tool NOT invoked |
| `tool_call_count(name, min, max)` | Assert call count |
| `all_tools_succeeded()` | No tool failures |
| `token_usage_under(max)` | Total tokens under budget |
| `slice_contains(type, pred, min)` | Custom slice assertion |

### Adapting Evaluators

`adapt(evaluator)` converts standard evaluators to session-aware signature.

## LLM-as-Judge

**Implementation:** `src/weakincentives/evals/_judge.py`

| Rating | Value | Passes |
|--------|-------|--------|
| `excellent` | 1.0 | Yes |
| `good` | 0.75 | Yes |
| `fair` | 0.5 | No |
| `poor` | 0.25 | No |
| `wrong` | 0.0 | No |

Factory: `llm_judge(adapter, criterion)` returns `Evaluator[str, str]`

## Running Evals

### EvalResult

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | `str` | Sample identifier |
| `score` | `Score` | Evaluation score |
| `latency_ms` | `int` | Processing time |
| `error` | `str \| None` | Error message |

### EvalReport

| Property | Description |
|----------|-------------|
| `total` | Sample count |
| `successful` | Completed without error |
| `pass_rate` | Fraction passed |
| `mean_score` | Average score |
| `mean_latency_ms` | Average latency |
| `failed_samples()` | Non-passing samples |

### EvalLoop

**Implementation:** `src/weakincentives/evals/_loop.py`

Mailbox-driven evaluation loop. Receives `EvalRequest`, executes through
MainLoop, scores with evaluator, sends `EvalResult` to results mailbox.

### Helper Functions

**Implementation:** `src/weakincentives/evals/_helpers.py`

| Function | Description |
|----------|-------------|
| `submit_dataset(dataset, experiment, requests)` | Submit all samples |
| `collect_results(results, expected_count, *, timeout_seconds)` | Collect into report |

## Distributed Deployment

EvalLoop supports distributed evaluation using Redis/SQS mailboxes.

### Mailbox-Based Architecture

EvalLoop communicates with MainLoop workers through a shared `mainloop_requests`
mailbox rather than holding a direct MainLoop reference. This enables:

- **Horizontal scaling**: Multiple MainLoop workers process requests in parallel
- **Process isolation**: EvalLoop and MainLoop can run in separate processes
- **Cluster deployment**: Workers distributed across machines via Redis mailbox

```
┌─────────────┐     EvalRequest      ┌─────────────────────┐
│             │──────────────────────▶│                     │
│   Client    │                       │  eval_requests      │
│             │◀──────────────────────│  (mailbox)          │
└─────────────┘     EvalResult        └──────────┬──────────┘
                                                 │
                                                 ▼
                                      ┌──────────────────────┐
                                      │    EvalLoop Worker   │
                                      │                      │
                                      │  1. Receive request  │
                                      │  2. Build MainLoop-  │
                                      │     Request          │
                                      │  3. Send to main-    │
                                      │     loop_requests    │
                                      │  4. Wait for result  │
                                      │  5. Score output     │
                                      │  6. Reply with       │
                                      │     EvalResult       │
                                      └──────────┬───────────┘
                                                 │
                                                 │ MainLoopRequest
                                                 ▼
                                      ┌──────────────────────┐
                                      │  mainloop_requests   │
                                      │  (mailbox)           │
                                      └──────────┬───────────┘
                 ┌───────────────────────────────┼───────────────────────────────┐
                 ▼                               ▼                               ▼
      ┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
      │ MainLoop Worker 1│           │ MainLoop Worker 2│           │ MainLoop Worker N│
      │                  │           │                  │           │                  │
      │ Execute prompt   │           │ Execute prompt   │           │ Execute prompt   │
      │ Reply with       │           │ Reply with       │           │ Reply with       │
      │ MainLoopResult   │           │ MainLoopResult   │           │ MainLoopResult   │
      └──────────────────┘           └──────────────────┘           └──────────────────┘
```

### Configuration

```python
# EvalLoop no longer requires a MainLoop instance
eval_loop = EvalLoop(
    evaluator=exact_match,
    requests=eval_requests_mailbox,
    mainloop_requests=mainloop_requests_mailbox,  # Target for MainLoopRequests
    config=EvalLoopConfig(
        mainloop_timeout=300.0,  # Max wait for MainLoop response
    ),
)
```

### Deployment Modes

**Worker Process:**

- Runs `EvalLoop.run()` indefinitely
- Polls with long polling (20s)
- Sends MainLoopRequest, waits for result, scores, acks/nacks

**Client Process:**

- Submits via `submit_dataset()`
- Collects via `collect_results()`
- Aggregates into EvalReport

**Scaling:**

- Multiple EvalLoop workers for horizontal eval scaling
- Multiple MainLoop workers for horizontal execution scaling
- Visibility timeout ensures at-least-once processing
- Failed evaluations retry with backoff

## Clustered Deployment Considerations

### Reply Mailbox Routing

Each EvalLoop worker creates a unique reply mailbox to receive `MainLoopResult`
messages. This mailbox name is included in the `reply_to` field of the
`MainLoopRequest`.

**In-process (InMemoryMailbox):**
- Reply routing uses direct object references
- No resolver configuration needed

**Distributed (RedisMailbox):**
- Reply mailbox names serialized as strings
- `MailboxResolver` reconstructs mailbox on MainLoop worker
- Use `CompositeResolver` with `RedisMailboxFactory` fallback

```python
# MainLoop worker configuration
resolver = CompositeResolver(
    registry={},
    factory=RedisMailboxFactory(client=redis, body_type=MainLoopResult),
)
mainloop = MyMainLoop(
    adapter=adapter,
    requests=RedisMailbox(
        name="mainloop-requests",
        client=redis,
        reply_resolver=resolver,
    ),
)
```

### Session Access for Session-Aware Evaluators

Session-aware evaluators require access to the session after MainLoop execution.
Two strategies:

**Strategy 1: Session Snapshot in Result**

MainLoop includes serialized session snapshot in `MainLoopResult.session_snapshot`.
EvalLoop deserializes for evaluator access.

```python
@dataclass(frozen=True)
class MainLoopResult[T]:
    # ... existing fields ...
    session_snapshot: bytes | None = None  # Serialized session state
```

Trade-offs:
- (+) Simple - session travels with result
- (-) Payload size increases with session size
- (-) Requires session serialization

**Strategy 2: Shared Session Store**

Sessions persisted to shared storage (Redis, database). MainLoop writes session
after execution; EvalLoop reads by session_id.

```python
# MainLoop writes after execution
session_store.save(session_id, session)

# EvalLoop reads for evaluator
session = session_store.load(result.session_id)
score = evaluator(output, expected, session)
```

Trade-offs:
- (+) Constant result payload size
- (+) Sessions available for debugging/replay
- (-) Additional infrastructure dependency
- (-) Eventual consistency concerns

### Heartbeat and Lease Extension

With mailbox-based communication, heartbeat propagation changes:

**Current (direct call):**
- EvalLoop passes `Heartbeat` to `MainLoop.execute()`
- Tool execution beats extend EvalLoop's message lease

**Mailbox-based:**
- EvalLoop cannot share heartbeat with remote MainLoop worker
- Each layer manages its own lease extension independently

```
EvalLoop                          MainLoop Worker
    │                                   │
    │ LeaseExtender on EvalRequest      │ LeaseExtender on MainLoopRequest
    │ extends during wait               │ extends during execution
    ▼                                   ▼
```

Configure visibility timeouts to accommodate:
- EvalLoop: `mainloop_timeout + scoring_time + buffer`
- MainLoop: `max_execution_time + buffer`

### Error Attribution

In clustered deployment, errors can occur at multiple layers:

| Error Source | Detection | Handling |
|--------------|-----------|----------|
| MainLoop timeout | No response within `mainloop_timeout` | Retry or DLQ |
| MainLoop execution failure | `MainLoopResult.error` populated | Score as failure |
| Network partition | `MailboxConnectionError` | Nack with backoff |
| Reply mailbox unreachable | `ReplyNotAvailableError` at MainLoop | MainLoop logs, EvalLoop times out |

### Ordering Guarantees

Mailbox-based architecture does **not** guarantee ordering:

- Multiple MainLoop workers may process requests concurrently
- Results may arrive out of order
- `EvalResult.sample_id` correlates results to samples

If ordering matters, use single MainLoop worker or add sequence numbers.

## Limitations

- **No caching**: Repeated samples re-execute
- **No checkpoints**: Cannot resume interrupted runs
- **Session-aware evaluators in cluster**: Require session snapshot or shared store
- **No ordering guarantees**: Results may arrive out of sample order

## Related Specifications

- `specs/DLQ.md` - Dead letter queue for failed samples
- `specs/MAIN_LOOP.md` - MainLoop orchestration
- `specs/MAILBOX.md` - Mailbox protocol
- `specs/LIFECYCLE.md` - LoopGroup coordination
- `specs/HEALTH.md` - Health checks and watchdog
