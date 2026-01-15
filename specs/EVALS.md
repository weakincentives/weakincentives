# Evals Specification

## Purpose

Minimal evaluation framework built on MainLoop. MainLoop handles orchestration;
this spec adds datasets and scoring.

**Implementation:** `src/weakincentives/evals/`

## Guiding Principles

- **EvalLoop wraps MainLoop** - Composition-based, event-driven, type-safe
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
| `submit_dataset(dataset, requests)` | Submit all samples |
| `collect_results(results, count, timeout)` | Collect into report |

## Distributed Deployment

EvalLoop supports distributed evaluation using Redis/SQS mailboxes.

### Deployment Modes

**Worker Process:**
- Runs `EvalLoop.run()` indefinitely
- Polls with long polling (20s)
- Executes, scores, acks/nacks

**Client Process:**
- Submits via `submit_dataset()`
- Collects via `collect_results()`
- Aggregates into EvalReport

**Scaling:**
- Multiple workers for horizontal scaling
- Visibility timeout ensures exactly-once
- Failed evaluations retry with backoff

## Limitations

- **Sequential execution**: MainLoop is synchronous
- **No caching**: Repeated samples re-execute
- **No checkpoints**: Cannot resume interrupted runs
- **Single loop**: One MainLoop per EvalLoop

## Related Specifications

- `specs/DLQ.md` - Dead letter queue for failed samples
- `specs/MAIN_LOOP.md` - MainLoop orchestration
- `specs/MAILBOX.md` - Mailbox protocol
- `specs/LIFECYCLE.md` - LoopGroup coordination
- `specs/HEALTH.md` - Health checks and watchdog
