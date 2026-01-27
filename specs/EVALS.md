# Evals Specification

## Purpose

Minimal evaluation framework built on AgentLoop. AgentLoop handles orchestration;
this spec adds datasets and scoring.

**Implementation:** `src/weakincentives/evals/`

## Guiding Principles

- **EvalLoop wraps AgentLoop** - Composition-based, event-driven, type-safe
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

### EvalRequest

Request to evaluate a sample under an experiment.

| Field | Type | Description |
|-------|------|-------------|
| `sample` | `Sample[InputT, ExpectedT]` | The sample to evaluate |
| `experiment` | `Experiment` | The experiment to evaluate under (required) |
| `request_id` | `UUID` | Unique request identifier (auto-generated) |
| `created_at` | `datetime` | When the request was created |

### Evaluator Types

| Type | Signature |
|------|-----------|
| `Evaluator` | `(output, expected) -> Score` |
| `SessionEvaluator` | `(output, expected, session) -> Score` |

Session-aware evaluators receive `SessionProtocol | SessionViewProtocol` as the session parameter.

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

### Session Access

Session-aware evaluators receive `SessionProtocol | SessionViewProtocol` which provides
read-only session access via `__getitem__(slice_type)` to query slices.

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

**Implementation:** `src/weakincentives/evals/_evaluators.py`

| Function | Description |
|----------|-------------|
| `adapt(evaluator)` | Convert standard evaluator to session-aware signature |
| `is_session_aware(fn)` | Check if evaluator accepts session parameter |

## Experiments

**Implementation:** `src/weakincentives/experiment.py`

Experiments enable A/B testing by bundling prompt overrides and feature flags.
The evals module re-exports `Experiment`, `BASELINE`, and `CONTROL` for convenience.

| Type | Description |
|------|-------------|
| `Experiment` | Named configuration variant with overrides tag and flags |
| `BASELINE` | Sentinel experiment with no overrides (alias: latest) |
| `CONTROL` | Explicit control group experiment (same as BASELINE) |

See `specs/EXPERIMENTS.md` for full experiment specification.

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
| `experiment_name` | `str` | Experiment name |
| `score` | `Score` | Evaluation score |
| `latency_ms` | `int` | Processing time |
| `error` | `str \| None` | Error message |
| `bundle_path` | `Path \| None` | Debug bundle path (if enabled) |

### ExperimentComparison

Statistical comparison between two experiments.

| Property | Description |
|----------|-------------|
| `baseline_name` | Name of the baseline experiment |
| `treatment_name` | Name of the treatment experiment |
| `baseline_pass_rate` | Pass rate for baseline |
| `treatment_pass_rate` | Pass rate for treatment |
| `pass_rate_delta` | Treatment minus baseline pass rate |
| `relative_improvement` | Percentage improvement over baseline (None if baseline is 0) |

### EvalReport

| Property | Description |
|----------|-------------|
| `total` | Sample count |
| `successful` | Completed without error |
| `pass_rate` | Fraction passed |
| `mean_score` | Average score |
| `mean_latency_ms` | Average latency |
| `failed_samples()` | Non-passing samples |
| `by_experiment()` | Group results by experiment name |
| `pass_rate_by_experiment()` | Pass rate per experiment |
| `mean_score_by_experiment()` | Mean score per experiment |
| `compare_experiments(baseline, treatment)` | Compare two experiments statistically |

### EvalLoopConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lease_extender` | `LeaseExtenderConfig \| None` | `None` | Message visibility extension |
| `debug_bundle_dir` | `Path \| None` | `None` | Debug bundle output directory |

When `debug_bundle_dir` is set, EvalLoop creates debug bundles for each
evaluation sample containing session state, logs, and eval metadata. See
`specs/DEBUG_BUNDLE.md` for bundle format details.

### EvalLoop

**Implementation:** `src/weakincentives/evals/_loop.py`

Mailbox-driven evaluation loop. Receives `EvalRequest`, executes through
AgentLoop, scores with evaluator, sends `EvalResult` to results mailbox.

### Helper Functions

**Implementation:** `src/weakincentives/evals/_helpers.py`

| Function | Description |
|----------|-------------|
| `submit_dataset(dataset, experiment, requests)` | Submit all samples under one experiment |
| `submit_experiments(dataset, experiments, requests)` | Submit dataset under multiple experiments for A/B testing |
| `collect_results(results, expected_count, *, timeout_seconds)` | Collect into report |

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

- **Sequential execution**: AgentLoop is synchronous
- **No caching**: Repeated samples re-execute
- **No checkpoints**: Cannot resume interrupted runs
- **Single loop**: One AgentLoop per EvalLoop

## Related Specifications

- `specs/DEBUG_BUNDLE.md` - Debug bundle format and EvalLoop integration
- `specs/DLQ.md` - Dead letter queue for failed samples
- `specs/EXPERIMENTS.md` - Experiment configuration for A/B testing
- `specs/AGENT_LOOP.md` - AgentLoop orchestration
- `specs/MAILBOX.md` - Mailbox protocol
- `specs/LIFECYCLE.md` - LoopGroup coordination
- `specs/HEALTH.md` - Health checks and watchdog
