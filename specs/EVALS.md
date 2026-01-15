# Evaluation Framework Specification

Minimal eval framework built on MainLoop: datasets, evaluators, and scoring.

**Source:** `src/weakincentives/evals/`

## Architecture

```
Dataset (samples) → EvalLoop → MainLoop → Adapter
                       │
                       ▼
                   Evaluator → EvalReport
```

## Core Types

### Sample / Dataset

**Definition:** `evals/_types.py`

```python
Sample[InputT, ExpectedT](id: str, input: InputT, expected: ExpectedT)
Dataset[InputT, ExpectedT](samples: tuple[Sample, ...])
```

Load from JSONL: `Dataset.load(path, input_type, expected_type)`

### Score

```python
Score(value: float, passed: bool, reason: str = "")
```

### Evaluator

```python
Evaluator = Callable[[OutputT, ExpectedT], Score]
SessionEvaluator = Callable[[OutputT, ExpectedT, SessionView], Score]
```

## Built-in Evaluators

**Definition:** `evals/_evaluators.py`

| Evaluator | Purpose |
|-----------|---------|
| `exact_match` | Strict equality |
| `contains` | Substring presence |
| `all_of(*evaluators)` | All must pass, mean score |
| `any_of(*evaluators)` | One must pass, max score |

## Session-Aware Evaluators

**Definition:** `evals/_session_evaluators.py`

| Evaluator | Purpose |
|-----------|---------|
| `tool_called(name)` | Tool was invoked |
| `tool_not_called(name)` | Tool was NOT invoked |
| `all_tools_succeeded()` | No tool failures |
| `token_usage_under(max)` | Token budget check |
| `slice_contains(type, pred)` | Custom slice assertion |

## LLM-as-Judge

**Definition:** `evals/_judge.py`

```python
llm_judge(adapter, criterion)  # Returns Evaluator[str, str]
```

Rating scale: excellent (1.0), good (0.75), fair (0.5), poor (0.25), wrong (0.0)

## Results

```python
EvalResult(sample_id, score, latency_ms, error)
EvalReport(results)
```

Report provides: `pass_rate`, `mean_score`, `mean_latency_ms`, `failed_samples()`

## EvalLoop

**Definition:** `evals/_loop.py`

```python
EvalLoop(
    loop: MainLoop,
    evaluator: Evaluator | SessionEvaluator,
    requests: Mailbox[EvalRequest],
    results: Mailbox[EvalResult],
)
```

Distributed deployment: submit via `submit_dataset()`, collect via `collect_results()`.

## Limitations

- Sequential execution (MainLoop is synchronous)
- No caching or checkpoints
- Single loop instance per EvalLoop
