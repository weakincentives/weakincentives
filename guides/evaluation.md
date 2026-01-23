# Evaluation with EvalLoop

*Canonical spec: [specs/EVALS.md](../specs/EVALS.md)*

Evaluation is built on the same composition pattern as everything else in WINK:
**EvalLoop wraps MainLoop**. Rather than a separate evaluation framework, evals
are just another way to drive your existing MainLoop with datasets and scoring.

This means a worker in production can run both your regular agent logic
(`MainLoop`) and your evaluation suite (`EvalLoop`) side by side—same prompt
templates, same tools, same adapters. Canary deployments become natural.

## The Composition Philosophy

```
┌─────────────┐     ┌──────────┐     ┌───────────┐     ┌────────────┐
│   Dataset   │────▶│ EvalLoop │────▶│ MainLoop  │────▶│  Adapter   │
│  (samples)  │     │.execute()│     │ .execute()│     │            │
└─────────────┘     └────┬─────┘     └───────────┘     └────────────┘
                         │
                         ▼
                   ┌───────────┐     ┌────────────┐
                   │ Evaluator │────▶│ EvalReport │
                   │ (scoring) │     │ (metrics)  │
                   └───────────┘     └────────────┘
```

`EvalLoop` orchestrates evaluation: for each sample, it executes through the
provided `MainLoop`, scores the output with an evaluator function, and
aggregates results into a report.

## Core Types

**Sample and Dataset:**

```python nocheck
from weakincentives.evals import Dataset, Sample

# A sample pairs input with expected output
sample = Sample(
    id="math-1",
    input="What is 2 + 2?",
    expected="4",
)

# Datasets are immutable collections of samples
dataset = Dataset(samples=(sample,))

# Or load from JSONL
dataset = Dataset.load(Path("qa.jsonl"), str, str)
```

**Score and Evaluator:**

Evaluators are pure functions—no side effects, no state:

```python nocheck
from weakincentives.evals import Score

def my_evaluator(output: str, expected: str) -> Score:
    passed = expected.lower() in output.lower()
    return Score(
        value=1.0 if passed else 0.0,
        passed=passed,
        reason="Found expected answer" if passed else "Missing expected answer",
    )
```

**Built-in evaluators:**

```text
from weakincentives.evals import exact_match, contains, all_of, any_of

# Strict equality
score = exact_match("hello", "hello")  # passed=True

# Substring presence
score = contains("The answer is 42.", "42")  # passed=True

# Combine evaluators
evaluator = all_of(contains, my_custom_check)  # All must pass
evaluator = any_of(exact_match, fuzzy_match)   # At least one must pass
```

## LLM-as-Judge

For subjective criteria, use an LLM to score outputs:

```text
from weakincentives.evals import llm_judge, all_of
from weakincentives.adapters.openai import OpenAIAdapter

judge_adapter = OpenAIAdapter(model="gpt-4o-mini")

evaluator = all_of(
    contains,
    llm_judge(judge_adapter, "Response is helpful"),
    llm_judge(judge_adapter, "No hallucinated info"),
)
```

The `llm_judge` factory creates an evaluator that prompts the model to rate
outputs as "excellent", "good", "fair", "poor", or "wrong"—each mapping to a
numeric value.

## Session Evaluators

Sometimes you need to evaluate not just *what* the agent produced, but *how* it
got there. Session evaluators receive a read-only `SessionView` and can assert
on tool usage patterns, token budgets, and custom state invariants.

**Built-in session evaluators:**

```python nocheck
from weakincentives.evals import (
    tool_called,
    tool_not_called,
    tool_call_count,
    all_tools_succeeded,
    token_usage_under,
    slice_contains,
    all_of,
    adapt,
)

# Combine output evaluation with behavioral assertions
evaluator = all_of(
    exact_match,                           # Output must match expected
    tool_called("search"),                 # Agent must have used search
    tool_not_called("fallback"),           # Should not have used fallback
    all_tools_succeeded(),                 # No tool failures
    token_usage_under(max_tokens=5000),    # Stay within budget
)
```

**Converting standard evaluators:**

Standard evaluators (that only see output and expected) can be converted to
session-aware evaluators using `adapt()`:

```python nocheck
from weakincentives.evals import adapt, exact_match, all_of, tool_called

evaluator = all_of(
    adapt(exact_match),    # Now works with session evaluators
    tool_called("search"),
)
```

## Running Evaluations

**EvalLoop wraps your MainLoop:**

```python nocheck
from weakincentives.evals import EvalLoop, EvalRequest, EvalResult, exact_match
from weakincentives.runtime import InMemoryMailbox

# Create mailbox for evaluation requests
eval_requests = InMemoryMailbox(name="eval-requests")

# Create EvalLoop wrapping your MainLoop
eval_loop = EvalLoop(
    loop=main_loop,
    evaluator=exact_match,
    requests=eval_requests,
)
```

**Submit samples and collect results:**

```python nocheck
from weakincentives.evals import submit_dataset, collect_results, Experiment

# Submit all samples to the evaluation mailbox
experiment = Experiment(name="my-eval")
submit_dataset(dataset, experiment=experiment, requests=eval_requests)

# Run the evaluation worker
eval_loop.run(max_iterations=1)

# Collect results into a report
report = collect_results(eval_results, expected_count=len(dataset))

# Inspect the report
print(f"Pass rate: {report.pass_rate:.1%}")
print(f"Mean score: {report.mean_score:.2f}")
print(f"Mean latency: {report.mean_latency_ms:.0f}ms")

# Review failures
for eval_result in report.results:
    if not eval_result.score.passed:
        print(f"Failed: {eval_result.sample_id} - {eval_result.score.reason}")
```

## Production Deployment Pattern

Run both `MainLoop` and `EvalLoop` workers from the same process. This ensures
your evaluation suite runs against the exact same configuration as your
production agent:

```python nocheck
from threading import Thread

# Production worker
def run_production():
    while True:
        for msg in prod_requests.receive():
            response, _session = main_loop.execute(msg.body)
            prod_results.send(response)
            msg.acknowledge()

# Eval worker (wraps the same MainLoop)
eval_loop = EvalLoop(loop=main_loop, evaluator=exact_match, requests=eval_requests)

# Run both in parallel
Thread(target=run_production, daemon=True).start()
eval_loop.run()
```

**Canary deployment:** Before rolling out changes, submit your eval dataset to
the new worker and verify the pass rate meets your threshold.

## Reply-To Routing

When workers need to send results to dynamic destinations, use the `reply_to`
pattern:

```python nocheck
# Client sends request with reply_to mailbox reference
requests.send(
    body=AnalysisRequest(query="Find all bugs"),
    reply_to=client_responses,
)

# Worker processes and replies
for msg in requests.receive():
    result = process(msg.body)
    msg.reply(result)  # Sends directly to client_responses
    msg.acknowledge()
```

This is how evaluation results flow back to the submitter, regardless of which
worker processed each sample.

## Experiments

Experiments enable A/B testing by running the same dataset through different
configurations. Each experiment bundles a name, prompt overrides tag, and
feature flags—allowing you to compare prompt variations, model settings, or
runtime behaviors systematically.

**Defining experiments:**

```python nocheck
from weakincentives.evals import Experiment, BASELINE

# The baseline uses default prompts (overrides_tag="latest")
baseline = BASELINE

# Treatment with different prompt overrides
treatment = Experiment(
    name="v2-concise-prompts",
    overrides_tag="v2",
    owner="alice@example.com",
    description="Test shorter, more direct prompt phrasing",
)

# Variant with feature flags
aggressive = Experiment(
    name="aggressive-tools",
    flags={"max_tool_calls": 10, "parallel_tools": True},
)
```

**Running A/B tests:**

Use `submit_experiments()` to run your dataset under multiple configurations:

```python nocheck
from weakincentives.evals import submit_experiments, collect_results

# Submit dataset under both experiments
count = submit_experiments(
    dataset,
    experiments=[baseline, treatment],
    requests=eval_requests,
)
print(f"Submitted {count} total requests")  # len(dataset) * 2

# Run evaluation worker
eval_loop.run(max_iterations=1)

# Collect all results
report = collect_results(eval_results, expected_count=count)
```

**Analyzing results by experiment:**

```python nocheck
# View pass rates for each experiment
for name, rate in report.pass_rate_by_experiment().items():
    print(f"{name}: {rate:.1%}")

# View mean scores
for name, score in report.mean_score_by_experiment().items():
    print(f"{name}: {score:.2f}")

# Access results grouped by experiment
for name, results in report.by_experiment().items():
    failures = [r for r in results if not r.score.passed]
    print(f"{name}: {len(failures)} failures")
```

**Statistical comparison:**

Compare treatment against baseline with `compare_experiments()`:

```python nocheck
comparison = report.compare_experiments("baseline", "v2-concise-prompts")

print(f"Baseline pass rate: {comparison.baseline_pass_rate:.1%}")
print(f"Treatment pass rate: {comparison.treatment_pass_rate:.1%}")
print(f"Delta: {comparison.pass_rate_delta:+.1%}")

if comparison.relative_improvement:
    print(f"Relative improvement: {comparison.relative_improvement:+.1%}")
```

**Feature flags:**

Experiments can carry feature flags that your agent code checks at runtime:

```python nocheck
# Define experiment with flags
exp = Experiment(
    name="high-retry",
    flags={"max_retries": 5, "timeout_seconds": 120},
)

# In your agent code, check flags
max_retries = experiment.get_flag("max_retries", default=3)
if experiment.has_flag("debug"):
    enable_debug_logging()
```

**Prompt overrides:**

The `overrides_tag` field controls which prompt overrides are loaded. Override
files live in `.weakincentives/prompts/overrides/{ns}/{key}/{tag}.json`:

```python nocheck
# Baseline uses "latest" tag (default prompts)
baseline = Experiment(name="baseline")

# Treatment uses "v2" overrides
treatment = Experiment(name="v2-test", overrides_tag="v2")

# Create variant from existing experiment
variant = treatment.with_tag("v3").with_flag("verbose", True)
```

**Multi-variant testing:**

Test more than two variants by including additional experiments:

```python nocheck
experiments = [
    BASELINE,
    Experiment(name="model-a", flags={"model": "gpt-4o"}),
    Experiment(name="model-b", flags={"model": "claude-3-sonnet"}),
    Experiment(name="model-c", flags={"model": "gemini-pro"}),
]

submit_experiments(dataset, experiments, requests)
```

## Next Steps

- [Lifecycle](lifecycle.md): Run MainLoop and EvalLoop together with LoopGroup
- [Debugging](debugging.md): Inspect failed evaluations
- [Testing](testing.md): Write unit tests for evaluators
