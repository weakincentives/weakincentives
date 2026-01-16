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

```python
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

```python
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

```python
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

```python
from weakincentives.evals import adapt, exact_match, all_of, tool_called

evaluator = all_of(
    adapt(exact_match),    # Now works with session evaluators
    tool_called("search"),
)
```

## Running Evaluations

**EvalLoop wraps your MainLoop:**

```python
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

```python
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

```python
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

```python
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

## Next Steps

- [Lifecycle](lifecycle.md): Run MainLoop and EvalLoop together with LoopGroup
- [Debugging](debugging.md): Inspect failed evaluations
- [Testing](testing.md): Write unit tests for evaluators
