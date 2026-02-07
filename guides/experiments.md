# Experiments and A/B Testing

*Canonical spec: [specs/EXPERIMENTS.md](../specs/EXPERIMENTS.md)*

This guide explains how to systematically compare agent variants using
WINK's experiments system. It builds on the evaluation guide—read
[Evaluation](evaluation.md) first if you haven't already.

## Why Experiment

Evaluation tells you whether your agent works. Experiments tell you
whether a change made it better. Without controlled comparison, prompt
engineering degenerates into guesswork: you tweak something, run a few
examples by hand, and convince yourself it improved. Maybe it did.
Maybe you just picked favorable inputs.

The experiments system gives you a repeatable process:

1. Define a baseline (the current behavior)
2. Define a treatment (the proposed change)
3. Run the same dataset through both
4. Compare results with real numbers

This is the same baseline-vs-treatment pattern used in any controlled
experiment. The only difference is that your "subjects" are agent runs
and your "intervention" is a prompt override, a feature flag, or both.

## The Mental Model

An `Experiment` bundles two things:

| Component | What It Controls | Example |
|-----------|-----------------|---------|
| Overrides tag | Which prompt text the agent sees | `"v2"` loads different section text |
| Feature flags | Runtime behavior switches | `{"max_retries": 5}` |

Together, these define a complete agent variant. The experiment has a
name for identification and optional metadata (owner, description) for
bookkeeping.

The key insight: **experiments compose existing mechanisms**. The
overrides tag hooks into the prompt overrides system (see
[Prompt Overrides](prompt-overrides.md)). Feature flags are arbitrary
key-value pairs your tools and agent logic can read. No new concepts—
just a named bundle that ties them together for evaluation.

## Baseline and Treatment

Every comparison needs a control group. WINK provides two pre-built
sentinel experiments for this:

```python nocheck
from weakincentives.evals import BASELINE, CONTROL

# Both are equivalent: name="baseline"/"control", tag="latest", no flags
```

`BASELINE` and `CONTROL` use the `"latest"` overrides tag and no
feature flags. This means they run your agent with whatever is in code
right now—no overrides applied. Use whichever name reads better in
your context.

A treatment is any `Experiment` with different configuration:

```python nocheck
from weakincentives.evals import Experiment

treatment = Experiment(
    name="concise-v2",
    overrides_tag="v2",
    flags={"max_tokens": 2000},
    description="Shorter prompts with token limit",
)
```

This treatment applies the `"v2"` overrides tag (loading different
prompt text from your overrides store) and sets a feature flag that
your tool logic or agent loop can read.

## Defining What Varies

### Prompt Overrides

The most common experiment changes prompt text. The workflow is:

1. Seed an override file for a new tag
   (`store.seed(prompt, tag="v2")`)
2. Edit the override file with your proposed changes
3. Create an experiment pointing to that tag

The override system handles the rest: hash validation ensures the
override applies to the right version of the prompt, and the tag
system keeps variants isolated. See
[Prompt Overrides](prompt-overrides.md) for the full override
workflow.

### Feature Flags

For changes that go beyond prompt text—different retry limits, model
parameters, tool policies—use feature flags:

```python nocheck
experiment = Experiment(
    name="aggressive-retry",
    overrides_tag="latest",
    flags={"max_retries": 5, "retry_backoff": 2.0},
)
```

Your agent logic reads flags via `experiment.get_flag()` or
`experiment.has_flag()`. Flags are untyped and unvalidated by
design—they're configuration knobs, not contracts.

### Both Together

The most powerful experiments combine prompt changes with behavioral
flags:

```python nocheck
experiment = Experiment(
    name="v3-with-guardrails",
    overrides_tag="v3",
    flags={"strict_validation": True, "verbose_logging": True},
)
```

This tests a new prompt variant with tighter runtime behavior
simultaneously—exactly how you'd deploy it in production.

## Running an Experiment

The experiment plugs into the evaluation flow you already know.
Instead of submitting your dataset once, you submit it once per
experiment:

```python nocheck
from weakincentives.evals import (
    BASELINE, Experiment, submit_experiments, collect_results,
)

baseline = BASELINE
treatment = Experiment(name="v2-concise", overrides_tag="v2")

# Submit dataset under both experiments
count = submit_experiments(
    dataset, [baseline, treatment], eval_requests,
)

# Run the eval loop (processes all requests)
eval_loop.run(max_iterations=count)

# Collect and compare
report = collect_results(eval_results, expected_count=count)
```

Each sample runs twice—once under each experiment. Results carry the
experiment name so the report can group and compare them.

## Comparing Results

`EvalReport` provides experiment-aware aggregation:

```python nocheck
# Per-experiment pass rates
for name, rate in report.pass_rate_by_experiment().items():
    print(f"{name}: {rate:.1%}")

# Direct comparison
comparison = report.compare_experiments("baseline", "v2-concise")
print(f"Baseline:    {comparison.baseline_pass_rate:.1%}")
print(f"Treatment:   {comparison.treatment_pass_rate:.1%}")
print(f"Delta:       {comparison.pass_rate_delta:+.1%}")
if comparison.relative_improvement is not None:
    print(f"Relative:    {comparison.relative_improvement:+.1%}")
```

The `ExperimentComparison` gives you:

| Property | Meaning |
|----------|---------|
| `baseline_pass_rate` | Fraction of baseline samples that passed |
| `treatment_pass_rate` | Fraction of treatment samples that passed |
| `pass_rate_delta` | Treatment minus baseline (positive = better) |
| `relative_improvement` | Delta as percentage of baseline |

These are straightforward pass-rate comparisons. For more nuanced
analysis, use `by_experiment()` to get raw results and compute your
own metrics (mean score, latency distributions, per-category
breakdowns).

## A Practical Workflow

Here is a workflow that works well for iterating on prompt quality:

1. **Establish a baseline.** Run your eval suite with `BASELINE` and
   record the pass rate. This is your current state of the world.

2. **Create overrides for the treatment.** Seed a new tag, edit the
   override files with your proposed changes.

3. **Run the comparison.** Submit the dataset under both experiments,
   collect results, check the delta.

4. **Iterate or promote.** If the treatment wins, update your code to
   incorporate the changes (or promote the override tag to production).
   If it loses, try a different approach.

5. **Repeat.** Each round, the previous winner becomes the new
   baseline. Improvements compound.

This is a tight loop: change, measure, decide. No guessing.

## Multiple Treatments

You are not limited to pairwise comparisons. Submit the same dataset
under as many experiments as you want:

```python nocheck
experiments = [
    BASELINE,
    Experiment(name="short-prompts", overrides_tag="short"),
    Experiment(name="detailed-prompts", overrides_tag="detailed"),
    Experiment(name="short-with-examples", overrides_tag="short",
               flags={"include_examples": True}),
]

count = submit_experiments(dataset, experiments, eval_requests)
```

Use `pass_rate_by_experiment()` to see all results at once, then
`compare_experiments()` to drill into specific pairs.

## Experiments vs RunContext

Both `Experiment` and `RunContext` can be attached to an
`AgentLoopRequest`, but they serve different purposes:

| Concept | Purpose | Affects behavior? |
|---------|---------|-------------------|
| `Experiment` | Configuration variant | Yes |
| `RunContext` | Execution metadata | No |

An experiment changes what the agent does (different prompts, different
flags). A run context records where and why it ran (trace IDs, user
info). They compose independently.

## When to Use Experiments

Use experiments when you want to answer questions like:

- Does the new prompt wording improve accuracy?
- Does increasing the retry limit help or just add latency?
- Which model produces better results for this task?
- Does adding examples to the prompt help on edge cases?

Do not use experiments for one-off debugging. If you're investigating
a single failure, use [debug bundles](debugging.md) instead.
Experiments are for systematic comparison across a dataset—they earn
their keep when you have enough samples to draw conclusions.

## Next Steps

- [Evaluation](evaluation.md): Core evaluation concepts this guide
  builds on
- [Prompt Overrides](prompt-overrides.md): Managing override files
  and tags
- [Debugging](debugging.md): Investigating individual failures
- [Lifecycle](lifecycle.md): Running eval workers in production
