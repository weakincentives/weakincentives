# Experiments Specification

## Purpose

Enable systematic evaluation of agent behavior variants through named
experiments. Bundles prompt overrides tag with feature flags for A/B testing,
optimization runs, and controlled rollouts.

**Implementation:**

- `src/weakincentives/experiment.py` - Experiment class, BASELINE, CONTROL
- `src/weakincentives/evals/_types.py` - EvalRequest, EvalResult, EvalReport, ExperimentComparison

## Core Types

### Experiment

| Field | Type | Default | Description |
| --------------- | ----------------------- | ---------- | ------------------------ |
| `name` | `str` | - | Unique identifier |
| `overrides_tag` | `str` | `"latest"` | Tag for prompt overrides |
| `flags` | `Mapping[str, object]` | `{}` | Feature flags |
| `owner` | `str \| None` | `None` | Owner identifier |
| `description` | `str \| None` | `None` | Human-readable description |

**Methods:**

| Method | Returns | Description |
| ------------------------- | ------------ | -------------------------------- |
| `with_flag(key, value)` | `Experiment` | New experiment with flag added |
| `with_tag(tag)` | `Experiment` | New experiment with new tag |
| `get_flag(key, default?)` | `object` | Get flag value or default |
| `has_flag(key)` | `bool` | Check if flag exists |

### Sentinel Experiments

```python
from weakincentives.evals import BASELINE, CONTROL, Experiment

# Pre-defined experiments for common patterns
BASELINE  # name="baseline", overrides_tag="latest", no flags
CONTROL   # name="control", overrides_tag="latest", no flags (alias)
```

Use `BASELINE` or `CONTROL` as the control group in A/B tests.

### ExperimentComparison

Returned by `EvalReport.compare_experiments(baseline, treatment)`:

| Field/Property | Type | Description |
| ---------------------- | --------------------------- | ------------------------------- |
| `baseline_name` | `str` | Baseline experiment name |
| `treatment_name` | `str` | Treatment experiment name |
| `baseline_results` | `tuple[EvalResult, ...]` | Results from baseline |
| `treatment_results` | `tuple[EvalResult, ...]` | Results from treatment |
| `baseline_pass_rate` | `float` (property) | Pass rate for baseline |
| `treatment_pass_rate` | `float` (property) | Pass rate for treatment |
| `pass_rate_delta` | `float` (property) | Treatment - baseline |
| `relative_improvement` | `float \| None` (property) | Delta / baseline (None if 0) |

## Request Integration

`AgentLoopRequest` accepts an optional `experiment` field; `EvalRequest` requires
one. Both are at `src/weakincentives/evals/_types.py`.

### EvalResult

Includes `experiment_name` for downstream aggregation.

## AgentLoop Integration

`prepare()` receives experiment to configure prompt and session:

| Integration Point | Pattern |
| ----------------- | ------------------------------------------------------- |
| Prompt overrides | `Prompt(template, overrides_tag=experiment.overrides_tag)` |
| Session tracking | `session[Experiment].seed(experiment)` |
| Feature flags | `if experiment.get_flag("verbose"): ...` |

## Feature Flags

| Flag Type | Example |
| --------- | ----------------------------------------- |
| Boolean | `{"verbose_logging": True}` |
| Numeric | `{"max_retries": 5}` |
| String | `{"model_override": "gpt-4o-mini"}` |
| Composite | `{"tool_policy": {"allow": ["read"]}}` |

## A/B Testing Workflow

1. Define a `BASELINE` experiment and one or more treatment `Experiment` instances
   with distinct names and `overrides_tag`/`flags`.
2. Submit `EvalRequest(sample=sample, experiment=experiment)` for each
   (sample, experiment) pair to the mailbox.
3. Collect results into an `EvalReport` and call `compare_experiments(baseline_name,
   treatment_name)` for a statistical comparison.

See `src/weakincentives/evals/_types.py` for `EvalReport` and `ExperimentComparison`.

## Result Aggregation

`EvalReport` provides:

| Method | Returns | Description |
| ---------------------------------------- | ----------------------------- | ------------------------ |
| `by_experiment()` | `dict[str, tuple[EvalResult]]`| Group by name |
| `pass_rate_by_experiment()` | `dict[str, float]` | Pass rates |
| `mean_score_by_experiment()` | `dict[str, float]` | Mean scores |
| `compare_experiments(baseline, treatment)`| `ExperimentComparison` | Statistical comparison |

## Relationship to RunContext

| Concept | Purpose | Affects Behavior |
| ------------ | ---------------------- | ----------------------- |
| `Experiment` | Configuration variant | Yes (prompts, flags) |
| `RunContext` | Execution metadata | No (tracing only) |

Both can be specified on `AgentLoopRequest` independently.

## Storage Layout

Experiments reference prompt overrides in:

```
.weakincentives/prompts/overrides/{ns}/{key}/{tag}.json
```

## Invariants

1. Experiment names unique within eval run
1. Missing overrides tags fall back to source (silent)
1. Flags not validated (invalid silently ignored)
1. Experiments are immutable (use `with_*` methods)
1. EvalRequest requires experiment
1. Equality is value-based (dataclass)

## Related Specifications

- `specs/PROMPTS.md` - Override system
- `specs/EVALS.md` - Evaluation framework
- `specs/AGENT_LOOP.md` - AgentLoop orchestration
- `specs/RUN_CONTEXT.md` - Execution metadata
