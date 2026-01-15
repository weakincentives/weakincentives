# Experiments Specification

## Purpose

Enable systematic evaluation of agent behavior variants through named
experiments. Bundles prompt overrides tag with feature flags for A/B testing,
optimization runs, and controlled rollouts.

**Implementation:**
- `src/weakincentives/evals/_experiment.py` - Experiment class
- `src/weakincentives/evals/_types.py` - EvalRequest, EvalResult, EvalReport, ExperimentComparison

## Core Type

### Experiment

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | - | Unique identifier |
| `overrides_tag` | `str` | `"latest"` | Tag for prompt overrides |
| `flags` | `Mapping[str, object]` | `{}` | Feature flags |
| `owner` | `str \| None` | `None` | Owner identifier |
| `description` | `str \| None` | `None` | Human-readable description |

Methods: `with_flag(key, value)`, `with_tag(tag)`, `get_flag(key, default)`,
`has_flag(key)`

## Request Integration

### MainLoopRequest

```python
request = MainLoopRequest(
    request=my_request,
    experiment=Experiment(name="v2-prompts", overrides_tag="v2"),
)
```

### EvalRequest

Experiment is **required** for evaluation requests:

```python
request = EvalRequest(sample=sample, experiment=experiment)
```

### EvalResult

Includes `experiment_name` for downstream aggregation.

## MainLoop Integration

`prepare()` receives experiment to configure prompt and session:

| Integration Point | Pattern |
|-------------------|---------|
| Prompt overrides | `Prompt(template, overrides_tag=experiment.overrides_tag)` |
| Session tracking | `session[Experiment].seed(experiment)` |
| Feature flags | `if experiment.get_flag("verbose"): ...` |

## Feature Flags

| Flag Type | Example |
|-----------|---------|
| Boolean | `{"verbose_logging": True}` |
| Numeric | `{"max_retries": 5}` |
| String | `{"model_override": "gpt-4o-mini"}` |
| Composite | `{"tool_policy": {"allow": ["read"]}}` |

## Dataset Submission Patterns

### Single Experiment

```python
for sample in dataset:
    mailbox.send(EvalRequest(sample=sample, experiment=experiment))
```

### Multi-Experiment (A/B)

```python
for experiment in experiments:
    for sample in dataset:
        mailbox.send(EvalRequest(sample=sample, experiment=experiment))
```

## Result Aggregation

`EvalReport` provides:
- `by_experiment()` - Group by name
- `pass_rate_by_experiment()` - Pass rates
- `mean_score_by_experiment()` - Mean scores
- `compare_experiments(baseline, treatment)` - Statistical comparison

## Relationship to RunContext

| Concept | Purpose | Affects Behavior |
|---------|---------|------------------|
| `Experiment` | Configuration variant | Yes (prompts, flags) |
| `RunContext` | Execution metadata | No (tracing only) |

Both can be specified on `MainLoopRequest` independently.

## Storage Layout

Experiments reference prompt overrides in:

```
.weakincentives/prompts/overrides/{ns}/{key}/{tag}.json
```

## Invariants

1. Experiment names unique within eval run
2. Missing overrides tags fall back to source (silent)
3. Flags not validated (invalid silently ignored)
4. Experiments are immutable (use `with_*` methods)
5. EvalRequest requires experiment
6. Equality is value-based

## Related Specifications

- `specs/PROMPTS.md` - Override system
- `specs/EVALS.md` - Evaluation framework
- `specs/MAIN_LOOP.md` - MainLoop orchestration
- `specs/RUN_CONTEXT.md` - Execution metadata
