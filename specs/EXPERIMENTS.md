# Experiments Specification

## Purpose

Define a data model for experiments that attaches to MainLoop execution. An
experiment captures feature flags and optimizer configuration options for
systematic evaluation during both manual testing and automated optimization
pipelines.

## Guiding Principles

- **Immutable configuration**: Experiments are frozen at creation; variants
  define variations
- **Composable**: Multiple experiments can be active simultaneously; conflicts
  are explicit
- **Observable**: Experiment assignment flows through events for analysis
- **Eval-native**: Designed for integration with `EvalLoop` and optimizer loops

## Architecture

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────┐
│    Experiment    │────▶│ ExperimentContext │────▶│ MainLoop     │
│ (flags, options) │     │ (active variants) │     │ .execute()   │
└──────────────────┘     └─────────┬─────────┘     └──────────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ MainLoopRequest   │
                         │ .experiment_ctx   │
                         └───────────────────┘
```

Experiments define **what** can vary. Variants define **how** it varies.
ExperimentContext binds specific variant assignments for a run.

## Core Types

### FeatureFlag

A binary or multi-value toggle affecting prompt or tool behavior:

```python
@dataclass(slots=True, frozen=True)
class FeatureFlag:
    """Single feature flag definition."""
    name: str                           # Unique identifier within experiment
    description: str                    # Human-readable purpose
    default: str                        # Default value
    values: tuple[str, ...] = ("off", "on")  # Allowed values

    def __post_init__(self) -> None:
        if self.default not in self.values:
            raise ValueError(f"default '{self.default}' not in values")
```

Flags use string values for flexibility. Boolean flags use `"on"`/`"off"`:

```python
# Binary flag
verbose_logging = FeatureFlag(
    name="verbose_logging",
    description="Enable detailed execution logs",
    default="off",
)

# Multi-value flag
model_tier = FeatureFlag(
    name="model_tier",
    description="Model quality/cost tradeoff",
    default="standard",
    values=("fast", "standard", "premium"),
)
```

### OptimizerOption

Configuration knob for optimizer behavior during evaluation:

```python
@dataclass(slots=True, frozen=True)
class OptimizerOption:
    """Optimizer configuration option."""
    name: str                      # Unique identifier within experiment
    description: str               # Human-readable purpose
    default: object                # Default value (JSON-serializable)
    schema: type | None = None     # Optional type hint for validation
```

Options store arbitrary JSON-serializable values:

```python
temperature = OptimizerOption(
    name="temperature",
    description="LLM sampling temperature for optimization prompts",
    default=0.7,
    schema=float,
)

max_iterations = OptimizerOption(
    name="max_iterations",
    description="Maximum optimization iterations before stopping",
    default=3,
    schema=int,
)
```

### Experiment

Container grouping related flags and options:

```python
@dataclass(slots=True, frozen=True)
class Experiment:
    """Experiment definition with flags and optimizer options."""
    name: str                                    # Unique experiment identifier
    description: str                             # Purpose of this experiment
    flags: tuple[FeatureFlag, ...] = ()          # Feature flags
    options: tuple[OptimizerOption, ...] = ()    # Optimizer options
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def flag_names(self) -> frozenset[str]:
        """All flag names in this experiment."""
        return frozenset(f.name for f in self.flags)

    @property
    def option_names(self) -> frozenset[str]:
        """All option names in this experiment."""
        return frozenset(o.name for o in self.options)

    def get_flag(self, name: str) -> FeatureFlag:
        """Get flag by name. Raises KeyError if not found."""
        for f in self.flags:
            if f.name == name:
                return f
        raise KeyError(f"flag '{name}' not in experiment '{self.name}'")

    def get_option(self, name: str) -> OptimizerOption:
        """Get option by name. Raises KeyError if not found."""
        for o in self.options:
            if o.name == name:
                return o
        raise KeyError(f"option '{name}' not in experiment '{self.name}'")
```

### ExperimentVariant

A specific configuration of flag and option values:

```python
@dataclass(slots=True, frozen=True)
class ExperimentVariant:
    """Specific flag and option assignments for an experiment."""
    experiment_name: str                         # Reference to parent experiment
    variant_name: str                            # Unique variant identifier
    flag_values: Mapping[str, str] = field(default_factory=dict)
    option_values: Mapping[str, object] = field(default_factory=dict)

    def get_flag(self, name: str, default: str | None = None) -> str:
        """Get flag value, falling back to default if not set."""
        return self.flag_values.get(name, default) if default else self.flag_values[name]

    def get_option(self, name: str, default: object = None) -> object:
        """Get option value, falling back to default if not set."""
        return self.option_values.get(name, default)
```

Variants inherit unspecified values from experiment defaults:

```python
# Control variant: all defaults
control = ExperimentVariant(
    experiment_name="prompt_v2",
    variant_name="control",
)

# Treatment: enable new feature
treatment = ExperimentVariant(
    experiment_name="prompt_v2",
    variant_name="treatment",
    flag_values={"new_section": "on"},
    option_values={"temperature": 0.5},
)
```

### ExperimentContext

Active experiment state attached to a MainLoop execution:

```python
@dataclass(slots=True, frozen=True)
class ExperimentContext:
    """Active experiment configuration for a MainLoop execution."""
    experiments: Mapping[str, Experiment] = field(default_factory=dict)
    variants: Mapping[str, ExperimentVariant] = field(default_factory=dict)
    run_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_enabled(self, experiment_name: str, flag_name: str) -> bool:
        """Check if flag is set to 'on' in active variant."""
        return self.get_flag(experiment_name, flag_name) == "on"

    def get_flag(self, experiment_name: str, flag_name: str) -> str:
        """Get effective flag value (variant override or experiment default)."""
        experiment = self.experiments.get(experiment_name)
        if experiment is None:
            raise KeyError(f"experiment '{experiment_name}' not in context")

        flag = experiment.get_flag(flag_name)
        variant = self.variants.get(experiment_name)

        if variant and flag_name in variant.flag_values:
            return variant.flag_values[flag_name]
        return flag.default

    def get_option(self, experiment_name: str, option_name: str) -> object:
        """Get effective option value (variant override or experiment default)."""
        experiment = self.experiments.get(experiment_name)
        if experiment is None:
            raise KeyError(f"experiment '{experiment_name}' not in context")

        option = experiment.get_option(option_name)
        variant = self.variants.get(experiment_name)

        if variant and option_name in variant.option_values:
            return variant.option_values[option_name]
        return option.default

    def with_variant(
        self,
        experiment: Experiment,
        variant: ExperimentVariant,
    ) -> ExperimentContext:
        """Return new context with additional experiment/variant binding."""
        return ExperimentContext(
            experiments={**self.experiments, experiment.name: experiment},
            variants={**self.variants, experiment.name: variant},
            run_id=self.run_id,
            created_at=self.created_at,
        )
```

### ExperimentRegistry

Central storage for experiment definitions:

```python
class ExperimentRegistry:
    """Registry for experiment definitions."""

    def __init__(self) -> None:
        self._experiments: dict[str, Experiment] = {}
        self._variants: dict[str, dict[str, ExperimentVariant]] = {}

    def register(self, experiment: Experiment) -> None:
        """Register an experiment definition."""
        if experiment.name in self._experiments:
            raise ValueError(f"experiment '{experiment.name}' already registered")
        self._experiments[experiment.name] = experiment
        self._variants[experiment.name] = {}

    def register_variant(self, variant: ExperimentVariant) -> None:
        """Register a variant for an experiment."""
        if variant.experiment_name not in self._experiments:
            raise ValueError(f"experiment '{variant.experiment_name}' not registered")
        self._variants[variant.experiment_name][variant.variant_name] = variant

    def get(self, name: str) -> Experiment:
        """Get experiment by name."""
        return self._experiments[name]

    def get_variant(self, experiment_name: str, variant_name: str) -> ExperimentVariant:
        """Get variant by experiment and variant name."""
        return self._variants[experiment_name][variant_name]

    def create_context(
        self,
        assignments: Mapping[str, str],  # experiment_name -> variant_name
    ) -> ExperimentContext:
        """Create context from variant assignments."""
        experiments: dict[str, Experiment] = {}
        variants: dict[str, ExperimentVariant] = {}

        for exp_name, var_name in assignments.items():
            experiments[exp_name] = self.get(exp_name)
            variants[exp_name] = self.get_variant(exp_name, var_name)

        return ExperimentContext(experiments=experiments, variants=variants)
```

## MainLoop Integration

### Extended MainLoopRequest

Experiments attach to requests via an optional `experiment_ctx` field:

```python
@FrozenDataclass()
class MainLoopRequest[UserRequestT]:
    """Request for MainLoop execution with optional constraints."""
    request: UserRequestT
    budget: Budget | None = None
    deadline: Deadline | None = None
    resources: ResourceRegistry | None = None
    experiment_ctx: ExperimentContext | None = None  # NEW
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### Accessing Experiments in Prompts

Prompt sections can check experiment flags during rendering:

```python
class ExperimentAwareSection(MarkdownSection[P]):
    """Section that conditionally renders based on experiment flags."""

    def __init__(
        self,
        *,
        experiment_name: str,
        flag_name: str,
        enabled_value: str = "on",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._experiment_name = experiment_name
        self._flag_name = flag_name
        self._enabled_value = enabled_value

    def render(self, params: P, *, context: RenderContext) -> str | None:
        experiment_ctx = context.experiment_ctx
        if experiment_ctx is None:
            return None  # No experiment context, skip section

        value = experiment_ctx.get_flag(self._experiment_name, self._flag_name)
        if value != self._enabled_value:
            return None

        return super().render(params, context=context)
```

### Accessing Experiments in Tool Handlers

Tool handlers receive experiment context via ToolContext:

```python
def my_handler(
    params: MyParams,
    *,
    context: ToolContext,
) -> ToolResult[MyOutput]:
    exp_ctx = context.experiment_ctx

    # Check feature flag
    if exp_ctx and exp_ctx.is_enabled("my_experiment", "new_algorithm"):
        return _new_algorithm(params)

    return _legacy_algorithm(params)
```

## EvalLoop Integration

### Experiment Matrix

Generate all variant combinations for comprehensive evaluation:

```python
def experiment_matrix(
    experiments: Sequence[Experiment],
    variants_per_experiment: Mapping[str, Sequence[str]],
) -> Sequence[ExperimentContext]:
    """Generate all combinations of experiment variants.

    Args:
        experiments: Experiments to include
        variants_per_experiment: Map of experiment name to variant names to test

    Returns:
        Sequence of ExperimentContexts, one per combination
    """
    # Implementation uses itertools.product over variant lists
    ...
```

### Running Experiments with EvalLoop

EvalLoop uses mailboxes for requests and results. Experiments attach to each
evaluation request:

```python
from weakincentives.evals import (
    Dataset, EvalLoop, EvalRequest, EvalResult,
    exact_match, submit_dataset, collect_results,
)
from weakincentives.runtime import InMemoryMailbox, Mailbox

# Define experiment
prompt_experiment = Experiment(
    name="prompt_v2",
    description="Test new prompt structure",
    flags=(
        FeatureFlag(name="new_section", description="Enable new section", default="off"),
        FeatureFlag(name="verbose", description="Verbose output", default="off"),
    ),
    options=(
        OptimizerOption(name="temperature", description="Sampling temp", default=0.7),
    ),
)

# Define variants
control = ExperimentVariant(experiment_name="prompt_v2", variant_name="control")
treatment_a = ExperimentVariant(
    experiment_name="prompt_v2",
    variant_name="treatment_a",
    flag_values={"new_section": "on"},
)
treatment_b = ExperimentVariant(
    experiment_name="prompt_v2",
    variant_name="treatment_b",
    flag_values={"new_section": "on", "verbose": "on"},
)

# Create mailboxes
requests: Mailbox[EvalRequest[str, str]] = InMemoryMailbox(name="eval-requests")
results: Mailbox[EvalResult] = InMemoryMailbox(name="eval-results")

# Run eval across variants
reports: dict[str, EvalReport] = {}
for variant in [control, treatment_a, treatment_b]:
    ctx = ExperimentContext(
        experiments={"prompt_v2": prompt_experiment},
        variants={"prompt_v2": variant},
    )

    # MainLoop receives experiment context via initialize()
    loop = ExperimentAwareLoop(
        adapter=adapter,
        requests=main_requests,
        responses=main_responses,
        experiment_ctx=ctx,
    )

    eval_loop = EvalLoop(
        loop=loop,
        evaluator=exact_match,
        requests=requests,
        results=results,
    )

    submit_dataset(dataset, requests)
    eval_loop.run(max_iterations=1)
    reports[variant.variant_name] = collect_results(results, expected_count=len(dataset))

# Compare results
for name, report in reports.items():
    print(f"{name}: pass_rate={report.pass_rate:.1%}, score={report.mean_score:.2f}")
```

### Experiment-Aware MainLoop

MainLoop implementations can access experiment context in `initialize()`:

```python
class ExperimentAwareLoop(MainLoop[str, str]):
    """MainLoop that adapts behavior based on experiment context."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[str],
        requests: Mailbox[MainLoopRequest[str]],
        responses: Mailbox[MainLoopResult[str]],
        experiment_ctx: ExperimentContext | None = None,
    ) -> None:
        super().__init__(adapter=adapter, requests=requests, responses=responses)
        self._experiment_ctx = experiment_ctx

    def initialize(self, request: str) -> tuple[Prompt[str], Session]:
        # Select template based on experiment flag
        if self._experiment_ctx and self._experiment_ctx.is_enabled("prompt_v2", "new_section"):
            template = self._template_v2
        else:
            template = self._template_v1

        prompt = Prompt(template).bind(Params(input=request))
        session = Session(tags={"experiment": self._experiment_ctx.run_id if self._experiment_ctx else None})
        return prompt, session
```

## Optimizer Integration

Optimizers access experiment options via OptimizationContext:

```python
@dataclass(slots=True, frozen=True)
class OptimizationContext:
    adapter: ProviderAdapter[object]
    dispatcher: Dispatcher
    deadline: Deadline | None = None
    overrides_store: PromptOverridesStore | None = None
    overrides_tag: str = "latest"
    optimization_session: Session | None = None
    experiment_ctx: ExperimentContext | None = None  # NEW
```

Optimizers read configuration from experiment options:

```python
class MyOptimizer(BasePromptOptimizer[I, O]):
    def optimize(self, prompt: Prompt[I], *, session: SessionProtocol) -> O:
        exp_ctx = self._context.experiment_ctx

        # Read optimizer options
        temperature = 0.7  # default
        max_iters = 3      # default

        if exp_ctx:
            temperature = exp_ctx.get_option("my_experiment", "temperature")
            max_iters = exp_ctx.get_option("my_experiment", "max_iterations")

        # Use in optimization logic
        ...
```

## Events

Experiment lifecycle emits events for observability:

```python
@dataclass(slots=True, frozen=True)
class ExperimentAssigned:
    """Emitted when experiment context is attached to a request."""
    request_id: UUID
    run_id: UUID
    experiment_name: str
    variant_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True, frozen=True)
class ExperimentCompleted:
    """Emitted when request with experiment context completes."""
    request_id: UUID
    run_id: UUID
    experiment_name: str
    variant_name: str
    success: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
```

## Serialization

Experiments serialize via the standard serde module:

```python
from weakincentives.serde import dump, parse

# Serialize experiment
data = dump(experiment)  # Returns JSON-compatible dict

# Deserialize
loaded = parse(Experiment, data)
```

**JSON representation:**

```json
{
  "name": "prompt_v2",
  "description": "Test new prompt structure",
  "flags": [
    {
      "name": "new_section",
      "description": "Enable new section",
      "default": "off",
      "values": ["off", "on"]
    }
  ],
  "options": [
    {
      "name": "temperature",
      "description": "Sampling temp",
      "default": 0.7,
      "schema": null
    }
  ],
  "metadata": {}
}
```

## Validation Rules

### Experiment Validation

- Experiment names must be non-empty and unique within a registry
- Flag names must be unique within an experiment
- Option names must be unique within an experiment
- Default values must be valid for their type/values

### Variant Validation

- Variant must reference a registered experiment
- Flag values must be in the flag's allowed values
- Option values should match the option's schema (if specified)
- Variant names must be unique per experiment

### Context Validation

- All referenced experiments must be present in the context
- All referenced variants must match their experiments

## Usage Examples

### Basic Feature Flag

```python
# Define experiment
experiment = Experiment(
    name="code_review_v2",
    description="Test enhanced code review prompt",
    flags=(
        FeatureFlag(
            name="include_examples",
            description="Include code examples in feedback",
            default="off",
        ),
    ),
)

# Create variants
control = ExperimentVariant(
    experiment_name="code_review_v2",
    variant_name="control",
)
with_examples = ExperimentVariant(
    experiment_name="code_review_v2",
    variant_name="with_examples",
    flag_values={"include_examples": "on"},
)

# Create context and run
ctx = ExperimentContext(
    experiments={"code_review_v2": experiment},
    variants={"code_review_v2": with_examples},
)

request = MainLoopRequest(
    request=ReviewRequest(file="main.py"),
    experiment_ctx=ctx,
)
requests_mailbox.send(request)
```

### Optimizer Tuning

```python
# Define experiment for optimizer tuning
tuning_experiment = Experiment(
    name="digest_optimization",
    description="Tune workspace digest generation",
    options=(
        OptimizerOption(
            name="temperature",
            description="LLM temperature for digest generation",
            default=0.3,
        ),
        OptimizerOption(
            name="max_tokens",
            description="Maximum digest length",
            default=2000,
        ),
    ),
)

# Test different configurations
for temp in [0.1, 0.3, 0.5, 0.7]:
    variant = ExperimentVariant(
        experiment_name="digest_optimization",
        variant_name=f"temp_{temp}",
        option_values={"temperature": temp},
    )
    ctx = ExperimentContext(
        experiments={"digest_optimization": tuning_experiment},
        variants={"digest_optimization": variant},
    )

    opt_ctx = OptimizationContext(
        adapter=adapter,
        dispatcher=dispatcher,
        experiment_ctx=ctx,
    )

    optimizer = WorkspaceDigestOptimizer(opt_ctx)
    result = optimizer.optimize(prompt, session=session)
    # ... evaluate quality ...
```

### Multi-Experiment Context

```python
# Multiple experiments active simultaneously
prompt_exp = Experiment(name="prompt_v2", ...)
tool_exp = Experiment(name="tool_config", ...)

ctx = ExperimentContext(
    experiments={
        "prompt_v2": prompt_exp,
        "tool_config": tool_exp,
    },
    variants={
        "prompt_v2": prompt_variant,
        "tool_config": tool_variant,
    },
)
```

## Limitations

- **No dynamic assignment**: Experiments are assigned before execution, not
  during
- **No statistical analysis**: Raw results only; use external tools for
  significance testing
- **No persistence**: ExperimentRegistry is in-memory; load from config files
  if needed
- **Single variant per experiment**: A context binds one variant per experiment
- **No gradual rollout**: All-or-nothing variant assignment; implement
  percentage-based assignment externally
