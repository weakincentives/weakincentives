# DSPy Integration Specification

## Purpose

Integrate DSPy as a foundational library for prompt optimization. This spec
defines the core building blocks: a language model adapter that routes DSPy
calls through weakincentives adapters, and the bridge layer that translates
between DSPy programs and weakincentives prompts. Training data uses the
`Dataset` type from evals; metrics bridge to `Evaluator`.

## Guiding Principles

- **Adapter-first**: All DSPy LLM calls flow through weakincentives adapters,
  inheriting throttling, deadlines, and observability.
- **Evals-native data**: Training data uses `Dataset[InputT, ExpectedT]`;
  metrics use `Evaluator[OutputT, ExpectedT]`.
- **Override-native persistence**: Optimized prompts persist through the
  existing override system.
- **Minimal surface area**: Expose only what's needed to run DSPy optimizers;
  avoid reimplementing DSPy concepts.

## Core Building Blocks

### WeakIncentivesLM

The foundational piece: a DSPy language model backed by a weakincentives
adapter.

```python
class WeakIncentivesLM(dspy.LM):
    """DSPy language model that delegates to a weakincentives adapter."""

    def __init__(
        self,
        adapter: ProviderAdapter[object],
        session: SessionProtocol,
    ) -> None:
        self._adapter = adapter
        self._session = session

    def __call__(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> list[str]:
        # Build a simple prompt from the DSPy request
        wink_prompt = self._build_prompt(prompt, **kwargs)

        # Execute through the adapter (telemetry via session.dispatcher)
        response = self._adapter.evaluate(
            wink_prompt,
            session=self._session,
        )

        # Return completions in DSPy's expected format
        return [response.text] if response.text else []

    def _build_prompt(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> Prompt[object]:
        # Convert DSPy prompt format to weakincentives Prompt
        ...
```

This ensures DSPy operations get:
- Throttle protection and retry logic
- Deadline enforcement
- Event emission via `session.dispatcher`
- Budget tracking

**Usage:**

```python
import dspy
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import Session, ControlDispatcher
from weakincentives.optimizers.dspy import WeakIncentivesLM

adapter = OpenAIAdapter(model="gpt-4o")
dispatcher = ControlDispatcher()
session = Session(dispatcher=dispatcher)

# Configure DSPy to use our adapter
lm = WeakIncentivesLM(adapter, session)
dspy.configure(lm=lm)

# Now DSPy programs run through weakincentives
```

### Prompt-to-Module Bridge

Convert weakincentives prompts to DSPy modules for optimization:

```python
def prompt_to_signature(
    prompt: Prompt[T],
) -> type[dspy.Signature]:
    """
    Extract a DSPy signature from a prompt's input/output types.

    The prompt's parameter dataclass becomes input fields.
    The prompt's output type becomes output fields.
    """
    ...

def prompt_to_module(
    prompt: Prompt[T],
) -> dspy.Module:
    """
    Wrap a prompt as a DSPy module for optimization.

    Returns a module whose forward() renders and evaluates the prompt.
    """
    ...
```

### Module-to-Override Bridge

Extract optimized content from DSPy and persist as overrides:

```python
def extract_optimized_sections(
    original_prompt: Prompt[T],
    optimized_module: dspy.Module,
) -> dict[tuple[str, ...], str]:
    """
    Extract section body replacements from an optimized DSPy module.

    Returns a mapping of section paths to new body content,
    including any synthesized few-shot examples.
    """
    ...

def persist_to_overrides(
    prompt: Prompt[T],
    sections: dict[tuple[str, ...], str],
    store: PromptOverridesStore,
    tag: str = "latest",
) -> None:
    """Persist optimized sections through the override system."""
    for path, body in sections.items():
        store.set_section_override(
            prompt,
            tag=tag,
            path=path,
            body=body,
        )
```

### Dataset Bridge

Convert between evals `Dataset` and DSPy trainset:

```python
def dataset_to_trainset(
    dataset: Dataset[InputT, ExpectedT],
    input_field: str = "input",
    output_field: str = "expected",
) -> list[dspy.Example]:
    """
    Convert a weakincentives Dataset to DSPy training examples.

    Args:
        dataset: Evals dataset with samples
        input_field: Name for the input field in DSPy Example
        output_field: Name for the output field in DSPy Example

    Returns:
        List of DSPy Examples with .with_inputs() called
    """
    examples = []
    for sample in dataset:
        ex = dspy.Example(**{
            input_field: sample.input,
            output_field: sample.expected,
        }).with_inputs(input_field)
        examples.append(ex)
    return examples
```

### Evaluator Bridge

Wrap an evals `Evaluator` as a DSPy metric:

```python
def evaluator_to_metric(
    evaluator: Evaluator[OutputT, ExpectedT],
    output_field: str = "expected",
) -> Callable[[dspy.Example, dspy.Prediction], bool]:
    """
    Wrap a weakincentives Evaluator as a DSPy metric function.

    Args:
        evaluator: Evals evaluator (output, expected) -> Score
        output_field: Field name containing expected value in Example

    Returns:
        DSPy-compatible metric function
    """
    def metric(example: dspy.Example, prediction: dspy.Prediction) -> bool:
        expected = getattr(example, output_field)
        output = prediction.get(output_field, "")
        score = evaluator(output, expected)
        return score.passed
    return metric
```

## BootstrapFewShotOptimizer

A `BasePromptOptimizer` implementation that wraps DSPy's BootstrapFewShot,
accepting evals `Dataset` and `Evaluator`:

```python
@dataclass(slots=True, frozen=True)
class BootstrapFewShotResult:
    """Result from bootstrap few-shot optimization."""
    optimized_sections: dict[tuple[str, ...], str]
    demos_count: int


class BootstrapFewShotOptimizer(BasePromptOptimizer[object, BootstrapFewShotResult]):
    """
    Generate few-shot demonstrations using DSPy's BootstrapFewShot.

    Wraps DSPy optimization in the weakincentives optimizer protocol,
    handling LM configuration, event emission, and override persistence.
    Accepts evals Dataset and Evaluator types.
    """

    def __init__(
        self,
        context: OptimizationContext,
        *,
        dataset: Dataset[Any, Any],
        evaluator: Evaluator[Any, Any],
        input_field: str = "input",
        output_field: str = "expected",
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
    ) -> None:
        super().__init__(context)
        self._trainset = dataset_to_trainset(dataset, input_field, output_field)
        self._metric = evaluator_to_metric(evaluator, output_field)
        self._max_bootstrapped_demos = max_bootstrapped_demos
        self._max_labeled_demos = max_labeled_demos

    @property
    def _optimizer_scope(self) -> str:
        return "dspy.bootstrap_few_shot"

    def optimize(
        self,
        prompt: Prompt[object],
        *,
        session: SessionProtocol,
    ) -> BootstrapFewShotResult:
        # Configure DSPy to use our adapter
        lm = WeakIncentivesLM(self._context.adapter, session)
        dspy.configure(lm=lm)

        # Convert prompt to DSPy module
        module = prompt_to_module(prompt)

        # Run DSPy optimization
        optimizer = dspy.BootstrapFewShot(
            metric=self._metric,
            max_bootstrapped_demos=self._max_bootstrapped_demos,
            max_labeled_demos=self._max_labeled_demos,
        )
        optimized = optimizer.compile(module, trainset=self._trainset)

        # Extract optimized sections
        sections = extract_optimized_sections(prompt, optimized)

        # Persist if store is configured
        if self._context.overrides_store is not None:
            persist_to_overrides(
                prompt,
                sections,
                self._context.overrides_store,
                tag=self._context.overrides_tag,
            )

        return BootstrapFewShotResult(
            optimized_sections=sections,
            demos_count=len(optimized.demos) if hasattr(optimized, "demos") else 0,
        )
```

**Usage:**

```python
from weakincentives.optimizers import OptimizationContext
from weakincentives.optimizers.dspy import BootstrapFewShotOptimizer
from weakincentives.evals import Dataset, Sample, exact_match

# Create dataset using evals types
dataset = Dataset(samples=(
    Sample(id="1", input="What is 2+2?", expected="4"),
    Sample(id="2", input="Capital of France?", expected="Paris"),
))

context = OptimizationContext(
    adapter=adapter,
    dispatcher=dispatcher,
    overrides_store=store,
    overrides_tag="bootstrap-v1",
)

optimizer = BootstrapFewShotOptimizer(
    context,
    dataset=dataset,
    evaluator=exact_match,  # From weakincentives.evals
)

result = optimizer.optimize(my_prompt, session=session)
print(f"Generated {result.demos_count} demonstrations")
```

## Low-Level Usage

For more control, use the building blocks directly:

```python
import dspy
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.prompt.overrides import LocalPromptOverridesStore
from weakincentives.runtime import Session, ControlDispatcher
from weakincentives.evals import Dataset, Sample, exact_match
from weakincentives.optimizers.dspy import (
    WeakIncentivesLM,
    prompt_to_module,
    extract_optimized_sections,
    persist_to_overrides,
    dataset_to_trainset,
    evaluator_to_metric,
)

# Setup
adapter = OpenAIAdapter(model="gpt-4o")
dispatcher = ControlDispatcher()
session = Session(dispatcher=dispatcher)
store = LocalPromptOverridesStore()

# Configure DSPy with our adapter
lm = WeakIncentivesLM(adapter, session)
dspy.configure(lm=lm)

# Training data using evals Dataset
dataset = Dataset(samples=(
    Sample(id="1", input="What is 2+2?", expected="4"),
    Sample(id="2", input="Capital of France?", expected="Paris"),
))
trainset = dataset_to_trainset(dataset)
metric = evaluator_to_metric(exact_match)

# Convert prompt to DSPy module
module = prompt_to_module(my_prompt)

# Run DSPy optimization
optimizer = dspy.BootstrapFewShot(metric=metric)
optimized = optimizer.compile(module, trainset=trainset)

# Extract and persist
sections = extract_optimized_sections(my_prompt, optimized)
persist_to_overrides(my_prompt, sections, store, tag="bootstrap-v1")

# Use optimized prompt
optimized_prompt = Prompt(
    my_template,
    overrides_store=store,
    overrides_tag="bootstrap-v1",
).bind(params)
response = adapter.evaluate(optimized_prompt, session=session)
```

## Metrics via Evals

Use built-in evaluators from `weakincentives.evals`:

```python
from weakincentives.evals import exact_match, contains, all_of, llm_judge

# Exact match
optimizer = BootstrapFewShotOptimizer(
    context,
    dataset=dataset,
    evaluator=exact_match,
)

# Substring match
optimizer = BootstrapFewShotOptimizer(
    context,
    dataset=dataset,
    evaluator=contains,
)

# Composite criteria
optimizer = BootstrapFewShotOptimizer(
    context,
    dataset=dataset,
    evaluator=all_of(contains, llm_judge(judge_adapter, "factually accurate")),
)
```

## File Layout

```
src/weakincentives/optimizers/dspy/
├── __init__.py              # Public exports
├── _lm.py                   # WeakIncentivesLM
├── _bridge.py               # prompt_to_module, extract_optimized_sections
├── _data.py                 # dataset_to_trainset, evaluator_to_metric
├── _persist.py              # persist_to_overrides
└── _bootstrap_few_shot.py   # BootstrapFewShotOptimizer
```

## Constraints

- DSPy calls are synchronous; long optimizations block
- Tools in prompts are not represented in DSPy signatures
- DSPy's caching is independent of weakincentives

## Dependencies

Optional dependency:

```bash
uv add weakincentives[dspy]
# or
uv add dspy-ai>=2.5
```

Import raises `ImportError` with instructions if DSPy is not installed.
