# DSPy Integration Specification

## Purpose

Integrate DSPy as a foundational library for prompt optimization. This spec
defines the core building blocks: a language model adapter that routes DSPy
calls through weakincentives adapters, and the bridge layer that translates
between DSPy programs and weakincentives prompts.

## Guiding Principles

- **Adapter-first**: All DSPy LLM calls flow through weakincentives adapters,
  inheriting throttling, deadlines, and observability.
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
        bus: EventBus,
        session: SessionProtocol,
    ) -> None:
        self._adapter = adapter
        self._bus = bus
        self._session = session

    def __call__(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> list[str]:
        # Build a simple prompt from the DSPy request
        wink_prompt = self._build_prompt(prompt, **kwargs)

        # Execute through the adapter
        response = self._adapter.evaluate(
            wink_prompt,
            bus=self._bus,
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
- Event emission for observability
- Budget tracking

**Usage:**

```python
import dspy
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import Session, InProcessEventBus
from weakincentives.optimizers.dspy import WeakIncentivesLM

adapter = OpenAIAdapter(model="gpt-4o")
bus = InProcessEventBus()
session = Session(bus=bus)

# Configure DSPy to use our adapter
lm = WeakIncentivesLM(adapter, bus, session)
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

## BootstrapFewShotOptimizer

A `BasePromptOptimizer` implementation that wraps DSPy's BootstrapFewShot:

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
    """

    def __init__(
        self,
        context: OptimizationContext,
        *,
        trainset: Sequence[dspy.Example],
        metric: Callable[[dspy.Example, dspy.Prediction], bool | float],
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
    ) -> None:
        super().__init__(context)
        self._trainset = trainset
        self._metric = metric
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
        lm = WeakIncentivesLM(
            self._context.adapter,
            self._context.event_bus,
            session,
        )
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

context = OptimizationContext(
    adapter=adapter,
    event_bus=bus,
    overrides_store=store,
    overrides_tag="bootstrap-v1",
)

trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question"),
]

optimizer = BootstrapFewShotOptimizer(
    context,
    trainset=trainset,
    metric=lambda ex, pred: ex.answer == pred.answer,
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
from weakincentives.runtime import Session, InProcessEventBus
from weakincentives.optimizers.dspy import (
    WeakIncentivesLM,
    prompt_to_module,
    extract_optimized_sections,
    persist_to_overrides,
)

# Setup
adapter = OpenAIAdapter(model="gpt-4o")
bus = InProcessEventBus()
session = Session(bus=bus)
store = LocalPromptOverridesStore()

# Configure DSPy with our adapter
lm = WeakIncentivesLM(adapter, bus, session)
dspy.configure(lm=lm)

# Training examples
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question"),
]

# Convert prompt to DSPy module
module = prompt_to_module(my_prompt)

# Run DSPy optimization
optimizer = dspy.BootstrapFewShot(metric=lambda ex, pred: ex.answer == pred.answer)
optimized = optimizer.compile(module, trainset=trainset)

# Extract and persist
sections = extract_optimized_sections(my_prompt, optimized)
persist_to_overrides(my_prompt, sections, store, tag="bootstrap-v1")

# Use optimized prompt
response = adapter.evaluate(
    my_prompt,
    bus=bus,
    session=session,
    overrides_store=store,
    overrides_tag="bootstrap-v1",
)
```

## Metric Functions

DSPy optimizers require a metric. Use simple callables:

```python
# Exact match
def exact_match(example, prediction) -> bool:
    return example.answer == prediction.answer

# Fuzzy match
def contains_answer(example, prediction) -> bool:
    return example.answer.lower() in prediction.answer.lower()

# LLM-as-judge (uses the same adapter)
def llm_judge(example, prediction) -> float:
    # Evaluate using a separate prompt through the adapter
    ...
```

## File Layout

```
src/weakincentives/optimizers/dspy/
├── __init__.py              # Public exports
├── _lm.py                   # WeakIncentivesLM
├── _bridge.py               # prompt_to_module, extract_optimized_sections
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
