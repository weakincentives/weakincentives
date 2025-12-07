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
from weakincentives.dspy import WeakIncentivesLM

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

## Example: BootstrapFewShot Optimization

Using the building blocks to run DSPy's BootstrapFewShot:

```python
import dspy
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.prompt.overrides import LocalPromptOverridesStore
from weakincentives.runtime import Session, InProcessEventBus
from weakincentives.dspy import (
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
src/weakincentives/dspy/
├── __init__.py          # Public exports
├── _lm.py               # WeakIncentivesLM
├── _bridge.py           # prompt_to_module, extract_optimized_sections
└── _persist.py          # persist_to_overrides
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
