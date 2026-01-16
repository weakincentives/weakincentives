# Testing and Reliability

WINK is designed so that most of your "agent logic" is testable without a model.
This guide explains the testing patterns that make agents reliable.

## The Testing Pyramid for Agents

Most agent testing can happen without calling a model:

1. **Prompt rendering tests**: Fast, deterministic, catch template regressions
2. **Tool handler tests**: Test side-effect logic in isolation
3. **Reducer tests**: Test state transitions as pure functions
4. **Integration tests**: Call the model, slow and expensive, run selectively

## Prompt Rendering Tests

Prompts are deterministic. Same inputs produce same outputs. This makes them
perfect for snapshot testing:

```python
from dataclasses import dataclass
from typing import Any
from weakincentives.prompt import Prompt, PromptTemplate, MarkdownSection
from weakincentives.runtime import Session


@dataclass(frozen=True)
class TestParams:
    question: str


template = PromptTemplate[Any](
    ns="test",
    key="snapshot",
    sections=(MarkdownSection(title="Q", key="q", template="Question: ${question}"),),
)
prompt = Prompt(template)
session = Session()


def test_prompt_renders_stably():
    rendered = prompt.bind(TestParams(question="x")).render(session=session)
    assert "Question: x" in rendered.text
```

The test doesn't call a model. It just verifies that the prompt renders as
expected. When prompts are deterministic, you can test them like regular code.

**What to test:**

- Placeholder substitution works correctly
- Conditional sections enable/disable as expected
- Tools are included when they should be
- Visibility overrides affect rendering correctly

## Tool Handler Tests

Call handlers directly with fake `ToolContext`:

```python
from weakincentives.prompt import ToolContext, ToolResult
from weakincentives.runtime import Session


def test_search_handler_success():
    context = ToolContext(
        prompt=None,
        rendered_prompt=None,
        adapter=None,
        session=Session(),
        resources=None,
    )

    result = search_handler(SearchParams(query="test"), context=context)

    assert result.success
    assert "test" in result.value.snippets[0]


def test_search_handler_error():
    context = ToolContext(...)

    result = search_handler(SearchParams(query=""), context=context)

    assert not result.success
    assert "empty query" in result.message.lower()
```

No model needed. You're testing the business logic in isolation.

## Reducer Tests

Reducers are pure functions. Test them directly:

```python
from weakincentives.runtime.session import SliceView, Append

def test_add_step_reducer():
    initial_state = SliceView((Plan(steps=()),))
    event = AddStep(step="read README")

    result = add_step_reducer(initial_state, event)

    assert isinstance(result, Append)
    assert result.value.steps == ("read README",)
```

Given this slice and this event, expect this new slice. Pure functions are easy
to test exhaustively.

## Integration Tests

Run `adapter.evaluate` behind a flag:

```python
import os
import pytest
from weakincentives.adapters.openai import OpenAIAdapter


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY"
)
def test_agent_answers_question():
    adapter = OpenAIAdapter(model="gpt-4o-mini")
    session = Session()

    prompt = Prompt(template).bind(QuestionParams(question="What is 2+2?"))
    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None
    assert "4" in response.output.answer
```

These are slow and cost money, so run them selectively:

- In CI, on pull requests to main
- Locally, when specifically requested
- Never as part of `make test` (that should be fast and free)

## Testing with Overrides

Test that overrides are applied correctly:

```python
def test_override_changes_prompt():
    store = InMemoryOverridesStore()
    store.save(override_with_custom_text)

    prompt = Prompt(template, overrides_store=store, overrides_tag="test")
    rendered = prompt.bind(params).render(session=session)

    assert "custom text" in rendered.text
```

## Testing Session State

Test that session state evolves correctly:

```python
def test_tool_invocation_updates_session():
    session = Session()
    session.install(Plan, initial=lambda: Plan(steps=()))

    # Simulate tool calls
    session.dispatch(AddStep(step="step 1"))
    session.dispatch(AddStep(step="step 2"))

    plan = session[Plan].latest()
    assert plan.steps == ("step 1", "step 2")
```

## What the WINK Codebase Does

WINK itself enforces:

- **100% coverage** for `src/weakincentives/`
- **Strict pytest config** with `--strict-config --strict-markers`
- **Automatic retries** for flaky tests (twice)

Run focused tests during development:

```bash
uv run pytest tests/path/to/test.py -v
```

Always finish with `make test` to verify coverage.

## Next Steps

- [Code Quality](code-quality.md): Types, contracts, and security scanning
- [Evaluation](evaluation.md): Systematic testing with datasets
- [Debugging](debugging.md): Understand failures when they happen
