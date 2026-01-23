# Testing and Reliability

WINK is designed so that most of your "agent logic" is testable without a model.
This guide explains the testing patterns that make agents reliable.

## The Testing Pyramid for Agents

Most agent testing can happen without calling a model:

1. **Prompt rendering tests**: Fast, deterministic, catch template regressions
1. **Tool handler tests**: Test side-effect logic in isolation
1. **Reducer tests**: Test state transitions as pure functions
1. **Integration tests**: Call the model, slow and expensive, run selectively

## Prompt Rendering Tests

Prompts are deterministic. Same inputs produce same outputs. This makes them
perfect for snapshot testing:

```python nocheck
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

```python nocheck
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

```python nocheck
from weakincentives.runtime.session import Append

def test_add_step_reducer():
    # Provide the current slice contents as a tuple
    current_state = (Plan(steps=()),)
    event = AddStep(step="read README")

    result = add_step_reducer(current_state, event)

    assert isinstance(result, Append)
    assert result.item.steps == ("read README",)
```

Given this slice and this event, expect this new slice. Pure functions are easy
to test exhaustively.

## Integration Tests

Run `adapter.evaluate` behind a flag:

```python nocheck
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

```python nocheck
def test_override_changes_prompt():
    store = InMemoryOverridesStore()
    store.save(override_with_custom_text)

    prompt = Prompt(template, overrides_store=store, overrides_tag="test")
    rendered = prompt.bind(params).render(session=session)

    assert "custom text" in rendered.text
```

## Testing Session State

Test that session state evolves correctly:

```python nocheck
def test_tool_invocation_updates_session():
    session = Session()
    session.install(Plan, initial=lambda: Plan(steps=()))

    # Simulate tool calls
    session.dispatch(AddStep(step="step 1"))
    session.dispatch(AddStep(step="step 2"))

    plan = session[Plan].latest()
    assert plan.steps == ("step 1", "step 2")
```

## Testing Time-Dependent Code with FakeClock

Time-dependent code (deadlines, timeouts, rate limiting) needs deterministic
testing. WINK provides `FakeClock` to advance time instantly without real delays:

```python nocheck
from weakincentives.clock import FakeClock

def test_timeout_logic():
    clock = FakeClock()

    start = clock.monotonic()
    clock.sleep(10)  # Advances instantly, no real delay
    assert clock.monotonic() - start == 10

    # Or advance manually
    clock.advance(60)
    assert clock.monotonic() - start == 70
```

**Key methods:**

| Method | Description |
| --- | --- |
| `monotonic()` | Return current monotonic time (seconds) |
| `utcnow()` | Return current wall-clock time (UTC datetime) |
| `sleep(seconds)` | Advance time instantly (calls `advance()`) |
| `advance(seconds)` | Advance both clocks by duration |
| `set_monotonic(value)` | Set monotonic time to absolute value |
| `set_wall(value)` | Set wall-clock time (must be timezone-aware) |

### Testing Deadlines

```python nocheck
from datetime import UTC, datetime, timedelta
from weakincentives import Deadline
from weakincentives.clock import FakeClock

def test_deadline_remaining():
    clock = FakeClock()
    clock.set_wall(datetime(2024, 6, 1, 12, 0, tzinfo=UTC))

    deadline = Deadline(
        expires_at=datetime(2024, 6, 1, 13, 0, tzinfo=UTC),
        clock=clock,
    )

    assert deadline.remaining() == timedelta(hours=1)

    clock.advance(1800)  # 30 minutes
    assert deadline.remaining() == timedelta(minutes=30)
```

### Using the Pytest Fixture

WINK provides a `fake_clock` fixture for convenience:

```python nocheck
from weakincentives.clock import FakeClock
from tests.helpers.time import fake_clock  # pytest fixture

def test_heartbeat_elapsed(fake_clock: FakeClock) -> None:
    hb = Heartbeat(clock=fake_clock)

    hb.beat()
    assert hb.elapsed() == 0.0

    fake_clock.advance(10)
    assert hb.elapsed() == 10.0
```

All clock-dependent components in WINK accept a `clock` parameter with a
sensible default (`SYSTEM_CLOCK`). Inject `FakeClock` for testing:

```python nocheck
# Production: uses real system time
deadline = Deadline(expires_at=datetime(2024, 12, 31, tzinfo=UTC))

# Testing: inject FakeClock for deterministic behavior
clock = FakeClock()
deadline = Deadline(expires_at=..., clock=clock)
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
