# Adapter Implementation Guide

## Purpose

This guide provides a minimal example of implementing a custom provider adapter
and documents common pitfalls. Use this when integrating a new LLM provider with
WINK.

## Minimal Adapter Example (~50 lines)

```python
"""Minimal adapter implementation for a hypothetical provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from weakincentives.adapters.core import (
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
)
from weakincentives.adapters._provider_protocols import (
    ProviderChoiceData,
    ProviderFunctionCallData,
    ProviderMessageData,
    ProviderToolCallData,
)
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.dataclasses import FrozenDataclass
from weakincentives.deadlines import Deadline
from weakincentives.prompt import Prompt
from weakincentives.prompt.tool import ResourceRegistry

if TYPE_CHECKING:
    from weakincentives.runtime.session.protocols import SessionProtocol

OutputT = TypeVar("OutputT")


@FrozenDataclass()
class MinimalConfig:
    """Configuration for the minimal adapter."""

    api_key: str | None = None
    timeout: float = 30.0


class MinimalAdapter(ProviderAdapter):
    """Minimal adapter demonstrating the required contract."""

    def __init__(
        self,
        model: str,
        *,
        config: MinimalConfig | None = None,
    ) -> None:
        self._model = model
        self._config = config or MinimalConfig()

    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        resources: ResourceRegistry | None = None,
    ) -> PromptResponse[OutputT]:
        # 1. Render the prompt
        rendered = prompt.render(session=session)

        # 2. Check deadline before calling provider
        if deadline and deadline.remaining().total_seconds() <= 0:
            raise PromptEvaluationError(
                "Deadline exceeded before provider call",
                prompt_name=prompt.name,
                phase="request",
            )

        # 3. Call the provider (simplified)
        try:
            response = self._call_provider(rendered.text)
        except Exception as e:
            raise PromptEvaluationError(
                f"Provider call failed: {e}",
                prompt_name=prompt.name,
                phase="request",
            ) from e

        # 4. Parse structured output if configured
        output = None
        if rendered.structured_output:
            output = self._parse_output(response, rendered.output_type)

        return PromptResponse(
            prompt_name=prompt.name,
            text=response,
            output=output,
        )

    def _call_provider(self, prompt_text: str) -> str:
        """Call the actual provider API."""
        # Implementation would go here
        raise NotImplementedError("Implement provider-specific logic")

    def _parse_output(self, text: str, output_type: type | None) -> object:
        """Parse structured output from response text."""
        # Implementation would go here
        raise NotImplementedError("Implement output parsing")
```

## Required Contract

All adapters must implement the `ProviderAdapter` abstract base class:

```python
class ProviderAdapter(ABC):
    @abstractmethod
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        resources: ResourceRegistry | None = None,
    ) -> PromptResponse[OutputT]: ...
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | `Prompt[OutputT]` | Yes | Prompt to evaluate |
| `session` | `SessionProtocol` | Yes | Session for state and events |
| `deadline` | `Deadline \| None` | No | Wall-clock time limit |
| `budget` | `Budget \| None` | No | Token/time budget limits |
| `budget_tracker` | `BudgetTracker \| None` | No | Shared tracker for limits |
| `resources` | `ResourceRegistry \| None` | No | Custom resources for tools |

### Return Value

```python
@FrozenDataclass()
class PromptResponse[OutputT]:
    prompt_name: str      # Name of the evaluated prompt
    text: str | None      # Raw text response from provider
    output: OutputT | None  # Parsed structured output
```

## Implementation Checklist

### 1. Configuration

Use frozen dataclasses for type-safe configuration:

```python
@FrozenDataclass()
class MyProviderConfig:
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 30.0
    max_retries: int = 3
```

### 2. Dynamic SDK Import

Import provider SDK lazily with helpful error messages:

```python
def _get_client(self):
    try:
        from myprovider import Client
    except ImportError as e:
        raise RuntimeError(
            "Install myprovider: pip install weakincentives[myprovider]"
        ) from e
    return Client(api_key=self._config.api_key)
```

### 3. Prompt Rendering

Render the prompt once, respecting session state:

```python
rendered = prompt.render(session=session)

# Access rendered components
text = rendered.text                    # Markdown prompt text
tools = rendered.tools                  # Tools from enabled sections
output_type = rendered.output_type      # Structured output type
deadline = rendered.deadline or deadline  # Prefer prompt deadline
```

### 4. Deadline Enforcement

Check deadlines at key points:

```python
def _check_deadline(self, deadline: Deadline | None, prompt_name: str) -> None:
    if deadline and deadline.remaining().total_seconds() <= 0:
        raise PromptEvaluationError(
            "Deadline exceeded",
            prompt_name=prompt_name,
            phase="request",
        )
```

Checkpoints:

- Before each provider call
- Before each tool execution
- Before response parsing

### 5. Budget Tracking

Track and enforce token budgets:

```python
def evaluate(self, prompt, *, budget=None, budget_tracker=None, **kwargs):
    # Create tracker if budget provided without tracker
    if budget and not budget_tracker:
        budget_tracker = BudgetTracker(budget)

    # After each provider response
    if budget_tracker:
        budget_tracker.record_cumulative(evaluation_id, token_usage)
        budget_tracker.check()  # Raises BudgetExceededError if exceeded
```

### 6. Tool Execution

Execute tools using the shared inner loop or custom logic:

```python
from weakincentives.adapters.shared import run_inner_loop

response = run_inner_loop(
    prompt=prompt,
    rendered=rendered,
    session=session,
    call_provider=self._call_provider,
    select_choice=lambda r: r.choices[0],
    deadline=deadline,
    budget_tracker=budget_tracker,
    resources=resources,
)
```

### 7. Event Emission

Publish telemetry events through the session dispatcher:

```python
from weakincentives.runtime.events import PromptRendered, PromptExecuted

# After rendering
session.dispatcher.dispatch(PromptRendered(
    prompt_ns=prompt.ns,
    prompt_key=prompt.key,
    prompt_name=prompt.name,
    adapter=self._adapter_name,
    session_id=session.session_id,
    render_inputs=tuple(params),
    rendered_prompt=rendered.text,
    created_at=datetime.now(UTC),
))

# After evaluation
session.dispatcher.dispatch(PromptExecuted(
    prompt_name=prompt.name,
    adapter=self._adapter_name,
    result=response.text,
    session_id=session.session_id,
    created_at=datetime.now(UTC),
    usage=token_usage,
    value=response.output,
))
```

### 8. Error Handling

Wrap provider errors with context:

```python
try:
    response = client.complete(...)
except ProviderRateLimitError as e:
    raise ThrottleError(
        "Rate limited",
        prompt_name=prompt.name,
        phase="request",
        details=ThrottleDetails(
            kind="rate_limit",
            retry_after=e.retry_after,
            attempts=attempt,
            retry_safe=True,
            provider_payload={"error": str(e)},
        ),
    ) from e
except ProviderError as e:
    raise PromptEvaluationError(
        f"Provider error: {e}",
        prompt_name=prompt.name,
        phase="request",
        provider_payload={"error": str(e)},
    ) from e
```

## Common Pitfalls

### 1. Not Checking Deadlines

**Problem**: Long-running evaluations ignore deadline constraints.

**Solution**: Check `deadline.remaining()` before provider calls and tool
execution.

```python
# Wrong - no deadline check
response = client.complete(prompt)

# Right - check before call
if deadline and deadline.remaining().total_seconds() <= 0:
    raise PromptEvaluationError(...)
response = client.complete(prompt)
```

### 2. Ignoring Session State

**Problem**: Not using session for visibility overrides or state queries.

**Solution**: Pass session to `prompt.render()` and tool handlers.

```python
# Wrong - ignores visibility overrides
rendered = prompt.render()

# Right - respects session state
rendered = prompt.render(session=session)
```

### 3. Swallowing Provider Errors

**Problem**: Generic `except Exception` loses error context.

**Solution**: Re-raise as `PromptEvaluationError` with phase and payload.

```python
# Wrong - loses context
try:
    response = client.complete(prompt)
except Exception:
    return PromptResponse(prompt_name=..., text=None, output=None)

# Right - preserves context
try:
    response = client.complete(prompt)
except Exception as e:
    raise PromptEvaluationError(
        str(e),
        prompt_name=prompt.name,
        phase="request",
        provider_payload={"error": str(e)},
    ) from e
```

### 4. Not Merging Resources

**Problem**: Custom resources don't merge with workspace defaults.

**Solution**: Use `ResourceRegistry.merge()` for layered resources.

```python
# Wrong - loses workspace resources
tool_resources = user_resources

# Right - merge with workspace defaults
workspace_resources = self._build_workspace_resources(prompt)
tool_resources = workspace_resources.merge(user_resources or ResourceRegistry())
```

### 5. Blocking on Tool Failures

**Problem**: Tool exceptions abort the entire evaluation.

**Solution**: Convert tool exceptions to `ToolResult(success=False)`.

```python
# Wrong - exception aborts evaluation
result = handler(params, context=context)

# Right - capture as failed result
try:
    result = handler(params, context=context)
except Exception as e:
    result = ToolResult(message=str(e), success=False)
```

### 6. Not Publishing Events

**Problem**: Missing telemetry makes debugging difficult.

**Solution**: Publish `PromptRendered`, `ToolInvoked`, and `PromptExecuted`.

### 7. Hardcoding Provider SDK

**Problem**: Tight coupling to specific SDK versions.

**Solution**: Use protocols for structural typing.

```python
# Wrong - requires specific OpenAI version
from openai import OpenAI

# Right - structural typing accepts compatible clients
class _ClientProtocol(Protocol):
    def create(self, **kwargs) -> ProviderCompletionResponse: ...
```

### 8. Not Handling Throttling

**Problem**: Rate limits cause immediate failures.

**Solution**: Implement exponential backoff with jitter.

```python
from weakincentives.adapters.shared import ThrottlePolicy, new_throttle_policy

policy = new_throttle_policy(
    max_attempts=5,
    base_delay=timedelta(milliseconds=500),
    max_delay=timedelta(seconds=8),
)
```

### 9. Mutating Session State Directly

**Problem**: Direct mutations bypass reducers and break determinism.

**Solution**: Use `session.dispatch()` for state changes.

```python
# Wrong - bypasses reducers
session._slices[Plan] = (new_plan,)

# Right - goes through dispatch
session.dispatch(UpdatePlan(plan=new_plan))
```

### 10. Forgetting Structured Output

**Problem**: Ignoring `rendered.structured_output` when present.

**Solution**: Parse and validate structured output from response.

```python
if rendered.structured_output:
    # Use provider-native JSON schema if supported
    response_format = build_json_schema(rendered.output_type)

    # Or parse from text
    output = parse_json_output(response.text, rendered.output_type)
```

## Testing Adapters

### Unit Tests

```python
def test_adapter_calls_provider():
    adapter = MinimalAdapter(model="test")
    prompt = Prompt[str](ns="test", key="test", sections=[...])

    # Mock the provider call
    adapter._call_provider = Mock(return_value="response text")

    session = Session()
    response = adapter.evaluate(prompt, session=session)

    assert response.text == "response text"
    adapter._call_provider.assert_called_once()
```

### Deadline Tests

```python
def test_adapter_respects_deadline():
    adapter = MinimalAdapter(model="test")
    prompt = Prompt[str](...)

    # Expired deadline
    expired = Deadline(expires_at=datetime.now(UTC) - timedelta(seconds=1))

    with pytest.raises(PromptEvaluationError) as exc_info:
        adapter.evaluate(prompt, session=Session(), deadline=expired)

    assert exc_info.value.phase == "request"
```

### Tool Execution Tests

```python
def test_adapter_executes_tools():
    tool = Tool(name="test_tool", handler=mock_handler)
    section = MarkdownSection(..., tools=(tool,))
    prompt = Prompt[str](..., sections=[section])

    # Configure mock to return tool call
    adapter._call_provider = Mock(side_effect=[
        create_tool_call_response("test_tool", '{"arg": "value"}'),
        create_text_response("final response"),
    ])

    response = adapter.evaluate(prompt, session=Session())

    assert mock_handler.called
```

## Related Specifications

- `specs/ADAPTERS.md` - Full adapter specification
- `specs/TOOLS.md` - Tool execution details
- `specs/SESSIONS.md` - Session state management
- `ARCHITECTURE.md` - Why protocols over ABCs (ADR-001)
