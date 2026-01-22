# Errors and Exceptions

*Canonical spec: [specs/ERRORS.md](../specs/ERRORS.md)*

WINK uses a structured approach to error handling built around two principles:

> **Exceptions are for infrastructure failures. ToolResult is for operational
> failures.**

When something breaks in the plumbing (can't connect to provider, invalid prompt
configuration, resource resolution failure), you get an exception. When a tool
can't complete its job (file not found, validation failed, API returned an
error), you get a `ToolResult.error()`. This distinction matters because models
can recover from tool failures but not from infrastructure failures.

## The Exception Hierarchy

All WINK exceptions inherit from `WinkError`, letting you catch library errors
in one place:

```python nocheck
from weakincentives import WinkError

try:
    response = adapter.evaluate(prompt, session=session)
except WinkError as e:
    # Handle any WINK error
    logger.error("WINK operation failed", error=str(e))
```

### Core Exceptions

| Exception | Inherits | When Raised |
| --- | --- | --- |
| `WinkError` | `Exception` | Base class for all library exceptions |
| `ToolValidationError` | `WinkError`, `ValueError` | Tool parameters fail validation |
| `DeadlineExceededError` | `WinkError`, `RuntimeError` | Execution exceeds deadline |
| `SnapshotError` | `WinkError`, `RuntimeError` | Base for snapshot failures |
| `SnapshotRestoreError` | `SnapshotError` | Snapshot restoration fails |
| `TransactionError` | `WinkError`, `RuntimeError` | Base for transaction failures |
| `RestoreFailedError` | `TransactionError` | Rollback during transaction fails |

### Prompt Errors

```python nocheck
from weakincentives.prompt import (
    PromptError,
    PromptValidationError,
    PromptRenderError,
    VisibilityExpansionRequired,
)
```

| Exception | When Raised |
| --- | --- |
| `PromptError` | Base class with structured context |
| `PromptValidationError` | Prompt construction validation fails |
| `PromptRenderError` | Template rendering fails |
| `VisibilityExpansionRequired` | Model requests section expansion |

Prompt errors carry contextual information:

```python nocheck
try:
    rendered = prompt.render(session=session)
except PromptValidationError as e:
    print(f"Section: {'.'.join(e.section_path)}")
    print(f"Placeholder: {e.placeholder}")
    print(f"Dataclass: {e.dataclass_type}")
```

### Adapter Errors

```python nocheck
from weakincentives.adapters import PromptEvaluationError
from weakincentives.adapters.throttle import ThrottleError
```

`PromptEvaluationError` tells you exactly where evaluation failed:

| Phase | Meaning |
| --- | --- |
| `"request"` | Failed sending request to provider |
| `"response"` | Failed parsing provider response |
| `"tool"` | Failed during tool execution |
| `"budget"` | Budget limits exceeded |

```python nocheck
try:
    response = adapter.evaluate(prompt, session=session)
except PromptEvaluationError as e:
    print(f"Prompt: {e.prompt_name}")
    print(f"Phase: {e.phase}")
    print(f"Payload: {e.provider_payload}")  # Provider-specific details
```

`ThrottleError` extends `PromptEvaluationError` with retry information:

```python nocheck
from weakincentives.adapters.throttle import ThrottleError, ThrottleKind

try:
    response = adapter.evaluate(prompt, session=session)
except ThrottleError as e:
    if e.details.retry_safe:
        # Safe to retry after delay
        time.sleep(e.details.retry_after.total_seconds())
    print(f"Kind: {e.details.kind}")  # rate_limit, quota_exhausted, timeout
    print(f"Attempts: {e.details.attempts}")
```

### Resource Errors

```python nocheck
from weakincentives.resources import (
    ResourceError,
    UnboundResourceError,
    CircularDependencyError,
    DuplicateBindingError,
    ProviderError,
)
```

| Exception | When Raised |
| --- | --- |
| `UnboundResourceError` | No binding for requested protocol |
| `CircularDependencyError` | A depends on B depends on A |
| `DuplicateBindingError` | Same protocol bound twice |
| `ProviderError` | Provider factory raised exception |

```python nocheck
try:
    client = prompt.resources.get(HTTPClient)
except UnboundResourceError as e:
    print(f"Missing binding for: {e.protocol.__name__}")
except CircularDependencyError as e:
    print(f"Cycle: {' -> '.join(t.__name__ for t in e.cycle)}")
```

## ToolResult: The Non-Exception Pattern

Tool handlers **never raise exceptions for operational failures**. Instead, they
return `ToolResult` objects:

```python nocheck
from weakincentives.prompt import ToolResult

def read_file(params: ReadParams, *, context: ToolContext) -> ToolResult[FileContent]:
    path = params.path
    if not path.exists():
        return ToolResult.error(f"File not found: {path}")

    content = path.read_text()
    return ToolResult.ok(FileContent(text=content), message="Read complete")
```

**Why this pattern?**

1. **Models can recover**: A failed `ToolResult` tells the model "try something
   else" rather than aborting the entire evaluation
1. **Consistent handling**: All tool outcomes flow through the same interface
1. **Transactional safety**: Failed tools trigger automatic state rollback

### ToolResult Fields

```python nocheck
@dataclass(slots=True)
class ToolResult[ResultValueT]:
    message: str              # Text forwarded to the model
    value: ResultValueT | None  # Typed payload (None on failure)
    success: bool = True      # False = operational failure
    exclude_value_from_context: bool = False  # Hide large payloads
```

### Factory Methods

```python nocheck
# Success with typed value
ToolResult.ok(MyResult(...), message="Done")

# Failure with descriptive message
ToolResult.error("Permission denied")
```

### What Happens on Tool Failure

When a tool returns `ToolResult.error()`:

1. WINK logs the failure with structured context
1. Session state is rolled back to pre-tool snapshot
1. The error message is sent to the model
1. Evaluation continues (model can try different approach)

Compare this to an unhandled exception:

1. Exception propagates up
1. Evaluation aborts entirely
1. No recovery possible

## Handling Exceptions in Tool Handlers

While tool handlers shouldn't raise exceptions for expected failures, they may
encounter unexpected exceptions. WINK catches these and converts them to tool
failures:

```python nocheck
def my_handler(params: Params, *, context: ToolContext) -> ToolResult[Result]:
    try:
        # External call that might fail unexpectedly
        data = external_api.fetch(params.query)
        return ToolResult.ok(Result(data=data))
    except requests.Timeout:
        # Expected failure: return error
        return ToolResult.error("API timeout - try again later")
    # Unexpected exceptions propagate up and become tool failures
```

**Best practice**: Catch expected failure modes explicitly and return meaningful
`ToolResult.error()` messages. Let unexpected exceptions propagate—WINK will
convert them to failures and log them for debugging.

## Error Context and Debugging

WINK errors carry structured context for debugging:

```python nocheck
try:
    response = adapter.evaluate(prompt, session=session)
except PromptEvaluationError as e:
    # Rich context for debugging
    context = {
        "prompt": e.prompt_name,
        "phase": e.phase,
        "provider_details": e.provider_payload,
    }
    logger.error("Evaluation failed", extra=context)
```

Enable debug logging to see error flow:

```python nocheck
from weakincentives.runtime import configure_logging
configure_logging(level="DEBUG")
```

## Exception Chaining

WINK preserves exception causes for debugging:

```python nocheck
try:
    params = parse_tool_params(tool, arguments)
except ToolValidationError as e:
    # e.__cause__ contains the original TypeError or ValueError
    print(f"Original cause: {e.__cause__}")
```

## Visibility Expansion Pattern

`VisibilityExpansionRequired` is a special exception for progressive disclosure.
When a model requests expansion of summarized sections:

```python nocheck
from weakincentives.prompt import VisibilityExpansionRequired

while True:
    try:
        response = adapter.evaluate(prompt, session=session)
        break
    except VisibilityExpansionRequired as e:
        # Apply requested overrides and retry
        for path, visibility in e.requested_overrides.items():
            session[VisibilityOverrides].set(path, visibility)
        # Loop continues with expanded sections
```

## Common Error Scenarios

| Error | Likely Cause | Fix |
| --- | --- | --- |
| `ToolValidationError` | Model sent invalid params | Improve tool description/examples |
| `UnboundResourceError` | Missing resource binding | Add binding to prompt.bind() |
| `PromptValidationError` | Template/dataclass mismatch | Check placeholder names match fields |
| `DeadlineExceededError` | Tool or model too slow | Increase deadline or optimize tools |
| `ThrottleError` | Provider rate limit | Implement backoff/retry logic |

## Best Practices

1. **Catch `WinkError` at the top level** for graceful degradation
1. **Use `ToolResult.error()` for recoverable failures** in tool handlers
1. **Let unexpected exceptions propagate** for proper logging
1. **Check error context** (`phase`, `section_path`, etc.) when debugging
1. **Implement retry logic** for `ThrottleError` with `retry_safe=True`
1. **Preserve exception chains** when wrapping errors

## Next Steps

- [Troubleshooting](troubleshooting.md): Common issues and fixes
- [Debugging](debugging.md): Inspect sessions and trace errors
- [Tools](tools.md): Understand ToolResult semantics
