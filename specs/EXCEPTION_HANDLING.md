# Exception Handling Specification

## Purpose

This document catalogs all exception types in WINK, describes which APIs throw
them, and provides troubleshooting guidance for common error scenarios.

## Exception Hierarchy

```
WinkError (base)
├── ToolValidationError
├── DeadlineExceededError
├── SnapshotError
│   └── SnapshotRestoreError
├── ExecutionStateError
│   ├── SnapshotMismatchError
│   └── RestoreFailedError
├── PromptError
│   ├── PromptValidationError
│   ├── PromptRenderError
│   └── VisibilityExpansionRequired
├── PromptEvaluationError
│   └── ThrottleError
├── OutputParseError
├── BudgetExceededError
├── PromptOverridesError
├── MailboxError
│   ├── ReceiptHandleExpiredError
│   ├── MailboxFullError
│   ├── SerializationError
│   └── MailboxConnectionError
├── SnapshotSerializationError
├── WorkspaceBudgetExceededError
└── WorkspaceSecurityError
```

## Exception Reference

### Core Exceptions (`weakincentives.errors`)

#### WinkError

Base class for all library exceptions. Catch this to handle any WINK error:

```python
try:
    response = adapter.evaluate(prompt, session=session)
except WinkError as e:
    logger.error(f"WINK operation failed: {e}")
```

**Location**: `src/weakincentives/errors.py:18`

---

#### ToolValidationError

**Inherits**: `WinkError`, `ValueError`

**Raised when**: Tool parameters fail validation.

**APIs that throw**:

- `Tool.validate_params()` - Invalid parameter structure
- Tool handlers - Via `ToolContext.validate()`

**Example**:

```python
from weakincentives.errors import ToolValidationError

try:
    result = tool.execute(invalid_params, context=context)
except ToolValidationError as e:
    print(f"Invalid params: {e}")
```

**Location**: `src/weakincentives/errors.py:31`

---

#### DeadlineExceededError

**Inherits**: `WinkError`, `RuntimeError`

**Raised when**: Operation cannot complete before deadline.

**APIs that throw**:

- `ProviderAdapter.evaluate()` - Deadline check before provider call
- Tool handlers - Via `ToolContext.check_deadline()`
- `ToolExecutor.execute()` - Pre-execution deadline check

**Example**:

```python
from weakincentives.errors import DeadlineExceededError

try:
    response = adapter.evaluate(prompt, session=session, deadline=deadline)
except DeadlineExceededError:
    print("Operation timed out")
```

**Location**: `src/weakincentives/errors.py:35`

---

#### SnapshotError

**Inherits**: `WinkError`, `RuntimeError`

**Base class for**: Snapshot-related errors.

**Location**: `src/weakincentives/errors.py:39`

---

#### SnapshotRestoreError

**Inherits**: `SnapshotError`

**Raised when**: Restoring from a snapshot fails.

**APIs that throw**:

- `Session.restore()` - Incompatible snapshot
- `Snapshot.from_json()` - Invalid JSON structure

**Example**:

```python
from weakincentives.errors import SnapshotRestoreError

try:
    session.restore(snapshot)
except SnapshotRestoreError as e:
    print(f"Cannot restore: {e}")
```

**Location**: `src/weakincentives/errors.py:43`

---

#### ExecutionStateError

**Inherits**: `WinkError`, `RuntimeError`

**Base class for**: Execution state container errors.

**Location**: `src/weakincentives/errors.py:47`

---

#### SnapshotMismatchError

**Inherits**: `ExecutionStateError`

**Raised when**: Snapshot structure doesn't match current state.

**APIs that throw**:

- `ExecutionState.restore()` - Incompatible snapshot structure

**Location**: `src/weakincentives/errors.py:51`

---

#### RestoreFailedError

**Inherits**: `ExecutionStateError`

**Raised when**: State restoration fails mid-operation.

**APIs that throw**:

- `ExecutionState.restore()` - Partial restore failure

**Location**: `src/weakincentives/errors.py:55`

---

### Prompt Exceptions (`weakincentives.prompt.errors`)

#### PromptError

**Inherits**: `WinkError`

**Base class for**: All prompt-related errors.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Error description |
| `section_path` | `tuple[str, ...]` | Path to failing section |
| `dataclass_type` | `type \| None` | Related dataclass type |
| `placeholder` | `str \| None` | Template placeholder involved |

**Location**: `src/weakincentives/prompt/errors.py:32`

---

#### PromptValidationError

**Inherits**: `PromptError`

**Raised when**: Prompt construction validation fails.

**APIs that throw**:

- `Prompt.__init__()` - Invalid section structure
- `Prompt.bind()` - Missing required parameters
- `PromptRenderer.build_param_lookup()` - Duplicate or unexpected params

**Examples**:

```python
# Duplicate params type
PromptValidationError("Duplicate params type supplied to prompt.")

# Unexpected params type
PromptValidationError("Unexpected params type supplied to prompt.")

# Non-dataclass instance
PromptValidationError("Prompt expects dataclass instances.")
```

**Location**: `src/weakincentives/prompt/errors.py:50`

---

#### PromptRenderError

**Inherits**: `PromptError`

**Raised when**: Rendering a prompt fails.

**APIs that throw**:

- `PromptRenderer.render()` - Template substitution failure
- `Section.render_body()` - Section-specific rendering failure

**Example**:

```python
from weakincentives.prompt.errors import PromptRenderError

try:
    rendered = prompt.render(session=session)
except PromptRenderError as e:
    print(f"Section {'.'.join(e.section_path)} failed: {e.message}")
```

**Location**: `src/weakincentives/prompt/errors.py:54`

---

#### VisibilityExpansionRequired

**Inherits**: `PromptError`

**Raised when**: Model requests section expansion.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `requested_overrides` | `Mapping[SectionPath, SectionVisibility]` | Sections to expand |
| `reason` | `str` | Why expansion was requested |
| `section_keys` | `tuple[str, ...]` | Affected section keys |

**APIs that throw**:

- `open_sections` tool handler - Model requested tool access
- `MainLoop.execute()` - During visibility expansion retry

**Handling**:

```python
from weakincentives.prompt.errors import VisibilityExpansionRequired

try:
    response = adapter.evaluate(prompt, session=session)
except VisibilityExpansionRequired as e:
    # Update session with requested overrides
    session[VisibilityOverrides].seed(
        VisibilityOverrides(overrides=dict(e.requested_overrides))
    )
    # Retry evaluation
    response = adapter.evaluate(prompt, session=session)
```

**Location**: `src/weakincentives/prompt/errors.py:58`

---

### Adapter Exceptions (`weakincentives.adapters`)

#### PromptEvaluationError

**Inherits**: `WinkError`, `RuntimeError`

**Raised when**: Evaluation against a provider fails.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Error description |
| `prompt_name` | `str` | Name of the failing prompt |
| `phase` | `PromptEvaluationPhase` | When error occurred |
| `provider_payload` | `dict \| None` | Provider-specific details |

**Phases**:

| Phase | Description |
|-------|-------------|
| `"request"` | Error issuing provider request |
| `"response"` | Error handling provider response |
| `"tool"` | Error during tool execution |
| `"budget"` | Budget limit exceeded |

**APIs that throw**:

- `ProviderAdapter.evaluate()` - Any evaluation failure
- `OpenAIAdapter.evaluate()` - OpenAI-specific errors
- `LiteLLMAdapter.evaluate()` - LiteLLM errors

**Example**:

```python
from weakincentives.adapters.core import PromptEvaluationError

try:
    response = adapter.evaluate(prompt, session=session)
except PromptEvaluationError as e:
    print(f"Evaluation failed in {e.phase} phase: {e.message}")
    if e.provider_payload:
        print(f"Provider details: {e.provider_payload}")
```

**Location**: `src/weakincentives/adapters/core.py:76`

---

#### ThrottleError

**Inherits**: `PromptEvaluationError`

**Raised when**: Provider rate limits or throttles request.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `details` | `ThrottleDetails` | Throttling details |
| `kind` | `ThrottleKind` | Type of throttling |
| `retry_after` | `timedelta \| None` | Suggested wait time |
| `attempts` | `int` | Number of attempts made |
| `retry_safe` | `bool` | Whether retry is safe |

**ThrottleKind values**:

| Kind | Description |
|------|-------------|
| `"rate_limit"` | Request rate exceeded |
| `"quota_exhausted"` | Account quota depleted |
| `"timeout"` | Request timed out |
| `"unknown"` | Unclassified throttling |

**APIs that throw**:

- `OpenAIAdapter.evaluate()` - OpenAI rate limits
- `LiteLLMAdapter.evaluate()` - Provider rate limits

**Example**:

```python
from weakincentives.adapters.throttle import ThrottleError
import time

try:
    response = adapter.evaluate(prompt, session=session)
except ThrottleError as e:
    if e.retry_safe and e.retry_after:
        time.sleep(e.retry_after.total_seconds())
        response = adapter.evaluate(prompt, session=session)
```

**Location**: `src/weakincentives/adapters/throttle.py:84`

---

### Budget Exceptions (`weakincentives.budget`)

#### BudgetExceededError

**Inherits**: `WinkError`, `RuntimeError`

**Raised when**: Budget limit is breached.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `budget` | `Budget` | The budget that was exceeded |
| `consumed` | `TokenUsage` | Tokens consumed |
| `exceeded_dimension` | `BudgetExceededDimension` | Which limit was hit |

**Dimensions**:

| Dimension | Description |
|-----------|-------------|
| `"deadline"` | Time budget exceeded |
| `"total_tokens"` | Total token limit exceeded |
| `"input_tokens"` | Input token limit exceeded |
| `"output_tokens"` | Output token limit exceeded |

**APIs that throw**:

- `BudgetTracker.check()` - Limit exceeded
- `ProviderAdapter.evaluate()` - During budget enforcement

**Example**:

```python
from weakincentives.budget import BudgetExceededError

try:
    response = adapter.evaluate(prompt, session=session, budget=budget)
except BudgetExceededError as e:
    print(f"Exceeded {e.exceeded_dimension}: {e.consumed}")
```

**Location**: `src/weakincentives/budget.py:78`

---

### Structured Output Exceptions (`weakincentives.prompt.structured_output`)

#### OutputParseError

**Inherits**: `WinkError`

**Raised when**: Structured output cannot be parsed.

**APIs that throw**:

- `parse_structured_output()` - JSON parsing failure
- `ProviderAdapter.evaluate()` - When structured output validation fails

**Example**:

```python
from weakincentives.prompt.structured_output import OutputParseError

try:
    response = adapter.evaluate(prompt_with_structured_output, session=session)
except OutputParseError as e:
    print(f"Failed to parse output: {e}")
```

**Location**: `src/weakincentives/prompt/structured_output.py:43`

---

### Mailbox Exceptions (`weakincentives.runtime.mailbox`)

#### MailboxError

**Inherits**: `WinkError`

**Base class for**: All mailbox-related errors.

**Location**: `src/weakincentives/runtime/mailbox/_types.py:26`

---

#### ReceiptHandleExpiredError

**Inherits**: `MailboxError`

**Raised when**: Message receipt handle is no longer valid.

**Causes**:

- Visibility timeout expired before acknowledge/nack
- Message was already acknowledged
- Message was redelivered with new receipt handle

**APIs that throw**:

- `Message.acknowledge()` - Expired handle
- `Message.nack()` - Expired handle
- `Message.extend_visibility()` - Expired handle

**Example**:

```python
from weakincentives.runtime.mailbox import ReceiptHandleExpiredError

try:
    message.acknowledge()
except ReceiptHandleExpiredError:
    # Message may have been redelivered to another worker
    logger.warning("Receipt handle expired, message may be reprocessed")
```

**Location**: `src/weakincentives/runtime/mailbox/_types.py:30`

---

#### MailboxFullError

**Inherits**: `MailboxError`

**Raised when**: Queue capacity is exceeded.

**Capacity limits**:

| Backend | Limit |
|---------|-------|
| SQS Standard | 120,000 in-flight messages |
| SQS FIFO | 20,000 in-flight messages |
| Redis | Depends on `maxmemory` |
| InMemory | Configurable `max_size` |

**APIs that throw**:

- `Mailbox.send()` - Queue at capacity

**Location**: `src/weakincentives/runtime/mailbox/_types.py:41`

---

#### SerializationError

**Inherits**: `MailboxError`

**Raised when**: Message cannot be serialized or deserialized.

**APIs that throw**:

- `Mailbox.send()` - Body not JSON-serializable
- `Mailbox.receive()` - Invalid message format

**Location**: `src/weakincentives/runtime/mailbox/_types.py:50`

---

#### MailboxConnectionError

**Inherits**: `MailboxError`

**Raised when**: Cannot connect to backend.

**Causes**:

- Redis: Connection refused, timeout, auth failure
- SQS: Invalid credentials, network unreachable

**APIs that throw**:

- `Mailbox.connect()` - Connection failure
- Any operation after connection loss

**Location**: `src/weakincentives/runtime/mailbox/_types.py:58`

---

### Session Snapshot Exceptions (`weakincentives.runtime.session.snapshots`)

#### SnapshotSerializationError

**Inherits**: `WinkError`, `RuntimeError`

**Raised when**: Snapshot cannot be serialized.

**APIs that throw**:

- `Snapshot.to_json()` - Non-serializable state
- `Session.snapshot()` - Serialization failure

**Location**: `src/weakincentives/runtime/session/snapshots.py:47`

---

### Workspace Exceptions (`weakincentives.adapters.claude_agent_sdk.workspace`)

#### WorkspaceBudgetExceededError

**Inherits**: `WinkError`

**Raised when**: Claude Agent SDK workspace budget is exceeded.

**Location**: `src/weakincentives/adapters/claude_agent_sdk/workspace.py:46`

---

#### WorkspaceSecurityError

**Inherits**: `WinkError`

**Raised when**: Workspace security constraint is violated.

**Location**: `src/weakincentives/adapters/claude_agent_sdk/workspace.py:52`

---

## Troubleshooting Guide

### Problem: "Deadline exceeded before provider call"

**Symptoms**: `DeadlineExceededError` raised immediately.

**Causes**:

1. Deadline already expired when evaluation started
2. Previous operations consumed available time

**Solutions**:

```python
# Check deadline before calling
if deadline.remaining().total_seconds() > min_required_time:
    response = adapter.evaluate(prompt, session=session, deadline=deadline)
else:
    # Handle insufficient time
    ...
```

---

### Problem: "Duplicate params type supplied to prompt"

**Symptoms**: `PromptValidationError` during `prompt.render()`.

**Causes**: Same dataclass type passed multiple times to render.

**Solutions**:

```python
# Wrong - duplicate types
prompt.render(MyParams(...), MyParams(...))

# Right - single instance per type
prompt.render(MyParams(...), OtherParams(...))
```

---

### Problem: "Unexpected params type supplied to prompt"

**Symptoms**: `PromptValidationError` during `prompt.render()`.

**Causes**: Passed a dataclass type not declared in any section.

**Solutions**:

```python
# Ensure section declares the params type
section = MarkdownSection(
    template="$value",
    key="my-section",
    default_params=MyParams,  # Declare expected type
)
```

---

### Problem: Rate limit errors from provider

**Symptoms**: `ThrottleError` with `kind="rate_limit"`.

**Solutions**:

1. **Implement backoff**:

   ```python
   try:
       response = adapter.evaluate(...)
   except ThrottleError as e:
       if e.retry_safe:
           delay = e.retry_after or timedelta(seconds=1)
           time.sleep(delay.total_seconds())
           response = adapter.evaluate(...)
   ```

2. **Use throttle policy**:

   ```python
   from weakincentives.adapters.shared import new_throttle_policy

   policy = new_throttle_policy(
       max_attempts=5,
       base_delay=timedelta(milliseconds=500),
   )
   ```

---

### Problem: Budget exceeded mid-evaluation

**Symptoms**: `BudgetExceededError` during tool loop.

**Solutions**:

1. **Increase budget**:

   ```python
   budget = Budget(max_total_tokens=50000)  # Higher limit
   ```

2. **Use streaming to monitor usage**:

   ```python
   tracker = BudgetTracker(budget)
   response = adapter.evaluate(..., budget_tracker=tracker)
   print(f"Used: {tracker.consumed}")
   ```

---

### Problem: Visibility expansion required

**Symptoms**: `VisibilityExpansionRequired` raised.

**Solutions**:

```python
from weakincentives.prompt.errors import VisibilityExpansionRequired
from weakincentives.prompt import VisibilityOverrides

max_retries = 3
for attempt in range(max_retries):
    try:
        response = adapter.evaluate(prompt, session=session)
        break
    except VisibilityExpansionRequired as e:
        # Apply requested overrides
        current = session[VisibilityOverrides].latest()
        merged = {**current.overrides, **e.requested_overrides}
        session[VisibilityOverrides].seed(VisibilityOverrides(overrides=merged))
```

---

### Problem: Snapshot restore fails

**Symptoms**: `SnapshotRestoreError` or `SnapshotMismatchError`.

**Causes**:

1. Snapshot from different WINK version
2. State slices changed since snapshot

**Solutions**:

```python
try:
    session.restore(snapshot)
except SnapshotRestoreError:
    # Start fresh if restore fails
    session.reset()
    # Re-seed required initial state
    session[Plan].seed(initial_plan)
```

---

### Problem: Message receipt handle expired

**Symptoms**: `ReceiptHandleExpiredError` during acknowledge.

**Causes**:

1. Processing took longer than visibility timeout
2. Message was redelivered to another worker

**Solutions**:

1. **Extend visibility during long operations**:

   ```python
   message.extend_visibility(timedelta(minutes=5))
   # Continue processing
   message.acknowledge()
   ```

2. **Increase default visibility timeout**:

   ```python
   mailbox = SQSMailbox(visibility_timeout=timedelta(minutes=10))
   ```

---

## Exception Handling Patterns

### Catch-All with Logging

```python
import logging

logger = logging.getLogger(__name__)

try:
    response = adapter.evaluate(prompt, session=session)
except ThrottleError as e:
    logger.warning(f"Throttled: {e.kind}, retry_after={e.retry_after}")
    raise
except PromptEvaluationError as e:
    logger.error(f"Evaluation failed in {e.phase}: {e.message}")
    raise
except WinkError as e:
    logger.error(f"WINK error: {type(e).__name__}: {e}")
    raise
```

### Retry with Backoff

```python
import time
from datetime import timedelta

def evaluate_with_retry(adapter, prompt, session, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return adapter.evaluate(prompt, session=session)
        except ThrottleError as e:
            if not e.retry_safe or attempt == max_attempts - 1:
                raise
            delay = e.retry_after or timedelta(seconds=2 ** attempt)
            time.sleep(delay.total_seconds())
    raise RuntimeError("Max retries exceeded")
```

### Graceful Degradation

```python
def evaluate_with_fallback(adapters, prompt, session):
    errors = []
    for adapter in adapters:
        try:
            return adapter.evaluate(prompt, session=session)
        except PromptEvaluationError as e:
            errors.append(e)
            continue
    raise ExceptionGroup("All adapters failed", errors)
```

## Related Specifications

- `specs/ADAPTERS.md` - Adapter error handling
- `specs/SESSIONS.md` - Session errors and snapshots
- `specs/MAILBOX.md` - Mailbox error semantics
- `specs/EXECUTION_STATE.md` - Execution state errors
