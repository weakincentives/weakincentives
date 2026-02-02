# Data Flow Bug Detection Report

This report documents potential bugs identified through comprehensive data flow analysis
of the weakincentives codebase. Bugs are organized by severity and category.

---

## Executive Summary

| Severity | Count |
|----------|-------|
| **CRITICAL** | 8 |
| **HIGH** | 12 |
| **MEDIUM** | 18 |
| **LOW** | 10 |
| **Total** | 48 |

### Top Priority Issues

1. **Race condition in reducer execution** - State corruption possible (session.py:698-716)
2. **Feedback collection after snapshot restoration** - Transaction boundary violation (tool_executor.py:744)
3. **Policy.on_result() exceptions skip cleanup** - Snapshot not restored (tool_executor.py:552)
4. **Token tracking key mismatch** - Budget tracking inconsistent across adapters
5. **Circular section reference detection missing** - Infinite recursion possible

---

## CRITICAL Bugs

### C1: Race Condition in Reducer Execution
**Location:** `src/weakincentives/runtime/session/session.py:698-716`
**Category:** Concurrency / State Corruption

**Problem:** The reducer execution releases the lock between creating a slice view and applying
the result. Another thread could modify the slice during this window.

```python
# Line 698-700: Lock held
with self.locked():
    slice_instance = self._store.get_or_create(slice_type)
    slice_view = slice_instance.view()  # Creates snapshot under lock
# Lock RELEASED here

# Line 702: Reducer called WITHOUT lock
op = registration.reducer(slice_view, event, context=context)

# Line 715-716: Lock re-acquired
with self.locked():
    apply_slice_op(op, slice_instance)
```

**Impact:** State inconsistency, data loss, or corruption in concurrent sessions.

**Recommendation:** Hold lock during entire reducer execution, or use optimistic locking with
retry on conflict.

---

### C2: Feedback Collection After Snapshot Restoration
**Location:** `src/weakincentives/adapters/tool_executor.py:744-750`
**Category:** Transaction / Data Flow

**Problem:** Feedback is collected AFTER the transactional context manager exits. If the tool
failed and snapshot was restored, feedback modifies the restored (pre-tool) state.

```python
with tool_execution(context=context, tool_call=tool_call) as outcome:
    invocation = dispatch_tool_invocation(context=context, outcome=outcome)
    # Context manager EXITS here - snapshot may be restored

# Feedback collected AFTER snapshot restoration
feedback_text = collect_feedback(...)  # Line 744
tool_result = _append_feedback_to_result(outcome.result, feedback_text)  # Line 750
```

**Impact:** Session state inconsistency; feedback may reference invalidated state.

**Recommendation:** Move feedback collection inside the transactional context.

---

### C3: Policy.on_result() Exceptions Skip Cleanup
**Location:** `src/weakincentives/adapters/tool_executor.py:552`
**Category:** Exception Handling / Transaction

**Problem:** `_notify_policies_of_result()` is in the `else` block of try/except. Exceptions
here bypass snapshot restoration.

```python
try:
    # ... tool execution ...
except Exception:
    # ... handle exception, restore snapshot ...
else:
    # THIS BLOCK IS OUTSIDE EXCEPTION HANDLER
    _notify_policies_of_result(...)  # Line 552: NOT PROTECTED
```

**Impact:** Successful tool execution can leave session in inconsistent state if policy fails.

**Recommendation:** Wrap policy notification in try/finally with snapshot restoration.

---

### C4: Outcome Used After Context Exit
**Location:** `src/weakincentives/adapters/tool_executor.py:835-850`
**Category:** Context Lifecycle / Use-After-Free

**Problem:** The `outcome` variable is bound inside the `with` block but used after context
manager exits. Snapshot restoration may invalidate the outcome.

```python
with tool_execution(context=execution_context, tool_call=tool_call) as outcome:
    _ = dispatch_tool_invocation(context=execution_context, outcome=outcome)
    # Context manager exits - snapshots may be restored

# outcome.result used AFTER context exit
feedback_text = collect_feedback(...)  # Line 845
tool_result = _append_feedback_to_result(outcome.result, feedback_text)  # Line 850
```

**Impact:** Tool result may reference invalidated state.

---

### C5: Token Tracking Key Mismatch Across Adapters
**Location:** Multiple files
**Category:** Data Consistency

**Problem:** Different adapters use different keys for budget tracking:
- InnerLoop (OpenAI/LiteLLM): Uses `evaluation_id` (inner_loop.py:223)
- Claude SDK: Uses `prompt_name` (adapter.py:1085)

**Impact:** Budget tracking may not aggregate correctly when using different adapters.

**Recommendation:** Standardize on a single key format across all adapters.

---

### C6: Circular Dependency Detection Loses Path Order
**Location:** `src/weakincentives/resources/context.py:106-108`
**Category:** Error Reporting

**Problem:** `self._resolving` is a `set`, so converting to tuple loses insertion order.

```python
if protocol in self._resolving:
    cycle = (*self._resolving, protocol)  # Line 107-108 - set loses order
    raise CircularDependencyError(cycle)
```

**Impact:** Error messages show incorrect dependency chain order.

**Recommendation:** Use `list` or `OrderedDict` to track resolution order.

---

### C7: No Circular Section Reference Detection
**Location:** `src/weakincentives/prompt/registry.py:496-504`
**Category:** Infinite Recursion

**Problem:** `_register_child_sections` recursively registers children without cycle detection.
If section A contains B which contains A, infinite recursion occurs.

**Impact:** Stack overflow during prompt construction.

**Recommendation:** Add visited set to detect cycles during section registration.

---

### C8: String-to-List Coercion Changes Data Semantics
**Location:** `src/weakincentives/serde/parse.py:319-326`
**Category:** Silent Data Transformation

**Problem:** With `coerce=True`, strings are silently wrapped in single-element lists.

```python
if config.coerce and isinstance(value, str):
    return [value]  # "hello" becomes ["hello"]
```

**Impact:** User mistakes are silently "corrected" with unexpected semantics.

**Recommendation:** Require explicit list even with coercion, or document prominently.

---

## HIGH Priority Bugs

### H1: ToolResult Mutation After Event Dispatch
**Location:** `src/weakincentives/adapters/tool_executor.py:685`
**Category:** Immutability Violation

**Problem:** ToolResult stored in ToolInvoked event is mutated after dispatch.

```python
invocation = ToolInvoked(..., result=cast(ToolResult[object], outcome.result))
dispatch_result = context.session.dispatcher.dispatch(invocation)
if not dispatch_result.ok:
    outcome.result.message = ...  # Line 685: Mutates stored object
```

**Impact:** Event contains mutated state, violating immutability contracts.

---

### H2: Feedback Provider Exceptions Uncaught
**Location:** `src/weakincentives/prompt/feedback.py:432-445`
**Category:** Exception Handling

**Problem:** `provider.provide()` can raise without being caught, after tool execution
and event dispatch already completed.

**Impact:** Unhandled exception; tool result inconsistent between dispatch and return.

---

### H3: Feedback vs Dispatch Inconsistency
**Location:** `src/weakincentives/adapters/tool_executor.py:738-750`
**Category:** Data Flow Divergence

**Problem:** ToolInvoked event stores result WITHOUT feedback, but caller receives result
WITH feedback appended.

**Impact:** Internal state and external API see different data.

---

### H4: Zero Token Bug - Falsy Value Fallback
**Location:** `src/weakincentives/adapters/token_usage.py:42-46`
**Category:** Logic Error

**Problem:** Uses `or` operator, so `input_tokens=0` falls back to `prompt_tokens`.

```python
input_tokens = _coerce_token_count(
    usage_payload.get("input_tokens") or usage_payload.get("prompt_tokens")
)
```

**Impact:** Zero input tokens incorrectly reported.

**Recommendation:** Use `if ... is not None` instead of `or`.

---

### H5: Claude SDK Potential Double Token Counting
**Location:** `src/weakincentives/adapters/claude_agent_sdk/adapter.py:1073-1084, 1279-1291`
**Category:** Duplicate Accounting

**Problem:** Tokens counted in both streaming message loop (line 1084) and
`_extract_result()` (line 1289).

**Impact:** Budget tracking may over-count tokens.

---

### H6: Concurrent Policy State Mutation Not Atomic
**Location:** `src/weakincentives/prompt/policy.py:164-173`
**Category:** Race Condition

**Problem:** Read-Modify-Write pattern on PolicyState not atomic.

```python
state = context.session[PolicyState].latest()  # Read
new_state = PolicyState(...)
context.session[PolicyState].seed(new_state)  # Write
```

**Impact:** Lost updates in concurrent tool execution.

---

### H7: System Mutation Events Bypass Reducer Dispatch
**Location:** `src/weakincentives/runtime/session/session.py:605-653`
**Category:** Architectural Violation

**Problem:** InitializeSlice and ClearSlice directly mutate state, bypassing registered
reducers.

**Impact:** Reducers can't intercept/validate these operations; audit trail incomplete.

---

### H8: Dispatcher Reassignment Race - Subscription Leak
**Location:** `src/weakincentives/runtime/session/session.py:745-754`
**Category:** Memory Leak

**Problem:** When session is attached to new dispatcher, old subscriptions remain on
previous dispatcher.

**Impact:** Memory leak; ghost event handlers; events routed to wrong dispatcher.

---

### H9: Temporary File Resource Leak in Podman
**Location:** `src/weakincentives/contrib/tools/podman.py:1170-1190`
**Category:** Resource Leak

**Problem:** Temporary file created with `delete=False` may not be deleted if exception
occurs. `suppress(OSError)` silently ignores cleanup failures.

**Impact:** Temporary files accumulate on disk.

---

### H10: Podman Client Connection Leak
**Location:** `src/weakincentives/contrib/tools/podman.py:1049-1071`
**Category:** Resource Leak

**Problem:** Early return in exception handler bypasses `client.close()`.

**Impact:** Podman connections accumulate without cleanup.

---

### H11: Thread Exhaustion in asteval
**Location:** `src/weakincentives/contrib/tools/asteval.py:515-541`
**Category:** Resource Exhaustion

**Problem:** Creates new daemon thread for every evaluation. Rapid consecutive calls
could exhaust thread resources.

**Impact:** System degradation with many evaluations.

---

### H12: Built-in Tool Names Not Reserved
**Location:** `src/weakincentives/prompt/registry.py:534-542`
**Category:** Name Collision

**Problem:** User-defined tools can have names like `open_sections` or `read_section`,
which conflict with injected tools at render time.

**Impact:** Tool collision error at runtime instead of registration time.

---

## MEDIUM Priority Bugs

### M1: TelemetryDispatcher.dispatch Without Lock
**Location:** `src/weakincentives/runtime/session/session.py:389-390`

Telemetry dispatched while session state is in flux; subscribers see inconsistent state.

---

### M2: Event Type Routing Error in SliceAccessor
**Location:** `src/weakincentives/runtime/session/slice_accessor.py:145-148, 165-168`

Dispatches InitializeSlice/ClearSlice with wrong event type; reducers for slice type
not invoked.

---

### M3: Unsafe Duck-Typing in Telemetry Extraction
**Location:** `src/weakincentives/runtime/session/session_telemetry.py:68-76`

Uses `hasattr(result, "value")` instead of proper `isinstance()` check.

---

### M4: Multi-Content Loss in OpenAI Responses
**Location:** `src/weakincentives/adapters/openai.py:480-482`

Only tracks first `content_output` with content; later content blocks ignored.

---

### M5: Structured Output Text Loss
**Location:** `src/weakincentives/adapters/response_parser.py:237`

Sets `text_value = None` when structured output parsed, losing original message text.

---

### M6: Tool Parameter Validation State Confusion
**Location:** `src/weakincentives/adapters/tool_executor.py:531-536, 562`

For parameterless tools, can't distinguish between "valid None" and "failed to parse".

---

### M7: ToolResult Not Frozen
**Location:** `src/weakincentives/prompt/tool_result.py:31`

`@dataclass(slots=True)` missing `frozen=True`; allows mutations after creation.

---

### M8: Policy State Fragmentation
**Location:** `src/weakincentives/adapters/tool_executor.py:256, policy.py:163-164`

Multiple policies may create separate PolicyState instances with same name.

---

### M9: Tool Scope Restoration Logic Flaw
**Location:** `src/weakincentives/resources/context.py:255-296`

Redundant removal and potential race condition in nested tool scopes.

---

### M10: Instantiation Order Not Tracking Scope Correctly
**Location:** `src/weakincentives/resources/context.py:79-82, 120, 204-231`

Scope stored may not match resource's actual scope during exception handling.

---

### M11: PostConstruct Failure Cleanup Under-Reported
**Location:** `src/weakincentives/resources/context.py:153-162`

`close()` failures during init only logged at WARNING level.

---

### M12: Missing Required Field Error Lacks Path Context
**Location:** `src/weakincentives/serde/parse.py:879-881`

Error shows field name but not nested path for deep structures.

---

### M13: __type__ Field Not Auto-Resolved
**Location:** `src/weakincentives/serde/parse.py:1025-1051`

Documentation suggests automatic polymorphic deserialization but it's not implemented.

---

### M14: Circular Import Workaround Loses Type Safety
**Location:** `src/weakincentives/serde/_utils.py:92-104`

`_ParseConfig.scope` typed as `object` instead of `SerdeScope`.

---

### M15: Inconsistent param_lookup Usage
**Location:** `src/weakincentives/prompt/rendering.py:192 vs 255, 277`

Copy passed to `_iter_enabled_sections` but original to visibility handlers.

---

### M16: Deserialization Size Limit Missing
**Location:** `src/weakincentives/contrib/mailbox/_redis.py:816-835`

`json.loads()` without size check could cause memory exhaustion.

---

### M17: Podman Workspace Handle Race Condition
**Location:** `src/weakincentives/contrib/tools/podman.py:960-980`

TOCTOU race between checking if overlay empty and copying files.

---

### M18: Format String Injection via Template Variables
**Location:** `src/weakincentives/contrib/tools/asteval.py:876-883`

`.format_map()` allows method calls on symtable objects.

---

## LOW Priority Bugs

### L1: Tool Call ID Conversion Edge Case
**Location:** `src/weakincentives/adapters/openai.py:215`

Complex string fallback could produce "None" string instead of None.

---

### L2: Content Type Inconsistency
**Location:** `src/weakincentives/adapters/response_parser.py:111-120`

Different behavior for string vs sequence content from providers.

---

### L3: Handler Null Check Error Message Ambiguity
**Location:** `src/weakincentives/adapters/tool_executor.py:156-171`

Error messages for "no tool" vs "no handler" could be clearer.

---

### L4: Regex Pattern Compiled on Every Validation
**Location:** `src/weakincentives/serde/_utils.py:202-209`

Pattern recompiled each time (Python caches, but still inefficient).

---

### L5: Type Coercion Key Collision
**Location:** `src/weakincentives/serde/parse.py:409-432`

Keys coerced to same value overwrite each other silently.

---

### L6: Constraint Normalization Before Validation
**Location:** `src/weakincentives/serde/_utils.py:123-136`

Validation runs on normalized value, may conflict with constraints.

---

### L7: _ExtrasDescriptor Uses id() - Memory Risk
**Location:** `src/weakincentives/serde/_utils.py:36-58`

id() reuse after GC could cause data sharing between instances.

---

### L8: None Values Render as "None" String
**Location:** `src/weakincentives/prompt/markdown.py:154-164`

Template substitution renders None-valued fields as literal "None".

---

### L9: Reaper Thread Exception Too Broad
**Location:** `src/weakincentives/contrib/mailbox/_redis.py:781-785`

`suppress(Exception)` hides real bugs in reaper loop.

---

### L10: Silent Write Failures in asteval
**Location:** `src/weakincentives/contrib/tools/asteval.py:425-440`

Writes that fail mode constraints are silently skipped.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

User Request
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ AgentLoop   │────▶│ Adapter      │────▶│ LLM Provider    │
│             │     │ (OpenAI/     │     │ (API Call)      │
│             │     │  LiteLLM/    │     │                 │
│ - Budget    │     │  Claude SDK) │     │                 │
│ - Deadline  │     │              │     │                 │
└─────────────┘     └──────────────┘     └─────────────────┘
    │                    │                      │
    │                    │                      │
    │     ┌──────────────┘                      │
    │     │ [BUG: Token key mismatch C5]       │
    │     │ [BUG: Zero token bug H4]           │
    │     ▼                                     │
    │ ┌──────────────┐                          │
    │ │ TokenUsage   │◀─────────────────────────┘
    │ │ Tracking     │     Response
    │ └──────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TOOL EXECUTION FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

Tool Call Request
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ ToolExecutor│────▶│ Transaction  │────▶│ Tool Handler    │
│             │     │ Context      │     │                 │
│             │     │              │     │                 │
│ - Parse     │     │ - Snapshot   │     │ - Execute       │
│ - Validate  │     │ - Restore    │     │ - Return Result │
└─────────────┘     └──────────────┘     └─────────────────┘
    │                    │                      │
    │ [BUG: C3 Policy    │ [BUG: C2 Feedback   │
    │  skip cleanup]     │  after restore]     │
    │                    │                      │
    ▼                    ▼                      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Policy      │     │ Session      │     │ ToolResult      │
│ Check       │     │ Dispatch     │     │                 │
│             │     │              │     │ [BUG: H1 mutated│
│ [BUG: H6    │     │ [BUG: C1     │     │  after dispatch]│
│  race]      │     │  race cond]  │     └─────────────────┘
└─────────────┘     └──────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           SESSION STATE FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Event Dispatch
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Session     │────▶│ Reducer      │────▶│ SliceStore      │
│ .dispatch() │     │ Registry     │     │                 │
│             │     │              │     │ - SliceOp       │
│ [BUG: H7    │     │ - Lookup     │     │ - Apply         │
│  system     │     │ - Execute    │     │                 │
│  events     │     │              │     │                 │
│  bypass]    │     │ [BUG: C1]    │     │                 │
└─────────────┘     └──────────────┘     └─────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Telemetry    │
                    │ Dispatcher   │
                    │              │
                    │ [BUG: M1     │
                    │  no lock]    │
                    │              │
                    │ [BUG: H8     │
                    │  subscription│
                    │  leak]       │
                    └──────────────┘
```

---

## Recommendations by Priority

### Immediate (Fix in next release)
1. Fix reducer race condition (C1) - Add lock or optimistic concurrency
2. Move feedback collection inside transaction (C2, C4)
3. Wrap policy notification in try/finally (C3)
4. Standardize token tracking keys (C5)
5. Add circular section detection (C7)

### High Priority (Fix soon)
1. Make ToolResult frozen (M7, H1)
2. Fix zero-token fallback logic (H4)
3. Reserve built-in tool names (H12)
4. Fix subscription cleanup in dispatcher (H8)
5. Add exception handling for feedback providers (H2)

### Medium Priority (Plan for future)
1. Fix system event dispatch bypass (H7)
2. Add atomic operations for policy state (H6)
3. Improve error messages with path context (M12)
4. Fix deserialization size limits (M16)

### Low Priority (Nice to have)
1. Improve diagnostic messages (L1-L3)
2. Optimize regex compilation (L4)
3. Use WeakKeyDictionary for extras (L7)

---

*Report generated by data flow analysis on 2026-02-02*
*Total files analyzed: 45+*
*Analysis method: Subagent-based parallel exploration*
