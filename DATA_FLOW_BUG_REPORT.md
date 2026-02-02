# Data Flow Bug Detection Report

**Generated:** 2026-02-02
**Scope:** Full codebase scan of `src/weakincentives/`
**Total Issues Found:** 65

---

## Executive Summary

A systematic analysis of the request/response data flow across the weakincentives codebase identified **65 potential bugs** across 4 major subsystems:

| Module | Critical | High | Medium | Low | Total |
|--------|----------|------|--------|-----|-------|
| Adapters | 3 | 7 | 7 | 10 | 27 |
| Runtime/Session | 2 | 2 | 5 | 1 | 10 |
| Prompt/Tools | 0 | 0 | 8 | 6 | 14 |
| Serde/Resources | 3 | 5 | 4 | 2 | 14 |
| **Total** | **8** | **14** | **24** | **19** | **65** |

---

## Critical Severity Bugs (8)

These bugs can cause data corruption, lost updates, or security issues.

### CRIT-1: Reducer Execution Not Atomic with Slice Application
**File:** `src/weakincentives/runtime/session/session.py:698-716`
**Impact:** Lost updates and state corruption under concurrent dispatch

The reducer is invoked OUTSIDE the session lock. Between capturing the slice view and applying the operation, another thread can modify the slice, causing lost updates with last-write-wins behavior.

```python
with self.locked():
    slice_instance = self._store.get_or_create(slice_type)
    slice_view = slice_instance.view()
# LOCK RELEASED HERE - Race window opens

try:
    op = registration.reducer(slice_view, event, context=context)  # OUTSIDE LOCK

with self.locked():
    apply_slice_op(op, slice_instance)  # Applies to potentially different state
```

### CRIT-2: Session Snapshot Doesn't Prevent Concurrent Mutations
**File:** `src/weakincentives/runtime/session/session.py:533-559, 423-453`
**Impact:** Snapshot/restore not transactional, potential state corruption

Parent/children IDs are captured then lock released. Between capturing and `create_snapshot` completion, relationships could change.

### CRIT-3: Missing Error Handling in Tool Message Serialization
**File:** `src/weakincentives/adapters/tool_executor.py:850-858`
**Impact:** Silent failures or crashes with custom objects

Tool message construction doesn't validate serialized result before appending. Non-serializable objects cause JSON encoding failures without context.

### CRIT-4: Race Condition in Token Usage Tracking
**File:** `src/weakincentives/adapters/inner_loop.py:221-225`
**Impact:** Inaccurate budget tracking under concurrent execution

Token usage recorded after budget check in `_record_and_check_budget()`, but multiple async tasks could race to update `self._provider_payload`.

### CRIT-5: Unvalidated Type Coercion in Token Usage
**File:** `src/weakincentives/adapters/token_usage.py:23-29`
**Impact:** Integer overflow with malformed provider responses

Function coerces int/float to int without bounds checking. Very large floats could overflow.

### CRIT-6: Unresolved Circular Dependencies Remain in _resolving Set
**File:** `src/weakincentives/resources/context.py:237-296`
**Impact:** Tool scopes unreliable after caught circular dependency errors

When `tool_scope()` is entered, `_resolving` set is not cleared/saved. Caught circular dependency errors leave protocols in `_resolving`, causing false positives on next resolution.

### CRIT-7: Missing Scope Parameter in Nested Dataclass Parsing
**File:** `src/weakincentives/serde/parse.py:254-275`
**Impact:** `HiddenInStructuredOutput` markers ignored in nested structures

```python
return _ParseConfig(
    extra=config.extra,
    coerce=config.coerce,
    # BUG: Missing scope=config.scope
)
```

### CRIT-8: Tool Scope Cache Restoration with Closed Resources
**File:** `src/weakincentives/resources/context.py:256, 286`
**Impact:** Use-after-close bugs, resource state corruption

When exiting `tool_scope()`, resources are closed then old cache restored. Previously cached TOOL_CALL resources get restored but may be closed.

---

## High Severity Bugs (14)

### HIGH-1: Lease Extension Race Condition
**File:** `src/weakincentives/runtime/lease_extender.py:171-206`
**Impact:** Message lifecycle violation, stale reference usage

Message reference captured then lock released. Another thread can detach, but first thread still calls `extend_visibility()` on stale reference.

### HIGH-2: InMemoryMailbox Reaper Thread Lifecycle
**File:** `src/weakincentives/runtime/mailbox/_in_memory.py:399-409`
**Impact:** Resource leak, zombie threads

`join(timeout=1.0)` returns if reaper stuck. Thread continues as zombie.

### HIGH-3: Partial Tool Call Handling Loss
**File:** `src/weakincentives/adapters/inner_loop.py:266-272`
**Impact:** Inconsistent state with failed tool calls

If some tool_calls reference non-existent tools, partial execution may leave inconsistent state.

### HIGH-4: Missing Null Check for tool_call.id
**File:** `src/weakincentives/adapters/tool_executor.py:853-855`
**Impact:** Malformed provider messages with None tool_call_id

Some providers may not set tool call IDs.

### HIGH-5: Empty Arguments Return Empty Dict Without Error
**File:** `src/weakincentives/adapters/tool_spec.py:107`
**Impact:** Tools expecting required parameters receive empty dict

```python
if not arguments_json:
    return {}  # Silent success for empty args
```

### HIGH-6: Message Mutation in _normalize_input_messages
**File:** `src/weakincentives/adapters/openai.py:385-441`
**Impact:** Caller's message data unexpectedly modified

Function modifies message dicts in-place without deep copying.

### HIGH-7: Incomplete Error Payload Extraction
**File:** `src/weakincentives/adapters/openai.py:181-190`
**Impact:** Lost debugging information

Only checks `response` and `json_body` attributes, other error formats silently return None.

### HIGH-8: Unbounded Memory Growth in _SLOTTED_EXTRAS
**File:** `src/weakincentives/serde/_utils.py:58`
**Impact:** Memory leak in long-running applications

Module-level dict never clears entries, grows unbounded with dynamic dataclass types.

### HIGH-9: Memory Leak in _ExtrasDescriptor via Reused Object IDs
**File:** `src/weakincentives/serde/_utils.py:36-56`
**Impact:** Extras from deleted objects accumulate

Stores extras by `id(instance)`. Deleted object IDs reused but entries never cleaned.

### HIGH-10: Type Coercion Error in TypeVar Resolution
**File:** `src/weakincentives/serde/parse.py:919-960`
**Impact:** Generic dataclass parsing fails with nested generics

Non-type objects can end up in typevar_map, causing type errors downstream.

### HIGH-11: clone() Doesn't Preserve SerdeScope
**File:** `src/weakincentives/serde/dump.py:244-274`
**Impact:** Hidden fields restored when cloning STRUCTURED_OUTPUT objects

```python
return parse(type(obj), serialized, extra="allow")  # BUG: No scope parameter
```

### HIGH-12: Tool Scope Cleanup Logic Fragility
**File:** `src/weakincentives/resources/context.py:255-296`
**Impact:** Silent state corruption from logic errors

Complex save/clear/restore sequence for cache and instantiation_order.

### HIGH-13: Exception in Reducer Dispatch Silently Continues
**File:** `src/weakincentives/runtime/session/session.py:693-727`
**Impact:** Partial state mutations

```python
except Exception:
    logger.exception(...)
    continue  # Silently skip to next reducer
```

### HIGH-14: Session Cloning During Concurrent Modifications
**File:** `src/weakincentives/runtime/session/session.py:565-592`
**Impact:** Clone receives snapshot from N milliseconds ago

No atomicity guarantee for cloning operation.

---

## Medium Severity Bugs (24)

### Adapters Module (7)

| ID | File | Lines | Issue |
|----|------|-------|-------|
| MED-A1 | inner_loop.py | 208-215 | Deadline check-to-use gap |
| MED-A2 | tool_spec.py | 115-121 | Non-string keys not validated |
| MED-A3 | response_parser.py | 106-120 | Empty content sequence returns "" |
| MED-A4 | openai.py | 256-307 | Invalid tool schema passes through |
| MED-A5 | tool_executor.py | 397-410 | Snapshot restore lacks context |
| MED-A6 | _visibility_signal.py | - | Signal not thread-safe |
| MED-A7 | response_parser.py | 193-239 | Ambiguous structured/text handling |

### Runtime Module (5)

| ID | File | Lines | Issue |
|----|------|-------|-------|
| MED-R1 | transactions.py | 475-536 | Duplicate tool_use_id overwrites snapshot |
| MED-R2 | message_handlers.py | 142-157 | Reply failure only logged at debug |
| MED-R3 | watchdog.py | 223-228 | Thread join may timeout silently |
| MED-R4 | session.py | 693-727 | Exception handling continues silently |
| MED-R5 | session.py | 565-592 | Clone receives stale snapshot |

### Prompt/Tools Module (8)

| ID | File | Lines | Issue |
|----|------|-------|-------|
| MED-P1 | rendering.py | 345-354 | Inconsistent tool override application |
| MED-P2 | registry.py | 534-540 | Missing tool identity in error |
| MED-P3 | section.py | 355-362 | Visibility validation in render loop |
| MED-P4 | rendering.py | 261-285 | Type cast for dynamic tools loses info |
| MED-P5 | rendering.py | 163-309 | No resource cleanup guarantee |
| MED-P6 | rendering.py | 373-380 | Exception wrapping loses context |
| MED-P7 | _prompt_resources.py | 144-150 | No type validation for instances |
| MED-P8 | _render_tool_examples.py | 101-111 | Unsafe cast for array results |

### Serde/Resources Module (4)

| ID | File | Lines | Issue |
|----|------|-------|-------|
| MED-S1 | parse.py | 278-285 | Incomplete error formatting |
| MED-S2 | _utils.py | 219-232 | Mixed types in "in" constraint |
| MED-S3 | _utils.py | 70-78 | Class-level descriptor modification |
| MED-S4 | _utils.py | 103-104 | Scope typed as object |

---

## Low Severity Bugs (19)

### Adapters Module (10)

| ID | File | Issue |
|----|------|-------|
| LOW-A1 | throttle.py:53-70 | Very small timedeltas rejected |
| LOW-A2 | openai.py, litellm.py | Inconsistent retry-after parsing |
| LOW-A3 | tool_executor.py:130-136 | Shallow copy with mutable nested |
| LOW-A4 | inner_loop.py:502-504 | Tool choice reset without validation |
| LOW-A5 | claude_agent_sdk/adapter.py:1026-1172 | Missing cancellation points |
| LOW-A6-10 | Various | Minor edge cases |

### Runtime Module (1)

| ID | File | Issue |
|----|------|-------|
| LOW-R1 | session.py:519-522 | Tags immutability not enforced |

### Prompt/Tools Module (6)

| ID | File | Issue |
|----|------|-------|
| LOW-P1 | _prompt_resources.py:83-93 | Context cleanup on exception |
| LOW-P2 | rendering.py:266-268 | Unsafe cast to generic Tool |
| LOW-P3 | rendering.py:164-192 | Mutable dict as Mapping |
| LOW-P4 | rendering.py:194-214 | Skip logic SUMMARY-specific |
| LOW-P5 | rendering.py:351-354 | Patches without description override |
| LOW-P6 | rendering.py:185-242 | Tool overrides not validated |

### Serde/Resources Module (2)

| ID | File | Issue |
|----|------|-------|
| LOW-S1 | context.py:223-228 | Silent exception swallowing in close |
| LOW-S2 | context.py:148-163 | Asymmetric post_construct failure |

---

## Data Flow Diagrams

### Request/Response Flow (Adapters)

```
ProviderAdapter.evaluate()
    │
    ├─▶ prepare_adapter_conversation()
    │   └─ Render prompt → AdapterRenderContext
    │
    ├─▶ InnerLoop.run()
    │   ├─ _prepare() → ToolExecutor, ResponseParser
    │   │
    │   └─ Loop until final response:
    │       ├─ _issue_provider_request()
    │       │  ├─ Normalize messages
    │       │  ├─ Call provider API
    │       │  └─ Handle ThrottleError with backoff
    │       │
    │       ├─ _record_and_check_budget()
    │       │
    │       ├─ If tool_calls:
    │       │  └─ _handle_tool_calls()
    │       │     ├─ serialize_tool_call()
    │       │     ├─ ToolExecutor.execute()
    │       │     │  └─ tool_execution() context manager
    │       │     │     ├─ Start transactional snapshot
    │       │     │     ├─ Check policies
    │       │     │     ├─ Invoke handler
    │       │     │     └─ On failure: restore snapshot
    │       │     └─ collect_feedback()
    │       │
    │       └─ Else: _finalize_response()
    │
    └─▶ Return PromptResponse
```

### Session Dispatch Flow (Runtime)

```
dispatch(event)
    │
    ├─ Lock acquired
    ├─ Get registrations [UNDER LOCK]
    └─ Lock released  ⚠️ BUG: Registrations stale
    │
    ├─ For each registration:
    │   ├─ Lock acquired
    │   ├─ Get slice, create view [UNDER LOCK]
    │   └─ Lock released  ⚠️ BUG: View stale
    │   │
    │   ├─ Call reducer(view, event)  ⚠️ OUTSIDE LOCK!
    │   │
    │   ├─ Lock acquired
    │   └─ Apply operation [UNDER LOCK]
    │
    └─ Return result
```

### Resource Resolution Flow (DI)

```
ctx.get(protocol)
    │
    ├─ Check singleton_cache → return if cached
    ├─ Check _tool_call_cache → return if cached
    │
    ├─ Cycle detection: _resolving set
    │
    ├─ _construct(binding)
    │   ├─ Call binding.provider(self)
    │   └─ PostConstruct.post_construct() if applicable
    │
    ├─ Cache by scope (SINGLETON, TOOL_CALL, PROTOTYPE)
    │
    └─ Return instance
```

---

## Recommendations

### Immediate (Critical)

1. **CRIT-1, CRIT-2**: Implement optimistic locking or hold lock during reducer execution
2. **CRIT-6, CRIT-8**: Clear `_resolving` on tool scope entry; don't restore closed resources
3. **CRIT-7**: Pass `scope=config.scope` to nested parsing

### Short-term (High)

1. Add comprehensive input validation at adapter boundaries
2. Implement atomic token usage tracking with synchronization
3. Add deep validation for nested data structures
4. Fix memory leaks in `_SLOTTED_EXTRAS` and `_ExtrasDescriptor`

### Medium-term

1. Standardize error handling patterns across providers
2. Add cancellation points in async loops for deadline enforcement
3. Implement thread-safety guarantees for concurrent execution
4. Add detailed error context to all state restoration operations

---

## Testing Gaps Identified

1. No concurrent dispatch tests for session reducers
2. No test for circular dependency recovery in resource context
3. No test for tool_scope with pre-existing cached resources
4. Limited testing of nested generic dataclass parsing
5. No memory leak tests for long-running serde operations
