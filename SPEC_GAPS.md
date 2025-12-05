# Specification vs Implementation Gap Analysis

This document identifies gaps between the specification documents in `specs/` and the actual implementation in `src/weakincentives/`.

## Summary

| Gap Type | Spec | Status |
|----------|------|--------|
| Missing Implementation | TODO_TOOL.md | **Critical** - No code exists |
| Stale Spec | DATACLASS_SERDE.md | **Minor** - Spec references non-existent modules |
| Stale Spec | LOGGING.md | **Minor** - Module paths outdated |

---

## Critical Gaps

### 1. TODO_TOOL.md - Missing Implementation

**Spec Location:** `specs/TODO_TOOL.md`

**What the spec describes:**
- A `TodoList` dataclass with an `items: list[str]` field
- `todo_write` tool to write/update the list
- `todo_read` tool to read the current list
- `TodoToolsSection` to register the tools and reducers
- Implementation should be in `weakincentives.tools.todo`

**Current state:**
- **No `todo.py` module exists** in `src/weakincentives/tools/`
- The `tools/__init__.py` exports no todo-related types
- No `TodoList`, `todo_write`, `todo_read`, or `TodoToolsSection` exist

**Recommendation:** Either implement the TODO tool as specified, or remove/archive the spec if the feature is no longer planned.

---

## Minor Gaps (Documentation Drift)

### 2. DATACLASS_SERDE.md - Stale Module References

**Spec Location:** `specs/DATACLASS_SERDE.md`

**Issue:** The "Behaviour Map" table references modules that don't exist:

| Spec Reference | Actual Location |
|----------------|-----------------|
| `src/weakincentives/serde/metadata.py` | Does not exist |
| `src/weakincentives/serde/coercion.py` | Does not exist |
| `src/weakincentives/serde/collections.py` | Does not exist |
| `src/weakincentives/serde/aliases.py` | Does not exist |
| `src/weakincentives/serde/errors.py` | Does not exist |
| `src/weakincentives/serde/clone.py` | `clone()` is in `dump.py` |

**Actual structure:**
```
src/weakincentives/serde/
├── __init__.py      # Exports: parse, dump, clone, schema
├── parse.py         # parse() function
├── dump.py          # dump() and clone() functions
├── schema.py        # schema() function
├── dataclass_serde.py  # Additional helpers
└── _utils.py        # Internal utilities
```

**Recommendation:** Update the spec's Behaviour Map table to reflect the actual consolidated module structure, or refactor the implementation to match the spec's modular design.

---

### 3. LOGGING.md - Outdated Module Paths

**Spec Location:** `specs/LOGGING.md`

**Issue:** The "Current Implementation" table references incorrect module paths:

| Spec Reference | Actual Location |
|----------------|-----------------|
| `src/weakincentives/events.py` | `src/weakincentives/runtime/events/__init__.py` |
| `src/weakincentives/session/session.py` | `src/weakincentives/runtime/session/session.py` |

**Recommendation:** Update the spec to use the correct paths under `runtime/`.

---

## Verified Implementations (No Gaps)

The following specs have been verified as correctly implemented:

- **ADAPTERS.md** - OpenAI and LiteLLM adapters exist and match spec
- **ASTEVAL.md** - `AstevalSection` and related types implemented in `tools/asteval.py`
- **DBC.md** - `@require`, `@ensure`, `@invariant`, `@pure` decorators in `dbc/__init__.py`
- **DEADLINES.md** - `Deadline` class with validation in `deadlines.py`
- **EVENTS.md** - `InProcessEventBus` with `RLock` thread safety implemented
- **FROZEN_DATACLASSES.md** - `FrozenDataclass` decorator with `__pre_init__`, `update`, `merge`, `map` helpers
- **LITE_LLM_ADAPTER.md** - LiteLLM adapter with proper dependency handling
- **NATIVE_OPENAI_STRUCTURED_OUTPUTS.md** - Native structured outputs implemented
- **OPENAI_RESPONSES_API.md** - Migration complete, uses `responses.create`
- **PLANNING_STRATEGIES.md** - `PlanningStrategy` enum with all three members
- **PLANNING_TOOL.md** - Full planning tool suite in `tools/planning.py`
- **PODMAN_SANDBOX.md** - Podman sandbox tools in `tools/podman.py`
- **PROMPTS.md** - Full prompt system implemented
- **PROMPTS_COMPOSITION.md** - `DelegationPrompt` and composition helpers in `prompt/composition.py`
- **PROMPT_OVERRIDES.md** - `LocalPromptOverridesStore` with full API
- **SESSIONS.md** - Session with Redux-like reducers and thread safety
- **STRUCTURED_OUTPUT.md** - `Prompt[OutputT]` specialization works as specified
- **SUBAGENTS.md** - Subagent dispatch tools in `tools/subagents.py`
- **THREAD_SAFETY.md** - `RLock` added to `InProcessEventBus` and `Session`
- **THROTTLING.md** - Throttle handling in adapters
- **TOOLS.md** - Tool system with `Tool.wrap` helper
- **VFS_TOOLS.md** - Virtual filesystem tools in `tools/vfs.py`
- **WINK_DEBUG.md** - Debug CLI and FastAPI server in `cli/`
- **WORKSPACE_DIGEST.md** - `WorkspaceDigestSection` and `optimize()` method

---

## Action Items

1. **TODO_TOOL.md**: Decide whether to implement the TODO tool or archive the spec
2. **DATACLASS_SERDE.md**: Update Behaviour Map to reflect actual module structure
3. **LOGGING.md**: Fix module paths to use `runtime/` prefix
