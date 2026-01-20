# Complexity Reduction Plan for weakincentives

This document outlines opportunities to reduce complexity in the weakincentives library while preserving its strength and flexibility.

## Executive Summary

The codebase consists of ~50,000 lines across 174 files. While the architecture is fundamentally sound with clean separation of concerns, there are opportunities to reduce complexity through:

1. **Consolidating fragmented packages** (session has 24 files for 4,174 LOC)
2. **Eliminating backward compatibility shims** (alpha software, per CLAUDE.md)
3. **Extracting shared adapter code** (duplicate retry-after logic)
4. **Reorganizing contrib/tools** (clearer boundaries)

---

## Priority 1: Quick Wins (Low Risk, High Impact)

### 1.1 Remove Backward Compatibility Shims

**File:** `src/weakincentives/runtime/session/dataclasses.py` (24 lines)

This file exists solely to re-export `is_dataclass_instance` for backward compatibility:

```python
"""Dataclass helper utilities for session modules.

This module re-exports the canonical ``is_dataclass_instance`` helper from
:mod:`weakincentives.types.dataclass` for backward compatibility with
existing imports within the session module.
"""
from ...types.dataclass import is_dataclass_instance
__all__ = ["is_dataclass_instance"]
```

**Action:** Delete this file. Per CLAUDE.md: "Alpha software. APIs may change. Delete unused code completely; no backward-compatibility shims."

**Impact:** -24 lines, -1 file

### 1.2 Consolidate Tiny Type Alias Files

**Files to merge:**
- `runtime/session/_slice_types.py` (24 lines) - just 2 type aliases
- `runtime/session/_types.py` (57 lines) - 3 types + protocols

**Action:** Merge `_slice_types.py` into `_types.py`

**Impact:** -24 lines, -1 file

### 1.3 Merge `slice_policy.py` into Related Module

**File:** `runtime/session/slice_policy.py` (45 lines)

Contains only a single enum `SlicePolicy` and a constant. Too small to justify its own file.

**Action:** Move into `snapshots.py` (which heavily uses it) or `slices/_protocols.py`

**Impact:** -45 lines, -1 file

---

## Priority 2: Adapter Code Deduplication (Medium Risk, High Impact)

### 2.1 Extract Shared Retry-After Logic

Both `openai.py` and `litellm.py` contain nearly identical implementations of:

- `_coerce_retry_after()` - Parse retry-after values
- `_retry_after_from_error()` - Extract retry-after from error objects
- `_error_payload()` - Extract error payload for debugging

**Current state:**
```
openai.py: lines 123-189 (~66 lines)
litellm.py: lines 95-145 (~50 lines)
```

**Action:** Create `adapters/_retry_utils.py` with shared implementations:

```python
# adapters/_retry_utils.py
"""Shared retry-after parsing utilities for provider adapters."""

def coerce_retry_after(value: object) -> timedelta | None: ...
def retry_after_from_headers(headers: Mapping[str, Any] | None) -> timedelta | None: ...
def retry_after_from_error(error: object) -> timedelta | None: ...
def extract_error_payload(error: object) -> dict[str, Any] | None: ...
```

**Impact:** -60 lines of duplication, clearer adapter code

---

## Priority 3: Session Package Consolidation (Medium Risk)

The session package has 24 files for 4,174 LOC and exports 52 symbols. This fragmentation increases cognitive load.

### Current Structure:
```
runtime/session/
├── __init__.py (122 lines, 52 exports)
├── session.py (827)
├── snapshots.py (532)
├── slices/ (6 files, 824 lines)
│   ├── _jsonl.py (319)
│   ├── _memory.py (155)
│   ├── _protocols.py (163)
│   ├── _ops.py (74)
│   ├── _config.py (75)
│   └── __init__.py (38)
├── slice_accessor.py (272)
├── state_slice.py (277)
├── protocols.py (156)
├── session_view.py (196)
├── session_cloning.py (180)
├── visibility_overrides.py (186)
├── session_telemetry.py (132)
├── session_dispatch.py (52)
├── reducers.py (116)
├── reducer_context.py (45)
├── slice_mutations.py (107)
├── slice_policy.py (45)
├── _types.py (57)
├── _slice_types.py (24)
└── dataclasses.py (24) [backward compat shim]
```

### Recommended Consolidation:

**Phase 1: Merge small files**
- Delete `dataclasses.py` (backward compat shim)
- Merge `_slice_types.py` into `_types.py`
- Merge `slice_policy.py` into `snapshots.py`
- Merge `session_dispatch.py` (52 lines) into `session.py`
- Merge `reducer_context.py` (45 lines) into `reducers.py`

**Result:** -5 files, ~200 lines saved

**Phase 2: Consider merging (optional)**
- `session_telemetry.py` → could be part of `session.py`
- `session_cloning.py` → could be part of `session.py`
- `visibility_overrides.py` → standalone feature, keep separate

---

## Priority 4: Evaluate Underutilized Packages

### 4.1 `formal` Package (759 lines)

TLA+ formal specification support. Currently only used by `contrib/mailbox/_redis.py`.

**Options:**
1. Move into `contrib/formal/` since it's specialized infrastructure
2. Keep as-is if TLA+ verification is a strategic priority
3. Inline the decorators into `_redis.py` if no other usage planned

**Recommendation:** Move to `contrib/formal/` - it's powerful but niche

### 4.2 `optimizers` Package (456 lines)

Base classes and protocols for prompt optimizers. Only implementation is `contrib/optimizers/workspace_digest.py`.

**Status:** This is appropriate - protocols in core, implementations in contrib. No change needed.

---

## Priority 5: Contrib/Tools Reorganization (Low Priority)

### Current Issues:
1. VFS code split across 3 files (vfs.py + vfs_types.py + vfs_mounts.py = 2,002 lines)
2. Podman code split across 3 files (podman.py + podman_connection.py + podman_eval.py = 1,786 lines)
3. `filesystem_memory.py` (682 lines) implements Filesystem protocol but lives in contrib

### Recommendations:

**5.1 VFS Organization**
Keep current split - it's intentional:
- `vfs.py` - Section and tools
- `vfs_types.py` - Data types (frozen dataclasses)
- `vfs_mounts.py` - Mount logic

This follows the pattern of separating concerns.

**5.2 Consider Moving `InMemoryFilesystem`**
`contrib/tools/filesystem_memory.py` implements the core `Filesystem` protocol.

**Options:**
1. Move to `filesystem/_memory.py` alongside `_host.py`
2. Keep in contrib if it's only used by tools

**Recommendation:** Keep in contrib - it's specifically for tool testing scenarios

---

## Not Recommended (Would Increase Complexity)

### Protocol Consolidation
The protocols are appropriately distributed:
- `prompt/protocols.py` - Prompt-related
- `runtime/session/protocols.py` - Session-related
- `resources/protocols.py` - Resource lifecycle

Consolidating would create a massive file with unrelated protocols.

### Claude Agent SDK Simplification
The `adapters/claude_agent_sdk/` package (3,500+ lines) is complex but necessarily so:
- `isolation.py` handles security boundaries
- `_hooks.py` manages the callback system
- This complexity is inherent to the integration

---

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours)
1. [ ] Delete `runtime/session/dataclasses.py`
2. [ ] Merge `_slice_types.py` into `_types.py`
3. [ ] Merge `slice_policy.py` into `snapshots.py`
4. [ ] Run `make check` to verify

### Phase 2: Adapter Cleanup (2-3 hours)
1. [ ] Create `adapters/_retry_utils.py`
2. [ ] Update `openai.py` to use shared utils
3. [ ] Update `litellm.py` to use shared utils
4. [ ] Run `make check` to verify

### Phase 3: Session Consolidation (3-4 hours)
1. [ ] Merge `session_dispatch.py` into `session.py`
2. [ ] Merge `reducer_context.py` into `reducers.py`
3. [ ] Update imports and re-exports
4. [ ] Run `make check` to verify

### Phase 4: Package Reorganization (optional, 2-3 hours)
1. [ ] Move `formal/` to `contrib/formal/`
2. [ ] Update imports in `contrib/mailbox/_redis.py`
3. [ ] Run `make check` to verify

---

## Metrics

**Before:**
- 174 files
- ~50,000 lines
- Session package: 24 files

**After Phase 1-3:**
- ~167 files (-7)
- ~49,700 lines (-300)
- Session package: 18 files

**Key Principle:** Every change must pass `make check`. This ensures we maintain the library's correctness while reducing complexity.
