# Test Coverage Gap Analysis

This document identifies important gaps in test coverage for the weakincentives codebase.

## Summary

The codebase has good overall test coverage with 82 test files covering 73 source files. However, several modules lack dedicated tests or have incomplete coverage.

## High Priority Gaps

### 1. `tools/digests.py` - WorkspaceDigest Module

**Location**: `src/weakincentives/tools/digests.py` (176 lines)

This module contains critical functionality with no dedicated test file:

- `set_workspace_digest()` - persists digests to session
- `clear_workspace_digest()` - removes cached digests
- `latest_workspace_digest()` - retrieves most recent digest
- `WorkspaceDigestSection` class - renders cached workspace digest

**Current coverage**: Only used indirectly in `test_core_optimize.py` and `test_prompt_utils.py`

**Recommended tests**:
- Unit tests for CRUD operations on workspace digests
- `WorkspaceDigestSection.render()` with and without cached digests
- `WorkspaceDigestSection.clone()` validation
- Edge cases: empty digests, key normalization, placeholder fallback

### 2. `tools/_context.py` - Tool Context Validation

**Location**: `src/weakincentives/tools/_context.py` (39 lines)

Contains `ensure_context_uses_session()` which validates tool execution context but has **zero direct tests**.

**Recommended tests**:
- Mismatched session raises `RuntimeError`
- Mismatched event bus raises `RuntimeError`
- Matching session/bus passes validation

### 3. `runtime/session/selectors.py` - Session Query Helpers

**Location**: `src/weakincentives/runtime/session/selectors.py` (57 lines)

Pure functions decorated with `@pure` but no direct tests:

- `select_all()` - return entire slice
- `select_latest()` - return most recent item
- `select_where()` - filter by predicate

**Recommended tests**:
- Empty session returns empty results
- Multiple items returns correct ordering
- Predicate filtering works correctly

### 4. `runtime/session/dataclasses.py` - TypeGuard Helper

**Location**: `src/weakincentives/runtime/session/dataclasses.py` (30 lines)

Contains `is_dataclass_instance()` TypeGuard function with no direct tests.

**Recommended tests**:
- Returns True for dataclass instances
- Returns False for dataclass types (classes, not instances)
- Returns False for non-dataclass objects

## Medium Priority Gaps

### 5. `prompt/response_format.py` - ResponseFormatSection

**Location**: `src/weakincentives/prompt/response_format.py` (76 lines)

Only has clone-related tests in `test_clone_contracts.py`. Missing:

- Tests for `ResponseFormatSection.render()` behavior
- Tests for template rendering with different `article`/`container` combinations
- Edge case: `clone()` when `default_params` is None

### 6. `prompt/_normalization.py` - Component Key Normalization

**Location**: `src/weakincentives/prompt/_normalization.py` (49 lines)

Has partial coverage in `test_section_base.py` but missing:

- Tests for boundary cases of the regex pattern (64-character keys)
- Tests for invalid characters at different positions
- Tests for whitespace stripping edge cases

### 7. `prompt/_structured_output_config.py` - StructuredOutputConfig

**Location**: `src/weakincentives/prompt/_structured_output_config.py` (32 lines)

Used across 6 test files but lacks focused tests for:

- Instantiation with various container types ("object" vs "array")
- Frozen dataclass immutability behavior
- Generic type parameter handling

### 8. `adapters/_names.py` - Adapter Name Constants

**Location**: `src/weakincentives/adapters/_names.py` (33 lines)

Simple constants module used in 3 test files but no explicit validation tests.

## Low Priority Gaps

### 9. `types/json.py` - JSON Type Aliases

**Location**: `src/weakincentives/types/json.py` (46 lines)

Type aliases used implicitly throughout the codebase. Consider adding:

- Import smoke tests
- Type validation tests using `typing.assert_type`

### 10. Serde Hypothesis Testing

**Location**: `tests/serde/test_dataclass_hypothesis.py` (111 lines)

Could be expanded for:

- More complex nested structures
- Union types with more than 2 alternatives
- Boundary values for numeric constraints
- Large collection sizes

## Coverage Summary Table

| Module | Priority | Lines | Current Tests | Gap Type |
|--------|----------|-------|---------------|----------|
| `tools/digests.py` | HIGH | 176 | 0 direct | No dedicated tests |
| `tools/_context.py` | HIGH | 39 | 0 | No tests |
| `runtime/session/selectors.py` | HIGH | 57 | 0 direct | No direct tests |
| `runtime/session/dataclasses.py` | MEDIUM | 30 | 0 | No tests |
| `prompt/response_format.py` | MEDIUM | 76 | Partial | Missing render tests |
| `prompt/_normalization.py` | MEDIUM | 49 | Partial | Missing edge cases |
| `prompt/_structured_output_config.py` | MEDIUM | 32 | Indirect | No focused tests |
| `adapters/_names.py` | LOW | 33 | Indirect | Constants validation |
| `types/json.py` | LOW | 46 | Indirect | Type validation |

## Recommended Actions

1. **Immediate**: Create dedicated test files for:
   - `tests/tools/test_digests.py`
   - `tests/tools/test_context.py`
   - `tests/runtime/test_selectors.py`

2. **Short-term**:
   - Expand `test_section_base.py` for `_normalization.py` edge cases
   - Add render tests for `ResponseFormatSection`
   - Add `is_dataclass_instance()` tests

3. **Longer-term**:
   - Run pytest-cov to get quantitative coverage metrics
   - Establish coverage thresholds (suggest 85%+ line, 90%+ branch)
   - Consider mutation testing to validate test quality
