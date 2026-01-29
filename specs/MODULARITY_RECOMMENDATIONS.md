# Modularity Recommendations for AI Context Management

This document analyzes the weakincentives codebase and provides recommendations
for improving modularity to make it easier for AI assistants to work with the
code in a compartmentalized fashion.

## Executive Summary

The codebase is generally well-organized with clear architectural layers. The
`runtime/session` package is exemplary—19 focused files averaging ~250 lines
each. However, several modules have grown large enough to impede efficient
AI-assisted development:

| File | Lines | Issue |
|------|-------|-------|
| `cli/query.py` | 2,043 | Mixed concerns: schema, SQL, parsing |
| `contrib/tools/podman.py` | 1,336 | Large tool implementation |
| `adapters/claude_agent_sdk/_hooks.py` | 1,240 | Multiple hook categories |
| `debug/bundle.py` | 1,219 | Bundle read/write/inspect mixed |
| `adapters/claude_agent_sdk/isolation.py` | 1,215 | Large single concern |
| `contrib/mailbox/_redis.py` | 1,181 | Complex implementation |
| `adapters/claude_agent_sdk/adapter.py` | 1,148 | Core adapter logic |
| `contrib/tools/asteval.py` | 1,099 | Large tool implementation |
| `serde/parse.py` | 1,024 | 46 functions, AST resolver |
| `runtime/agent_loop.py` | 1,006 | Core orchestration |
| `prompt/overrides/validation.py` | 852 | Validation logic |
| `prompt/registry.py` | 812 | Registry implementation |
| `dbc/__init__.py` | 762 | All decorators in root |
| `formal/__init__.py` | 702 | Utility module |
| `prompt/tool.py` | 658 | Tool definitions |

---

## Priority 1: High-Impact Splits

### 1.1 Split `cli/query.py` (2,043 lines)

**Current state**: 40+ functions mixed with 6 dataclasses for schema output,
SQL conversion, transcript parsing, and slice extraction.

**Recommended structure**:
```
cli/
├── query.py              # Main CLI entry point (~300 lines)
├── _query/
│   ├── __init__.py       # Re-exports
│   ├── schema.py         # ColumnInfo, TableInfo, SchemaOutput (~200 lines)
│   ├── sql.py            # SQL generation and coercion (~300 lines)
│   ├── transcript.py     # Transcript extraction utilities (~250 lines)
│   └── slices.py         # Slice/snapshot parsing logic (~200 lines)
```

**Benefits**:
- AI can read schema definitions without loading SQL conversion code
- Changes to transcript parsing don't require context of entire file
- Each module fits in ~2,000 token context window

### 1.2 Split `serde/parse.py` (1,024 lines)

**Current state**: 46 functions for type coercion, AST-based type resolution,
and normalization all in one file.

**Recommended structure**:
```
serde/
├── parse.py              # Main parse() function, dispatch (~300 lines)
├── _coercers.py          # Primitive coercion: bool, int, datetime (~200 lines)
├── _ast_resolver.py      # AST-based type resolution (~400 lines)
├── _normalizers.py       # Sequence/mapping normalization (~150 lines)
```

**Benefits**:
- AST resolver is complex and self-contained—isolate it
- Coercion strategies are lookup tables—easy to test independently
- Main parse() becomes a dispatcher calling into focused modules

### 1.3 Extract `dbc/_pure.py` from `dbc/__init__.py` (762 lines)

**Current state**: All decorators (`@require`, `@ensure`, `@invariant`, `@pure`)
plus 27 internal functions in the root `__init__.py`.

**Recommended structure**:
```
dbc/
├── __init__.py           # Re-exports only (~50 lines)
├── _decorators.py        # @require, @ensure, @invariant (~300 lines)
├── _pure.py              # @pure decorator and guards (~200 lines)
├── _checking.py          # Contract evaluation logic (~200 lines)
```

**Benefits**:
- `@pure` has distinct implementation (attribute patching)
- Contract checking logic is reusable infrastructure
- Reduces cognitive load when working on specific decorator

---

## Priority 2: Medium-Impact Improvements

### 2.1 Split `adapters/claude_agent_sdk/_hooks.py` (1,240 lines)

**Recommended**: Separate hooks by lifecycle phase:
```
adapters/claude_agent_sdk/
├── _hooks/
│   ├── __init__.py
│   ├── tool_hooks.py     # Tool execution hooks
│   ├── turn_hooks.py     # Turn lifecycle hooks
│   └── agent_hooks.py    # Agent-level hooks
```

### 2.2 Split `debug/bundle.py` (1,219 lines)

**Recommended**: Separate read/write/inspect concerns:
```
debug/
├── bundle.py             # BundleWriter, main API
├── _bundle_reader.py     # Reading and parsing bundles
├── _bundle_inspect.py    # Inspection and query utilities
```

### 2.3 Consider `contrib/tools/podman.py` (1,336 lines)

**Recommended**: If the tool has distinct phases, split by concern:
```
contrib/tools/
├── podman/
│   ├── __init__.py
│   ├── container.py      # Container management
│   ├── image.py          # Image building
│   └── network.py        # Network configuration
```

---

## Priority 3: Documentation and Navigation Aids

### 3.1 Add Module-Level Documentation Headers

For files >300 lines, add a structured header:

```python
"""Module name.

Quick Reference
---------------
- Class1: Brief description
- Class2: Brief description
- function1(): Brief description

Dependencies
------------
- module_a: What it provides
- module_b: What it provides

See Also
--------
- specs/RELATED_SPEC.md
- Related module: some.other.module
"""
```

**Benefits**: AI can read first 50 lines to understand module purpose and
decide if full context is needed.

### 3.2 Add ARCHITECTURE.md with Module Index

Create `/specs/ARCHITECTURE.md` with:

```markdown
# Architecture Overview

## Module Index

Quick reference for common tasks:

| Task | Primary Module | Related Specs |
|------|----------------|---------------|
| Parse dataclass from JSON | `serde.parse` | DATACLASSES.md |
| Define a tool | `prompt.tool` | TOOLS.md |
| Manage session state | `runtime.session` | SESSIONS.md, SLICES.md |
| Run agent loop | `runtime.agent_loop` | AGENT_LOOP.md |
| Design-by-contract | `dbc` | DBC.md |

## Dependency Graph (Simplified)

[ASCII art or Mermaid diagram showing key dependencies]
```

**Benefits**: AI can quickly identify which module to examine for a given task.

### 3.3 Add `_README.md` to Complex Subpackages

For packages with 5+ modules, add internal documentation:

```
runtime/session/_README.md
adapters/claude_agent_sdk/_README.md
prompt/overrides/_README.md
```

Content should include:
- Purpose of the subpackage
- How modules relate to each other
- Which module to modify for common changes

---

## Structural Patterns to Adopt

### Pattern A: Exemplary Decomposition (`runtime/session`)

The session subpackage demonstrates excellent modularity:

```
session/
├── __init__.py           # 406 lines - comprehensive re-exports + docs
├── session.py            # 838 lines - main Session class
├── _types.py             # Type definitions (private)
├── _slice_types.py       # Slice type definitions (private)
├── protocols.py          # Public protocols
├── reducers.py           # Built-in reducer functions
├── slice_accessor.py     # Slice access patterns
├── snapshots.py          # Snapshot implementation
└── ... (19 files total, avg 253 lines)
```

**Key characteristics**:
- Single responsibility per file
- Private modules (`_*.py`) for implementation details
- Rich `__init__.py` documentation
- Average file size ~250 lines

### Pattern B: Use `_subpackage/` for Implementation

When a public module grows large, extract implementation to private subpackage:

```python
# Before: one large file
module.py  # 1,500 lines

# After: thin public API with private implementation
module.py              # 200 lines - public API, imports from _module/
_module/
├── __init__.py        # Re-exports for internal use
├── implementation.py  # Core logic
├── helpers.py         # Utility functions
└── types.py           # Type definitions
```

### Pattern C: Spec-Driven Module Boundaries

Each major module should have a corresponding spec:

| Module | Spec |
|--------|------|
| `runtime/session/` | `specs/SESSIONS.md`, `specs/SLICES.md` |
| `prompt/` | `specs/PROMPTS.md`, `specs/TOOLS.md` |
| `adapters/` | `specs/ADAPTERS.md` |
| `dbc/` | `specs/DBC.md` |

**Benefits**: AI can read the spec first to understand intent before diving into
implementation.

---

## Immediate Actions

### Action 1: Split the Top 3 Files

Target these files for immediate refactoring:

1. `cli/query.py` → `cli/_query/` subpackage
2. `serde/parse.py` → Extract `_ast_resolver.py`
3. `dbc/__init__.py` → `dbc/_decorators.py`, `dbc/_pure.py`

### Action 2: Add Navigation Documentation

1. Create `specs/ARCHITECTURE.md` with module index
2. Add structured headers to files >500 lines
3. Add `_README.md` to complex subpackages

### Action 3: Establish File Size Guidelines

Add to `CLAUDE.md`:

```markdown
## File Size Guidelines

- **Target**: 200-400 lines per module
- **Warning**: >600 lines suggests splitting opportunity
- **Maximum**: 800 lines before mandatory review

When a file grows beyond 600 lines:
1. Identify distinct concerns (types, logic, utilities)
2. Extract to private modules (`_name.py`)
3. Keep public API in original module
4. Update `__init__.py` re-exports if needed
```

---

## Metrics for Success

After implementing these recommendations, target:

| Metric | Current | Target |
|--------|---------|--------|
| Files >600 lines | 16 | ≤5 |
| Avg lines/file | 334 | ~250 |
| Files with structured headers | ~30% | 100% for files >300 lines |
| Subpackages with `_README.md` | 0 | All complex subpackages |

---

## Summary

The codebase follows good architectural principles with clear layer separation.
The main opportunities for improvement are:

1. **Split large files** - Several files exceed 1,000 lines
2. **Add navigation aids** - Module index, structured headers, internal READMEs
3. **Establish conventions** - File size guidelines, documentation standards

These changes will allow AI assistants to:
- Load relevant context without reading entire packages
- Navigate to the right module quickly for common tasks
- Understand module boundaries without reading all code
- Make focused changes without loading unrelated code

The `runtime/session` package provides an excellent template—19 focused files
with clear responsibilities and comprehensive `__init__.py` documentation.
