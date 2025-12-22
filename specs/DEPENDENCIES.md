# Module Dependencies Specification

## Purpose

This document maps dependencies between core entities in weakincentives,
identifies coupling issues, and provides refactoring recommendations. It serves
as a guide for understanding the architecture and addressing technical debt.

## Dependency Hierarchy

The codebase follows a layered architecture with dependencies flowing downward:

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 6: Entry Points                                           │
│   └── cli                                                       │
├─────────────────────────────────────────────────────────────────┤
│ Layer 5: Extensions                                             │
│   └── contrib/tools, contrib/optimizers                         │
├─────────────────────────────────────────────────────────────────┤
│ Layer 4: Integration                                            │
│   └── adapters (openai, litellm, claude_agent_sdk)             │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: Core Systems                                           │
│   └── prompt, runtime, optimizers                              │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: Infrastructure                                         │
│   └── dataclasses, serde, budget                               │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1: Foundation                                             │
│   └── types, errors, deadlines, dbc                            │
└─────────────────────────────────────────────────────────────────┘
```

## Module Dependency Map

### Foundation Layer (No Internal Dependencies)

| Module | Purpose | Exports |
|--------|---------|---------|
| `types/` | JSON types, dataclass protocols | `JSONValue`, `SupportsDataclass` |
| `errors.py` | Exception hierarchy root | `WinkError`, `ToolValidationError` |
| `deadlines.py` | Wall-clock expiration | `Deadline` |
| `dbc/` | Design-by-contract decorators | `@require`, `@ensure`, `@pure` |

### Infrastructure Layer

| Module | Depends On | Exports |
|--------|------------|---------|
| `dataclasses/` | (stdlib only) | `FrozenDataclass` |
| `serde/` | `types`, `errors` | `dump`, `parse`, `schema`, `clone` |
| `budget/` | `types`, `errors`, `deadlines`, `dataclasses` | `Budget`, `BudgetTracker` |

### Core Systems Layer

| Module | Depends On | Key Exports |
|--------|------------|-------------|
| `prompt/` | `types`, `dbc`, `dataclasses`, `budget` | `Prompt`, `Tool`, `ToolContext`, `Section` |
| `runtime/` | `types`, `dbc`, `budget`, `deadlines`, `dataclasses` | `Session`, `EventBus`, `ExecutionState` |
| `optimizers/` | `prompt`, `runtime` | `PromptOptimizer`, `OptimizationContext` |

### Integration Layer

| Module | Depends On | Key Exports |
|--------|------------|-------------|
| `adapters/core` | `budget`, `deadlines`, `prompt`, `runtime` | `ProviderAdapter`, `PromptResponse` |
| `adapters/openai` | core + `serde`, `contrib.tools` | `OpenAIAdapter` |
| `adapters/litellm` | core + `serde` | `LiteLLMAdapter` |
| `adapters/claude_agent_sdk/` | all above | `ClaudeAgentSDKAdapter` |

### Extensions Layer

| Module | Depends On | Key Exports |
|--------|------------|-------------|
| `contrib/tools/` | `prompt`, `runtime`, `serde`, `dataclasses` | `Filesystem`, `VfsToolsSection`, `Plan` |
| `contrib/optimizers/` | `optimizers`, `contrib/tools` | `WorkspaceDigestOptimizer` |

## Critical Coupling Issues

### Issue 1: Prompt ↔ Runtime Circular Dependency

**Status**: MANAGED but PROBLEMATIC

**Location**:
- `prompt/visibility_overrides.py` ↔ `runtime/session/state_slice.py`
- `runtime/session/session.py` ↔ `prompt/visibility_overrides.py`

**Details**:

The `visibility_overrides` module is architecturally misplaced. It implements:
1. Session state slice functionality (belongs in `runtime.session`)
2. Prompt visibility logic (belongs in `prompt`)

This forces circular imports suppressed via `# pyright: reportImportCycles=false`.

```python
# visibility_overrides.py imports reducer from runtime
from ..runtime.session.state_slice import reducer

# session.py imports visibility_overrides at runtime
from ...prompt.visibility_overrides import register_visibility_reducers
```

**Impact**:
- Prompt module cannot be used without runtime session infrastructure
- Visibility feature cannot be disabled without breaking imports
- Import happens in a method to avoid import-time resolution

**Recommendation**: Move `visibility_overrides.py` to `runtime/session/` and
re-export from prompt for backward compatibility.

### Issue 2: Core ← Contrib Backward Dependency

**Status**: CRITICAL architectural violation

**Location**:
- `prompt/tool.py` imports `Filesystem` from `contrib.tools`
- `runtime/execution_state.py` imports `Filesystem`, `InMemoryFilesystem`
- `adapters/openai.py` imports `Filesystem`

**Details**:

The `Filesystem` protocol lives in contrib but is used by core modules:

```python
# prompt/tool.py
from ..contrib.tools.filesystem import Filesystem

class ToolContext:
    @property
    def filesystem(self) -> Filesystem: ...
```

This violates the "batteries" philosophy where contrib should depend on core,
not vice versa.

**Impact**:
- Core modules require contrib to function
- Cannot use core without contrib installed
- Blocks alternative filesystem implementations
- All adapters hardcoded to same Filesystem type

**Recommendation**: Move `Filesystem` protocol to `runtime/` or a new
`protocols/` module. Keep implementations in contrib.

### Issue 3: Adapter Tool Execution Duplication

**Status**: HIGH technical debt

**Location**:
- `adapters/tool_executor.py` (693 lines)
- `adapters/tool_runner.py` (256 lines)

**Details**:

Two tool execution patterns exist:

| Feature | tool_executor | tool_runner |
|---------|---------------|-------------|
| Snapshot/restore | ✓ Complex | ✓ Simple |
| Event publishing | ✓ | ✗ |
| Budget tracking | ✓ | ✗ |
| Error handling | 4 handlers | 1 handler |
| Used by | OpenAI, LiteLLM | None (orphaned) |

**Impact**:
- `tool_runner.py` is exported but never used
- Confusion about which pattern to use
- Duplicate error handling logic

**Recommendation**: Delete `tool_runner.py` and consolidate its test coverage.

### Issue 4: Claude Agent SDK Complexity

**Status**: MEDIUM - manageable but high cognitive load

**Location**: `adapters/claude_agent_sdk/` (2977 lines across 9 files)

**Details**:

The Claude Agent SDK adapter recreates machinery:
- Hook system (`_hooks.py`, 769 lines) duplicates event bus functionality
- Tool bridging (`_bridge.py`) duplicates `tool_executor` logic
- Bidirectional state sync creates coupling between two complex systems

**Impact**:
- Difficult to understand data flow
- Maintenance burden for parallel implementations
- High barrier to contribution

**Recommendation**: Document architectural decisions. Evaluate whether hooks
can be replaced by event bus. Consider consolidating tool bridging.

## Hub Modules (High Inbound Dependencies)

Ranked by number of modules that depend on them:

1. **`types/`** (15+ modules) - Type definitions
2. **`errors/`** (12+ modules) - Exception types
3. **`runtime/session/`** (11+ modules) - State management
4. **`prompt/`** (10+ modules) - Prompt composition
5. **`budget/`** (8+ modules) - Budget tracking
6. **`adapters/core`** (7+ modules) - Adapter interface
7. **`dataclasses/`** (7+ modules) - Immutability utilities

## Leaf Modules (No Internal Dependencies)

- `types/json.py`
- `types/dataclass.py`
- `errors.py`
- `deadlines.py`
- `runtime/snapshotable.py`
- `adapters/_names.py`
- `optimizers/_protocol.py`

## TYPE_CHECKING Usage

30+ files use `TYPE_CHECKING` guards to break import cycles:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime.session import SessionProtocol
```

This pattern is correctly applied for:
- Protocols used only in type hints
- Cross-layer references needed for static analysis
- Breaking circular dependencies at compile time

## Refactoring Priorities

### Priority 1: Move Filesystem to Core

**Action**: Move `contrib/tools/filesystem.py` (protocol only) to `runtime/`

**Files to modify**:
- Create `runtime/filesystem.py` with protocol
- Update `prompt/tool.py` imports
- Update `runtime/execution_state.py` imports
- Update adapter imports
- Keep implementations in `contrib/tools/`

**Lines affected**: ~50 lines of imports

### Priority 2: Relocate visibility_overrides

**Action**: Move `prompt/visibility_overrides.py` to `runtime/session/`

**Files to modify**:
- Move file to `runtime/session/visibility_overrides.py`
- Create re-export in `prompt/visibility_overrides.py`
- Update `session.py` internal import

**Lines affected**: ~20 lines of imports

### Priority 3: Remove tool_runner.py

**Action**: Delete orphaned `adapters/tool_runner.py`

**Files to modify**:
- Delete `adapters/tool_runner.py`
- Update `adapters/__init__.py` exports
- Move relevant tests to `tool_executor` test file

**Lines affected**: -256 lines

### Priority 4: Split utilities.py

**Action**: Break `adapters/utilities.py` (395 lines) into focused modules

**New modules**:
- `adapters/tool_specs.py` - Tool specification building
- `adapters/tool_parsing.py` - Argument parsing
- `adapters/response_utils.py` - Response extraction
- `adapters/config_utils.py` - Deadline/config utilities

**Lines affected**: ~50 lines of new import statements

## Dependency Health Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Total Python files | 103 | - |
| Files with TYPE_CHECKING guards | 30 | ✓ Good |
| Files with circular import suppression | 8 | ⚠ Needs reduction |
| Runtime circular imports | 0 | ✓ Good |
| Core modules importing from contrib | 3 | ✗ Should be 0 |
| Average dependencies per module | 2.1 | ✓ Good |
| Max dependencies (openai.py) | 8 | ⚠ Monitor |

## Invariants

The following dependency rules should be maintained:

1. **Foundation modules import nothing from weakincentives**
   - `types/`, `errors.py`, `deadlines.py` have zero internal imports

2. **Layering is strictly downward**
   - Higher layers may import lower layers
   - Lower layers never import higher layers
   - Exception: TYPE_CHECKING guards for protocols

3. **Contrib depends on core, never reverse**
   - Core modules should not import from `contrib/`
   - Resource injection via protocols, not direct imports

4. **Adapters are read-only consumers**
   - Adapters import from core but core never imports from adapters
   - Provider-specific logic stays encapsulated

5. **CLI imports everything**
   - CLI is the top layer with full access
   - No other module imports from CLI

## See Also

- `specs/ADAPTERS.md` - Adapter protocol specification
- `specs/SESSIONS.md` - Session state management
- `specs/TOOLS.md` - Tool system specification
- `specs/PROMPTS.md` - Prompt composition system
