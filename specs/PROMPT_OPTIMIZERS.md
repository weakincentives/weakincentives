# Prompt Optimizer Specification

## Overview

**Scope:** This document defines the `PromptOptimizer` abstraction—a generic
interface for algorithms that transform, enhance, or generate content for
`Prompt` objects. It covers refactoring `ProviderAdapter.optimize()` into the
new `WorkspaceDigestOptimizer` class and establishes the contract future
optimization algorithms must follow.

**Design goals**

- Decouple optimization logic from provider adapters so optimizers can be
  composed, tested, and extended independently.
- Present a uniform interface so callers can swap optimization strategies
  without modifying orchestration code.
- Support both prompt-level transformations (digest generation, compression)
  and section-level refinements (few-shot synthesis, instruction rewriting).
- Preserve deterministic, side-effect-free execution aligned with session
  reducers and the DbC enforcement model.
- Keep optimization observable through the `EventBus` while isolating internal
  evaluation from the caller's session state.

**Constraints**

- Optimizers must run synchronously and honor caller-supplied deadlines.
- Optimization results must be serializable for persistence to override stores
  or session snapshots.
- Optimizers that require provider evaluation must accept an adapter
  dependency at construction time rather than importing concrete adapters.
- Internal sessions created by optimizers must not mutate the caller's session;
  use session cloning or isolated instances.

**Rationale for the abstraction**

- The current `ProviderAdapter.optimize()` method conflates adapter
  responsibilities (provider communication) with optimization logic (prompt
  composition, digest extraction, persistence). Extracting optimizers allows
  adapters to focus solely on evaluation.
- Different optimization strategies (workspace digest, prompt compression,
  few-shot generation, instruction tuning) share common patterns: they inspect
  a prompt, possibly evaluate helper prompts, and produce structured results.
  A shared interface captures this pattern.
- Optimization is inherently experimental; a pluggable interface encourages
  exploration without destabilizing core adapter code.

## Core Interfaces

### PromptOptimizer Protocol

```python
from typing import Protocol, TypeVar
from weakincentives.prompt import Prompt
from weakincentives.runtime.session.protocols import SessionProtocol

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)


class PromptOptimizer(Protocol[InputT, OutputT]):
    """Protocol for prompt optimization algorithms."""

    def optimize(
        self,
        prompt: Prompt[InputT],
        *,
        session: SessionProtocol,
    ) -> OutputT:
        """
        Apply the optimization algorithm to the given prompt.

        Args:
            prompt: The prompt to optimize.
            session: The caller's session for state queries and result
                persistence. Optimizers must not mutate this session
                during internal evaluation; use isolated sessions instead.

        Returns:
            An optimization result whose shape depends on the algorithm.
        """
        ...
```

The protocol is parameterized by:

- `InputT` – the prompt's output type, allowing optimizers to constrain which
  prompts they accept.
- `OutputT` – the result type returned by the optimizer, enabling type-safe
  consumption of optimization artifacts.

### OptimizationContext

Optimizers that require provider evaluation receive dependencies through an
`OptimizationContext` dataclass rather than direct adapter references:

```python
from dataclasses import dataclass
from weakincentives.adapters.core import ProviderAdapter
from weakincentives.deadlines import Deadline
from weakincentives.prompt.overrides import PromptOverridesStore
from weakincentives.runtime.events._types import EventBus
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class OptimizationContext:
    """Immutable context bundle for optimization algorithms."""

    adapter: ProviderAdapter[object]
    """Provider adapter for evaluating helper prompts."""

    event_bus: EventBus
    """Event bus for optimization telemetry."""

    deadline: Deadline | None = None
    """Optional deadline enforced during optimization."""

    overrides_store: PromptOverridesStore | None = None
    """Optional store for persisting or reading prompt overrides."""

    overrides_tag: str = "latest"
    """Tag for override versioning."""

    optimization_session: Session | None = None
    """
    Optional pre-configured session for optimization evaluation.
    When None, optimizers create isolated sessions internally.
    """
```

Optimizers that need evaluation capabilities accept the context at construction
or as an additional `optimize()` parameter, depending on whether the context is
fixed or varies per invocation.

### BasePromptOptimizer

A convenience base class provides common infrastructure:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from weakincentives.prompt import Prompt
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.protocols import SessionProtocol

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(slots=True, frozen=True)
class OptimizerConfig:
    """Base configuration shared across optimizers."""

    accepts_overrides: bool = True
    """Whether the optimization prompt respects override stores."""


class BasePromptOptimizer(ABC, Generic[InputT, OutputT]):
    """Abstract base for prompt optimizers with shared utilities."""

    def __init__(
        self,
        context: OptimizationContext,
        *,
        config: OptimizerConfig | None = None,
    ) -> None:
        self._context = context
        self._config = config or OptimizerConfig()

    @abstractmethod
    def optimize(
        self,
        prompt: Prompt[InputT],
        *,
        session: SessionProtocol,
    ) -> OutputT:
        """Subclasses implement optimization logic here."""
        ...

    def _create_optimization_session(
        self, prompt: Prompt[InputT]
    ) -> Session:
        """
        Create an isolated session for optimization evaluation.

        Reuses the context's optimization_session if provided; otherwise
        creates a fresh Session tagged for the optimization scope.
        """
        if self._context.optimization_session is not None:
            return self._context.optimization_session
        prompt_name = prompt.name or prompt.key
        return Session(
            tags={
                "scope": f"{self._optimizer_scope}_optimization",
                "prompt": prompt_name,
            }
        )

    @property
    @abstractmethod
    def _optimizer_scope(self) -> str:
        """Return a short identifier for session tagging."""
        ...
```

## Optimization Result Types

Each optimizer declares its own result type. Common patterns include:

### Generic OptimizationResult

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

from weakincentives.adapters.core import PromptResponse

ArtifactT = TypeVar("ArtifactT")


@dataclass(slots=True, frozen=True)
class OptimizationResult(Generic[ArtifactT]):
    """Generic result container for optimization algorithms."""

    response: PromptResponse[object] | None
    """
    The provider response from the optimization prompt, if evaluation
    occurred. None for purely analytical optimizers.
    """

    artifact: ArtifactT
    """The primary optimization artifact (digest, compressed prompt, etc.)."""

    metadata: dict[str, object]
    """
    Algorithm-specific metadata (token counts, compression ratio,
    section paths, etc.).
    """
```

### WorkspaceDigestResult

The existing `OptimizationResult` from `adapters.core` becomes a specific
instance of this pattern:

```python
from dataclasses import dataclass
from enum import StrEnum

from weakincentives.adapters.core import PromptResponse


class PersistenceScope(StrEnum):
    """Where optimization artifacts are stored."""

    SESSION = "session"
    GLOBAL = "global"


@dataclass(slots=True, frozen=True)
class WorkspaceDigestResult:
    """Result of workspace digest optimization."""

    response: PromptResponse[object]
    """The provider response from the optimization prompt."""

    digest: str
    """The extracted workspace digest text."""

    scope: PersistenceScope
    """Where the digest was persisted."""

    section_key: str
    """The WorkspaceDigestSection key that was updated."""
```

## Workspace Digest Optimizer

The current `ProviderAdapter.optimize()` method is refactored into a dedicated
`WorkspaceDigestOptimizer` class:

```python
from weakincentives.prompt import Prompt
from weakincentives.runtime.session.protocols import SessionProtocol


class WorkspaceDigestOptimizer(BasePromptOptimizer[object, WorkspaceDigestResult]):
    """
    Generate a workspace digest for prompts containing WorkspaceDigestSection.

    This optimizer composes an internal prompt that explores the mounted
    workspace and produces a task-agnostic summary. The result can be
    persisted to the session (SESSION scope) or the override store
    (GLOBAL scope).
    """

    def __init__(
        self,
        context: OptimizationContext,
        *,
        config: OptimizerConfig | None = None,
        store_scope: PersistenceScope = PersistenceScope.SESSION,
    ) -> None:
        super().__init__(context, config=config)
        self._store_scope = store_scope

    @property
    def _optimizer_scope(self) -> str:
        return "workspace_digest"

    def optimize(
        self,
        prompt: Prompt[object],
        *,
        session: SessionProtocol,
    ) -> WorkspaceDigestResult:
        """
        Generate and persist a workspace digest for the given prompt.

        Raises:
            PromptEvaluationError: If the prompt lacks required sections
                or digest extraction fails.
        """
        ...
```

### Implementation Workflow

The optimizer's `optimize()` method follows the same workflow as the current
`ProviderAdapter.optimize()`:

1. **Validate prerequisites** – Locate the `WorkspaceDigestSection` and
   workspace section (`PodmanSandboxSection` or `VfsToolsSection`). Raise
   `PromptEvaluationError` if either is missing.

2. **Create isolated session** – Use `_create_optimization_session()` to obtain
   a session that will not mutate the caller's state during tool execution.

3. **Clone session-aware sections** – Call `section.clone(session=...,
   bus=...)` on workspace and tool sections to bind them to the optimization
   session.

4. **Compose optimization prompt** – Build a `PromptTemplate` containing:
   - `MarkdownSection` for "Optimization Goal"
   - `MarkdownSection` for "Expectations"
   - `PlanningToolsSection` with `GOAL_DECOMPOSE_ROUTE_SYNTHESISE` strategy
   - Cloned tool sections (`AstevalSection`, etc.)
   - Cloned workspace section

5. **Evaluate** – Call `context.adapter.evaluate()` with the optimization
   prompt and isolated session.

6. **Extract digest** – Parse the response output or text to obtain the digest
   string.

7. **Persist result** – Based on `store_scope`:
   - `SESSION`: Call `set_workspace_digest()` on the caller's session.
   - `GLOBAL`: Call `overrides_store.set_section_override()` and clear the
     session entry via `clear_workspace_digest()`.

8. **Return result** – Package the response, digest, scope, and section key
   into `WorkspaceDigestResult`.

## Adapter Refactoring

### Before (Current Implementation)

```python
class ProviderAdapter(ABC):
    @abstractmethod
    def evaluate(self, prompt, *, ...) -> PromptResponse[OutputT]: ...

    def optimize(self, prompt, *, ...) -> OptimizationResult:
        # 200+ lines of optimization logic embedded in adapter
        ...
```

### After (Proposed Implementation)

```python
class ProviderAdapter(ABC):
    @abstractmethod
    def evaluate(self, prompt, *, ...) -> PromptResponse[OutputT]: ...

    # optimize() method removed from ProviderAdapter


# Standalone optimizer usage:
from weakincentives.optimizers import WorkspaceDigestOptimizer, OptimizationContext

context = OptimizationContext(
    adapter=adapter,
    event_bus=session.event_bus,
    overrides_store=overrides_store,
)
optimizer = WorkspaceDigestOptimizer(context, store_scope=PersistenceScope.SESSION)
result = optimizer.optimize(prompt, session=session)
```

### Refactoring Scope

The refactoring requires the following changes:

1. **Create `src/weakincentives/optimizers/` module** – Add `PromptOptimizer`
   protocol, `OptimizationContext`, `BasePromptOptimizer`, and
   `WorkspaceDigestOptimizer`.

2. **Delete `ProviderAdapter.optimize()`** – Remove the method and all helper
   methods (`_resolve_workspace_section`, `_resolve_tool_sections`,
   `_clone_section`, `_require_workspace_digest_section`, `_find_section_path`,
   `_extract_digest`) from `adapters/core.py`.

3. **Move result types** – Relocate `OptimizationScope` and `OptimizationResult`
   from `adapters/core.py` to the optimizers module. Delete
   `_OptimizationResponse` (internal to optimizer).

4. **Update callers** – Modify `code_reviewer_example.py` and any other code
   that calls `adapter.optimize()` to instantiate and use
   `WorkspaceDigestOptimizer` directly.

5. **Update ADAPTERS.md** – Remove the "Optimization API" and "Optimization
   Workflow" sections; add a cross-reference to this spec.

6. **Update WORKSPACE_DIGEST.md** – Replace references to
   `ProviderAdapter.optimize()` with `WorkspaceDigestOptimizer`.

## Future Optimizer Examples

The abstraction supports diverse optimization algorithms:

### PromptCompressor

Reduces token usage by summarizing verbose sections:

```python
@dataclass(slots=True, frozen=True)
class CompressionResult:
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    compressed_prompt: Prompt[object]


class PromptCompressor(BasePromptOptimizer[object, CompressionResult]):
    """Compress prompt sections to reduce token usage."""

    def optimize(
        self,
        prompt: Prompt[object],
        *,
        session: SessionProtocol,
    ) -> CompressionResult:
        # Analyze sections, identify verbose content, generate summaries
        ...
```

### FewShotSynthesizer

Generates example invocations for tools:

```python
@dataclass(slots=True, frozen=True)
class FewShotResult:
    tool_name: str
    examples: tuple[ToolExample[object, object], ...]


class FewShotSynthesizer(BasePromptOptimizer[object, tuple[FewShotResult, ...]]):
    """Generate few-shot examples for prompt tools."""

    def optimize(
        self,
        prompt: Prompt[object],
        *,
        session: SessionProtocol,
    ) -> tuple[FewShotResult, ...]:
        # Evaluate tools, generate representative examples
        ...
```

### InstructionRefiner

Improves section instructions based on evaluation feedback:

```python
@dataclass(slots=True, frozen=True)
class RefinementResult:
    section_path: tuple[str, ...]
    original_template: str
    refined_template: str
    improvement_rationale: str


class InstructionRefiner(BasePromptOptimizer[object, tuple[RefinementResult, ...]]):
    """Refine section instructions based on model feedback."""

    def optimize(
        self,
        prompt: Prompt[object],
        *,
        session: SessionProtocol,
    ) -> tuple[RefinementResult, ...]:
        # Evaluate prompt, analyze failures, suggest improvements
        ...
```

## Module Structure

```
src/weakincentives/optimizers/
├── __init__.py           # Public exports
├── _context.py           # OptimizationContext dataclass
├── _protocol.py          # PromptOptimizer protocol
├── _base.py              # BasePromptOptimizer, OptimizerConfig
├── _results.py           # Shared result types
├── workspace_digest.py   # WorkspaceDigestOptimizer
└── (future modules)      # Additional optimizer implementations
```

## Events and Observability

Optimizers publish events through the `OptimizationContext.event_bus`. The
shared event types include:

- `OptimizationStarted` – Emitted when `optimize()` begins, capturing the
  prompt descriptor and optimizer type.
- `OptimizationCompleted` – Emitted on successful completion with the result
  summary.
- `OptimizationFailed` – Emitted when optimization raises an exception,
  capturing the error context.

Internal evaluation events (`PromptRendered`, `ToolInvoked`, `PromptExecuted`)
flow through the optimization session's event bus and are isolated from the
caller's bus unless the caller provides a shared `optimization_session`.

## Design-by-Contract Considerations

Optimizers follow DbC conventions from `specs/DBC.md`:

- **Preconditions** – `optimize()` validates that required sections exist
  before proceeding. Missing prerequisites raise `PromptEvaluationError` with
  `phase="request"`.
- **Postconditions** – Results are immutable frozen dataclasses. Digest
  strings are stripped of leading/trailing whitespace.
- **Invariants** – The caller's session is never mutated during internal
  evaluation. All mutations occur through explicit persistence calls after
  optimization completes.

## Testing Requirements

- Unit tests cover optimizer construction, prerequisite validation, and result
  serialization.
- Integration tests verify end-to-end optimization with real provider calls
  (requires `OPENAI_API_KEY`).
- Mock adapters validate that optimizers correctly delegate evaluation without
  coupling to specific providers.
- Session isolation tests confirm that internal tool execution does not affect
  the caller's session state.

## Limitations and Caveats

- **Synchronous execution** – Optimizers run synchronously. Long-running
  optimizations should consider chunking or progress callbacks in future
  iterations.
- **Provider dependency** – Optimizers that evaluate helper prompts require a
  functional adapter. Pure analytical optimizers (e.g., token counters) can
  operate without provider access.
- **Override store coupling** – `GLOBAL` persistence requires a configured
  override store. Callers must ensure the store is writable and the tag is
  appropriate for the deployment context.
- **Alpha stability** – The optimizer interface may evolve as additional
  algorithms are implemented. Backward compatibility shims will not be added;
  migrate promptly when interfaces change.

## Related Specifications

| Spec | Relationship |
|------|--------------|
| `ADAPTERS.md` | Documents the `ProviderAdapter` being refactored |
| `WORKSPACE_DIGEST.md` | Covers digest lifecycle and section behavior |
| `PROMPT_OVERRIDES.md` | Details the override store used for GLOBAL persistence |
| `SESSIONS.md` | Explains session isolation and state management |
| `PROMPTS.md` | Describes `Prompt` and `PromptTemplate` structures |
| `TOOLS.md` | Covers tool sections cloned during optimization |
| `DBC.md` | Guides contract enforcement in optimizer methods |
