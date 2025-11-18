# Prompt Optimizers

## Purpose

Prompt optimizers analyze reducer snapshots from completed Sessions and emit prompt
override plans so future executions inherit improvements without manual edits.
This document defines the shared contract that every optimizer must honor in
order to integrate cleanly with the runtime and override system.

## Core Concepts

- **Prompt**: Structured authoring artifact described in `specs/PROMPTS.md`.
- **Adapter**: Component that renders a prompt for a target model or toolchain
  and produces a `Session`.
- **Session**: Event-sourced execution trace composed of the `PromptRendered`,
  `ToolInvoked`, and `PromptExecuted` events defined in `specs/EVENTS.md`.
  Sessions are constructed through the reducer rules outlined in
  `specs/SESSIONS.md`.
- **Prompt override**: Persistent override payloads (`PromptOverride`,
  `SectionOverride`, `ToolOverride`, and associated hashes) defined in
  `specs/PROMPT_OVERRIDES.md`.
- **Prompt override store**: Any `PromptOverridesStore` implementation that
  mediates `resolve`, `upsert`, `delete`, and `seed_if_necessary` operations.

## Optimizer Lifecycle

1. The optimizer receives a batch of reducer snapshots plus access to the
   override store.
1. All snapshots in the batch MUST resolve to the same `PromptDescriptor`.
   Diverging descriptors are a planning error and MUST be detected by deriving
   descriptors from the snapshots.
1. The optimizer produces an immutable plan describing intended store
   mutations. Planning MUST NOT mutate snapshots or the store.
1. The runtime validates the plan against the current prompt descriptor.
1. If validation succeeds, the optimizer applies the plan atomically to the
   override store.
1. Optimizers SHOULD produce identical plans when re-run over the same batch.
   When the optimization strategy is inherently stochastic (for example, when
   driven by an external model), the implementation MUST clearly document the
   nondeterministic behavior and, when possible, expose controls (e.g., explicit
   seeds or reproducibility toggles) that callers can use to regain determinism.

## Required Interfaces

### Data Structures and Protocol

```python
from typing import NamedTuple, Protocol, Sequence

from weakincentives.prompt.overrides.versioning import PromptDescriptor
from weakincentives.runtime.session.snapshots import Snapshot
from weakincentives.prompt.overrides.versioning import PromptOverridesStore


class PromptOptimizer(Protocol):
    """Shared contract for prompt optimizers."""

    def plan(
        self,
        snapshots: Sequence[Snapshot],
        *,
        tag: str = "latest",
    ) -> OverridePlan: ...

    def apply(
        self,
        plan: OverridePlan,
        store: PromptOverridesStore,
    ) -> tuple[PromptOverride | None, ...]: ...
```

- `plan` inspects one or more snapshots and produces an `OverridePlan` scoped
  to a single override tag. Empty batches MUST raise. Callers SHOULD pass the
  full snapshot sequence for a session so planners can analyze how execution
  evolved before emitting overrides.
- `apply` commits the plan using `PromptOverridesStore.upsert` and
  `PromptOverridesStore.delete`. The return value lists the resulting overrides
  in mutation order, inserting `None` entries for deletions. Validation failures
  MUST surface as `PromptOverridesError`.

### Override Plan Contract

`OverridePlan` is an immutable record containing:

- `descriptor`: The `PromptDescriptor` derived from the canonical prompt in the
  snapshots at planning time. Plans MUST persist this descriptor so apply time
  validation can detect drift.
- `upserts`: Ordered `PromptOverride` payloads ready for
  `PromptOverridesStore.upsert`. Each payload targets the `(ns, prompt_key, tag)`
  triple referenced by `descriptor` and MUST contain validated section and tool
  overrides.
- `deletions`: Ordered `OverrideLocator` values describing which override files
  to remove via `PromptOverridesStore.delete`.
- `rationale`: Optional human-readable explanation of the plan.
- `metadata`: Optional machine-readable diagnostics (confidence scores, etc.).

`OverrideLocator` is a frozen dataclass with `ns: str`, `prompt_key: str`, and
`tag: str`. Plans MUST be serializable for auditing. Planning MUST confirm that
all snapshots map back to the same descriptor; mismatches MUST raise.

## Validation Requirements

Before applying overrides, the optimizer (or caller) MUST:

1. Recompute the `PromptDescriptor` for each snapshot sequence under
   consideration by deriving it from the appropriate snapshots and confirm
   every descriptor matches `OverridePlan.descriptor` (namespace, prompt key,
   section hashes, and tag).
1. Validate each `PromptOverride` using the hashing and identifier rules in
   `specs/PROMPT_OVERRIDES.md`, including tool contract hashes.
1. Treat deletions targeting missing files as no-ops while continuing to ensure
   the `(ns, prompt_key, tag)` triples align with the descriptor namespace.
1. Abort the entire apply step when any validation fails. Rely on the atomic
   file replacement semantics provided by `PromptOverridesStore` rather than
   implementing partial writes.

## Session Analysis Expectations

- Planning SHOULD draw conclusions from the `PromptRendered`, `ToolInvoked`, and
  `PromptExecuted` event slices held in each snapshot sequence provided to the
  optimizer.
- Optimizers MAY use additional analytics or scoring helpers, but they MUST NOT
  perform network I/O unless configuration explicitly enables it.
- Session analysis MUST be idempotent and side-effect free. Reducer-derived
  state is a read-only input.

## Override Store Interaction

- Use `PromptOverridesStore.seed_if_necessary` only when a backing file must be
  created before mutations occur.
- Stage work in `plan` and commit only during `apply`; never mutate the store
  during planning.
- Depend on `PromptOverridesStore.upsert` for atomic replacement instead of
  crafting bespoke merge logic.

## Observability

- Optimizers rely exclusively on the existing session event stream. Do **not**
  introduce new event types for the optimization lifecycle.
- Structured logging is optional and MUST reuse existing telemetry sinks; no
  additional telemetry hooks are required by default.

## Error Handling

- `plan` SHOULD raise deterministic, typed exceptions for invalid input or
  conflicting state.
- `apply` MUST raise `PromptOverridesError` whenever store validation fails or
  the descriptor has drifted, enabling callers to re-plan with fresh
  snapshots.
- Exceptions SHOULD preserve references to the descriptor and snapshot data that
  triggered them.

## Extensibility Guidelines

- Optimizers MAY expose additional configuration while preserving the base
  protocol.
- Extensions to `OverridePlan` MUST be backward compatible.
- Shared utilities belong under `src/weakincentives/runtime/optimizers/` and
  MUST reuse the validators in
  `src/weakincentives/prompt/overrides/validation.py` when touching override
  payloads.

## Testing Requirements

- Unit tests MUST exercise both planning and apply paths, including validation
  errors and descriptor drift scenarios.
- Integration tests SHOULD verify that plans produced by the optimizer update
  the override store correctly when executed through the runtime orchestration
  layer.
- Regression tests SHOULD accompany fixes for previously observed defects.

## Documentation Expectations

- Each optimizer MUST document its strategy, configuration options, and runtime
  assumptions.
- Update `CHANGELOG.md` whenever the shared contract changes or when introducing
  a new optimizer implementation.
