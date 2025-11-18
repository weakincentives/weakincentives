# Trajectory Specification

## Purpose

Trajectories capture the full execution context required for downstream
optimizers and analytics to reason about how a prompt evolves over multiple
renderings. Each trajectory binds the ordered reducer snapshots produced after
every render so consumers can reconstruct the complete story of an execution
without re-hydrating intermediate runtime state or storing prompt bodies.

## Core Concepts

- **Trajectory**: An immutable record containing the reducer snapshots captured
  after each rendering in execution order. The final snapshot reflects the
  terminal session state. Trajectories guarantee that every snapshot in the
  sequence maps back to the same `(ns, key, tag)` tuple when its descriptor is
  re-derived.
- **Snapshot**: The serialized reducer output defined in
  `specs/SESSIONS.md`. Snapshots contain every `PromptRendered`, `ToolInvoked`,
  and `PromptExecuted` event emitted across the prompt sequence up to the point
  they were captured. Each snapshot also records the `PromptDescriptor`
  necessary to fetch the canonical prompt and construct overrides.
- **Prompt descriptor**: The `PromptDescriptor` computed for the canonical prompt
  resolved by the runtime. Descriptors encode namespace, prompt key, tag, and the
  hash lineage for sections and tools as described in
  `specs/PROMPT_OVERRIDES.md`. Trajectories do not persist prompt bodies; callers
  MUST use the descriptors embedded in snapshots to rehydrate the prompt when
  needed.

## Data Model

```python
from typing import NamedTuple, Sequence

from weakincentives.runtime.session.snapshots import Snapshot


class Trajectory(NamedTuple):
    snapshots: Sequence[Snapshot]
```

- `snapshots`: Ordered reducer snapshots recorded after each prompt evaluation.
  The sequence MUST contain at least one snapshot. The final snapshot MUST
  represent the authoritative end state and be object-equal to the snapshot that
  would be produced by rerunning the reducer on the full event stream.

## Construction Rules

1. After every render completes, snapshot the session using the reducer output
   rules in `specs/SESSIONS.md` and append it to the trajectory.
1. After the final render completes, compute the prompt descriptor for the
   canonical prompt using the final snapshot.
1. Validate that every snapshot in the sequence resolves to the same descriptor
   triple. Divergence MUST raise and the trajectory MUST be discarded.
1. Freeze the snapshot sequence to guarantee immutability before exposing the
   trajectory to consumers.

## Session Semantics

- Reducer snapshots are authoritative for events. Consumers MUST rely on the
  event list contained in each snapshot instead of replaying prompts or tools.
- Trajectories MUST NOT alter snapshot ordering or contents. Any enrichment
  occurs in caller-managed metadata that lives outside the core trajectory
  payload.
- When prompts trigger concurrent tool execution, the reducer ordering rules in
  `specs/SESSIONS.md` dictate how events appear in each snapshot.

## Validation Requirements

- Trajectory constructors MUST reject empty snapshot sequences.
- If the reducer emits validation errors, the trajectory MUST NOT be created.
- When trajectories serialize to disk, snapshots MUST serialize using the codecs
  in `src/weakincentives/runtime/session/serde.py`. Prompt bodies SHOULD be
  rehydrated lazily from the override store using the descriptors embedded in
  each snapshot.

## Persistence Guidance

- Persist trajectories as append-only logs so historical executions remain
  accessible for auditing and offline analysis.
- Persist ancillary diagnostics separately from the trajectory payload. Large
  attachments (e.g., binary artifacts) SHOULD live in external object storage
  referenced by durable identifiers.
- When persisting batches, ensure they remain addressable by the descriptor so
  optimizers can retrieve the relevant history without scanning unrelated runs.

## Usage Guidelines

- Optimizers MUST treat trajectories as read-only inputs. Plans generated from a
  batch MUST re-derive the descriptor from the snapshots contained in each
  trajectory.
- Analytics pipelines MAY derive additional metrics but MUST persist results
  separately to avoid mutating the underlying trajectory record.
- When replaying executions for debugging, prefer the recorded snapshots over
  re-running prompts unless investigating non-deterministic adapter behavior.
