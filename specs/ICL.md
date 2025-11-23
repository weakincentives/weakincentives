# In-Context Learning (ICL) Optimizer

A lightweight contract for wiring an "optimize" command that refreshes repository instructions by scanning the workspace, exercising tools, and reflecting on recent turns. Keep the surface small so orchestrators can drop it into any session.

## Core Contract

- Run inside a temporary `Session` so the caller's planning state stays untouched while history is still readable.
- Expect a structured response with `instructions`, `workspace_digest`, `tool_digest`, and `history_digest`; fall back to plain text when adapters cannot emit structured data.
- Persist `instructions` into the `repository-instructions` override; other digests are optional telemetry.

### API Sketch

```python
@dataclass(slots=True)
class OptimizationObjective:
    focus: str

@dataclass(slots=True)
class OptimizationResponse:
    instructions: str
    workspace_digest: str
    tool_digest: str
    history_digest: str

class InContextLearningOptimizer:
    def __init__(self, adapter, overrides_store, overrides_tag):
        ...

    def run(self, *, session, objective) -> OptimizationResponse | None:
        ...
```

## Prompt Shape

1. **Optimization Brief**: Goal, repository overview, and JSON response contract.
1. **History Retrospective**: Latest session snapshot (limit to a few turns) with explicit mention when empty.
1. **Planning Tools Section**: Default to PAR strategy and include subagents where available.
1. **Workspace Tools Section**: Prefer Podman sandbox when available; otherwise use VFS.
1. **Optimization Objective**: User-provided focus text.

## Exploration Expectations

- **Filesystem**: Breadth-first directory scan followed by targeted reads of canonical docs (README, build scripts, specs). Record languages, frameworks, and commands.
- **Tools**: Exercise at least one planning cycle and inspect tool registries to surface quotas, limits, and best practices.
- **History**: Call out repeated mistakes or tactics from recent turns; state when no history exists.

### VFS Optimization Target

- Render an empty subsection reserved for repository-specific VFS guidance.
- Allow overrides to populate it so optimized instructions can bake in VFS quirks without code changes.

### Podman Optimization Target

- Render an empty subsection reserved for Podman sandbox guidance.
- Accept overrides to inject sandbox expectations (mount points, execution limits) when present.

## Error Handling & Normalization

- Trim and normalize Markdown before persisting instructions.
- Log and return `None` on adapter failures or missing outputs; callers handle messaging.

## Testing Notes

- Snapshot the rendered prompt to assert required sections (brief, history, planning, workspace, objective, VFS, Podman) appear in order.
- Use a mock adapter in integration tests to ensure overrides persist `instructions` into the repository override store.
- Unit test history sourcing by feeding fake events into a `Session` and verifying the retrospective content.
