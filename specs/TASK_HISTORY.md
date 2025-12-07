# Task History Section Specification

## Status

**Exploratory Draft** - This specification captures design thinking for a
plan-aligned conversation history system. Implementation details are subject
to change.

## Purpose

The `TaskHistorySection` provides a structured approach to conversation history
management where context is organized around plan steps rather than raw message
sequences. This enables model-managed context compaction aligned with task
semantics.

Traditional compaction algorithms treat conversation history as an opaque blob
to compress. This specification proposes treating history as structured work
aligned with the planning task graph, allowing the model to make semantic
decisions about what context to retain, summarize, or expand.

## Guiding Principles

- **Task-Aligned Granularity**: History organizes around plan steps, not
  arbitrary message counts. Summarization boundaries follow task boundaries.
- **Model-Managed Context**: The model decides when to compact via tools rather
  than external heuristics. Models have semantic understanding of importance.
- **Progressive Disclosure Integration**: Reuses the existing `SectionVisibility`
  and `open_sections` machinery at step-level granularity.
- **Session-First Storage**: History lives in the Session as typed slices,
  enabling snapshots, rollback, and inspection.
- **Composable with Planning**: Integrates naturally with `PlanningToolsSection`
  to track which turns belong to which steps.

## Core Data Model

### Turn

Individual conversation turn within a step's context:

```python
@FrozenDataclass()
class Turn:
    """Single conversation turn associated with a plan step."""

    turn_id: int
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_name: str | None = None  # For role="tool"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def render(self) -> str:
        prefix = f"[{self.role}]"
        if self.tool_name:
            prefix = f"[tool:{self.tool_name}]"
        return f"{prefix} {self.content}"
```

### StepHistory

Accumulated history for a single plan step with summarization support:

```python
StepVisibility = Literal["full", "summary"]

@FrozenDataclass()
class StepHistory:
    """Conversation history scoped to a single plan step."""

    step_id: int
    step_title: str
    turns: tuple[Turn, ...] = ()
    summary: str | None = None  # Populated when summarized
    visibility: StepVisibility = "full"

    def render(self, *, include_turns: bool = True) -> str:
        lines = [f"Step {self.step_id}: {self.step_title}"]
        if self.visibility == "summary" and self.summary:
            lines.append(f"[Summary] {self.summary}")
            lines.append("[Call `expand_step` with this step_id for full history]")
        elif include_turns:
            for turn in self.turns:
                lines.append(f"  {turn.render()}")
        return "\n".join(lines)
```

### TaskHistory

Top-level container linking history to the active plan:

```python
@FrozenDataclass()
class TaskHistory:
    """Plan-aligned conversation history with per-step summarization."""

    objective: str
    active_step_id: int | None = None  # Current step receiving turns
    steps: tuple[StepHistory, ...] = ()

    def render(self) -> str:
        lines = [f"Task: {self.objective}", ""]
        if not self.steps:
            lines.append("<no history recorded>")
        else:
            for step in self.steps:
                lines.append(step.render())
                lines.append("")
        return "\n".join(lines)
```

## TaskHistorySection

Custom section that renders plan-aligned history from session state:

```python
class TaskHistorySection(Section[TaskHistory]):
    """Renders conversation history organized by plan steps."""

    def __init__(
        self,
        *,
        session: Session,
        title: str = "Task History",
        key: str = "task-history",
        accepts_overrides: bool = False,
    ) -> None:
        self._session = session
        self._initialize_session(session)
        tools = _build_history_tools(section=self)
        super().__init__(
            title=title,
            key=key,
            default_params=None,
            tools=tools,
            accepts_overrides=accepts_overrides,
            visibility=SectionVisibility.FULL,
        )

    @staticmethod
    def _initialize_session(session: Session) -> None:
        """Register reducers for history accumulation."""
        session.mutate(TaskHistory).register(TaskHistory, replace_latest)
        session.mutate(TaskHistory).register(AppendTurn, _append_turn_reducer)
        session.mutate(TaskHistory).register(SummarizeStep, _summarize_step_reducer)
        session.mutate(TaskHistory).register(ExpandStep, _expand_step_reducer)
        session.mutate(TaskHistory).register(SetActiveStep, _set_active_step_reducer)

    def render(
        self,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        *,
        visibility: SectionVisibility | None = None,
    ) -> str:
        history = self._session.query(TaskHistory).latest()
        if history is None:
            return self._render_empty(depth, number)

        lines = [self._heading(depth, number)]
        lines.append(f"**Objective**: {history.objective}")
        lines.append("")

        for i, step in enumerate(history.steps):
            step_num = f"{number}.{i + 1}"
            lines.append(self._render_step(step, depth + 1, step_num))

        return "\n".join(lines)

    def _render_step(
        self,
        step: StepHistory,
        depth: int,
        number: str,
    ) -> str:
        hashes = "#" * (depth + 2)
        status_marker = "[active]" if self._is_active(step) else ""
        heading = f"{hashes} {number}. {step.step_title} {status_marker}".strip()

        if step.visibility == "summary":
            return (
                f"{heading}\n\n"
                f"{step.summary}\n\n"
                f"_Call `expand_step({step.step_id})` for full conversation._"
            )

        lines = [heading, ""]
        for turn in step.turns:
            lines.append(self._render_turn(turn))
        return "\n".join(lines)

    def _render_turn(self, turn: Turn) -> str:
        if turn.role == "tool":
            return f"**[{turn.tool_name}]**: {turn.content}"
        return f"**{turn.role.title()}**: {turn.content}"

    def _is_active(self, step: StepHistory) -> bool:
        history = self._session.query(TaskHistory).latest()
        return history is not None and history.active_step_id == step.step_id
```

## Model-Managed Context Tools

### summarize_step

Compresses a step's conversation history to free context space:

```python
@FrozenDataclass()
class SummarizeStepParams:
    step_id: int = field(
        metadata={"description": "ID of the step to summarize."}
    )
    summary: str = field(
        metadata={"description": "Concise summary of the step's conversation (1-3 sentences)."}
    )

summarize_step_tool = Tool[SummarizeStepParams, SummarizeStepParams](
    name="summarize_step",
    description=(
        "Compress a step's conversation history into a summary. "
        "Use this to free context space after completing a step or when "
        "the full history is no longer needed."
    ),
    handler=summarize_step_handler,
)
```

The model generates the summary itself, ensuring semantic preservation of
important details. This is preferable to external summarization because:

1. The model has full context about what information matters
2. The model can preserve details relevant to future steps
3. Summarization aligns with the model's understanding of the task

### expand_step

Restores full history for a summarized step:

```python
@FrozenDataclass()
class ExpandStepParams:
    step_id: int = field(
        metadata={"description": "ID of the step to expand."}
    )
    reason: str = field(
        metadata={"description": "Why the full history is needed."}
    )

expand_step_tool = Tool[ExpandStepParams, StepHistory](
    name="expand_step",
    description=(
        "Restore full conversation history for a summarized step. "
        "Use when you need to review details that were compressed."
    ),
    handler=expand_step_handler,
)
```

The handler raises `VisibilityExpansionRequired` to signal re-rendering,
mirroring the `open_sections` pattern.

### context_pressure (Optional)

Exposes context utilization to guide compaction decisions:

```python
@FrozenDataclass()
class ContextPressureParams:
    pass

@FrozenDataclass()
class ContextPressure:
    used_tokens: int
    max_tokens: int
    utilization: float  # 0.0 to 1.0
    recommendation: Literal["ok", "consider_summarizing", "urgent"]

context_pressure_tool = Tool[ContextPressureParams, ContextPressure](
    name="context_pressure",
    description="Check current context utilization to decide if summarization is needed.",
    handler=context_pressure_handler,
)
```

This enables proactive compaction: the model can check pressure and summarize
completed steps before hitting limits.

## Integration with Planning Tools

### Automatic Step Tracking

When `PlanningToolsSection` and `TaskHistorySection` coexist, turns
automatically associate with the active plan step:

```python
def _append_turn_reducer(
    slice_values: tuple[TaskHistory, ...],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> tuple[TaskHistory, ...]:
    history = _latest_history(slice_values)
    if history is None or history.active_step_id is None:
        return slice_values

    turn = cast(AppendTurn, event.value).turn
    updated_steps = []
    for step in history.steps:
        if step.step_id == history.active_step_id:
            updated_steps.append(
                StepHistory(
                    step_id=step.step_id,
                    step_title=step.step_title,
                    turns=(*step.turns, turn),
                    summary=step.summary,
                    visibility=step.visibility,
                )
            )
        else:
            updated_steps.append(step)

    return (TaskHistory(
        objective=history.objective,
        active_step_id=history.active_step_id,
        steps=tuple(updated_steps),
    ),)
```

### Plan Step Lifecycle Hooks

When planning tools update step status, history tracking responds:

| Planning Event | History Behavior |
|----------------|------------------|
| `SetupPlan` | Initialize `TaskHistory` with matching steps |
| `AddStep` | Add corresponding `StepHistory` entries |
| `UpdateStep` (to `in_progress`) | Set `active_step_id` |
| `UpdateStep` (to `done`) | Optionally trigger summarization prompt |

Example integration:

```python
def _sync_plan_to_history(session: Session, plan: Plan) -> None:
    """Synchronize TaskHistory with Plan state."""
    history = session.query(TaskHistory).latest()

    # Create history if plan exists but history doesn't
    if history is None and plan is not None:
        step_histories = tuple(
            StepHistory(step_id=s.step_id, step_title=s.title)
            for s in plan.steps
        )
        session.mutate(TaskHistory).seed(TaskHistory(
            objective=plan.objective,
            steps=step_histories,
        ))
        return

    # Sync new steps
    existing_ids = {s.step_id for s in history.steps}
    new_steps = [s for s in plan.steps if s.step_id not in existing_ids]
    if new_steps:
        session.mutate(TaskHistory).dispatch(AddHistorySteps(
            steps=tuple(
                StepHistory(step_id=s.step_id, step_title=s.title)
                for s in new_steps
            )
        ))

    # Update active step
    in_progress = next(
        (s for s in plan.steps if s.status == "in_progress"),
        None,
    )
    if in_progress:
        session.mutate(TaskHistory).dispatch(
            SetActiveStep(step_id=in_progress.step_id)
        )
```

## Auto-Append on Re-render

When a prompt containing `TaskHistorySection` is re-rendered (e.g., after
visibility expansion or tool calls), the section automatically reflects
current session state:

```python
# Caller loop pattern
visibility_overrides = {}
prompt = Prompt(template).bind(*params)

while True:
    try:
        # TaskHistorySection queries session.query(TaskHistory).latest()
        # on every render, reflecting accumulated turns
        rendered = prompt.render(visibility_overrides=visibility_overrides)
        response = adapter.evaluate(prompt, session=session, bus=bus)
        break
    except VisibilityExpansionRequired as e:
        visibility_overrides.update(e.requested_overrides)
```

Each render cycle:
1. Queries `TaskHistory` from session
2. Renders each step according to its visibility
3. Summarized steps show summary + expansion hint
4. Full steps show complete turn history

## Turn Accumulation

Turns accumulate into session state via event dispatch. The adapter or
caller is responsible for dispatching turns:

```python
# After receiving assistant response
session.mutate(TaskHistory).dispatch(AppendTurn(
    turn=Turn(
        turn_id=next_turn_id(),
        role="assistant",
        content=response.text,
    )
))

# After tool execution
session.mutate(TaskHistory).dispatch(AppendTurn(
    turn=Turn(
        turn_id=next_turn_id(),
        role="tool",
        content=result.message,
        tool_name=tool_name,
    )
))
```

## Visibility Resolution

Per-step visibility follows the progressive disclosure pattern:

```python
class StepVisibilityResolver:
    """Resolves visibility for individual history steps."""

    def __init__(
        self,
        *,
        default: StepVisibility = "full",
        overrides: Mapping[int, StepVisibility] | None = None,
    ) -> None:
        self._default = default
        self._overrides = overrides or {}

    def resolve(self, step_id: int, stored: StepVisibility) -> StepVisibility:
        # Explicit override takes precedence
        if step_id in self._overrides:
            return self._overrides[step_id]
        # Then stored visibility (from summarize_step)
        return stored
```

## Summarization Strategies

### Model Self-Summarization (Recommended)

The model generates summaries via `summarize_step`, leveraging its full
context understanding:

```python
def summarize_step_handler(
    params: SummarizeStepParams,
    *,
    context: ToolContext,
) -> ToolResult[SummarizeStepParams]:
    session = context.session
    history = session.query(TaskHistory).latest()

    if history is None:
        raise ToolValidationError("No task history to summarize.")

    step = _find_step(history, params.step_id)
    if step is None:
        raise ToolValidationError(f"Step {params.step_id} not found.")

    # Validate summary
    summary = params.summary.strip()
    if not summary:
        raise ToolValidationError("Summary must not be empty.")
    if len(summary) > 1000:
        raise ToolValidationError("Summary must be <= 1000 characters.")

    return ToolResult(
        message=f"Step {params.step_id} summarized.",
        value=params,
    )
```

### Prompted Summarization (Alternative)

For automated summarization, a nested prompt can generate summaries:

```python
async def auto_summarize_step(
    session: Session,
    step_id: int,
    adapter: ProviderAdapter,
) -> str:
    """Generate summary via nested prompt evaluation."""
    history = session.query(TaskHistory).latest()
    step = _find_step(history, step_id)

    # Build summarization prompt
    prompt = Prompt(summarization_template).bind(
        SummarizationParams(
            step_title=step.step_title,
            turns=step.turns,
        )
    )

    response = await adapter.evaluate(prompt, session=session)
    return response.output.summary
```

### Extractive Summarization (Lightweight)

For resource-constrained scenarios, extract key turns without LLM calls:

```python
def extractive_summary(step: StepHistory, max_turns: int = 3) -> str:
    """Keep first and last turns, plus any tool results."""
    if len(step.turns) <= max_turns:
        return "\n".join(t.render() for t in step.turns)

    key_turns = [step.turns[0]]  # First turn
    key_turns.extend(t for t in step.turns if t.role == "tool")
    key_turns.append(step.turns[-1])  # Last turn

    # Dedupe and limit
    seen = set()
    unique = []
    for t in key_turns:
        if t.turn_id not in seen:
            seen.add(t.turn_id)
            unique.append(t)

    return "\n".join(t.render() for t in unique[:max_turns])
```

## Usage Example

```python
from weakincentives.runtime import Session, InProcessEventBus
from weakincentives.prompt import Prompt, MarkdownSection
from weakincentives.tools.planning import PlanningToolsSection
from weakincentives.tools.task_history import TaskHistorySection

# Setup
bus = InProcessEventBus()
session = Session(bus=bus)

# Build prompt with both planning and history sections
template = PromptTemplate[TaskResult](
    ns="agents/worker",
    key="task-executor",
    sections=[
        MarkdownSection[TaskParams](
            title="Task",
            key="task",
            template="Complete the following: ${objective}",
        ),
        PlanningToolsSection(session=session),
        TaskHistorySection(session=session),
    ],
)

# Evaluation loop
prompt = Prompt(template).bind(TaskParams(objective="Refactor auth module"))
visibility_overrides = {}

while not is_complete(session):
    try:
        rendered = prompt.render(visibility_overrides=visibility_overrides)
        response = adapter.evaluate(
            prompt,
            session=session,
            bus=bus,
            visibility_overrides=visibility_overrides,
        )

        # Accumulate assistant turn
        session.mutate(TaskHistory).dispatch(AppendTurn(
            turn=Turn(
                turn_id=next_id(),
                role="assistant",
                content=response.raw_text,
            )
        ))

    except VisibilityExpansionRequired as e:
        visibility_overrides.update(e.requested_overrides)

# Final state
history = session.query(TaskHistory).latest()
print(history.render())
```

## Open Questions

### Summarization Timing

When should summarization occur?

1. **On step completion**: Auto-summarize when `UpdateStep(status="done")`
2. **On context pressure**: Model checks `context_pressure` and decides
3. **Explicit only**: Model calls `summarize_step` when it wants to
4. **Hybrid**: Auto-suggest but require model confirmation

Recommendation: Start with explicit model control (option 3) to understand
usage patterns before adding automation.

### Turn Association

How do turns associate with the active step?

1. **Implicit via active_step_id**: All turns go to the current step
2. **Explicit step_id on each turn**: Caller specifies association
3. **Inferred from tool context**: Tool calls associate with their invoking step

Recommendation: Implicit association with `active_step_id` for simplicity,
with explicit override capability for edge cases.

### Cross-Step References

What if the model needs context from multiple non-adjacent steps?

1. **Expand individually**: Model calls `expand_step` for each needed step
2. **Working set concept**: Introduce `pin_step` to keep specific steps expanded
3. **Reference tool**: Add `reference_step(step_id)` that inlines content without full expansion

This requires usage data to inform the design.

### History Persistence

Should `TaskHistory` survive across sessions?

1. **Session-scoped**: History dies with session (simpler)
2. **Snapshot-based**: Persist via existing snapshot mechanism
3. **Dedicated storage**: Separate persistence layer for history

The existing `Snapshot` mechanism may suffice for most use cases.

### Visibility Inheritance

If the entire `TaskHistorySection` is summarized via `open_sections`, what
happens to individual steps?

1. **Collapse all**: All steps become summarized
2. **Preserve state**: Steps keep their individual visibility
3. **Show outline only**: Render step titles without content

This intersects with the progressive disclosure system design.

## Comparison: Task History vs. Traditional Compaction

| Aspect | Traditional Compaction | Task History |
|--------|------------------------|--------------|
| Granularity | Message count / token budget | Plan step boundaries |
| Decision maker | External algorithm | Model via tools |
| Semantic awareness | None (treats text as blob) | Full (model understands importance) |
| Reversibility | Usually lossy | Expand restores full history |
| Task alignment | None | Tied to planning structure |
| Complexity | Simple (sliding window) | Higher (section + tools + state) |

## Future Directions

### Multi-Level Summarization

Steps could have multiple summary levels:

- **Full**: All turns
- **Condensed**: Key turns only (extractive)
- **Summary**: Model-generated summary
- **Title only**: Just the step heading

### Cross-Session History

For long-running agents, history could persist and load across sessions:

```python
# Save
snapshot = session.snapshot()
history_json = snapshot.slice_to_json(TaskHistory)

# Restore
session.mutate(TaskHistory).seed(
    TaskHistory.from_json(history_json)
)
```

### Automatic Context Budget

Integrate with `BudgetTracker` to auto-summarize when approaching limits:

```python
def check_and_compact(session: Session, budget: BudgetTracker) -> None:
    pressure = budget.consumed.total_tokens / budget.budget.max_total_tokens
    if pressure > 0.8:
        # Summarize oldest completed steps
        history = session.query(TaskHistory).latest()
        for step in history.steps:
            if step.visibility == "full" and is_completed(step):
                trigger_summarization(session, step.step_id)
                break
```

### Hierarchical Tasks

For complex agents with sub-tasks, history could nest:

```python
@FrozenDataclass()
class TaskHistory:
    objective: str
    steps: tuple[StepHistory, ...]
    subtasks: tuple[TaskHistory, ...] = ()  # Nested task histories
```

## Limitations

- **Requires planning integration**: Meaningless without plan step structure
- **Model cooperation required**: Model must use tools appropriately
- **Storage growth**: Full history accumulates until summarized
- **Summarization quality**: Depends on model's summarization ability
- **No mid-step expansion**: Cannot expand part of a step's history
