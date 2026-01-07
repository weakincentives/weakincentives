# Trajectory Observer Specification

## Purpose

Trajectory observers provide ongoing assessment of agent progress during
unattended execution. Unlike tool policies that gate individual calls,
observers analyze patterns over time and inject feedback into the agent's
context. This enables soft course-correction without hard intervention.

## Guiding Principles

- **Non-blocking feedback**: Observers produce guidance, not gates. The agent
  decides how to respond to observations.
- **Evidence-backed**: Observations cite specific tool calls and patterns.
  Vague warnings are not actionable.
- **Session-stored**: Assessments live in session slices, making them available
  for prompt injection and preserving them across snapshot/restore.
- **Inline execution**: Observers run synchronously after tool calls. No
  background threads or async complexity.
- **Composable**: Multiple observers can run independently; their assessments
  are merged into a single context block.

```mermaid
flowchart TB
    subgraph ToolExecution["Tool Execution"]
        Call["Tool call completes"]
        Record["ToolInvoked stored in slice"]
    end

    subgraph Observer["Trajectory Observer"]
        Trigger["should_run()?"]
        Observe["observe()"]
        Assess["Assessment"]
    end

    subgraph Session["Session State"]
        Store["Append to Assessment slice"]
    end

    subgraph Prompt["Next Prompt"]
        Inject["Render assessment in context"]
    end

    Call --> Record
    Record --> Trigger
    Trigger -->|threshold met| Observe
    Trigger -->|not yet| Done["Continue"]
    Observe --> Assess
    Assess --> Store
    Store -.-> Inject
```

## Core Types

### TrajectoryObserver Protocol

```python
class TrajectoryObserver(Protocol):
    """Programmatic assessment of agent trajectory."""

    @property
    def name(self) -> str:
        """Unique identifier for this observer."""
        ...

    def should_run(
        self,
        session: Session,
        *,
        context: ObserverContext,
    ) -> bool:
        """Determine if assessment threshold has been met."""
        ...

    def observe(
        self,
        session: Session,
        *,
        context: ObserverContext,
    ) -> Assessment:
        """Analyze trajectory and produce feedback."""
        ...
```

### Observation

```python
@dataclass(frozen=True)
class Observation:
    """Single observation about the trajectory."""

    category: str  # "loop", "error_rate", "drift", "stall", etc.
    description: str
    evidence: str | None = None  # Specific tool calls, patterns
```

### Assessment (Session Slice)

Assessments are appended to a session slice. All history is retained; use
`.latest()` for context injection:

```python
@dataclass(frozen=True)
class Assessment:
    """Structured output from trajectory observation."""

    observer_name: str
    summary: str
    observations: tuple[Observation, ...] = ()
    suggestions: tuple[str, ...] = ()
    severity: Literal["info", "caution", "warning"] = "info"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    call_index: int = 0  # Tool call index when generated

    def render(self) -> str:
        """Render as concise markdown for context injection."""
        lines = [
            "## Trajectory Assessment",
            "",
            f"_Generated after tool call #{self.call_index}_",
            "",
            f"### {self.observer_name} [{self.severity}]",
            "",
            self.summary,
        ]

        if self.observations:
            lines.append("")
            for obs in self.observations:
                lines.append(f"**{obs.category}**: {obs.description}")
                if obs.evidence:
                    lines.append(f"```\n{obs.evidence}\n```")

        if self.suggestions:
            lines.append("")
            lines.append("**Suggestions**:")
            for suggestion in self.suggestions:
                lines.append(f"- {suggestion}")

        return "\n".join(lines)

    def is_stale(self, current_call_index: int, max_age_calls: int = 20) -> bool:
        """Check if assessment is too old to be relevant."""
        return (current_call_index - self.call_index) > max_age_calls
```

### ObserverContext

```python
@dataclass(frozen=True)
class ObserverContext:
    """Context provided to observers during assessment."""

    session: Session

    @property
    def last_assessment(self) -> Assessment | None:
        """Most recent assessment, if any."""
        return self.session[Assessment].latest()

    @property
    def tool_call_count(self) -> int:
        """Total tool calls in session."""
        return len(self.session[ToolInvoked].all())

    def tool_calls_since_last_assessment(self) -> int:
        """Number of tool calls since last assessment."""
        last = self.last_assessment
        if last is None:
            return self.tool_call_count
        return self.tool_call_count - last.call_index

    def recent_tool_calls(self, n: int) -> Sequence[ToolInvoked]:
        """Retrieve the N most recent tool invocations."""
        records = self.session[ToolInvoked].all()
        return records[-n:] if len(records) >= n else records

    def error_rate(self, window: int) -> float:
        """Calculate error rate over the last N calls."""
        recent = self.recent_tool_calls(window)
        if not recent:
            return 0.0
        return sum(1 for r in recent if not _is_success(r)) / len(recent)
```

### ToolInvoked (Existing Event)

Trajectory observers use the existing `ToolInvoked` event from
`weakincentives.runtime.events`. This event is dispatched after each tool
execution and can be stored in a session slice by registering a reducer:

```python
# ToolInvoked is already defined in runtime.events
@FrozenDataclass()
class ToolInvoked:
    prompt_name: str
    adapter: AdapterName
    name: str              # Tool name
    params: Any            # Full parameters
    result: Any            # ToolResult with success/error
    session_id: UUID | None
    created_at: datetime
    usage: TokenUsage | None
    rendered_output: str
    call_id: str | None
    event_id: UUID

# Register a reducer to store ToolInvoked events in a slice
session[ToolInvoked].register(
    ToolInvoked,
    lambda state, event: Append(event),
)
```

The observer extracts what it needs:

```python
def _is_success(event: ToolInvoked) -> bool:
    """Check if tool invocation succeeded."""
    result = event.result
    if hasattr(result, "success"):
        return result.success
    return True  # Assume success if no flag

def _get_error_message(event: ToolInvoked) -> str | None:
    """Extract error message from failed invocation."""
    result = event.result
    if hasattr(result, "message") and not _is_success(event):
        return result.message
    return None
```

## Trigger Configuration

Observers declare when they should run via `ObserverTrigger`:

```python
@dataclass(frozen=True)
class ObserverTrigger:
    """Conditions that trigger observer execution."""

    # Run every N tool calls
    every_n_calls: int | None = None

    # Run after N consecutive errors
    after_consecutive_errors: int | None = None

    # Run every N seconds (wall clock)
    every_n_seconds: float | None = None

    # Always run (observer decides internally)
    on_every_call: bool = False
```

Multiple triggers are OR'd together: if any condition is met, the observer
runs.

**Example:**

```python
trigger = ObserverTrigger(
    every_n_calls=15,
    after_consecutive_errors=3,
)
# Runs after every 15 calls OR after 3 consecutive errors
```

## Observer Configuration

```python
@dataclass(frozen=True)
class ObserverConfig:
    """Configuration for a trajectory observer."""

    observer: TrajectoryObserver
    trigger: ObserverTrigger
```

Multiple observers can be configured; each appends to the `Assessment` slice.

## Prompt Integration

### Reading Assessment from Session

The prompt system reads the latest assessment from the session slice and
injects it into context:

```python
def build_context(session: Session, *, max_age_calls: int = 20) -> str:
    """Build additional context including trajectory assessment."""
    current_call_index = len(session[ToolInvoked].all())

    # Get latest assessment if not stale
    assessment = session[Assessment].latest()
    if assessment and not assessment.is_stale(current_call_index, max_age_calls):
        return assessment.render()

    return ""
```

### Prompt Declaration

```python
template = PromptTemplate(
    ns="my-agent",
    key="main",
    sections=[
        MarkdownSection(
            title="Instructions",
            key="instructions",
            template="...",
        ),
        # ... other sections
    ],
    observers=[
        ObserverConfig(
            observer=ResourceObserver(),
            trigger=ObserverTrigger(every_n_calls=10),
        ),
    ],
)
```

### Context Injection Point

Assessment context is injected when building the prompt for the next LLM call:

```python
def build_prompt_messages(
    prompt: Prompt,
    session: Session,
    *,
    include_assessment: bool = True,
) -> list[Message]:
    """Build messages for LLM call, including trajectory context."""
    messages = prompt.render_messages()

    if include_assessment:
        assessment_context = build_context(session)
        if assessment_context:
            # Inject as system context or append to user message
            messages = inject_context(messages, assessment_context)

    return messages
```

## Execution Flow

`ToolInvoked` events are already dispatched by adapters after tool execution.
The observer hooks into this existing event flow:

```python
def on_tool_invoked(
    event: ToolInvoked,
    *,
    session: Session,
    prompt: Prompt,
) -> None:
    """Run trajectory observers after tool execution."""

    # ToolInvoked is already stored in slice via registered reducer
    # Build context for observers
    context = ObserverContext(session=session)
    call_index = context.tool_call_count

    # Check each observer
    for config in prompt.observers:
        if _should_trigger(config.trigger, context, event):
            if config.observer.should_run(session, context=context):
                assessment = config.observer.observe(session, context=context)
                # Append assessment with call index
                assessment = replace(assessment, call_index=call_index)
                session.dispatch(RecordAssessment(assessment))


def _should_trigger(
    trigger: ObserverTrigger,
    context: ObserverContext,
    event: ToolInvoked,
) -> bool:
    """Check if any trigger condition is met."""

    if trigger.on_every_call:
        return True

    if trigger.every_n_calls:
        if context.tool_calls_since_last_assessment() >= trigger.every_n_calls:
            return True

    if trigger.after_consecutive_errors:
        recent = context.recent_tool_calls(trigger.after_consecutive_errors)
        if len(recent) >= trigger.after_consecutive_errors:
            if all(not _is_success(r) for r in recent):
                return True

    if trigger.every_n_seconds:
        last = context.last_assessment
        if last:
            elapsed = (datetime.utcnow() - last.timestamp).total_seconds()
            if elapsed >= trigger.every_n_seconds:
                return True
        else:
            return True  # No previous assessment, run now

    return False
```

## Built-in Observer

### ResourceObserver

The primary built-in observer reports time and token budget constraints in
natural language. This gives the agent clear visibility into remaining runway.

```python
@dataclass(frozen=True)
class ResourceObserver:
    """Report remaining time and token budget in natural language."""

    # Thresholds for severity escalation (percentage remaining)
    caution_threshold: float = 0.3  # 30% remaining
    warning_threshold: float = 0.1  # 10% remaining

    @property
    def name(self) -> str:
        return "Resources"

    def should_run(self, session: Session, *, context: ObserverContext) -> bool:
        # Always run when triggered - let trigger config control frequency
        return True

    def observe(self, session: Session, *, context: ObserverContext) -> Assessment:
        budget = context.budget
        statements: list[str] = []
        severity: Literal["info", "caution", "warning"] = "info"

        # Time remaining
        if budget.deadline is not None:
            remaining_seconds = (budget.deadline - datetime.utcnow()).total_seconds()
            if remaining_seconds <= 0:
                statements.append("You have reached the time deadline.")
                severity = "warning"
            else:
                time_str = self._format_duration(remaining_seconds)
                elapsed = budget.elapsed_seconds or 0
                total = elapsed + remaining_seconds
                pct_remaining = remaining_seconds / total if total > 0 else 0

                statements.append(f"You have {time_str} remaining before the deadline.")

                if pct_remaining <= self.warning_threshold:
                    severity = "warning"
                elif pct_remaining <= self.caution_threshold:
                    severity = max(severity, "caution")

        # Token budget
        if budget.max_tokens is not None:
            used = budget.tokens_used or 0
            remaining = budget.max_tokens - used
            pct_used = used / budget.max_tokens

            if remaining <= 0:
                statements.append("You have exhausted your token budget.")
                severity = "warning"
            else:
                statements.append(
                    f"You have used {used:,} of {budget.max_tokens:,} tokens "
                    f"({pct_used:.0%} of budget). {remaining:,} tokens remaining."
                )

                pct_remaining = remaining / budget.max_tokens
                if pct_remaining <= self.warning_threshold:
                    severity = "warning"
                elif pct_remaining <= self.caution_threshold:
                    severity = max(severity, "caution")

        # Tool call budget
        if budget.max_tool_calls is not None:
            used = len(context.recent_tool_calls(1000))  # All calls
            remaining = budget.max_tool_calls - used
            pct_used = used / budget.max_tool_calls

            if remaining <= 0:
                statements.append("You have exhausted your tool call budget.")
                severity = "warning"
            else:
                statements.append(
                    f"You have made {used} of {budget.max_tool_calls} allowed tool calls. "
                    f"{remaining} calls remaining."
                )

                pct_remaining = remaining / budget.max_tool_calls
                if pct_remaining <= self.warning_threshold:
                    severity = "warning"
                elif pct_remaining <= self.caution_threshold:
                    severity = max(severity, "caution")

        # Build suggestions based on severity
        suggestions: list[str] = []
        if severity == "warning":
            suggestions.append("Prioritize completing the most critical remaining work.")
            suggestions.append("Consider wrapping up with a summary of progress and remaining tasks.")
        elif severity == "caution":
            suggestions.append("Be mindful of remaining resources when planning next steps.")

        # Compose summary
        if not statements:
            summary = "No resource constraints configured."
        else:
            summary = " ".join(statements)

        return Assessment(
            observer_name=self.name,
            summary=summary,
            suggestions=tuple(suggestions),
            severity=severity,
        )

    def _format_duration(self, seconds: float) -> str:
        """Format seconds as human-readable duration."""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            hours = seconds / 3600
            if hours < 24:
                return f"{hours:.1f} hours"
            else:
                days = hours / 24
                return f"{days:.1f} days"
```

### Budget Configuration

The observer reads budget constraints from the session:

```python
@dataclass(frozen=True)
class Budget:
    """Resource constraints for the session."""

    # Time constraint
    deadline: datetime | None = None
    elapsed_seconds: float | None = None

    # Token constraint
    max_tokens: int | None = None
    tokens_used: int | None = None

    # Tool call constraint
    max_tool_calls: int | None = None
```

Budget is typically set at session initialization and updated by the adapter
after each LLM call:

```python
# Initialize with constraints
session[Budget].seed(Budget(
    deadline=datetime.utcnow() + timedelta(minutes=30),
    max_tokens=50_000,
    max_tool_calls=100,
))

# Adapter updates token usage after each call
current = session[Budget].latest()
session[Budget].seed(replace(current, tokens_used=current.tokens_used + response.usage.total))
```

### ObserverContext Extension

The observer context provides access to budget (extending the base definition):

```python
@property
def budget(self) -> Budget:
    """Current resource budget."""
    return self.session[Budget].latest() or Budget()
```

## Example Rendered Assessment

When injected into context, an assessment renders as natural language:

```markdown
## Trajectory Assessment

_Generated after tool call #47_

### Resources [caution]

You have 8 minutes remaining before the deadline. You have used 35,000 of
50,000 tokens (70% of budget). 15,000 tokens remaining. You have made 47 of
100 allowed tool calls. 53 calls remaining.

**Suggestions**:
- Be mindful of remaining resources when planning next steps.
```

### Severity Examples

**Info** (plenty of runway):
```markdown
### Resources [info]

You have 25 minutes remaining before the deadline. You have used 12,000 of
50,000 tokens (24% of budget). 38,000 tokens remaining.
```

**Caution** (approaching limits):
```markdown
### Resources [caution]

You have 6 minutes remaining before the deadline. You have used 42,000 of
50,000 tokens (84% of budget). 8,000 tokens remaining.

**Suggestions**:
- Be mindful of remaining resources when planning next steps.
```

**Warning** (critical):
```markdown
### Resources [warning]

You have 2 minutes remaining before the deadline. You have used 48,500 of
50,000 tokens (97% of budget). 1,500 tokens remaining.

**Suggestions**:
- Prioritize completing the most critical remaining work.
- Consider wrapping up with a summary of progress and remaining tasks.
```

## State Management

All observer state lives in session slices:

| Slice | Purpose | Mutation |
|-------|---------|----------|
| `Budget` | Time/token/call constraints | Replace after LLM calls |
| `ToolInvoked` | Append-only log of tool invocations | Append after each call |
| `Assessment` | Append-only log of assessments | Append after observer runs |

```python
# Budget tracking
session[Budget].latest()
# Budget(
#     deadline=datetime(2024, 1, 15, 15, 0),
#     max_tokens=50000,
#     tokens_used=35000,
#     max_tool_calls=100,
# )

# Tool call history
len(session[ToolInvoked].all())  # 51 calls
session[ToolInvoked].latest()
# ToolInvoked(name="edit_file", params={...}, result=..., created_at=...)

# Assessment history (all retained, query latest for injection)
len(session[Assessment].all())  # 5 assessments
session[Assessment].latest()
# Assessment(
#     observer_name="Resources",
#     summary="You have 8 minutes remaining...",
#     severity="caution",
#     call_index=47,
# )
```

**Snapshot/restore**: All slices are captured in session snapshots. Restoring
a snapshot resets state to that point.

## Design Decisions

### Why session-based instead of file-based?

1. **Simpler**: No filesystem operations, directory creation, or path handling
2. **Atomic**: Assessment storage is transactional with session state
3. **Bounded**: Only inject latest; history available for analysis
4. **Integrated**: Naturally participates in snapshot/restore

### Why no escalation path?

For unattended agents with budget/time limits, the agent should self-correct
based on feedback. Hard intervention would require:

- Human availability (not guaranteed for unattended)
- Clear escalation criteria (domain-specific)
- Recovery procedures (complex state management)

The observer provides feedback; budget exhaustion provides the backstop.

### Why inline execution?

1. **Simplicity**: No threading, no race conditions
2. **Consistency**: Assessment reflects state at a known point
3. **Predictability**: Observer cost is visible in timing

### Why staleness check?

Assessments become misleading if too old. The `is_stale()` check prevents
injecting outdated feedback that no longer reflects current trajectory.
Default threshold of 20 calls balances relevance with persistence.

## Limitations

- **No cross-session state**: Observers reset with each session
- **Synchronous only**: Observers block tool execution briefly
- **Single assessment**: Only latest assessment is retained
- **Text-based feedback**: Agent must interpret markdown guidance

## Future Observers

The following observers are out of scope for the initial implementation but
represent natural extensions:

### StallDetector

Detect repetitive tool call patterns suggesting the agent is stuck:

- Track tool call frequency in sliding window
- Alert on repeated calls to same tool (e.g., `edit_file` 5x in 10 calls)
- Detect read→write→read→write thrashing patterns

### ErrorCascadeDetector

Detect consecutive failure sequences:

- Track consecutive errors (e.g., 3+ failures in a row)
- Group similar error messages to identify systemic issues
- Suggest reassessment when error patterns emerge

### DriftDetector

Detect when agent wanders from original task scope:

- Compare recent file paths to initial task context
- Alert when majority of activity is on unrelated files
- Useful for preventing scope creep in long-running tasks

## Future Considerations

Out of scope for initial implementation:

- **Assessment history**: Keep last N assessments for trend analysis
- **Observer composition**: Combine observers into pipelines
- **Severity trends**: Track escalating/de-escalating patterns
- **Structured response**: Allow agent to acknowledge/dismiss observations
- **Pace estimation**: Project whether budget will last based on consumption rate
