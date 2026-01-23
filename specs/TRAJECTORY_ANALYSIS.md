# Trajectory Analysis Specification

## Purpose

Analyze agent execution trajectories to detect patterns, anomalies, and
opportunities for intervention. Provides the analytical foundation that
guidance classifiers use to determine when and what guidance to inject.

**Planned implementation:** `src/weakincentives/runtime/trajectory.py`

## Principles

- **Observation over mutation**: Analyze; never modify session state
- **Incremental computation**: Update metrics incrementally, not full recompute
- **Pattern library**: Common patterns detected automatically
- **Extensible detectors**: Custom patterns via detector protocol
- **Cheap queries**: O(1) lookups for common metrics

## Core Types

### TrajectoryAnalysis

Primary interface for trajectory insights. Built from session state.

| Property | Type | Description |
|----------|------|-------------|
| `turn_count` | `int` | Total agent turns |
| `tool_call_count` | `int` | Total tool invocations |
| `error_count` | `int` | Total tool failures |
| `consecutive_errors` | `int` | Current error streak |
| `unique_tools_used` | `frozenset[str]` | Distinct tool names invoked |
| `duration` | `timedelta` | Elapsed time since start |

| Method | Description |
|--------|-------------|
| `recent_tool_calls(n)` | Last N `ToolInvoked` events |
| `recent_errors(n)` | Last N failed tool results |
| `tool_frequency()` | `dict[str, int]` of tool usage counts |
| `error_frequency()` | `dict[str, int]` of error types |
| `patterns()` | Detected `TrajectoryPattern` instances |
| `has_pattern(pattern_type)` | Check for specific pattern |

### TrajectoryPattern

Base class for detected patterns.

| Field | Type | Description |
|-------|------|-------------|
| `pattern_type` | `str` | Pattern identifier |
| `confidence` | `float` | 0.0-1.0 detection confidence |
| `evidence` | `tuple[str, ...]` | Supporting observations |
| `first_seen` | `datetime` | When pattern first detected |
| `occurrences` | `int` | Times pattern matched |

### PatternDetector Protocol

Interface for pattern detection algorithms.

| Method | Description |
|--------|-------------|
| `name` | Unique detector identifier |
| `detect(trajectory)` | Returns `TrajectoryPattern \| None` |
| `reset()` | Clear internal state |

## Built-in Patterns

### DoomLoop

Circular behavior indicating stuck agent.

| Field | Type | Description |
|-------|------|-------------|
| `pattern_type` | `Literal["doom_loop"]` | Fixed identifier |
| `cycle_length` | `int` | Actions in repeated cycle |
| `repetitions` | `int` | Times cycle repeated |
| `cycle_actions` | `tuple[str, ...]` | Tool names in cycle |

Detection criteria:
- Same sequence of N tools repeated M times
- Configurable similarity threshold for fuzzy matching
- Minimum cycle length to avoid false positives

### RepeatedError

Same error occurring multiple times.

| Field | Type | Description |
|-------|------|-------------|
| `pattern_type` | `Literal["repeated_error"]` | Fixed identifier |
| `error_signature` | `str` | Normalized error identifier |
| `count` | `int` | Occurrences |
| `tool_name` | `str` | Tool producing error |

Detection criteria:
- Same error message (normalized) appears N times
- Within configurable window of recent calls
- Same or similar tool invocations

### ProgressStall

No meaningful progress over window.

| Field | Type | Description |
|-------|------|-------------|
| `pattern_type` | `Literal["progress_stall"]` | Fixed identifier |
| `stall_duration` | `timedelta` | Time without progress |
| `actions_during_stall` | `int` | Tool calls during stall |

Detection criteria:
- No successful tool calls for N consecutive attempts
- Or no new files/resources accessed for M calls
- Configurable progress indicators

### ToolAvoidance

Consistently avoiding recommended tool.

| Field | Type | Description |
|-------|------|-------------|
| `pattern_type` | `Literal["tool_avoidance"]` | Fixed identifier |
| `avoided_tool` | `str` | Tool not being used |
| `alternative_attempts` | `int` | Other approaches tried |

Detection criteria:
- Guidance suggested tool N times
- Agent used different tools instead
- Task context suggests avoided tool appropriate

### ContextDegradation

Signs of context window pressure.

| Field | Type | Description |
|-------|------|-------------|
| `pattern_type` | `Literal["context_degradation"]` | Fixed identifier |
| `symptoms` | `tuple[str, ...]` | Observed degradation signs |

Detection criteria:
- Repeated questions about already-discussed topics
- Contradictory actions within short window
- Forgetting recent tool results

## Trajectory Builder

Incrementally constructs `TrajectoryAnalysis` from session events.

```python
class TrajectoryBuilder:
    def __init__(
        self,
        detectors: tuple[PatternDetector, ...] = DEFAULT_DETECTORS,
        window_size: int = 50,
    ) -> None: ...

    def on_event(self, event: ToolInvoked | PromptExecuted) -> None:
        """Update analysis with new event."""

    def build(self) -> TrajectoryAnalysis:
        """Produce current analysis snapshot."""

    def reset(self) -> None:
        """Clear accumulated state."""
```

### Event Integration

Subscribe to session events for automatic updates:

```python
session.dispatcher.subscribe(ToolInvoked, builder.on_event)
session.dispatcher.subscribe(PromptExecuted, builder.on_event)
```

Or manual integration in adapter hooks.

## Pattern Detection Algorithms

### Sequence Similarity

For doom loop detection, compare action sequences using:

```python
def sequence_similarity(seq_a: tuple[str, ...], seq_b: tuple[str, ...]) -> float:
    """
    Jaccard similarity of action n-grams.
    Returns 0.0-1.0 similarity score.
    """
```

### Error Normalization

For repeated error detection, normalize error messages:

```python
def normalize_error(message: str) -> str:
    """
    Remove variable parts (paths, line numbers, timestamps).
    Returns canonical error signature.
    """
```

Normalization rules:
- Strip absolute paths to basenames
- Remove line/column numbers
- Collapse whitespace
- Remove timestamps and IDs

### Progress Indicators

Configurable signals of forward progress:

| Indicator | Description |
|-----------|-------------|
| `new_file_created` | Filesystem write to new path |
| `test_passed` | Test tool returned success |
| `build_succeeded` | Build tool returned success |
| `user_acknowledged` | User provided positive feedback |
| `plan_step_completed` | Plan step marked done |

## Configuration

### TrajectoryAnalysisConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `window_size` | `int` | 50 | Events to retain |
| `doom_loop_threshold` | `int` | 3 | Cycle repetitions for detection |
| `doom_loop_min_length` | `int` | 2 | Minimum cycle length |
| `similarity_threshold` | `float` | 0.8 | Sequence similarity threshold |
| `error_window` | `int` | 10 | Window for repeated error detection |
| `stall_threshold` | `int` | 5 | Failed calls for stall detection |
| `detectors` | `tuple[PatternDetector, ...]` | `DEFAULT_DETECTORS` | Active detectors |

```python
config = TrajectoryAnalysisConfig(
    doom_loop_threshold=5,
    similarity_threshold=0.9,
)
builder = TrajectoryBuilder.from_config(config)
```

## Session Integration

### As Session Slice

```python
@dataclass(frozen=True)
class TrajectoryState:
    analysis: TrajectoryAnalysis

    @reducer(on=ToolInvoked)
    def on_tool_invoked(self, event: ToolInvoked) -> Replace["TrajectoryState"]:
        updated = self.analysis.with_event(event)
        return Replace((TrajectoryState(analysis=updated),))
```

Install with `session.install(TrajectoryState)`.

### Query Patterns

```python
# Check for specific pattern
trajectory = session[TrajectoryState].latest().analysis
if trajectory.has_pattern("doom_loop"):
    loop = trajectory.patterns()["doom_loop"]
    print(f"Doom loop detected: {loop.cycle_actions}")

# Get consecutive error count
if trajectory.consecutive_errors >= 3:
    recent = trajectory.recent_errors(3)
    print(f"Error streak: {[e.message for e in recent]}")
```

## Custom Detector Example

```python
@dataclass(frozen=True)
class FileChurnDetector:
    """Detect repeatedly modifying same file."""

    churn_threshold: int = 5
    window: int = 10

    @property
    def name(self) -> str:
        return "file_churn"

    def detect(self, trajectory: TrajectoryAnalysis) -> FileChurnPattern | None:
        recent = trajectory.recent_tool_calls(self.window)
        write_calls = [c for c in recent if c.tool_name in ("write_file", "edit_file")]

        file_counts: dict[str, int] = {}
        for call in write_calls:
            path = call.params.get("path", "")
            file_counts[path] = file_counts.get(path, 0) + 1

        for path, count in file_counts.items():
            if count >= self.churn_threshold:
                return FileChurnPattern(
                    pattern_type="file_churn",
                    confidence=min(1.0, count / (self.churn_threshold * 2)),
                    evidence=(f"{path} modified {count} times",),
                    churned_file=path,
                    modification_count=count,
                )
        return None

    def reset(self) -> None:
        pass  # Stateless detector
```

## Performance Considerations

### Memory Bounds

- Fixed window size limits memory growth
- Patterns evicted when no longer detected
- Metrics computed incrementally

### Computation Bounds

- Similarity computed only on window, not full history
- Detectors run after each event, not on query
- Pattern cache invalidated on new events

### Complexity

| Operation | Complexity |
|-----------|------------|
| `recent_tool_calls(n)` | O(min(n, window)) |
| `tool_frequency()` | O(window) |
| `patterns()` | O(1) cached |
| `on_event()` | O(detectors * window) |

## Relationship to Guidance

`TrajectoryAnalysis` feeds into `GuidanceContext`:

```python
@dataclass(frozen=True)
class GuidanceContext:
    session: SessionProtocol
    trajectory: TrajectoryAnalysis  # <-- Built from session
    # ...

def build_guidance_context(session: Session) -> GuidanceContext:
    trajectory = session[TrajectoryState].latest().analysis
    return GuidanceContext(
        session=session,
        trajectory=trajectory,
        # ...
    )
```

Classifiers then query trajectory patterns:

```python
class DoomLoopClassifier:
    def classify(self, context: GuidanceContext) -> ClassificationResult:
        if context.trajectory.has_pattern("doom_loop"):
            pattern = context.trajectory.patterns()["doom_loop"]
            return ClassificationResult(
                relevant=True,
                confidence=pattern.confidence,
                reason=f"Detected {pattern.repetitions}x cycle",
            )
        return ClassificationResult(relevant=False, confidence=0.0)
```

## Limitations

- **Bounded history**: Only analyzes recent window, not full trajectory
- **Heuristic patterns**: Detection is approximate, not formal verification
- **Single-session scope**: No cross-session pattern learning
- **Synchronous updates**: Detector runs block event processing briefly
- **No causal inference**: Detects correlation, not causation

## Public API

```python
from weakincentives.runtime import (
    TrajectoryAnalysis,      # Primary analysis interface
    TrajectoryAnalysisConfig,# Configuration
    TrajectoryBuilder,       # Incremental builder
    TrajectoryPattern,       # Base pattern class
    PatternDetector,         # Custom detector protocol
    # Built-in patterns
    DoomLoop,
    RepeatedError,
    ProgressStall,
    ToolAvoidance,
    ContextDegradation,
    # Built-in detectors
    DoomLoopDetector,
    RepeatedErrorDetector,
    ProgressStallDetector,
    ToolAvoidanceDetector,
    ContextDegradationDetector,
    DEFAULT_DETECTORS,
)
```

## Related Specifications

- `specs/DECISION_TIME_GUIDANCE.md` - Consumes trajectory analysis
- `specs/GUIDANCE_CLASSIFIERS.md` - Classifiers query patterns
- `specs/SESSIONS.md` - Event system for updates
- `specs/FEEDBACK_PROVIDERS.md` - Similar context access patterns
