# Guidance Classifiers Specification

## Purpose

Lightweight classifiers that determine when guidance should be injected.
The key insight: selectivity matters more than comprehensiveness. A classifier
that fires on every turn provides no value; one that fires precisely when
needed prevents failures without context pollution.

**Planned implementation:** `src/weakincentives/prompt/classifiers.py`

## Principles

- **Fast over accurate**: Microseconds, not milliseconds; heuristics over ML
- **Recall over precision**: False positives are cheap (guidance is soft)
- **Stateless execution**: Depend only on `GuidanceContext`, no external calls
- **Composable logic**: Combine simple classifiers into complex decisions
- **Fail-open**: Errors in classifiers → skip guidance, never block execution

## Core Types

### ClassificationResult

Output of classifier decision.

| Field | Type | Description |
|-------|------|-------------|
| `relevant` | `bool` | Whether guidance applies |
| `confidence` | `float` | 0.0-1.0 certainty |
| `reason` | `str \| None` | Explanation for decision |
| `metadata` | `dict[str, Any]` | Additional context for provider |

Factory methods:

```python
ClassificationResult.yes(confidence=0.9, reason="Detected doom loop")
ClassificationResult.no()
ClassificationResult.maybe(confidence=0.5, reason="Possible stall")
```

### GuidanceClassifier Protocol

Interface all classifiers implement.

| Method | Description |
|--------|-------------|
| `name` | Unique classifier identifier |
| `classify(context)` | Returns `ClassificationResult` |

### ClassifierConfig

Configuration for classifier behavior.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_confidence` | `float` | 0.5 | Threshold for "relevant" |
| `cooldown_turns` | `int` | 0 | Turns to skip after firing |
| `max_fires_per_session` | `int \| None` | None | Limit total activations |

## Classifier Categories

### Pattern-Based Classifiers

Detect trajectory patterns from `specs/TRAJECTORY_ANALYSIS.md`.

#### DoomLoopClassifier

Fires when circular behavior detected.

```python
@dataclass(frozen=True)
class DoomLoopClassifier:
    min_repetitions: int = 3
    min_cycle_length: int = 2

    @property
    def name(self) -> str:
        return "doom_loop"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        pattern = context.trajectory.patterns().get("doom_loop")
        if pattern is None:
            return ClassificationResult.no()

        if pattern.repetitions >= self.min_repetitions:
            return ClassificationResult.yes(
                confidence=min(1.0, pattern.repetitions / (self.min_repetitions * 2)),
                reason=f"Cycle {pattern.cycle_actions} repeated {pattern.repetitions}x",
                metadata={"cycle": pattern.cycle_actions},
            )
        return ClassificationResult.no()
```

#### ErrorStreakClassifier

Fires on consecutive tool failures.

```python
@dataclass(frozen=True)
class ErrorStreakClassifier:
    threshold: int = 3

    @property
    def name(self) -> str:
        return "error_streak"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        streak = context.trajectory.consecutive_errors
        if streak >= self.threshold:
            recent = context.trajectory.recent_errors(self.threshold)
            return ClassificationResult.yes(
                confidence=min(1.0, streak / (self.threshold * 2)),
                reason=f"{streak} consecutive errors",
                metadata={"errors": [e.message for e in recent]},
            )
        return ClassificationResult.no()
```

#### ProgressStallClassifier

Fires when no forward progress detected.

```python
@dataclass(frozen=True)
class ProgressStallClassifier:
    stall_threshold: int = 5

    @property
    def name(self) -> str:
        return "progress_stall"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        pattern = context.trajectory.patterns().get("progress_stall")
        if pattern and pattern.actions_during_stall >= self.stall_threshold:
            return ClassificationResult.yes(
                confidence=0.8,
                reason=f"No progress in {pattern.actions_during_stall} actions",
            )
        return ClassificationResult.no()
```

### Heuristic Classifiers

Simple rule-based classifiers.

#### HighToolCountClassifier

Fires after excessive tool usage.

```python
@dataclass(frozen=True)
class HighToolCountClassifier:
    threshold: int = 50
    warning_ratio: float = 0.8

    @property
    def name(self) -> str:
        return "high_tool_count"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        count = context.tool_call_count
        if count >= self.threshold:
            return ClassificationResult.yes(
                confidence=1.0,
                reason=f"{count} tool calls exceeds threshold",
            )
        if count >= self.threshold * self.warning_ratio:
            return ClassificationResult.maybe(
                confidence=0.6,
                reason=f"{count} tool calls approaching limit",
            )
        return ClassificationResult.no()
```

#### SingleToolRepeatedClassifier

Fires when same tool called repeatedly without variation.

```python
@dataclass(frozen=True)
class SingleToolRepeatedClassifier:
    window: int = 5
    threshold: int = 4

    @property
    def name(self) -> str:
        return "single_tool_repeated"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        recent = context.trajectory.recent_tool_calls(self.window)
        if len(recent) < self.threshold:
            return ClassificationResult.no()

        tool_names = [c.tool_name for c in recent]
        if len(set(tool_names)) == 1:
            return ClassificationResult.yes(
                confidence=0.7,
                reason=f"{tool_names[0]} called {len(tool_names)}x consecutively",
                metadata={"tool": tool_names[0]},
            )
        return ClassificationResult.no()
```

#### SequentialWhenParallelClassifier

Fires when independent tools called sequentially.

```python
@dataclass(frozen=True)
class SequentialWhenParallelClassifier:
    independent_tools: frozenset[str] = frozenset({"read_file", "search", "grep"})
    threshold: int = 3

    @property
    def name(self) -> str:
        return "sequential_when_parallel"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        recent = context.trajectory.recent_tool_calls(self.threshold)
        if len(recent) < self.threshold:
            return ClassificationResult.no()

        # All recent calls are single-tool turns with independent tools
        independent_count = sum(
            1 for c in recent if c.tool_name in self.independent_tools
        )
        if independent_count >= self.threshold:
            return ClassificationResult.yes(
                confidence=0.6,
                reason=f"{independent_count} independent tools called sequentially",
            )
        return ClassificationResult.no()
```

### Content-Based Classifiers

Analyze tool parameters or results.

#### LargeOutputClassifier

Fires when tool output is very large.

```python
@dataclass(frozen=True)
class LargeOutputClassifier:
    size_threshold: int = 10000  # characters

    @property
    def name(self) -> str:
        return "large_output"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        recent = context.trajectory.recent_tool_calls(1)
        if not recent:
            return ClassificationResult.no()

        last_result = recent[0].result
        if last_result and len(str(last_result.message)) > self.size_threshold:
            return ClassificationResult.yes(
                confidence=0.7,
                reason="Large tool output may overwhelm context",
            )
        return ClassificationResult.no()
```

#### SensitiveContentClassifier

Fires when content suggests sensitive operation.

```python
@dataclass(frozen=True)
class SensitiveContentClassifier:
    patterns: tuple[str, ...] = (
        r"password",
        r"secret",
        r"api[_-]?key",
        r"credential",
        r"token",
    )

    @property
    def name(self) -> str:
        return "sensitive_content"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        if not context.pending_tool_calls:
            return ClassificationResult.no()

        for call in context.pending_tool_calls:
            params_str = str(call.params).lower()
            for pattern in self.patterns:
                if re.search(pattern, params_str):
                    return ClassificationResult.yes(
                        confidence=0.9,
                        reason=f"Sensitive pattern detected: {pattern}",
                    )
        return ClassificationResult.no()
```

## Composite Classifiers

Combine multiple classifiers with logic operators.

### AllOfClassifier

All sub-classifiers must fire.

```python
@dataclass(frozen=True)
class AllOfClassifier:
    classifiers: tuple[GuidanceClassifier, ...]

    @property
    def name(self) -> str:
        names = ", ".join(c.name for c in self.classifiers)
        return f"all_of({names})"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        results = [c.classify(context) for c in self.classifiers]
        if all(r.relevant for r in results):
            avg_confidence = sum(r.confidence for r in results) / len(results)
            return ClassificationResult.yes(
                confidence=avg_confidence,
                reason="; ".join(r.reason for r in results if r.reason),
            )
        return ClassificationResult.no()
```

### AnyOfClassifier

Any sub-classifier firing triggers.

```python
@dataclass(frozen=True)
class AnyOfClassifier:
    classifiers: tuple[GuidanceClassifier, ...]

    @property
    def name(self) -> str:
        names = ", ".join(c.name for c in self.classifiers)
        return f"any_of({names})"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        for classifier in self.classifiers:
            result = classifier.classify(context)
            if result.relevant:
                return result
        return ClassificationResult.no()
```

### NotClassifier

Inverts classifier result.

```python
@dataclass(frozen=True)
class NotClassifier:
    classifier: GuidanceClassifier

    @property
    def name(self) -> str:
        return f"not({self.classifier.name})"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        result = self.classifier.classify(context)
        if result.relevant:
            return ClassificationResult.no()
        return ClassificationResult.yes(
            confidence=1.0 - result.confidence,
            reason=f"Inverse of: {result.reason}",
        )
```

### ThresholdClassifier

Requires minimum confidence from sub-classifier.

```python
@dataclass(frozen=True)
class ThresholdClassifier:
    classifier: GuidanceClassifier
    min_confidence: float

    @property
    def name(self) -> str:
        return f"threshold({self.classifier.name}, {self.min_confidence})"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        result = self.classifier.classify(context)
        if result.relevant and result.confidence >= self.min_confidence:
            return result
        return ClassificationResult.no()
```

## Classifier Execution

### Execution Order

Classifiers execute in declaration order. First match with sufficient
confidence triggers guidance:

```python
def run_classifiers(
    classifiers: Sequence[GuidanceClassifier],
    context: GuidanceContext,
    min_confidence: float = 0.5,
) -> tuple[GuidanceClassifier, ClassificationResult] | None:
    for classifier in classifiers:
        try:
            result = classifier.classify(context)
            if result.relevant and result.confidence >= min_confidence:
                return (classifier, result)
        except Exception:
            # Log and continue; classifiers fail open
            continue
    return None
```

### Cooldown Management

Track classifier fires to prevent spam:

```python
@dataclass
class ClassifierCooldownTracker:
    fires: dict[str, int] = field(default_factory=dict)  # classifier -> last_turn
    counts: dict[str, int] = field(default_factory=dict)  # classifier -> total_fires

    def can_fire(
        self,
        classifier: GuidanceClassifier,
        current_turn: int,
        config: ClassifierConfig,
    ) -> bool:
        name = classifier.name

        # Check cooldown
        if name in self.fires:
            if current_turn - self.fires[name] < config.cooldown_turns:
                return False

        # Check max fires
        if config.max_fires_per_session is not None:
            if self.counts.get(name, 0) >= config.max_fires_per_session:
                return False

        return True

    def record_fire(self, classifier: GuidanceClassifier, turn: int) -> None:
        name = classifier.name
        self.fires[name] = turn
        self.counts[name] = self.counts.get(name, 0) + 1
```

## LLM-Based Classifiers

For complex decisions, use small/fast model:

### LLMClassifier

Delegates classification to external model.

```python
@dataclass(frozen=True)
class LLMClassifier:
    """
    Use small, fast model for classification.
    Reserved for cases where heuristics insufficient.
    """

    model: str = "claude-3-haiku"  # Fast, cheap
    prompt_template: str = ""
    timeout_ms: int = 500  # Strict timeout

    @property
    def name(self) -> str:
        return f"llm({self.model})"

    def classify(self, context: GuidanceContext) -> ClassificationResult:
        # Build minimal context for classification
        summary = self._build_summary(context)

        # Call with strict timeout
        try:
            response = self._call_model(summary)
            return self._parse_response(response)
        except TimeoutError:
            return ClassificationResult.no()  # Fail open

    def _build_summary(self, context: GuidanceContext) -> str:
        """Extract minimal relevant context for classification."""
        recent = context.trajectory.recent_tool_calls(5)
        return f"""
Recent actions: {[c.tool_name for c in recent]}
Errors: {context.trajectory.consecutive_errors}
Turn: {context.turn_count}
"""
```

**Usage guidance:**
- Reserve for complex pattern detection
- Always set strict timeout
- Fail open on errors
- Cache results when possible

## Configuration Examples

### Minimal Configuration

```python
guidance_providers = (
    GuidanceProviderConfig(
        provider=DiagnosticSignalProvider(),
        classifier=ErrorStreakClassifier(threshold=3),
    ),
)
```

### Layered Configuration

```python
guidance_providers = (
    # High-priority: doom loops
    GuidanceProviderConfig(
        provider=DoomLoopBreaker(),
        classifier=ThresholdClassifier(
            DoomLoopClassifier(min_repetitions=3),
            min_confidence=0.8,
        ),
        max_per_turn=1,
    ),
    # Medium-priority: error streaks
    GuidanceProviderConfig(
        provider=DiagnosticSignalProvider(),
        classifier=ErrorStreakClassifier(threshold=3),
    ),
    # Low-priority: efficiency hints
    GuidanceProviderConfig(
        provider=ParallelToolReminder(),
        classifier=SequentialWhenParallelClassifier(),
        cooldown_turns=10,
    ),
)
```

### Composite Configuration

```python
# Fire only if doom loop AND high tool count
classifier = AllOfClassifier(
    classifiers=(
        DoomLoopClassifier(min_repetitions=2),
        HighToolCountClassifier(threshold=30),
    ),
)

# Fire if any error condition
classifier = AnyOfClassifier(
    classifiers=(
        ErrorStreakClassifier(threshold=3),
        ProgressStallClassifier(stall_threshold=5),
        DoomLoopClassifier(min_repetitions=4),
    ),
)
```

## Performance Requirements

| Metric | Requirement |
|--------|-------------|
| Latency per classifier | < 1ms |
| Memory per classifier | < 1KB |
| Total classification time | < 10ms |
| LLM classifier timeout | 500ms |

### Benchmarking

```python
def benchmark_classifier(
    classifier: GuidanceClassifier,
    contexts: Sequence[GuidanceContext],
) -> ClassifierBenchmark:
    """Measure classifier performance."""
    timings = []
    for context in contexts:
        start = time.perf_counter_ns()
        classifier.classify(context)
        timings.append(time.perf_counter_ns() - start)

    return ClassifierBenchmark(
        classifier_name=classifier.name,
        mean_ns=statistics.mean(timings),
        p99_ns=statistics.quantiles(timings, n=100)[98],
        max_ns=max(timings),
    )
```

## Testing Patterns

### Unit Testing

```python
def test_error_streak_classifier():
    classifier = ErrorStreakClassifier(threshold=3)

    # Build context with error streak
    context = build_context_with_errors(count=3)

    result = classifier.classify(context)
    assert result.relevant
    assert result.confidence >= 0.5
    assert "consecutive errors" in result.reason
```

### Property Testing

```python
from hypothesis import given, strategies as st

@given(error_count=st.integers(0, 10))
def test_error_streak_monotonic(error_count: int):
    classifier = ErrorStreakClassifier(threshold=3)
    context = build_context_with_errors(error_count)

    result = classifier.classify(context)

    if error_count >= 3:
        assert result.relevant
    else:
        assert not result.relevant
```

### Integration Testing

```python
def test_classifier_with_session():
    session = Session()
    builder = TrajectoryBuilder()

    # Simulate tool failures
    for i in range(5):
        event = ToolInvoked(tool_name="test", result=ToolResult.error("fail"))
        session.dispatch(event)
        builder.on_event(event)

    context = GuidanceContext(
        session=session,
        trajectory=builder.build(),
    )

    classifier = ErrorStreakClassifier(threshold=3)
    result = classifier.classify(context)

    assert result.relevant
    assert result.confidence > 0.5
```

## Limitations

- **Heuristic-based**: Not ML-trained; may miss novel patterns
- **Context-limited**: Only sees what's in `GuidanceContext`
- **No learning**: Static rules; doesn't adapt to agent behavior
- **Single-turn**: Classification happens per-turn, no multi-turn patterns
- **No feedback loop**: Doesn't learn from guidance effectiveness

## Public API

```python
from weakincentives.prompt import (
    # Core types
    ClassificationResult,
    GuidanceClassifier,
    ClassifierConfig,
    ClassifierCooldownTracker,
    # Pattern-based
    DoomLoopClassifier,
    ErrorStreakClassifier,
    ProgressStallClassifier,
    # Heuristic
    HighToolCountClassifier,
    SingleToolRepeatedClassifier,
    SequentialWhenParallelClassifier,
    # Content-based
    LargeOutputClassifier,
    SensitiveContentClassifier,
    # Composite
    AllOfClassifier,
    AnyOfClassifier,
    NotClassifier,
    ThresholdClassifier,
    # LLM-based
    LLMClassifier,
    # Utilities
    run_classifiers,
    benchmark_classifier,
)
```

## Related Specifications

- `specs/DECISION_TIME_GUIDANCE.md` - Uses classifiers for selective injection
- `specs/TRAJECTORY_ANALYSIS.md` - Provides patterns classifiers query
- `specs/FEEDBACK_PROVIDERS.md` - Similar trigger patterns
