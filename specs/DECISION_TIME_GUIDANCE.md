# Decision-Time Guidance Specification

## Purpose

Inject short, situational instructions at decision points during agent execution.
Unlike static prompts that front-load all constraints, decision-time guidance
intervenes selectively—only when relevant, only where it matters.

**Planned implementation:** `src/weakincentives/prompt/guidance.py`

## Principles

- **Selective over exhaustive**: Inject only relevant guidance; irrelevant rules pollute context
- **Ephemeral by default**: Guidance doesn't persist in conversation history
- **Recency-exploiting**: Place guidance at end of context to leverage recency bias
- **Classifier-gated**: Lightweight analysis determines what guidance applies
- **Soft not hard**: Guidance suggests; policies gate—distinct mechanisms
- **Cache-preserving**: Core prompt unchanged; guidance appended dynamically

## Problem Statement

Static prompts fail on long trajectories:

1. **Learned priors override rules**: Models fall back to pre-training when rules are verbose or ambiguous
2. **Instruction-following degrades**: Mid-context rules lose influence (primacy/recency bias)
3. **More rules have diminishing returns**: Priority ambiguity leads to partial compliance

Decision-time guidance addresses these by injecting short instructions at the
moment of decision, filtered by a classifier that determines relevance.

## Core Types

### GuidanceInjection

Frozen dataclass representing a single piece of guidance.

| Field | Type | Description |
|-------|------|-------------|
| `key` | `str` | Unique identifier (pattern: `^[a-z0-9][a-z0-9._-]{0,63}$`) |
| `content` | `str` | Guidance text (1-500 chars) |
| `priority` | `int` | Lower = higher priority (default: 100) |
| `category` | `str` | Grouping for analysis (e.g., "diagnostic", "consultation") |

### GuidanceContext

Context provided to classifiers and providers for decision-making.

| Property/Method | Description |
|-----------------|-------------|
| `session` | Session protocol with full history |
| `prompt` | Active prompt protocol |
| `rendered_prompt` | Current rendered state |
| `trajectory` | `TrajectoryAnalysis` with pattern detection |
| `pending_tool_calls` | Tool calls about to execute |
| `last_response` | Most recent model response |
| `turn_count` | Number of agent turns so far |
| `tool_call_count` | Total tool invocations |

### GuidanceProvider Protocol

Generates guidance based on context. Similar to `FeedbackProvider` but for
pre-decision injection.

| Method/Property | Description |
|-----------------|-------------|
| `name` | Unique provider identifier |
| `category` | Guidance category (e.g., "diagnostic", "consultation") |
| `classify(context)` | Returns `ClassificationResult` indicating relevance |
| `provide(context)` | Produces `GuidanceInjection` when classified as relevant |

### ClassificationResult

Result of classifier determining whether guidance applies.

| Field | Type | Description |
|-------|------|-------------|
| `relevant` | `bool` | Whether guidance should be injected |
| `confidence` | `float` | 0.0-1.0 confidence score |
| `reason` | `str \| None` | Explanation for classification |

### GuidanceProviderConfig

Configuration binding provider to prompt.

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `GuidanceProvider` | Provider implementation |
| `max_per_turn` | `int` | Max injections per turn (default: 3) |
| `min_confidence` | `float` | Threshold for injection (default: 0.5) |

## Decision Points

Guidance can be injected at specific lifecycle points:

| Point | When | Use Case |
|-------|------|----------|
| `PRE_RENDER` | Before prompt rendering | Context-aware section selection |
| `PRE_TOOL_SELECTION` | Before model chooses tools | Guide tool choice |
| `PRE_TOOL_EXECUTION` | After parsing, before execution | Parameter validation guidance |
| `POST_TOOL_RESULT` | After tool completes | Next-step guidance (overlaps with FeedbackProviders) |
| `PRE_RESPONSE` | Before final response assembly | Format/content guidance |

```python
class DecisionPoint(Enum):
    PRE_RENDER = "pre_render"
    PRE_TOOL_SELECTION = "pre_tool_selection"
    PRE_TOOL_EXECUTION = "pre_tool_execution"
    POST_TOOL_RESULT = "post_tool_result"
    PRE_RESPONSE = "pre_response"
```

## Execution Flow

1. Agent reaches decision point
2. Build `GuidanceContext` with current state and trajectory analysis
3. For each configured `GuidanceProvider`:
   a. Call `classify(context)` with lightweight classifier
   b. If `relevant` and confidence >= threshold, call `provide(context)`
4. Sort injections by priority
5. Apply `max_per_turn` limit
6. Inject as ephemeral context (position based on decision point)
7. Continue execution; guidance not persisted to history

### Injection Positions

| Decision Point | Injection Position |
|----------------|-------------------|
| `PRE_RENDER` | Added to render context |
| `PRE_TOOL_SELECTION` | Appended to system/assistant message |
| `PRE_TOOL_EXECUTION` | Prepended to tool execution context |
| `POST_TOOL_RESULT` | Appended to tool result (like FeedbackProviders) |
| `PRE_RESPONSE` | Appended before response parsing |

## Built-in Providers

### DiagnosticSignalProvider

Detects repeated errors and prompts agent to address them.

```python
provider = DiagnosticSignalProvider(
    error_threshold=3,  # Trigger after N consecutive errors
    log_tool_name="view_logs",  # Tool to suggest
)
```

Output example:
```
Found 3 new console errors. Use the view_logs tool to examine before continuing.
```

### DoomLoopDetector

Detects circular patterns indicating stuck agent.

```python
provider = DoomLoopDetector(
    similarity_threshold=0.85,  # Action similarity threshold
    window_size=5,  # Actions to compare
    max_repetitions=3,  # Trigger after N similar sequences
)
```

Output example:
```
Detected repeated unsuccessful pattern. Consider a different approach or consult
the planning tool to reassess strategy.
```

### ConsultationPromptProvider

Suggests consulting external agent when local attempts fail.

```python
provider = ConsultationPromptProvider(
    failure_threshold=5,  # Failed attempts before suggesting
    consultation_tool="request_review",
)
```

Output example:
```
Multiple attempts unsuccessful. Consider using request_review to get fresh
perspective from specialized agent.
```

### ParallelToolReminder

Encourages parallel tool execution when applicable.

```python
provider = ParallelToolReminder(
    sequential_threshold=3,  # Trigger after N sequential single-tool calls
)
```

Output example:
```
Multiple independent operations detected. Consider executing tools in parallel
for efficiency.
```

## Prompt Integration

Configure at template level:

```python
template = PromptTemplate(
    ns="agent",
    key="main",
    sections=(...),
    guidance_providers=(
        GuidanceProviderConfig(
            provider=DiagnosticSignalProvider(error_threshold=3),
            max_per_turn=1,
        ),
        GuidanceProviderConfig(
            provider=DoomLoopDetector(window_size=5),
            min_confidence=0.7,
        ),
    ),
)
```

## Adapter Integration

| Adapter | Delivery Method |
|---------|-----------------|
| Claude Agent SDK | `PreToolUse` / `PostToolUse` hooks, `additionalContext` |
| OpenAI | Appended to appropriate message based on decision point |

### Claude Agent SDK Example

```python
class GuidanceHookHandler:
    def pre_tool_use(self, hook_data: dict, context: HookContext) -> dict:
        guidance = collect_guidance(
            context,
            decision_point=DecisionPoint.PRE_TOOL_EXECUTION,
        )
        if guidance:
            return {"additionalContext": format_guidance(guidance)}
        return {}
```

## State Management

### GuidanceDelivered Event

| Field | Type | Description |
|-------|------|-------------|
| `provider_name` | `str` | Source provider |
| `injection` | `GuidanceInjection` | What was injected |
| `decision_point` | `DecisionPoint` | Where injected |
| `classification` | `ClassificationResult` | Why injected |
| `timestamp` | `datetime` | When delivered |

Stored via `session.dispatch(GuidanceDelivered(...))` for debugging and analysis.

### Ephemeral Semantics

Guidance is **not** added to conversation history. It affects the current
decision only. This:

- Prevents context pollution over long trajectories
- Preserves prompt cache (core prompt unchanged)
- Allows aggressive injection without history bloat

## Classifier Requirements

Classifiers must be:

- **Fast**: Run on every decision point; microseconds not milliseconds
- **Cheap**: Use heuristics or small models, not primary LLM
- **Stateless**: Depend only on `GuidanceContext`, no external calls
- **Tolerant**: False positives acceptable (guidance is soft)

See `specs/GUIDANCE_CLASSIFIERS.md` for classifier implementation patterns.

## Relationship to Existing Systems

| System | Relationship |
|--------|--------------|
| `FeedbackProviders` | Guidance is pre-decision; feedback is post-tool. Complementary. |
| `ToolPolicies` | Policies gate (allow/deny); guidance suggests. Distinct concerns. |
| `TaskCompletionChecker` | Completion checks end-state; guidance shapes path. |
| `PromptOverrides` | Overrides modify prompt source; guidance is ephemeral context. |
| `Progressive Disclosure` | Disclosure expands sections; guidance injects new content. |

## Design Rationale

- **False positives are cheap**: Soft suggestions; model ignores if irrelevant
- **Guidance is ephemeral**: No history pollution; fresh each turn
- **Caching preserved**: Core prompt unchanged; only append dynamic content
- **Priority ordering**: Most important guidance first within limit
- **Category separation**: Avoid conflicting guidance from same category

## Limitations

- **No guarantee of compliance**: Guidance is advisory; model may ignore
- **Single-turn scope**: Cannot inject multi-turn conversation patterns
- **Classifier accuracy**: Depends on trajectory analysis quality
- **No rollback**: Once injected, cannot undo for current turn
- **Ordering sensitivity**: Multiple injections may interact unpredictably

## Public API

```python
from weakincentives.prompt import (
    GuidanceInjection,       # Injection dataclass
    GuidanceContext,         # Context for providers
    GuidanceProvider,        # Protocol
    GuidanceProviderConfig,  # Configuration
    ClassificationResult,    # Classifier output
    DecisionPoint,           # Injection points enum
    GuidanceDelivered,       # Event for tracking
    collect_guidance,        # Primary entry point
    # Built-in providers
    DiagnosticSignalProvider,
    DoomLoopDetector,
    ConsultationPromptProvider,
    ParallelToolReminder,
)
```

## Related Specifications

- `specs/TRAJECTORY_ANALYSIS.md` - Pattern detection for classifiers
- `specs/GUIDANCE_CLASSIFIERS.md` - Classifier implementation patterns
- `specs/FEEDBACK_PROVIDERS.md` - Post-tool feedback (complementary)
- `specs/TOOLS.md` - Tool policies (distinct from guidance)
- `specs/SESSIONS.md` - Event dispatch for tracking
