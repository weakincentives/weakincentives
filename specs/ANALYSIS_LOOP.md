# Analysis Loop Specification

## Purpose

The best way to debug a complex agent is with another agent.

`AnalysisLoop` is a specialized AgentLoop that analyzes debug bundles and
produces human-readable analysis bundles with actionable insights. It receives
requests on its own mailbox and runs independently of the loops it analyzes.

**Implementation:** `src/weakincentives/analysis/`

## Motivation

Debug bundles are complete sources of information regarding past runs. Evals
provide the signal needed to know if the system behaves as expected. But
analyzing the data that emerges from evals and production runs is difficult
and labor-intensive:

- A single eval run produces hundreds of bundles
- Understanding *why* failures occur requires deep trace inspection
- Patterns span multiple samples and require aggregation
- Manual analysis does not scale

AnalysisLoop automates this work. It uses `wink query` to explore bundle
contents and produces markdown reports that developers can act on immediately.

## Principles

- **Loosely coupled**: Receives requests via mailbox; no direct loop dependencies
- **Always-on**: Runs continuously with sampling; not just on-demand
- **Budget-aware**: Operates within token/cost constraints via sampling rate
- **Self-contained output**: Analysis bundles include source debug bundles
- **Human-first**: Produces markdown for reading, not structured data for machines
- **Single tool dependency**: Only `wink query`; no other tools required
- **Prebuilt with hooks**: Works out of the box; customize via prompt overrides

## Architecture

AnalysisLoop stays decoupled from AgentLoop and EvalLoop through a notification
and forwarding pattern:

```
┌─────────────┐     ┌─────────────┐
│  AgentLoop  │     │  EvalLoop   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       │ CompletionNotification
       ▼                   ▼
┌──────────────────────────────────┐
│      Notifications Mailbox       │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│       AnalysisForwarder          │
│  - sampling logic                │
│  - budget tracking               │
│  - request construction          │
└────────────────┬─────────────────┘
                 │ AnalysisRequest
                 ▼
┌──────────────────────────────────┐
│   AnalysisLoop Requests Mailbox  │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│          AnalysisLoop            │
│  - load bundles                  │
│  - run wink query                │
│  - generate report               │
│  - write analysis bundle         │
└──────────────────────────────────┘
```

### Completion Notifications

AgentLoop and EvalLoop emit notifications when work completes:

```python
@dataclass(frozen=True)
class CompletionNotification:
    """Emitted when a loop completes an execution."""

    source: Literal["agent_loop", "eval_loop"]
    """Which loop type emitted this notification."""

    bundle_path: Path
    """Path to the debug bundle."""

    request_id: UUID
    """Original request identifier."""

    success: bool
    """Whether the execution succeeded."""

    # EvalLoop-specific (None for AgentLoop)
    passed: bool | None = None
    """Whether the eval passed (EvalLoop only)."""

    score: float | None = None
    """Eval score if applicable."""

    completed_at: datetime = field(default_factory=lambda: now(UTC))
```

Loops emit to a shared notifications mailbox without knowing who consumes:

```python
# In AgentLoop, after execution completes
self._notifications.send(CompletionNotification(
    source="agent_loop",
    bundle_path=bundle.path,
    request_id=request.request_id,
    success=result.error is None,
))

# In EvalLoop, after evaluation completes
self._notifications.send(CompletionNotification(
    source="eval_loop",
    bundle_path=bundle.path,
    request_id=request.request_id,
    success=result.error is None,
    passed=result.passed,
    score=result.score,
))
```

### AnalysisForwarder

A lightweight consumer that handles sampling and forwards to AnalysisLoop:

```python
@dataclass(frozen=True)
class AnalysisForwarderConfig:
    """Configuration for the analysis forwarder."""

    objective: str
    """Research question for analysis."""

    sample_rate: float = 0.1
    """Fraction of notifications to forward (0.0-1.0)."""

    always_forward_failures: bool = True
    """Forward failed executions regardless of sample rate."""

    budget: AnalysisBudget = field(default_factory=AnalysisBudget)
    """Token and bundle limits."""

    seed: int | None = None
    """Random seed for reproducible sampling."""
```

```python
class AnalysisForwarder:
    """Consumes notifications and forwards sampled requests to AnalysisLoop."""

    def __init__(
        self,
        notifications: Mailbox[CompletionNotification, None],
        analysis_requests: Mailbox[AnalysisRequest, AnalysisBundle],
        config: AnalysisForwarderConfig,
    ):
        self._notifications = notifications
        self._analysis_requests = analysis_requests
        self._config = config
        self._rng = Random(config.seed)
        self._budget_tracker = BudgetTracker(config.budget)

    def run(self) -> None:
        """Process notifications and forward sampled requests."""
        for msg in self._notifications.receive():
            notification = msg.body

            if self._should_forward(notification):
                request = AnalysisRequest(
                    objective=self._config.objective,
                    bundles=(notification.bundle_path,),
                    source=notification.source,
                    eval_context=EvalContext(
                        passed=notification.passed,
                        score=notification.score,
                    ) if notification.source == "eval_loop" else None,
                )
                self._analysis_requests.send(request)
                self._budget_tracker.record_forwarded()

            msg.acknowledge()

    def _should_forward(self, notification: CompletionNotification) -> bool:
        """Determine if this notification should trigger analysis."""
        if self._budget_tracker.exhausted:
            return False

        if self._config.always_forward_failures and not notification.success:
            return True

        return self._rng.random() < self._config.sample_rate
```

### AnalysisLoop

Receives requests on its own mailbox, independent of notification source:

```python
class AnalysisLoop(AgentLoop[AnalysisRequest, AnalysisBundle]):
    """Specialized loop for debug bundle analysis."""

    def __init__(
        self,
        adapter: Adapter,
        requests: Mailbox[AnalysisRequest, AnalysisBundle],
        config: AnalysisLoopConfig | None = None,
    ):
        ...
```

## Auto-Connect API

Helper functions wire up the notification pattern:

```python
from weakincentives.analysis import connect_analysis

# Returns (forwarder, analysis_loop) tuple
# Creates mailboxes and wiring automatically
forwarder, analysis = connect_analysis(
    notifications=notifications_mailbox,  # Shared notifications mailbox
    objective="Identify patterns in failing samples",
    sample_rate=0.1,
    budget=AnalysisBudget(max_tokens=100_000),
)

# Run all components together
group = LoopGroup(loops=[agent_loop, eval_loop, forwarder, analysis])
group.run()
```

For convenience, loops can be configured to emit to a notifications mailbox:

```python
# Create shared notifications mailbox
notifications = InMemoryMailbox[CompletionNotification, None](name="notifications")

# Configure loops to emit notifications
agent_loop = AgentLoop(
    ...,
    notifications=notifications,  # Emits CompletionNotification on completion
)

eval_loop = EvalLoop(
    ...,
    notifications=notifications,  # Emits CompletionNotification on completion
)

# Wire up analysis
forwarder, analysis = connect_analysis(
    notifications=notifications,
    objective="Monitor agent behavior",
    sample_rate=0.1,
)
```

### Sampling

The `sample_rate` parameter controls what fraction of notifications trigger
analysis:

| Sample Rate | Behavior |
|-------------|----------|
| `1.0` | Analyze every execution (expensive) |
| `0.1` | Analyze ~10% of executions |
| `0.0` | Disable automatic analysis |

Sampling is random but reproducible given a seed. Failed executions can be
configured to always trigger analysis regardless of sample rate.

### Budget Constraints

The forwarder tracks budget and stops forwarding when exhausted:

```python
@dataclass(frozen=True)
class AnalysisBudget:
    max_requests: int = 100
    """Maximum analysis requests per interval."""

    reset_interval: timedelta = timedelta(hours=1)
    """Budget resets after this interval."""
```

## Analysis Request and Response

### AnalysisRequest

```python
@dataclass(frozen=True)
class AnalysisRequest:
    """Request to analyze debug bundles."""

    objective: str
    """Research question to investigate."""

    bundles: tuple[Path, ...]
    """Debug bundle paths to analyze."""

    source: Literal["agent_loop", "eval_loop", "manual"] = "manual"
    """Where this request originated."""

    eval_context: EvalContext | None = None
    """Eval-specific context (pass/fail, score) if from EvalLoop."""
```

### AnalysisBundle (Response)

The output is an **analysis bundle**: a self-contained archive for human
consumption.

```
analysis-2024-01-15T10-30-00.zip
├── report.md              # Main analysis report (markdown)
├── evidence/              # Supporting evidence extracted from bundles
│   ├── error-patterns.md
│   ├── tool-sequences.md
│   └── sample-details/
│       ├── sample-001.md
│       └── sample-002.md
├── queries/               # wink query results referenced in report
│   ├── failing-samples.csv
│   └── tool-frequency.csv
├── source-bundles/        # Original debug bundles (for drill-down)
│   ├── bundle-abc123.zip
│   └── bundle-def456.zip
└── metadata.json          # Analysis metadata
```

### Report Format

The main `report.md` follows a consistent structure:

```markdown
# Analysis Report

**Objective:** Why do 30% of samples fail the accuracy evaluator?
**Bundles Analyzed:** 12
**Generated:** 2024-01-15T10:30:00Z

## Executive Summary

[2-3 sentence summary of key findings]

## Findings

### Finding 1: Token truncation in long inputs

**Severity:** Critical
**Frequency:** 8/12 samples (67%)

[Detailed description with evidence references]

**Evidence:** See `evidence/error-patterns.md#truncation`

### Finding 2: ...

## Recommendations

1. **[High]** Increase context window for long-input samples
   - Rationale: ...
   - Expected impact: ...

2. **[Medium]** Add input length validation
   - ...

## Appendix

- Query results: `queries/`
- Sample details: `evidence/sample-details/`
- Source bundles: `source-bundles/`
```

### Self-Containment

Analysis bundles include the source debug bundles that were analyzed:

```python
@dataclass(frozen=True)
class AnalysisLoopConfig:
    include_source_bundles: bool = True
    """Embed analyzed debug bundles in the output."""

    max_source_bundle_size: int = 50_000_000  # 50MB
    """Skip embedding bundles larger than this."""
```

## wink query Integration

The analysis agent has a single tool: `wink query`. This provides SQL-based
access to bundle contents and is sufficient for all analysis tasks.

```bash
# Examples the agent can run:

# Find failing samples
wink query ./bundles/ "SELECT sample_id, score FROM eval WHERE passed = false"

# Analyze tool usage patterns
wink query ./bundles/ "SELECT tool_name, COUNT(*) FROM tool_invocations GROUP BY tool_name"

# Search error logs
wink query ./bundles/ "SELECT * FROM logs WHERE level = 'ERROR'"

# Compare timing across samples
wink query ./bundles/ "SELECT sample_id, duration_ms FROM metrics ORDER BY duration_ms DESC"
```

The agent writes query results to files and references them in the markdown
report. This keeps the report readable while preserving detailed data.

See `specs/WINK_QUERY.md` for full query capabilities.

## Prebuilt Agent with Hooks

AnalysisLoop ships with a prebuilt analysis agent that works out of the box.
The underlying prompt can be customized via section overrides for different
analysis styles.

### Default Sections

| Section | Purpose |
|---------|---------|
| `objective` | Research question to investigate |
| `methodology` | How to approach the analysis |
| `output-format` | Report structure and formatting |
| `evidence-gathering` | How to use wink query effectively |

### Customization

Override sections to change analysis behavior:

```python
from weakincentives.analysis import AnalysisPromptOverrides

# Security-focused analysis
security_overrides = AnalysisPromptOverrides(
    methodology="""
    Focus on security-relevant patterns:
    - Credential handling in tool calls
    - Input validation failures
    - Error message information leakage
    """,
    output_format="""
    Use security severity levels (Critical/High/Medium/Low).
    Include CVE-style identifiers for patterns.
    """,
)

forwarder, analysis = connect_analysis(
    notifications=notifications,
    objective="Security audit of agent behavior",
    overrides=security_overrides,
)
```

### Common Override Patterns

| Pattern | Override Focus |
|---------|----------------|
| **Performance** | Timing analysis, token efficiency |
| **Reliability** | Error rates, retry patterns, recovery |
| **Security** | Input validation, credential handling |
| **Cost** | Token usage, API call frequency |
| **UX** | Response quality, user-facing errors |

## Configuration

```python
@dataclass(frozen=True)
class AnalysisLoopConfig:
    """Configuration for AnalysisLoop."""

    # Output
    output_dir: Path = Path("./analysis-bundles/")
    """Where to write analysis bundles."""

    include_source_bundles: bool = True
    """Embed debug bundles in analysis output."""

    max_source_bundle_size: int = 50_000_000
    """Skip embedding bundles larger than this."""

    # Agent
    overrides: AnalysisPromptOverrides | None = None
    """Prompt section overrides."""
```

## Usage

### Continuous Background Analysis

```python
from weakincentives.analysis import connect_analysis
from weakincentives.runtime import AgentLoop, LoopGroup, InMemoryMailbox

# Create shared notifications mailbox
notifications = InMemoryMailbox(name="notifications")

# Create loops that emit notifications
agent_loop = AgentLoop(..., notifications=notifications)
eval_loop = EvalLoop(..., notifications=notifications)

# Wire up analysis
forwarder, analysis = connect_analysis(
    notifications=notifications,
    objective="Monitor for degraded performance patterns",
    sample_rate=0.1,
)

# Run all together
group = LoopGroup(loops=[agent_loop, eval_loop, forwarder, analysis])
group.run()

# Analysis bundles appear in ./analysis-bundles/
```

### On-Demand Analysis

```python
from weakincentives.analysis import AnalysisLoop, AnalysisRequest

# Create analysis loop with its own mailbox
requests = InMemoryMailbox(name="analysis-requests")
analysis = AnalysisLoop(adapter=adapter, requests=requests)

# Submit request directly
requests.send(AnalysisRequest(
    objective="Deep dive on this specific failure",
    bundles=(Path("./debug/bundle-001.zip"),),
))

# Run and get result
bundle = analysis.run_once()
print(f"Report: {bundle.report_path}")
```

### Reading Analysis Bundles

```python
from weakincentives.analysis import AnalysisBundle

bundle = AnalysisBundle.load(Path("./analysis-bundles/analysis-2024-01-15.zip"))

# Read the main report
print(bundle.report)

# Access evidence files
for evidence in bundle.evidence_files:
    print(f"{evidence.name}: {evidence.read_text()}")

# Access source debug bundles for drill-down
for debug_bundle in bundle.source_bundles:
    print(f"Source: {debug_bundle.path}")
```

## Limitations

- **Single objective**: Each AnalysisLoop instance targets one research question
- **Sampling lag**: Background analysis may not catch every issue immediately
- **Bundle size**: Very large debug bundles may be excluded from embedding
- **wink query only**: Complex analysis requiring custom tools needs a different approach

## Related Specifications

- `specs/AGENT_LOOP.md` - Base loop abstraction
- `specs/EVALS.md` - EvalLoop and evaluation framework
- `specs/DEBUG_BUNDLE.md` - Debug bundle format
- `specs/MAILBOX.md` - Message passing protocol
- `specs/WINK_QUERY.md` - SQL queries against bundles
- `specs/LIFECYCLE.md` - Loop coordination
