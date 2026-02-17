# Analysis Loop Specification

## Purpose

The best way to debug a complex agent is with another agent.

Debug bundles capture everything about an execution. Evals tell us pass or fail.
But understanding *why* an agent behaves a certain way requires analyzing the
data—and that work is labor-intensive. AnalysisLoop automates it.

**Implementation:** Not yet implemented (specification only).

## Core Ideas

**Debug bundles are complete.** They contain inputs, outputs, session state,
tool invocations, logs, and metrics. Everything needed to understand what
happened is there.

**Evals provide signal, not insight.** A failing eval tells you something is
wrong. It doesn't tell you why, or what pattern of failures you're seeing
across samples.

**Manual analysis doesn't scale.** A single eval run produces hundreds of
bundles. Finding patterns requires comparing executions, aggregating data,
and forming hypotheses. This is exactly what an agent is good at.

**The analysis agent uses `wink query`.** SQL-based access to bundle contents
is sufficient for all analysis tasks. One tool, not a toolkit.

## Architecture

Three components, loosely coupled:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   AgentLoop ─┐                                                  │
│              ├──► CompletionNotification ──► Notifications      │
│   EvalLoop ──┘                               Mailbox            │
│                                                 │               │
│                                                 ▼               │
│                                        AnalysisForwarder        │
│                                        (sampling, budget)       │
│                                                 │               │
│                                                 ▼               │
│                                        AnalysisRequest          │
│                                                 │               │
│                                                 ▼               │
│                                        AnalysisLoop             │
│                                        Requests Mailbox         │
│                                                 │               │
│                                                 ▼               │
│                                           AnalysisLoop          │
│                                                 │               │
│                                                 ▼               │
│                                          AnalysisBundle         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**AgentLoop and EvalLoop** emit notifications when work completes. They don't
know about analysis—they just announce completion.

**AnalysisForwarder** consumes notifications, decides what to analyze (sampling,
budget), and sends requests to AnalysisLoop.

**AnalysisLoop** is a standard AgentLoop with its own mailbox. It receives
requests, analyzes bundles, and produces analysis bundles.

## Completion Notifications

Loops emit a notification when execution finishes:

```python
@dataclass(frozen=True)
class CompletionNotification:
    source: Literal["agent_loop", "eval_loop"]
    bundle_path: Path
    request_id: UUID
    success: bool
    passed: bool | None = None   # EvalLoop only
    score: float | None = None   # EvalLoop only
    completed_at: datetime = field(default_factory=lambda: now(UTC))
```

Emit to a shared mailbox:

```python
agent_loop = AgentLoop(..., notifications=notifications_mailbox)
eval_loop = EvalLoop(..., notifications=notifications_mailbox)
```

## AnalysisForwarder

Handles sampling and budget, forwards selected notifications as analysis
requests:

```python
@dataclass(frozen=True)
class AnalysisForwarderConfig:
    objective: str
    sample_rate: float = 0.1
    always_forward_failures: bool = True
    budget: AnalysisBudget = field(default_factory=AnalysisBudget)


@dataclass(frozen=True)
class AnalysisBudget:
    max_requests: int = 100
    reset_interval: timedelta = timedelta(hours=1)
```

The forwarder:

- Samples notifications at the configured rate
- Always forwards failures (if configured)
- Stops forwarding when budget exhausted
- Resets budget after the interval

## AnalysisLoop

A specialized AgentLoop that analyzes debug bundles:

```python
class AnalysisLoop(AgentLoop[AnalysisRequest, AnalysisBundle]):
    def __init__(
        self,
        adapter: Adapter,
        requests: Mailbox[AnalysisRequest, AnalysisBundle],
        config: AnalysisLoopConfig | None = None,
    ): ...
```

Receives requests on its own mailbox. Doesn't know or care where they came from.

### AnalysisRequest

```python
@dataclass(frozen=True)
class AnalysisRequest:
    objective: str
    bundles: tuple[Path, ...]
    source: Literal["agent_loop", "eval_loop", "manual"] = "manual"
    eval_context: EvalContext | None = None
```

### AnalysisBundle

The output is a self-contained archive:

```
analysis-2024-01-15T10-30-00.zip
├── report.md           # Markdown report for humans
├── evidence/           # Supporting details
├── queries/            # wink query results (CSV)
├── source-bundles/     # Original debug bundles
└── metadata.json
```

The report follows a consistent structure:

```markdown
# Analysis Report

**Objective:** [research question]
**Bundles Analyzed:** [count]

## Executive Summary
[2-3 sentences]

## Findings
### Finding 1: [title]
**Severity:** Critical | Warning | Info
**Frequency:** X/Y samples
[description]
**Evidence:** See `evidence/...`

## Recommendations
1. **[High]** [action]
   - Rationale: ...
   - Expected impact: ...
```

Source bundles are embedded so the analysis is self-contained and reproducible.

## wink query

The analysis agent has one tool: `wink query`. SQL access to bundle contents.

```bash
wink query ./bundles/ "SELECT sample_id, score FROM eval WHERE passed = false"
wink query ./bundles/ "SELECT tool_name, COUNT(*) FROM tool_invocations GROUP BY tool_name"
wink query ./bundles/ "SELECT * FROM logs WHERE level = 'ERROR'"
```

Query results go to files. The report references them. This keeps the report
readable while preserving detailed data.

See `specs/WINK_QUERY.md` for schema and capabilities.

## Prebuilt Agent

AnalysisLoop ships with a prebuilt prompt that works out of the box. Override
sections for different analysis styles:

| Section | Purpose |
|---------|---------|
| `methodology` | How to approach the analysis |
| `output-format` | Report structure and formatting |
| `evidence-gathering` | How to use wink query |

```python
overrides = AnalysisPromptOverrides(
    methodology="Focus on security patterns: credential handling, input validation...",
    output_format="Use CVE-style severity levels...",
)
```

## Wiring It Up

Helper function creates the forwarder and analysis loop:

```python
from weakincentives.analysis import connect_analysis

forwarder, analysis = connect_analysis(
    notifications=notifications_mailbox,
    objective="Identify patterns in failing samples",
    sample_rate=0.1,
)

group = LoopGroup(loops=[agent_loop, eval_loop, forwarder, analysis])
group.run()
```

Or wire manually for more control:

```python
notifications = InMemoryMailbox(name="notifications")
analysis_requests = InMemoryMailbox(name="analysis-requests")

agent_loop = AgentLoop(..., notifications=notifications)
eval_loop = EvalLoop(..., notifications=notifications)

forwarder = AnalysisForwarder(
    notifications=notifications,
    analysis_requests=analysis_requests,
    config=AnalysisForwarderConfig(objective="...", sample_rate=0.1),
)

analysis = AnalysisLoop(adapter=adapter, requests=analysis_requests)
```

## On-Demand Analysis

Skip the notification machinery for one-off analysis:

```python
requests = InMemoryMailbox(name="analysis-requests")
analysis = AnalysisLoop(adapter=adapter, requests=requests)

requests.send(AnalysisRequest(
    objective="Why did this specific execution fail?",
    bundles=(Path("./debug/bundle-001.zip"),),
))

bundle = analysis.run_once()
print(bundle.report)
```

## Configuration

```python
@dataclass(frozen=True)
class AnalysisLoopConfig:
    output_dir: Path = Path("./analysis-bundles/")
    include_source_bundles: bool = True
    max_source_bundle_size: int = 50_000_000  # 50MB
    overrides: AnalysisPromptOverrides | None = None
```

## Limitations

- **Single objective per loop**: Create multiple forwarders for different questions
- **wink query only**: Complex analysis needing custom tools requires a different approach
- **Sampling lag**: Background analysis won't catch every issue immediately

## Related Specifications

- `specs/AGENT_LOOP.md` – Base loop abstraction
- `specs/EVALS.md` – EvalLoop and evaluation framework
- `specs/DEBUG_BUNDLE.md` – Debug bundle format
- `specs/MAILBOX.md` – Message passing protocol
- `specs/WINK_QUERY.md` – SQL queries against bundles
