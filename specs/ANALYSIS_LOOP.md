# Analysis Loop Specification

## Purpose

The best way to debug a complex agent is with another agent.

`AnalysisLoop` is a specialized agent loop purpose-built to analyze debug bundles
given a research objective related to improving the target agent. It automates
the labor-intensive work of understanding agent behavior from evaluation data
and execution traces.

**Implementation:** `src/weakincentives/analysis/`

## Motivation

### The Analysis Burden

Evaluations provide the signal needed to determine if an agent behaves as
expected. However, analyzing the data that emerges from an evaluation is a
difficult and labor-intensive task:

- **Volume**: A single eval run can produce hundreds of debug bundles
- **Complexity**: Each bundle contains session state, logs, tool calls, and outputs
- **Patterns**: Failure modes often span multiple samples and require aggregation
- **Context**: Understanding *why* an agent failed requires deep trace inspection
- **Iteration**: Improvement hypotheses must be tested across representative samples

Manual analysis does not scale. An agent designed specifically for this task can
systematically process bundles, identify patterns, and surface actionable insights.

### Agents Debugging Agents

Debug bundles are complete sources of information regarding past runs. They
capture everything needed to understand an execution: inputs, outputs, session
state transitions, tool invocations, errors, and metrics. This makes them ideal
inputs for an analysis agent.

The AnalysisLoop treats bundles as structured documents to be queried, compared,
and reasoned about. By leveraging the same agentic capabilities as the system
under study, it can perform deep investigation that would be impractical manually.

## Principles

- **Research-objective-driven**: Every analysis session targets a specific question
- **Bundle-native**: Debug bundles are first-class inputs, not afterthoughts
- **Two operational modes**: Works with EvalLoop (dataset-driven) or AgentLoop (retrospective)
- **Actionable output**: Produces findings, hypotheses, and concrete recommendations
- **Compositional**: Built on AgentLoop; reuses mailbox, lifecycle, and resource patterns

## Core Concepts

### Research Objective

A research objective defines what the analysis aims to discover:

```python
@dataclass(frozen=True)
class ResearchObjective:
    """Defines the goal of an analysis session."""

    question: str
    """Primary question to answer (e.g., 'Why do tool calls fail on long inputs?')."""

    hypothesis: str | None = None
    """Optional hypothesis to test against the data."""

    focus_areas: tuple[str, ...] = ()
    """Specific aspects to examine (e.g., 'tool_invocations', 'token_usage')."""

    success_criteria: str | None = None
    """What constitutes a satisfactory answer."""
```

### Analysis Modes

AnalysisLoop operates in two complementary modes:

| Mode | Input | Use Case |
|------|-------|----------|
| **EvalLoop Integration** | Dataset + EvalLoop mailbox | Systematic analysis with expected outputs |
| **AgentLoop Integration** | Debug bundles directly | Retrospective trajectory analysis |

#### EvalLoop Integration Mode

When an input dataset is available, AnalysisLoop orchestrates evaluation and
analysis as a unified workflow:

1. Split dataset into micro-batches
1. Dispatch batches to EvalLoop via mailbox
1. Collect debug bundles as evaluations complete
1. Analyze each bundle in relationship to its input sample
1. Aggregate findings across the batch

This mode answers questions like:

- "Why do 30% of samples fail the accuracy evaluator?"
- "What patterns distinguish passing from failing samples?"
- "Which tool sequences correlate with success?"

#### AgentLoop Integration Mode

When there is no clear expected output or the goal is understanding trajectory
quality, AnalysisLoop performs retrospective analysis:

1. Receive debug bundles from AgentLoop executions
1. Analyze whether the trajectory was sound given the objective
1. Identify decision points where the agent could have done better
1. Surface patterns across multiple retrospectives

This mode answers questions like:

- "Did the agent take a reasonable path to the solution?"
- "Where did the agent get stuck and why?"
- "What information was the agent missing?"

### Analysis Scope

| Scope | Description |
|-------|-------------|
| `SINGLE_BUNDLE` | Deep analysis of one execution |
| `BATCH` | Comparative analysis across a micro-batch |
| `AGGREGATE` | Pattern synthesis across all batches |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AnalysisLoop                                   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     Research Objective                              │ │
│  │  "Why do long-context samples fail more frequently?"                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────┐       ┌──────────────────────────────────────┐ │
│  │   EvalLoop Mode     │       │   AgentLoop Mode                      │ │
│  │                     │       │                                       │ │
│  │  Dataset            │       │  Debug Bundles                        │ │
│  │    ↓                │       │    ↓                                  │ │
│  │  Micro-batches      │       │  Trajectory Analysis                  │ │
│  │    ↓                │       │    ↓                                  │ │
│  │  EvalLoop Mailbox   │       │  Soundness Assessment                 │ │
│  │    ↓                │       │                                       │ │
│  │  Bundle Collection  │       │                                       │ │
│  │    ↓                │       │                                       │ │
│  │  Sample Analysis    │       │                                       │ │
│  └─────────┬───────────┘       └──────────────────┬────────────────────┘ │
│            │                                      │                      │
│            └──────────────────┬───────────────────┘                      │
│                               ↓                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Analysis Agent                                   │ │
│  │                                                                     │ │
│  │  Tools:                                                             │ │
│  │  - bundle_query: SQL-like queries against bundle data               │ │
│  │  - bundle_compare: Diff two bundles                                 │ │
│  │  - slice_inspect: Deep-dive into session slices                     │ │
│  │  - log_search: Search logs with filters                             │ │
│  │  - pattern_extract: Identify recurring patterns                     │ │
│  │  - hypothesis_test: Validate hypothesis against evidence            │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                               ↓                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Analysis Report                                  │ │
│  │                                                                     │ │
│  │  - Findings: Observed patterns and anomalies                        │ │
│  │  - Evidence: Supporting data from bundles                           │ │
│  │  - Hypotheses: Testable explanations                                │ │
│  │  - Recommendations: Concrete improvement actions                    │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Types

### AnalysisRequest

```python
@dataclass(frozen=True)
class AnalysisRequest:
    """Request to analyze debug bundles given a research objective."""

    objective: ResearchObjective
    """What question to answer."""

    bundles: tuple[Path, ...] | None = None
    """Direct bundle paths (AgentLoop mode)."""

    dataset: Dataset[Any, Any] | None = None
    """Dataset to evaluate (EvalLoop mode)."""

    experiment: Experiment = BASELINE
    """Experiment configuration for evaluation."""

    batch_size: int = 10
    """Micro-batch size for EvalLoop mode."""

    scope: Literal["single_bundle", "batch", "aggregate"] = "aggregate"
    """Analysis granularity."""

    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: now(UTC))
```

### AnalysisReport

```python
@dataclass(frozen=True)
class AnalysisReport:
    """Output of an analysis session."""

    objective: ResearchObjective
    """The research objective that was investigated."""

    findings: tuple[Finding, ...]
    """Discovered patterns and observations."""

    evidence: tuple[Evidence, ...]
    """Supporting data extracted from bundles."""

    hypotheses: tuple[Hypothesis, ...]
    """Testable explanations for observed behavior."""

    recommendations: tuple[Recommendation, ...]
    """Concrete actions to improve the agent."""

    bundles_analyzed: int
    """Number of bundles processed."""

    samples_analyzed: int
    """Number of samples analyzed (EvalLoop mode)."""

    confidence: float
    """Confidence in findings (0.0-1.0)."""

    request_id: UUID
    completed_at: datetime
```

### Finding

```python
@dataclass(frozen=True)
class Finding:
    """An observed pattern or anomaly."""

    summary: str
    """One-line description of the finding."""

    description: str
    """Detailed explanation."""

    severity: Literal["info", "warning", "critical"]
    """Impact level."""

    frequency: float
    """How often this occurs (0.0-1.0)."""

    affected_samples: tuple[str, ...]
    """Sample IDs where this was observed."""

    bundle_refs: tuple[str, ...]
    """Bundle IDs containing evidence."""
```

### Hypothesis

```python
@dataclass(frozen=True)
class Hypothesis:
    """A testable explanation for observed behavior."""

    statement: str
    """The hypothesis to test."""

    supporting_evidence: tuple[str, ...]
    """Evidence IDs that support this hypothesis."""

    contradicting_evidence: tuple[str, ...]
    """Evidence IDs that contradict this hypothesis."""

    confidence: float
    """Confidence level (0.0-1.0)."""

    test_suggestion: str
    """How to further test this hypothesis."""
```

### Recommendation

```python
@dataclass(frozen=True)
class Recommendation:
    """A concrete action to improve the agent."""

    action: str
    """What to change."""

    rationale: str
    """Why this should help."""

    expected_impact: str
    """Predicted improvement."""

    priority: Literal["low", "medium", "high"]
    """Implementation urgency."""

    related_findings: tuple[str, ...]
    """Finding IDs this addresses."""
```

## Analysis Tools

AnalysisLoop provides specialized tools for bundle inspection:

### bundle_query

SQL-like queries against bundle data:

```python
def bundle_query(
    params: BundleQueryParams,
    *,
    context: ToolContext,
) -> ToolResult[QueryResult]:
    """Execute a query against loaded bundles.

    Example queries:
    - SELECT sample_id, score FROM eval WHERE passed = false
    - SELECT tool_name, COUNT(*) FROM tool_invocations GROUP BY tool_name
    - SELECT * FROM logs WHERE level = 'ERROR'
    """
```

### bundle_compare

Diff two bundles to identify differences:

```python
def bundle_compare(
    params: BundleCompareParams,
    *,
    context: ToolContext,
) -> ToolResult[ComparisonResult]:
    """Compare two bundles and highlight differences.

    Compares:
    - Session state transitions
    - Tool call sequences
    - Output differences
    - Timing profiles
    """
```

### slice_inspect

Deep-dive into session slices:

```python
def slice_inspect(
    params: SliceInspectParams,
    *,
    context: ToolContext,
) -> ToolResult[SliceDetail]:
    """Inspect a specific slice type across bundles.

    Examines state evolution, identifies anomalies,
    and correlates with outcomes.
    """
```

### log_search

Search logs with structured filters:

```python
def log_search(
    params: LogSearchParams,
    *,
    context: ToolContext,
) -> ToolResult[LogSearchResult]:
    """Search logs across bundles.

    Supports:
    - Level filtering (DEBUG, INFO, WARNING, ERROR)
    - Time range filtering
    - Event type filtering
    - Full-text search in messages
    """
```

### pattern_extract

Identify recurring patterns:

```python
def pattern_extract(
    params: PatternExtractParams,
    *,
    context: ToolContext,
) -> ToolResult[PatternResult]:
    """Extract patterns from bundle data.

    Identifies:
    - Common tool sequences
    - Recurring error patterns
    - Successful vs failing trajectories
    - Token usage patterns
    """
```

### hypothesis_test

Validate hypothesis against evidence:

```python
def hypothesis_test(
    params: HypothesisTestParams,
    *,
    context: ToolContext,
) -> ToolResult[HypothesisTestResult]:
    """Test a hypothesis against available evidence.

    Searches bundles for supporting and contradicting
    evidence, calculates confidence score.
    """
```

## Analysis Agent Environment

### wink query Integration

The analysis agent makes heavy use of `wink query` for structured bundle
exploration. This CLI tool is installed in the analysis agent's sandbox and
provides SQL-based access to bundle contents:

```bash
# Query tool invocations across bundles
wink query ./bundles/ "SELECT tool_name, COUNT(*) as calls FROM tool_invocations GROUP BY tool_name"

# Find error patterns
wink query ./bundles/ "SELECT * FROM logs WHERE level = 'ERROR' ORDER BY timestamp"

# Compare token usage across experiments
wink query ./bundles/ "SELECT experiment_name, AVG(total_tokens) FROM metrics GROUP BY experiment_name"

# Identify failing samples
wink query ./bundles/ "SELECT sample_id, score, reason FROM eval WHERE passed = false"
```

The `bundle_query` tool wraps `wink query` for programmatic access within the
analysis agent, providing the same SQL interface with structured result handling.

See `specs/WINK_QUERY.md` for full query capabilities and schema documentation.

### Collocated Tool Access

Because AnalysisLoop runs as a specialized AgentLoop, the analysis agent has
access to the same tool environment as any other agent in the system:

| Tool Category | Examples | Purpose in Analysis |
|---------------|----------|---------------------|
| **Bundle tools** | `bundle_query`, `bundle_compare` | Direct bundle inspection |
| **Filesystem tools** | `read_file`, `list_directory` | Examine bundle contents on disk |
| **Shell tools** | `bash`, `wink query` | Run CLI queries and scripts |
| **Search tools** | `grep`, `glob` | Find patterns across extracted data |

This enables sophisticated analysis workflows that combine structured queries
with ad-hoc exploration.

### Original Prompt Tool Access

For deep behavioral analysis, the analysis agent can optionally access the
complete tool set from the original prompt being analyzed:

```python
@dataclass(frozen=True)
class AnalysisLoopConfig:
    # ... other fields ...

    mount_original_tools: bool = False
    """If True, mount tools from the analyzed prompt for replay/inspection."""

    original_prompt_template: PromptTemplate | None = None
    """The prompt template whose tools should be mounted."""
```

When enabled, this allows the analysis agent to:

- **Replay tool calls**: Re-execute tool invocations from the bundle to verify behavior
- **Inspect tool schemas**: Understand what capabilities were available
- **Test hypotheses**: Try alternative tool sequences to validate findings

This is particularly useful for understanding whether failures stem from tool
limitations, incorrect tool selection, or flawed tool usage patterns.

### Sandboxed Execution

The analysis agent runs in an isolated environment with:

- Read-only access to bundle directories
- Write access to a scratch workspace for intermediate results
- Network access for LLM API calls (analysis agent's own inference)
- No access to production systems or sensitive credentials

This ensures analysis cannot inadvertently affect the systems being studied.

## EvalLoop Integration

### Micro-batch Dispatch

When analyzing a dataset, AnalysisLoop splits samples into micro-batches and
dispatches them through the EvalLoop mailbox:

```python
class AnalysisLoop(AgentLoop[AnalysisRequest, AnalysisReport]):
    def __init__(
        self,
        adapter: Adapter,
        requests: Mailbox[AgentLoopRequest[AnalysisRequest], AgentLoopResult[AnalysisReport]],
        eval_requests: Mailbox[EvalRequest, EvalResult],
        eval_results: Mailbox[EvalResult, None],
        config: AnalysisLoopConfig | None = None,
    ):
        ...
```

### Workflow

```python
async def _execute_eval_mode(
    self,
    request: AnalysisRequest,
) -> tuple[AnalysisReport, Session]:
    """Execute analysis in EvalLoop integration mode."""

    # 1. Split dataset into micro-batches
    batches = self._create_micro_batches(request.dataset, request.batch_size)

    # 2. Submit batches to EvalLoop
    pending: list[UUID] = []
    for batch in batches:
        for sample in batch:
            eval_request = EvalRequest(
                sample=sample,
                experiment=request.experiment,
            )
            msg_id = self._eval_requests.send(eval_request, reply_to=self._eval_results)
            pending.append(msg_id)

    # 3. Collect results and bundles
    collected: list[tuple[EvalResult, DebugBundle]] = []
    for msg in self._eval_results.receive(timeout=self._config.collection_timeout):
        result = msg.body
        if result.bundle_path:
            bundle = DebugBundle.load(result.bundle_path)
            collected.append((result, bundle))
        msg.acknowledge()

    # 4. Load bundles into analysis context
    self._load_bundles([b for _, b in collected])

    # 5. Run analysis agent
    return await self._analyze(request, collected)
```

### Sample-Bundle Correlation

Each bundle is analyzed in relationship to its originating sample:

```python
@dataclass(frozen=True)
class SampleAnalysis:
    """Analysis of a single sample's execution."""

    sample_id: str
    input_summary: str
    expected_summary: str
    actual_output: str
    passed: bool
    score: float

    trajectory_summary: str
    """High-level description of execution path."""

    decision_points: tuple[DecisionPoint, ...]
    """Key moments where agent made choices."""

    anomalies: tuple[Anomaly, ...]
    """Unexpected behaviors observed."""
```

## AgentLoop Integration

### Retrospective Analysis

For AgentLoop integration, bundles come directly without expected outputs:

```python
async def _execute_retrospective_mode(
    self,
    request: AnalysisRequest,
) -> tuple[AnalysisReport, Session]:
    """Execute analysis in retrospective mode."""

    # 1. Load provided bundles
    bundles = [DebugBundle.load(p) for p in request.bundles]
    self._load_bundles(bundles)

    # 2. Run trajectory analysis
    return await self._analyze_trajectories(request, bundles)
```

### Trajectory Soundness

Without expected outputs, analysis focuses on trajectory quality:

```python
@dataclass(frozen=True)
class TrajectoryAssessment:
    """Assessment of whether a trajectory was sound."""

    bundle_id: str
    objective_understood: bool
    """Did the agent correctly understand the task?"""

    approach_reasonable: bool
    """Was the chosen approach sensible?"""

    execution_sound: bool
    """Did the agent execute the approach correctly?"""

    recovery_appropriate: bool
    """Did the agent recover appropriately from setbacks?"""

    outcome_achieved: bool
    """Did the agent achieve the objective?"""

    improvement_opportunities: tuple[str, ...]
    """Where could the agent have done better?"""

    overall_assessment: Literal["excellent", "good", "fair", "poor"]
```

## Configuration

### AnalysisLoopConfig

```python
@dataclass(frozen=True)
class AnalysisLoopConfig:
    """Configuration for AnalysisLoop."""

    # Execution
    deadline: Deadline | None = None
    budget: Budget | None = None

    # EvalLoop integration
    collection_timeout: timedelta = timedelta(minutes=30)
    """Max time to wait for eval results."""

    max_concurrent_batches: int = 3
    """Maximum micro-batches in flight."""

    # Analysis
    max_bundles_in_memory: int = 50
    """Bundle cache size."""

    analysis_depth: Literal["shallow", "standard", "deep"] = "standard"
    """How thoroughly to analyze each bundle."""

    # Output
    report_format: Literal["structured", "narrative"] = "structured"
    """Report output style."""

    include_evidence: bool = True
    """Include raw evidence in report."""
```

## Session State

AnalysisLoop maintains state across analysis phases:

### Slices

| Slice | Purpose |
|-------|---------|
| `LoadedBundle` | Metadata for bundles in analysis context |
| `SampleAnalysis` | Per-sample analysis results |
| `TrajectoryAssessment` | Per-bundle trajectory assessments |
| `Finding` | Accumulated findings |
| `Hypothesis` | Generated hypotheses |
| `AnalysisProgress` | Current phase and progress |

### Progress Tracking

```python
@dataclass(frozen=True)
class AnalysisProgress:
    """Tracks analysis progress."""

    phase: Literal["loading", "analyzing", "synthesizing", "reporting"]
    bundles_loaded: int
    bundles_analyzed: int
    total_bundles: int
    current_batch: int | None
    total_batches: int | None
```

## Usage

### EvalLoop Mode

```python
from weakincentives.analysis import AnalysisLoop, AnalysisRequest, ResearchObjective
from weakincentives.evals import EvalLoop, Dataset

# Setup mailboxes
analysis_requests = InMemoryMailbox(name="analysis-requests")
eval_requests = InMemoryMailbox(name="eval-requests")
eval_results = InMemoryMailbox(name="eval-results")

# Create loops
eval_loop = EvalLoop(
    loop=agent_loop,
    evaluator=my_evaluator,
    requests=eval_requests,
    config=EvalLoopConfig(debug_bundle_dir=Path("./bundles/")),
)

analysis_loop = AnalysisLoop(
    adapter=analysis_adapter,
    requests=analysis_requests,
    eval_requests=eval_requests,
    eval_results=eval_results,
)

# Submit analysis request
analysis_requests.send(
    AgentLoopRequest(
        request=AnalysisRequest(
            objective=ResearchObjective(
                question="Why do tool calls fail on inputs > 10000 tokens?",
                hypothesis="Token truncation causes context loss",
                focus_areas=("tool_invocations", "token_usage"),
            ),
            dataset=my_dataset,
            batch_size=5,
        ),
    ),
    reply_to=analysis_responses,
)

# Run both loops
group = LoopGroup(loops=[eval_loop, analysis_loop])
group.run()
```

### AgentLoop Mode

```python
# Collect bundles from AgentLoop executions
bundle_paths = list(Path("./debug/").glob("*.zip"))

# Submit retrospective analysis
analysis_requests.send(
    AgentLoopRequest(
        request=AnalysisRequest(
            objective=ResearchObjective(
                question="Did the agent take reasonable paths to solutions?",
                focus_areas=("decision_points", "recovery_behavior"),
            ),
            bundles=tuple(bundle_paths),
        ),
    ),
    reply_to=analysis_responses,
)

analysis_loop.run()
```

### Direct Execution

```python
report, session = analysis_loop.execute(
    AnalysisRequest(
        objective=ResearchObjective(
            question="What patterns distinguish passing from failing samples?",
        ),
        bundles=tuple(bundle_paths),
    )
)

print(f"Analyzed {report.bundles_analyzed} bundles")
for finding in report.findings:
    print(f"[{finding.severity}] {finding.summary}")
for rec in report.recommendations:
    print(f"[{rec.priority}] {rec.action}")
```

## Implementation Notes

AnalysisLoop is a specialized built-in implementation of AgentLoop. It uses the
same underlying patterns:

- **Mailbox-driven**: Requests via mailbox; results via `msg.reply()`
- **Factory-based**: Owns prompt and session construction
- **Resource-managed**: Analysis tools registered as prompt resources
- **Lifecycle-aware**: Integrates with LoopGroup and ShutdownCoordinator

The analysis agent itself is instantiated with a purpose-built prompt template
that includes:

1. The research objective as primary context
1. Bundle inspection tools
1. Structured output schema for reports
1. Guidelines for evidence-based reasoning

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Bundle load failure | Skip bundle, log warning, continue |
| EvalLoop timeout | Return partial results with warning |
| Analysis budget exceeded | Finalize with available findings |
| Invalid research objective | Return error in AnalysisReport |

## Limitations

- **Sequential bundle loading**: Large analyses may be memory-constrained
- **Single adapter**: One analysis agent per AnalysisLoop
- **No incremental results**: Full report returned on completion
- **EvalLoop dependency**: Dataset mode requires running EvalLoop

## Future Considerations

- **Streaming analysis**: Incremental findings as bundles are processed
- **Distributed analysis**: Shard bundle analysis across workers
- **Interactive mode**: REPL-style bundle exploration
- **Diff-based analysis**: Compare analyses across agent versions
- **Integration with experiment tracking**: Auto-link findings to experiments

## Related Specifications

- `specs/AGENT_LOOP.md` - Base loop abstraction
- `specs/EVALS.md` - EvalLoop and evaluation framework
- `specs/DEBUG_BUNDLE.md` - Bundle format and creation
- `specs/MAILBOX.md` - Message passing protocol
- `specs/LIFECYCLE.md` - Loop coordination
- `specs/WINK_QUERY.md` - SQL queries against bundles
