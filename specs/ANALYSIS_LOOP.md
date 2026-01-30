# Analysis Loop Specification

## Purpose

The best way to debug a complex agent is with another agent.

`AnalysisLoop` is a specialized AgentLoop that analyzes debug bundles and
produces human-readable analysis bundles with actionable insights. It runs
continuously in the background, sampling executions from EvalLoop or AgentLoop
within configurable budget constraints.

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

- **Always-on**: Runs continuously with sampling; not just on-demand
- **Budget-aware**: Operates within token/cost constraints via sampling rate
- **Self-contained output**: Analysis bundles include source debug bundles
- **Human-first**: Produces markdown for reading, not structured data for machines
- **Single tool dependency**: Only `wink query`; no other tools required
- **Prebuilt with hooks**: Works out of the box; customize via prompt overrides

## Auto-Connect API

AnalysisLoop provides a simple API to attach to existing loops for continuous
background analysis:

```python
from weakincentives.analysis import AnalysisLoop

# Attach to EvalLoop - analyzes eval results with expected outputs
analysis = AnalysisLoop.connect_to_eval_loop(
    eval_loop=eval_loop,
    objective="Identify patterns in failing samples",
    sample_rate=0.1,  # Analyze 10% of executions
    budget=Budget(max_tokens=100_000),
)

# Attach to AgentLoop - retrospective trajectory analysis
analysis = AnalysisLoop.connect_to_agent_loop(
    agent_loop=agent_loop,
    objective="Assess trajectory soundness",
    sample_rate=0.05,  # Analyze 5% of executions
    budget=Budget(max_tokens=50_000),
)

# Both return the same AnalysisLoop instance configured appropriately
# Run as part of a LoopGroup for lifecycle management
group = LoopGroup(loops=[eval_loop, analysis])
group.run()
```

### Sampling

The `sample_rate` parameter controls what fraction of executions trigger
analysis. This enables continuous background operation within budget:

| Sample Rate | Behavior |
|-------------|----------|
| `1.0` | Analyze every execution (expensive) |
| `0.1` | Analyze ~10% of executions |
| `0.0` | Disable automatic analysis |

Sampling is random but reproducible given a seed. Failed executions can be
configured to always trigger analysis regardless of sample rate.

### Budget Constraints

Analysis operates within the specified budget:

```python
@dataclass(frozen=True)
class AnalysisBudget:
    max_tokens: int = 100_000
    """Maximum tokens per analysis session."""

    max_bundles: int = 10
    """Maximum bundles to analyze per session."""

    reset_interval: timedelta = timedelta(hours=1)
    """Budget resets after this interval."""
```

When budget is exhausted, analysis pauses until the reset interval elapses.

## Analysis Bundle

The output of AnalysisLoop is an **analysis bundle**: a self-contained archive
designed for human consumption.

### Structure

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

Analysis bundles include the source debug bundles that were analyzed. This
ensures the analysis is reproducible and allows drill-down without requiring
access to the original bundle storage.

```python
@dataclass(frozen=True)
class AnalysisBundleConfig:
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
from weakincentives.analysis import AnalysisLoop, AnalysisPromptOverrides

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

analysis = AnalysisLoop.connect_to_eval_loop(
    eval_loop=eval_loop,
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

    # Sampling
    sample_rate: float = 0.1
    """Fraction of executions to analyze (0.0-1.0)."""

    always_analyze_failures: bool = True
    """Analyze failed executions regardless of sample rate."""

    # Budget
    budget: AnalysisBudget = field(default_factory=AnalysisBudget)
    """Token and bundle limits."""

    # Output
    output_dir: Path = Path("./analysis-bundles/")
    """Where to write analysis bundles."""

    include_source_bundles: bool = True
    """Embed debug bundles in analysis output."""

    # Agent
    overrides: AnalysisPromptOverrides | None = None
    """Prompt section overrides."""
```

## Operational Modes

### EvalLoop Integration

When connected to EvalLoop, analysis has access to expected outputs and can
focus on pass/fail patterns:

```
EvalLoop                    AnalysisLoop
    │                            │
    ├─── eval complete ─────────►│ (sampled)
    │    + debug bundle          │
    │                            ├── load bundle
    │                            ├── run wink query
    │                            ├── generate report
    │                            └── write analysis bundle
```

Questions this mode answers:

- "Why do samples fail?"
- "What distinguishes passing from failing samples?"
- "Which tool sequences correlate with success?"

### AgentLoop Integration

When connected to AgentLoop (production), there are no expected outputs.
Analysis focuses on trajectory soundness:

```
AgentLoop                   AnalysisLoop
    │                            │
    ├─── execution complete ────►│ (sampled)
    │    + debug bundle          │
    │                            ├── load bundle
    │                            ├── assess trajectory
    │                            ├── generate report
    │                            └── write analysis bundle
```

Questions this mode answers:

- "Did the agent take a reasonable path?"
- "Where did the agent get stuck?"
- "What information was missing?"

## Usage

### Continuous Background Analysis

```python
from weakincentives.analysis import AnalysisLoop
from weakincentives.runtime import AgentLoop, LoopGroup

# Create your agent loop
agent_loop = AgentLoop(...)

# Attach analysis with 10% sampling
analysis = AnalysisLoop.connect_to_agent_loop(
    agent_loop=agent_loop,
    objective="Monitor for degraded performance patterns",
    sample_rate=0.1,
)

# Run together
group = LoopGroup(loops=[agent_loop, analysis])
group.run()

# Analysis bundles appear in ./analysis-bundles/
```

### On-Demand Analysis

```python
from weakincentives.analysis import AnalysisLoop

# Analyze specific bundles
bundle = AnalysisLoop.analyze(
    bundles=[Path("./debug/bundle-001.zip")],
    objective="Deep dive on this specific failure",
)

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

## Implementation

AnalysisLoop is a specialized AgentLoop with:

- A prebuilt prompt template focused on bundle analysis
- `wink query` as the single registered tool
- Markdown file output instead of structured responses
- Analysis bundle packaging on completion

```python
class AnalysisLoop(AgentLoop[AnalysisTrigger, AnalysisBundle]):
    """Specialized loop for debug bundle analysis."""

    @classmethod
    def connect_to_eval_loop(
        cls,
        eval_loop: EvalLoop,
        objective: str,
        sample_rate: float = 0.1,
        **config_kwargs,
    ) -> AnalysisLoop:
        """Create an AnalysisLoop connected to an EvalLoop."""
        ...

    @classmethod
    def connect_to_agent_loop(
        cls,
        agent_loop: AgentLoop,
        objective: str,
        sample_rate: float = 0.1,
        **config_kwargs,
    ) -> AnalysisLoop:
        """Create an AnalysisLoop connected to an AgentLoop."""
        ...

    @classmethod
    def analyze(
        cls,
        bundles: Sequence[Path],
        objective: str,
        **config_kwargs,
    ) -> AnalysisBundle:
        """One-shot analysis of specific bundles."""
        ...
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
- `specs/WINK_QUERY.md` - SQL queries against bundles
- `specs/LIFECYCLE.md` - Loop coordination
