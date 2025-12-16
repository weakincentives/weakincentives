# Evals Specification

## Purpose

This specification defines a minimal evaluation framework for measuring prompt
and agent quality in WINK. The design leverages existing session infrastructure,
event-driven state management, and provider-agnostic adapters to enable
reproducible, type-safe evaluations without introducing heavyweight abstractions.

## Guiding Principles

- **Composition over framework** - Evals are built from existing primitives
  (sessions, events, state slices), not a parallel evaluation runtime
- **Dataclass-first metrics** - All evaluation results are frozen dataclasses
  for type safety and serializability
- **Evaluator as function** - Evaluators are pure functions, not classes with
  lifecycle methods
- **Sample isolation** - Each sample runs in its own session for reproducibility
- **Provider-agnostic judging** - LLM-as-judge evaluators use the same adapter
  interface as production prompts

## Core Abstractions

### Sample

A single evaluation input with expected output and metadata:

```python
from dataclasses import dataclass
from typing import Any

@dataclass(slots=True, frozen=True)
class Sample[InputT, ExpectedT]:
    """Single evaluation sample."""

    id: str
    input: InputT
    expected: ExpectedT
    metadata: dict[str, Any] = field(default_factory=dict)
```

Samples are generic over input and expected types, enabling strongly-typed
datasets for different evaluation scenarios.

### Dataset

An immutable sequence of samples:

```python
from collections.abc import Sequence

@dataclass(slots=True, frozen=True)
class Dataset[InputT, ExpectedT]:
    """Immutable collection of evaluation samples."""

    name: str
    samples: tuple[Sample[InputT, ExpectedT], ...]
    version: str = "1.0"

    def __iter__(self) -> Iterator[Sample[InputT, ExpectedT]]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)
```

Datasets are loaded from JSON, JSONL, or constructed programmatically:

```python
from weakincentives.evals import Dataset, Sample, load_jsonl

# From file
dataset = load_jsonl("tests/fixtures/qa_samples.jsonl", QAInput, QAExpected)

# Programmatic
dataset = Dataset(
    name="math-eval",
    samples=tuple(
        Sample(id=str(i), input=MathInput(problem=p), expected=MathExpected(answer=a))
        for i, (p, a) in enumerate(problems)
    ),
)
```

### Score

The result of evaluating a single criterion:

```python
@dataclass(slots=True, frozen=True)
class Score:
    """Result of a single evaluation criterion."""

    criterion: str
    value: float  # 0.0 to 1.0 normalized
    passed: bool
    rationale: str = ""
```

### EvalResult

The complete evaluation result for a single sample:

```python
@dataclass(slots=True, frozen=True)
class EvalResult[OutputT]:
    """Complete evaluation result for one sample."""

    sample_id: str
    output: OutputT
    scores: tuple[Score, ...]
    latency_ms: int
    tokens_used: int
    success: bool
    error: str | None = None

    @property
    def aggregate_score(self) -> float:
        """Mean of all criterion scores."""
        if not self.scores:
            return 0.0
        return sum(s.value for s in self.scores) / len(self.scores)
```

### Evaluator Protocol

Evaluators are functions that score outputs against criteria:

```python
from typing import Protocol

class Evaluator[InputT, OutputT, ExpectedT](Protocol):
    """Protocol for evaluation functions."""

    def __call__(
        self,
        sample: Sample[InputT, ExpectedT],
        output: OutputT,
        *,
        context: EvalContext,
    ) -> tuple[Score, ...]:
        """Score output against sample criteria."""
        ...
```

The `EvalContext` provides access to the adapter (for LLM-as-judge) and session:

```python
@dataclass(slots=True, frozen=True)
class EvalContext:
    """Context available to evaluators."""

    adapter: ProviderAdapter[Any]
    session: Session
    run_id: str
```

## Built-in Evaluators

### Exact Match

```python
def exact_match[T](
    sample: Sample[Any, T],
    output: T,
    *,
    context: EvalContext,
) -> tuple[Score, ...]:
    """Check if output exactly matches expected."""
    passed = output == sample.expected
    return (
        Score(
            criterion="exact_match",
            value=1.0 if passed else 0.0,
            passed=passed,
        ),
    )
```

### Contains

```python
def contains(
    sample: Sample[Any, str],
    output: str,
    *,
    context: EvalContext,
) -> tuple[Score, ...]:
    """Check if output contains expected substring."""
    passed = sample.expected in output
    return (
        Score(
            criterion="contains",
            value=1.0 if passed else 0.0,
            passed=passed,
        ),
    )
```

### LLM-as-Judge

For subjective criteria, use the adapter to evaluate:

```python
@dataclass(slots=True, frozen=True)
class JudgeInput:
    criterion: str
    output: str
    expected: str | None

@dataclass(slots=True, frozen=True)
class JudgeOutput:
    score: float  # 0.0 to 1.0
    passed: bool
    rationale: str

JUDGE_PROMPT = PromptTemplate[JudgeOutput](
    ns="wink.evals",
    key="llm-judge",
    name="llm_judge",
    sections=[
        MarkdownSection(
            title="Evaluation Task",
            template="""
You are an evaluation judge. Score the following output on the criterion.

## Criterion
$criterion

## Output to Evaluate
$output

## Expected (if provided)
$expected

Provide a score from 0.0 to 1.0 and explain your reasoning.
""",
            key="task",
        ),
    ],
)

def llm_judge(
    criterion: str,
) -> Evaluator[Any, str, str | None]:
    """Create an LLM-as-judge evaluator for a criterion."""

    def evaluate(
        sample: Sample[Any, str | None],
        output: str,
        *,
        context: EvalContext,
    ) -> tuple[Score, ...]:
        prompt = JUDGE_PROMPT.bind(
            criterion=criterion,
            output=output,
            expected=sample.expected or "Not provided",
        )
        response = context.adapter.evaluate(
            prompt,
            session=context.session,
        )
        return (
            Score(
                criterion=criterion,
                value=response.output.score,
                passed=response.output.passed,
                rationale=response.output.rationale,
            ),
        )

    return evaluate
```

### Composite Evaluator

Combine multiple evaluators:

```python
def composite[I, O, E](
    *evaluators: Evaluator[I, O, E],
) -> Evaluator[I, O, E]:
    """Combine multiple evaluators into one."""

    def evaluate(
        sample: Sample[I, E],
        output: O,
        *,
        context: EvalContext,
    ) -> tuple[Score, ...]:
        scores: list[Score] = []
        for evaluator in evaluators:
            scores.extend(evaluator(sample, output, context=context))
        return tuple(scores)

    return evaluate
```

## Eval Suite

The `EvalSuite` orchestrates running evaluations across a dataset:

```python
@dataclass(slots=True, frozen=True)
class EvalConfig:
    """Configuration for an evaluation run."""

    max_concurrent: int = 1
    timeout_per_sample_ms: int = 30000
    max_retries: int = 0
    stop_on_error: bool = False

@dataclass(slots=True)
class EvalSuite[InputT, OutputT, ExpectedT]:
    """Orchestrates evaluation across a dataset."""

    prompt: PromptTemplate[OutputT]
    dataset: Dataset[InputT, ExpectedT]
    evaluator: Evaluator[InputT, OutputT, ExpectedT]
    adapter: ProviderAdapter[OutputT]
    config: EvalConfig = field(default_factory=EvalConfig)

    def run(
        self,
        *,
        bus: EventBus | None = None,
        tags: dict[str, str] | None = None,
    ) -> EvalReport[OutputT]:
        """Execute evaluation and return report."""
        bus = bus or InProcessEventBus()
        run_id = uuid4().hex
        results: list[EvalResult[OutputT]] = []

        for sample in self.dataset:
            result = self._evaluate_sample(sample, bus, run_id, tags)
            results.append(result)
            bus.publish(SampleEvaluated(run_id=run_id, result=result))

            if not result.success and self.config.stop_on_error:
                break

        return EvalReport(
            run_id=run_id,
            dataset_name=self.dataset.name,
            results=tuple(results),
            config=self.config,
        )

    def _evaluate_sample(
        self,
        sample: Sample[InputT, ExpectedT],
        bus: EventBus,
        run_id: str,
        tags: dict[str, str] | None,
    ) -> EvalResult[OutputT]:
        """Evaluate a single sample in its own session."""
        session = Session(
            bus=bus,
            tags={
                "eval_run_id": run_id,
                "sample_id": sample.id,
                **(tags or {}),
            },
        )
        context = EvalContext(
            adapter=self.adapter,
            session=session,
            run_id=run_id,
        )

        start = time.monotonic()
        try:
            prompt = self.prompt.bind(**asdict(sample.input))
            response = self.adapter.evaluate(prompt, session=session)
            latency_ms = int((time.monotonic() - start) * 1000)

            scores = self.evaluator(sample, response.output, context=context)

            return EvalResult(
                sample_id=sample.id,
                output=response.output,
                scores=scores,
                latency_ms=latency_ms,
                tokens_used=response.usage.total_tokens,
                success=True,
            )
        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            return EvalResult(
                sample_id=sample.id,
                output=None,  # type: ignore
                scores=(),
                latency_ms=latency_ms,
                tokens_used=0,
                success=False,
                error=str(e),
            )
```

## Eval Report

Aggregate results with computed metrics:

```python
@dataclass(slots=True, frozen=True)
class EvalReport[OutputT]:
    """Aggregate evaluation report."""

    run_id: str
    dataset_name: str
    results: tuple[EvalResult[OutputT], ...]
    config: EvalConfig

    @property
    def total_samples(self) -> int:
        return len(self.results)

    @property
    def successful_samples(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def pass_rate(self) -> float:
        """Fraction of samples where all criteria passed."""
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.success and all(s.passed for s in r.scores))
        return passed / len(self.results)

    @property
    def mean_score(self) -> float:
        """Mean aggregate score across all samples."""
        successful = [r for r in self.results if r.success]
        if not successful:
            return 0.0
        return sum(r.aggregate_score for r in successful) / len(successful)

    @property
    def scores_by_criterion(self) -> dict[str, float]:
        """Mean score per criterion."""
        by_criterion: dict[str, list[float]] = {}
        for result in self.results:
            if not result.success:
                continue
            for score in result.scores:
                by_criterion.setdefault(score.criterion, []).append(score.value)
        return {k: sum(v) / len(v) for k, v in by_criterion.items()}

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_used for r in self.results)

    @property
    def mean_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)
```

## Events

Evaluation emits events for observability:

```python
@dataclass(slots=True, frozen=True)
class EvalRunStarted:
    """Emitted when evaluation run begins."""
    run_id: str
    dataset_name: str
    sample_count: int

@dataclass(slots=True, frozen=True)
class SampleEvaluated:
    """Emitted after each sample is evaluated."""
    run_id: str
    result: EvalResult[Any]

@dataclass(slots=True, frozen=True)
class EvalRunCompleted:
    """Emitted when evaluation run finishes."""
    run_id: str
    pass_rate: float
    mean_score: float
    total_tokens: int
```

## Integration with LangSmith

When LangSmith tracing is enabled, evaluation runs are automatically traced:

```python
from weakincentives.contrib.langsmith import configure_wink

configure_wink(
    project="my-evals",
    tracing_enabled=True,
)

# Runs traced to LangSmith with:
# - Run-level metadata (run_id, dataset_name)
# - Sample-level spans (one per sample)
# - Scores as span outputs
# - Tokens and latency as metrics
```

## State Slice for Accumulation

For long-running or streaming evaluations, results accumulate in session state:

```python
# Register the slice
session.mutate(EvalResult).seed(())

# Results accumulate as events are processed
session.mutate(EvalResult).register(
    SampleEvaluated,
    lambda state, event: (*state, event.result),
)

# Query intermediate results
current_results = session.query(EvalResult).all()
```

## Usage Example

```python
from weakincentives import PromptTemplate, MarkdownSection
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.evals import (
    Dataset,
    Sample,
    EvalSuite,
    EvalConfig,
    exact_match,
    llm_judge,
    composite,
    load_jsonl,
)

# Define prompt under test
@dataclass(slots=True, frozen=True)
class QAInput:
    question: str

@dataclass(slots=True, frozen=True)
class QAOutput:
    answer: str

qa_prompt = PromptTemplate[QAOutput](
    ns="my-app",
    key="qa",
    name="qa_prompt",
    sections=[
        MarkdownSection(
            title="Question Answering",
            template="Answer the question: $question",
            key="main",
        ),
    ],
)

# Load dataset
dataset: Dataset[QAInput, str] = load_jsonl(
    "tests/fixtures/qa_eval.jsonl",
    QAInput,
    str,
)

# Define evaluator
evaluator = composite(
    exact_match,
    llm_judge("Answers the question correctly and completely"),
    llm_judge("Response is concise and well-formatted"),
)

# Run evaluation
adapter = OpenAIAdapter(model="gpt-4o")
suite = EvalSuite(
    prompt=qa_prompt,
    dataset=dataset,
    evaluator=evaluator,
    adapter=adapter,
    config=EvalConfig(max_concurrent=4, timeout_per_sample_ms=30000),
)

report = suite.run(tags={"experiment": "baseline"})

# Inspect results
print(f"Pass rate: {report.pass_rate:.1%}")
print(f"Mean score: {report.mean_score:.2f}")
print(f"By criterion: {report.scores_by_criterion}")
print(f"Total tokens: {report.total_tokens}")
```

## CLI Integration

Evals can be run from the CLI:

```bash
# Run evaluation
wink eval run --prompt qa --dataset tests/fixtures/qa_eval.jsonl --model gpt-4o

# Compare models
wink eval compare \
    --prompt qa \
    --dataset tests/fixtures/qa_eval.jsonl \
    --models gpt-4o gpt-4o-mini claude-3-opus

# Export results
wink eval export --run-id abc123 --format csv > results.csv
```

## Testing Evaluators

Evaluators are tested with fixture data:

```python
def test_exact_match_passes():
    sample = Sample(id="1", input="x", expected="hello")
    context = EvalContext(adapter=mock_adapter, session=session, run_id="test")

    scores = exact_match(sample, "hello", context=context)

    assert len(scores) == 1
    assert scores[0].passed is True
    assert scores[0].value == 1.0

def test_exact_match_fails():
    sample = Sample(id="1", input="x", expected="hello")
    context = EvalContext(adapter=mock_adapter, session=session, run_id="test")

    scores = exact_match(sample, "world", context=context)

    assert scores[0].passed is False
    assert scores[0].value == 0.0
```

## Limitations

- **No automatic parallelism** - `max_concurrent` requires explicit async
  implementation; the sync API processes samples sequentially
- **No caching** - Identical samples are re-evaluated; caching is left to the
  adapter layer
- **No warm-up** - Cold start latency affects first sample timing
- **Single judge model** - LLM-as-judge uses the same adapter; cross-model
  judging requires separate adapter instances

## Future Considerations

These are explicitly out of scope for the initial implementation but may be
added later based on usage patterns:

- **Streaming results** - Emit results as they complete for long-running evals
- **Resume from checkpoint** - Continue interrupted eval runs
- **A/B comparison** - First-class support for comparing prompt versions
- **Regression detection** - Automatic alerting on score degradation
- **Human-in-the-loop** - UI for manual review of edge cases
