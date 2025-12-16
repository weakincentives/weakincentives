# Evals Specification

## Purpose

Minimal evaluation framework built on MainLoop. MainLoop handles orchestration;
this spec adds datasets and scoring.

## Guiding Principles

- **MainLoop is the runner** - No parallel orchestration; MainLoop already does this
- **Evaluators are functions** - `(output, expected) -> Score`
- **Datasets are tuples** - Immutable, typed, simple

## Core Types

### Sample

```python
@dataclass(slots=True, frozen=True)
class Sample[InputT, ExpectedT]:
    """Single evaluation input."""
    id: str
    input: InputT
    expected: ExpectedT
```

### Dataset

A dataset is just a tuple of samples:

```python
Dataset = tuple[Sample[InputT, ExpectedT], ...]
```

Load from JSONL:

```python
def load_jsonl[I, E](
    path: Path,
    input_type: type[I],
    expected_type: type[E],
) -> tuple[Sample[I, E], ...]:
    """Load samples from JSONL file."""
    ...
```

### Score

```python
@dataclass(slots=True, frozen=True)
class Score:
    """Result of scoring one output."""
    value: float      # 0.0 to 1.0
    passed: bool
    reason: str = ""
```

### Evaluator

An evaluator is a callable:

```python
Evaluator = Callable[[OutputT, ExpectedT], Score]
```

## Built-in Evaluators

```python
def exact_match[T](output: T, expected: T) -> Score:
    """Exact equality check."""
    passed = output == expected
    return Score(value=1.0 if passed else 0.0, passed=passed)


def contains(output: str, expected: str) -> Score:
    """Substring check."""
    passed = expected in output
    return Score(value=1.0 if passed else 0.0, passed=passed)


def all_of(*evaluators: Evaluator[O, E]) -> Evaluator[O, E]:
    """Combine evaluators; all must pass."""
    def evaluate(output: O, expected: E) -> Score:
        scores = [e(output, expected) for e in evaluators]
        passed = all(s.passed for s in scores)
        value = sum(s.value for s in scores) / len(scores)
        return Score(value=value, passed=passed)
    return evaluate
```

### LLM-as-Judge

Creates an evaluator that uses an adapter for scoring:

```python
@dataclass(slots=True, frozen=True)
class JudgeOutput:
    score: float
    passed: bool
    reason: str


JUDGE_TEMPLATE = PromptTemplate[JudgeOutput](...)


def llm_judge(
    adapter: ProviderAdapter[JudgeOutput],
    criterion: str,
) -> Evaluator[str, str]:
    """Create evaluator that uses LLM to judge output."""
    def evaluate(output: str, expected: str) -> Score:
        prompt = Prompt(JUDGE_TEMPLATE).bind(
            criterion=criterion,
            output=output,
            expected=expected,
        )
        response = adapter.evaluate(prompt)
        return Score(
            value=response.output.score,
            passed=response.output.passed,
            reason=response.output.reason,
        )
    return evaluate
```

## Running Evals

### EvalResult

```python
@dataclass(slots=True, frozen=True)
class EvalResult:
    """Result for one sample."""
    sample_id: str
    score: Score
    latency_ms: int
    tokens: int
    error: str | None = None
```

### EvalReport

```python
@dataclass(slots=True, frozen=True)
class EvalReport:
    """Aggregate results."""
    results: tuple[EvalResult, ...]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.score.passed) / len(self.results)

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score.value for r in self.results) / len(self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)
```

### run_eval

```python
def run_eval[I, O, E](
    loop: MainLoop[I, O],
    dataset: tuple[Sample[I, E], ...],
    evaluator: Evaluator[O, E],
) -> EvalReport:
    """Run evaluation using MainLoop."""
    results: list[EvalResult] = []

    for sample in dataset:
        start = time.monotonic()
        try:
            response, _ = loop.execute(sample.input)
            latency_ms = int((time.monotonic() - start) * 1000)
            score = evaluator(response.output, sample.expected)
            results.append(EvalResult(
                sample_id=sample.id,
                score=score,
                latency_ms=latency_ms,
                tokens=response.usage.total_tokens,
            ))
        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            results.append(EvalResult(
                sample_id=sample.id,
                score=Score(value=0.0, passed=False, reason=str(e)),
                latency_ms=latency_ms,
                tokens=0,
                error=str(e),
            ))

    return EvalReport(results=tuple(results))
```

## Usage

```python
from weakincentives import MainLoop, PromptTemplate, Prompt
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.evals import (
    Sample, load_jsonl, run_eval, exact_match, llm_judge, all_of,
)

# Define the loop under test
class QALoop(MainLoop[str, str]):
    def create_prompt(self, question: str) -> Prompt[str]:
        return Prompt(self._template).bind(question=question)

    def create_session(self) -> Session:
        return Session(bus=self._bus)

# Load dataset
dataset = load_jsonl("tests/fixtures/qa.jsonl", str, str)

# Create loop
adapter = OpenAIAdapter(model="gpt-4o")
loop = QALoop(adapter=adapter, bus=InProcessEventBus())

# Run with simple evaluator
report = run_eval(loop, dataset, exact_match)
print(f"Pass rate: {report.pass_rate:.1%}")

# Run with LLM judge
judge_adapter = OpenAIAdapter(model="gpt-4o-mini")
evaluator = all_of(
    exact_match,
    llm_judge(judge_adapter, "Answer is correct and concise"),
)
report = run_eval(loop, dataset, evaluator)
```

## Events

MainLoop already emits `MainLoopCompleted` and `MainLoopFailed`. No additional
events needed - subscribe to those for observability.

## Limitations

- Sequential execution (MainLoop is synchronous)
- No caching of repeated samples
- No resume from checkpoint
