# Evals Specification

## Purpose

Minimal evaluation framework built on MainLoop. MainLoop handles orchestration;
this spec adds datasets and scoring.

## Guiding Principles

- **EvalLoop wraps MainLoop** - Composition-based, event-driven, type-safe
- **Evaluators are functions** - Pure `(output, expected) -> Score`
- **Datasets are immutable** - Frozen dataclass with typed samples

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌───────────┐     ┌────────────┐
│   Dataset   │────▶│ EvalLoop │────▶│ MainLoop  │────▶│  Adapter   │
│  (samples)  │     │.execute()│     │ .execute()│     │            │
└─────────────┘     └────┬─────┘     └───────────┘     └────────────┘
                         │
                         ▼
                   ┌───────────┐     ┌────────────┐
                   │ Evaluator │────▶│ EvalReport │
                   │ (scoring) │     │ (metrics)  │
                   └───────────┘     └────────────┘
```

`EvalLoop` orchestrates evaluation: for each sample, it executes through the
provided `MainLoop`, scores the output, and aggregates results into a report.

## Core Types

### Sample

A single evaluation case pairing an input with its expected output:

```python
@dataclass(slots=True, frozen=True)
class Sample[InputT, ExpectedT]:
    """Single evaluation input."""
    id: str
    input: InputT
    expected: ExpectedT
```

The generic parameters allow typed datasets—`Sample[MathProblem, int]` for math
evals, `Sample[str, str]` for QA, etc.

### Dataset

An immutable collection of samples. The class provides a clean API for loading
and accessing evaluation data:

```python
@dataclass(slots=True, frozen=True)
class Dataset[InputT, ExpectedT]:
    """Immutable collection of evaluation samples."""
    samples: tuple[Sample[InputT, ExpectedT], ...]

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample[InputT, ExpectedT]]:
        """Iterate over samples."""
        return iter(self.samples)

    def __getitem__(self, index: int) -> Sample[InputT, ExpectedT]:
        """Get sample by index."""
        return self.samples[index]

    @staticmethod
    def load[I, E](
        path: Path,
        input_type: type[I],
        expected_type: type[E],
    ) -> Dataset[I, E]:
        """Load dataset from JSONL file.

        Each line must be a JSON object with "id", "input", and "expected" keys.
        Primitives (str, int, float, bool) are used directly; mappings are
        deserialized into dataclasses via serde.parse.

        Args:
            path: Path to JSONL file
            input_type: Type for deserializing input field
            expected_type: Type for deserializing expected field

        Returns:
            Dataset containing all samples from the file
        """
        samples: list[Sample[I, E]] = []
        with path.open() as f:
            for line in f:
                obj = json.loads(line)
                samples.append(Sample(
                    id=obj["id"],
                    input=_coerce(obj["input"], input_type),
                    expected=_coerce(obj["expected"], expected_type),
                ))
        return Dataset(samples=tuple(samples))


def _coerce[T](value: object, target: type[T]) -> T:
    """Coerce JSON value to target type.

    Primitives (str, int, float, bool) pass through directly.
    Mappings are parsed as dataclasses via serde.parse.
    """
    if target in (str, int, float, bool):
        if not isinstance(value, target):
            raise TypeError(f"expected {target.__name__}, got {type(value).__name__}")
        return value  # type: ignore[return-value]
    if isinstance(value, Mapping):
        return parse(target, value)
    raise TypeError(f"cannot coerce {type(value).__name__} to {target.__name__}")
```

**JSONL format:**

```jsonl
{"id": "1", "input": "What is 2+2?", "expected": "4"}
{"id": "2", "input": "Capital of France?", "expected": "Paris"}
```

For complex types, input/expected can be nested objects that deserialize into
dataclasses.

### Score

The result of evaluating a single output:

```python
@dataclass(slots=True, frozen=True)
class Score:
    """Result of scoring one output."""
    value: float      # 0.0 to 1.0, normalized
    passed: bool      # Binary pass/fail
    reason: str = ""  # Explanation (useful for LLM judges)
```

The `value` field enables ranking and aggregation; `passed` provides a binary
threshold for pass rates.

### Evaluator

An evaluator is any callable matching this signature:

```python
Evaluator = Callable[[OutputT, ExpectedT], Score]
```

Evaluators are pure functions—no side effects, no state. This makes them easy
to test, compose, and reason about.

## Built-in Evaluators

### exact_match

Strict equality check:

```python
def exact_match[T](output: T, expected: T) -> Score:
    """Exact equality check."""
    passed = output == expected
    return Score(value=1.0 if passed else 0.0, passed=passed)
```

### contains

Substring presence check:

```python
def contains(output: str, expected: str) -> Score:
    """Check if expected appears in output."""
    passed = expected in output
    return Score(value=1.0 if passed else 0.0, passed=passed)
```

### Combinators

Combine evaluators for multi-criteria scoring:

```python
def all_of(*evaluators: Evaluator[O, E]) -> Evaluator[O, E]:
    """All evaluators must pass. Score is the mean."""
    def evaluate(output: O, expected: E) -> Score:
        scores = [e(output, expected) for e in evaluators]
        passed = all(s.passed for s in scores)
        value = sum(s.value for s in scores) / len(scores)
        reasons = [s.reason for s in scores if s.reason]
        return Score(value=value, passed=passed, reason="; ".join(reasons))
    return evaluate


def any_of(*evaluators: Evaluator[O, E]) -> Evaluator[O, E]:
    """At least one evaluator must pass. Score is the max."""
    def evaluate(output: O, expected: E) -> Score:
        scores = [e(output, expected) for e in evaluators]
        passed = any(s.passed for s in scores)
        value = max(s.value for s in scores)
        reasons = [s.reason for s in scores if s.reason]
        return Score(value=value, passed=passed, reason="; ".join(reasons))
    return evaluate
```

### Custom Evaluators

Write domain-specific evaluators as simple functions:

```python
def within_tolerance(tolerance: float) -> Evaluator[float, float]:
    """Check if output is within tolerance of expected."""
    def evaluate(output: float, expected: float) -> Score:
        diff = abs(output - expected)
        passed = diff <= tolerance
        value = max(0.0, 1.0 - diff / tolerance) if tolerance > 0 else float(passed)
        return Score(value=value, passed=passed, reason=f"diff={diff:.4f}")
    return evaluate


def json_subset(output: dict, expected: dict) -> Score:
    """Check if expected keys/values are present in output."""
    for key, value in expected.items():
        if key not in output or output[key] != value:
            return Score(value=0.0, passed=False, reason=f"missing or wrong: {key}")
    return Score(value=1.0, passed=True)
```

## LLM-as-Judge

For subjective criteria, use an LLM to score outputs. Rather than asking for
numerical scores (which LLMs calibrate poorly), the judge selects from a fixed
set of rating labels that map to values.

### Rating Scale

```python
Rating = Literal["excellent", "good", "fair", "poor", "wrong"]

RATING_VALUES: dict[Rating, float] = {
    "excellent": 1.0,   # Fully meets criterion
    "good": 0.75,       # Meets criterion with minor issues
    "fair": 0.5,        # Partially meets criterion
    "poor": 0.25,       # Mostly fails criterion
    "wrong": 0.0,       # Completely fails criterion
}

PASSING_RATINGS: frozenset[Rating] = frozenset({"excellent", "good"})
```

### Judge Output

```python
@dataclass(slots=True, frozen=True)
class JudgeOutput:
    """Structured output from judge prompt."""
    rating: Rating  # Categorical label
    reason: str     # Brief explanation


JUDGE_TEMPLATE = PromptTemplate[JudgeOutput](
    ns="wink.evals",
    key="llm-judge",
    name="llm_judge",
    sections=[
        MarkdownSection(
            title="Evaluation Task",
            template="""You are an evaluation judge. Rate the output on the given criterion.

## Criterion
$criterion

## Output to Evaluate
$output

## Reference Answer
$expected

## Rating Scale
- **excellent**: Fully meets the criterion
- **good**: Meets the criterion with minor issues
- **fair**: Partially meets the criterion
- **poor**: Mostly fails the criterion
- **wrong**: Completely fails the criterion

Select one rating and explain your reasoning briefly.""",
            key="task",
        ),
    ],
)
```

### llm_judge Factory

```python
def llm_judge(
    adapter: ProviderAdapter[JudgeOutput],
    criterion: str,
    *,
    bus: Dispatcher,
) -> Evaluator[str, str]:
    """Create evaluator that uses LLM to judge output.

    Args:
        adapter: Provider adapter configured for JudgeOutput
        criterion: What to evaluate (e.g., "factual accuracy", "clarity")
        bus: Dispatcher for creating judge sessions

    Returns:
        Evaluator function that scores string outputs
    """
    def evaluate(output: str, expected: str) -> Score:
        prompt = Prompt(JUDGE_TEMPLATE).bind(
            criterion=criterion,
            output=output,
            expected=expected,
        )
        session = Session(bus=bus, tags={"judge_criterion": criterion})
        response = adapter.evaluate(prompt, session=session)
        rating = response.output.rating
        return Score(
            value=RATING_VALUES[rating],
            passed=rating in PASSING_RATINGS,
            reason=response.output.reason,
        )
    return evaluate
```

**Usage:**

```python
# Use a smaller/cheaper model for judging
judge_adapter = OpenAIAdapter[JudgeOutput](model="gpt-4o-mini")
judge_bus = InProcessDispatcher()

evaluator = all_of(
    contains,  # Must contain expected answer
    llm_judge(judge_adapter, "Response is helpful and well-formatted", bus=judge_bus),
    llm_judge(judge_adapter, "No hallucinated information", bus=judge_bus),
)
```

## Running Evals

### EvalResult

Result for a single sample:

```python
@dataclass(slots=True, frozen=True)
class EvalResult:
    """Result for one sample."""
    sample_id: str
    score: Score
    latency_ms: int
    error: str | None = None

    @property
    def success(self) -> bool:
        """True if no error occurred."""
        return self.error is None
```

### EvalReport

Aggregate results with computed metrics:

```python
@dataclass(slots=True, frozen=True)
class EvalReport:
    """Aggregate evaluation results."""
    results: tuple[EvalResult, ...]

    @property
    def total(self) -> int:
        """Total number of samples."""
        return len(self.results)

    @property
    def successful(self) -> int:
        """Samples that completed without error."""
        return sum(1 for r in self.results if r.success)

    @property
    def pass_rate(self) -> float:
        """Fraction of successful samples that passed."""
        successful = [r for r in self.results if r.success]
        if not successful:
            return 0.0
        return sum(1 for r in successful if r.score.passed) / len(successful)

    @property
    def mean_score(self) -> float:
        """Mean score across successful samples."""
        successful = [r for r in self.results if r.success]
        if not successful:
            return 0.0
        return sum(r.score.value for r in successful) / len(successful)

    @property
    def mean_latency_ms(self) -> float:
        """Mean latency per sample."""
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    def failed_samples(self) -> tuple[EvalResult, ...]:
        """Samples that did not pass."""
        return tuple(r for r in self.results if r.success and not r.score.passed)
```

### EvalLoop

The core class for running evaluations. Takes a `MainLoop` instance for
executing samples and an optional `Mailbox` for durable result delivery:

```python
class EvalLoop(Generic[InputT, OutputT, ExpectedT]):
    """Orchestrates evaluation over a dataset.

    Uses the provided MainLoop to execute each sample, then scores
    the output with the evaluator. Results are sent to a Mailbox for
    durable, at-least-once delivery.
    """

    def __init__(
        self,
        *,
        loop: MainLoop[InputT, OutputT],
        dataset: Dataset[InputT, ExpectedT],
        evaluator: Evaluator[OutputT, ExpectedT],
        results: Mailbox[EvalResult] | None = None,
    ) -> None:
        self._loop = loop
        self._dataset = dataset
        self._evaluator = evaluator
        self._results = results

    def execute(self) -> EvalReport:
        """Run evaluation on all samples.

        For each sample:
        1. Execute through MainLoop
        2. Score with evaluator
        3. Record timing
        4. Send result to mailbox (if provided)

        Results are sent to the mailbox before being added to the report,
        ensuring durability even if the process crashes mid-evaluation.

        Returns:
            EvalReport with all results and aggregate metrics
        """
        results: list[EvalResult] = []

        for sample in self._dataset:
            start = time.monotonic()
            try:
                response, _ = self._loop.execute(sample.input)
                latency_ms = int((time.monotonic() - start) * 1000)
                score = self._evaluator(response.output, sample.expected)
                result = EvalResult(
                    sample_id=sample.id,
                    score=score,
                    latency_ms=latency_ms,
                )
            except Exception as e:
                latency_ms = int((time.monotonic() - start) * 1000)
                result = EvalResult(
                    sample_id=sample.id,
                    score=Score(value=0.0, passed=False, reason=str(e)),
                    latency_ms=latency_ms,
                    error=str(e),
                )

            # Send to mailbox first for durability
            if self._results is not None:
                self._results.send(result)

            results.append(result)

        return EvalReport(results=tuple(results))
```

**Why Mailbox over Dispatcher?**

- **Durability**: Results survive process crashes with persistent backends
- **At-least-once delivery**: Results won't be lost in transit
- **Distributed consumption**: Multiple consumers can process results
- **Cross-process**: Works in distributed deployments (Redis, SQS)

## Usage Examples

### Basic Evaluation

```python
from weakincentives import MainLoop, PromptTemplate, Prompt, Session
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import InProcessDispatcher, InMemoryMailbox
from weakincentives.evals import Dataset, EvalLoop, EvalResult, exact_match

# Define the MainLoop for QA
class QALoop(MainLoop[str, str]):
    def __init__(self, adapter, bus):
        super().__init__(adapter=adapter, bus=bus)
        self._template = PromptTemplate[str](
            ns="qa",
            key="answer",
            name="qa",
            sections=[
                MarkdownSection(
                    title="Question",
                    template="Answer concisely: $question",
                    key="q",
                ),
            ],
        )

    def initialize(self, question: str) -> tuple[Prompt[str], Session]:
        prompt = Prompt(self._template).bind(question=question)
        session = Session(bus=self._bus)
        return prompt, session


# Load dataset
dataset = Dataset.load(Path("tests/fixtures/qa.jsonl"), str, str)

# Create loop and run evaluation
adapter = OpenAIAdapter(model="gpt-4o")
loop = QALoop(adapter=adapter, bus=InProcessDispatcher())

eval_loop = EvalLoop(loop=loop, dataset=dataset, evaluator=exact_match)
report = eval_loop.execute()

# Inspect results
print(f"Pass rate: {report.pass_rate:.1%}")
print(f"Mean score: {report.mean_score:.2f}")
print(f"Mean latency: {report.mean_latency_ms:.0f}ms")

# Review failures
for result in report.failed_samples():
    print(f"Failed: {result.sample_id} - {result.score.reason}")
```

### Multi-Criteria Evaluation

```python
from weakincentives.evals import EvalLoop, all_of, llm_judge, contains

# Create judge adapter and bus
judge_adapter = OpenAIAdapter[JudgeOutput](model="gpt-4o-mini")
judge_bus = InProcessDispatcher()

# Compose multiple criteria
evaluator = all_of(
    contains,  # Must contain expected substring
    llm_judge(judge_adapter, "Factually accurate", bus=judge_bus),
    llm_judge(judge_adapter, "Well-structured response", bus=judge_bus),
)

# Run with composite evaluator
eval_loop = EvalLoop(loop=loop, dataset=dataset, evaluator=evaluator)
report = eval_loop.execute()
```

### Programmatic Dataset

```python
from weakincentives.evals import Dataset, Sample

# Build dataset in code
samples = tuple(
    Sample(
        id=str(i),
        input=f"What is {a} + {b}?",
        expected=str(a + b),
    )
    for i, (a, b) in enumerate([(1, 1), (2, 3), (10, 20)])
)
dataset = Dataset(samples=samples)

# Run evaluation
eval_loop = EvalLoop(loop=loop, dataset=dataset, evaluator=contains)
report = eval_loop.execute()
```

## Observability

### Durable Result Collection

Pass a `Mailbox` to `EvalLoop` for durable result delivery:

```python
from weakincentives.runtime import InMemoryMailbox
from weakincentives.evals import EvalResult

# Create mailbox for results
results_mailbox: Mailbox[EvalResult] = InMemoryMailbox(name="eval-results")

# Run evaluation with durable results
eval_loop = EvalLoop(
    loop=loop,
    dataset=dataset,
    evaluator=exact_match,
    results=results_mailbox,
)
report = eval_loop.execute()

# Results are available in the mailbox for downstream processing
for msg in results_mailbox.receive(max_messages=10):
    result = msg.body
    status = "PASS" if result.score.passed else "FAIL"
    print(f"[{status}] {result.sample_id}: {result.score.value:.2f}")
    msg.acknowledge()
```

### Distributed Processing

For production deployments, use `RedisMailbox` or `SQSMailbox`:

```python
from weakincentives.runtime import RedisMailbox
from redis import Redis

# Redis-backed mailbox for cross-process durability
results_mailbox = RedisMailbox[EvalResult](
    name="eval-results",
    client=Redis(host="localhost", port=6379),
)

eval_loop = EvalLoop(
    loop=loop,
    dataset=dataset,
    evaluator=exact_match,
    results=results_mailbox,
)
report = eval_loop.execute()
```

With LangSmith enabled, MainLoop executions are automatically traced.

## Testing Evaluators

Evaluators are pure functions—test them directly:

```python
def test_exact_match_pass():
    score = exact_match("hello", "hello")
    assert score.passed is True
    assert score.value == 1.0


def test_exact_match_fail():
    score = exact_match("hello", "world")
    assert score.passed is False
    assert score.value == 0.0


def test_contains_pass():
    score = contains("The answer is 42.", "42")
    assert score.passed is True


def test_all_of_requires_all():
    evaluator = all_of(exact_match, contains)
    score = evaluator("hello", "hello")
    assert score.passed is True

    score = evaluator("hello world", "hello")
    assert score.passed is False  # exact_match fails


def test_any_of_requires_one():
    evaluator = any_of(exact_match, contains)
    score = evaluator("hello world", "hello")
    assert score.passed is True  # contains passes
```

## Limitations

- **Sequential execution** - MainLoop is synchronous; samples run one at a time
- **No caching** - Repeated samples re-execute; add caching at adapter level
- **No checkpoints** - Cannot resume interrupted runs
- **Single loop** - Each `EvalLoop.execute()` uses one MainLoop instance
