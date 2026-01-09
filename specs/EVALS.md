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

An evaluator is any callable matching one of these signatures:

```python
# Standard evaluator - scores output against expected
Evaluator = Callable[[OutputT, ExpectedT], Score]

# Session-aware evaluator - also receives session for behavioral assertions
SessionEvaluator = Callable[[OutputT, ExpectedT, SessionView], Score]
```

Evaluators are pure functions—no side effects, no state. This makes them easy
to test, compose, and reason about.

Session-aware evaluators receive a `SessionView`—a read-only view of session
state that provides access to tool invocations, token usage, and custom slices
without allowing mutations.

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

## Session-Aware Evaluators

Session-aware evaluators enable behavioral assertions—checking not just _what_
the agent produced, but _how_ it got there. This includes verifying tool usage
patterns, token budgets, and custom state invariants.

### SessionView Protocol

`SessionView` provides read-only access to session state:

```python
class SessionView(Protocol):
    """Read-only view of session state for evaluators."""

    @property
    def session_id(self) -> UUID:
        """Unique session identifier."""
        ...

    def __getitem__[T](self, slice_type: type[T]) -> SliceView[T]:
        """Access a slice by type. Returns read-only view."""
        ...
```

The `SliceView` provides the same query API as `SliceAccessor` but without
mutation methods:

```python
class SliceView(Protocol[T]):
    """Read-only slice access."""

    def all(self) -> tuple[T, ...]:
        """Return all items in the slice."""
        ...

    def latest(self) -> T | None:
        """Return the most recent item, or None if empty."""
        ...

    def where(self, predicate: Callable[[T], bool]) -> tuple[T, ...]:
        """Return items matching the predicate."""
        ...
```

### Built-in Session Evaluators

#### tool_called

Assert that a specific tool was invoked:

```python
def tool_called(name: str) -> SessionEvaluator[Any, Any]:
    """Assert that a tool was called at least once.

    Args:
        name: The tool name to check for.

    Returns:
        SessionEvaluator that passes if the tool was called.
    """
    def evaluate(output: Any, expected: Any, session: SessionView) -> Score:
        calls = session[ToolInvoked].all()
        matching = [c for c in calls if c.name == name]
        passed = len(matching) > 0
        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"tool '{name}' called {len(matching)} time(s)",
        )
    return evaluate
```

#### tool_not_called

Assert that a tool was NOT invoked:

```python
def tool_not_called(name: str) -> SessionEvaluator[Any, Any]:
    """Assert that a tool was never called.

    Args:
        name: The tool name that should not appear.

    Returns:
        SessionEvaluator that passes if the tool was not called.
    """
    def evaluate(output: Any, expected: Any, session: SessionView) -> Score:
        calls = session[ToolInvoked].all()
        matching = [c for c in calls if c.name == name]
        passed = len(matching) == 0
        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"tool '{name}' called {len(matching)} time(s)" if not passed else "",
        )
    return evaluate
```

#### tool_call_count

Assert tool was called within count bounds:

```python
def tool_call_count(
    name: str,
    *,
    min_count: int = 0,
    max_count: int | None = None,
) -> SessionEvaluator[Any, Any]:
    """Assert tool call count is within bounds.

    Args:
        name: The tool name to count.
        min_count: Minimum number of calls required (inclusive).
        max_count: Maximum number of calls allowed (inclusive). None = no limit.

    Returns:
        SessionEvaluator that passes if count is within bounds.
    """
    def evaluate(output: Any, expected: Any, session: SessionView) -> Score:
        calls = session[ToolInvoked].all()
        count = sum(1 for c in calls if c.name == name)
        passed = count >= min_count and (max_count is None or count <= max_count)

        if max_count is None:
            bounds = f">= {min_count}"
        else:
            bounds = f"{min_count}-{max_count}"

        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"tool '{name}' called {count} times (expected {bounds})",
        )
    return evaluate
```

#### all_tools_succeeded

Assert all tool calls returned success:

```python
def all_tools_succeeded() -> SessionEvaluator[Any, Any]:
    """Assert all tool invocations succeeded.

    Checks the 'success' field in each ToolInvoked.result dict.
    Tools without a 'success' field are assumed to have succeeded.

    Returns:
        SessionEvaluator that passes if no tool failures occurred.
    """
    def evaluate(output: Any, expected: Any, session: SessionView) -> Score:
        calls = session[ToolInvoked].all()
        if not calls:
            return Score(value=1.0, passed=True)

        failures = []
        for call in calls:
            result = call.result
            if isinstance(result, dict) and result.get("success") is False:
                failures.append(call.name)

        passed = len(failures) == 0
        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"failed tools: {failures}" if failures else "",
        )
    return evaluate
```

#### token_usage_under

Assert total token usage stayed under budget:

```python
def token_usage_under(max_tokens: int) -> SessionEvaluator[Any, Any]:
    """Assert total token usage is under budget.

    Sums input_tokens + output_tokens across all PromptExecuted events.

    Args:
        max_tokens: Maximum total tokens allowed.

    Returns:
        SessionEvaluator that passes if usage is under budget.
    """
    def evaluate(output: Any, expected: Any, session: SessionView) -> Score:
        executions = session[PromptExecuted].all()
        total = 0
        for ex in executions:
            if ex.usage:
                total += (ex.usage.input_tokens or 0) + (ex.usage.output_tokens or 0)

        passed = total <= max_tokens
        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"used {total} tokens (limit: {max_tokens})",
        )
    return evaluate
```

#### slice_contains

Assert a custom slice contains expected values:

```python
def slice_contains[T](
    slice_type: type[T],
    predicate: Callable[[T], bool],
    *,
    min_count: int = 1,
) -> SessionEvaluator[Any, Any]:
    """Assert slice contains items matching predicate.

    Args:
        slice_type: The slice type to query.
        predicate: Function to test each item.
        min_count: Minimum matching items required.

    Returns:
        SessionEvaluator that passes if enough items match.
    """
    def evaluate(output: Any, expected: Any, session: SessionView) -> Score:
        items = session[slice_type].where(predicate)
        passed = len(items) >= min_count
        return Score(
            value=1.0 if passed else 0.0,
            passed=passed,
            reason=f"found {len(items)} matching items (need >= {min_count})",
        )
    return evaluate
```

### Adapting Standard Evaluators

Standard evaluators work with session-aware combinators via the `adapt`
function:

```python
def adapt[O, E](evaluator: Evaluator[O, E]) -> SessionEvaluator[O, E]:
    """Adapt a standard evaluator to session-aware signature.

    The session parameter is ignored, allowing standard evaluators
    to compose with session-aware evaluators.
    """
    def evaluate(output: O, expected: E, session: SessionView) -> Score:
        return evaluator(output, expected)
    return evaluate
```

### Combinators

The `all_of` and `any_of` combinators work with both evaluator types:

```python
def all_of[O, E](
    *evaluators: Evaluator[O, E] | SessionEvaluator[O, E],
) -> SessionEvaluator[O, E]:
    """All evaluators must pass. Score is the mean.

    Automatically adapts standard evaluators to session-aware signature.
    """
    adapted = [
        e if _is_session_aware(e) else adapt(e)
        for e in evaluators
    ]

    def evaluate(output: O, expected: E, session: SessionView) -> Score:
        scores = [e(output, expected, session) for e in adapted]
        passed = all(s.passed for s in scores)
        value = sum(s.value for s in scores) / len(scores)
        reasons = [s.reason for s in scores if s.reason]
        return Score(value=value, passed=passed, reason="; ".join(reasons))
    return evaluate


def any_of[O, E](
    *evaluators: Evaluator[O, E] | SessionEvaluator[O, E],
) -> SessionEvaluator[O, E]:
    """At least one evaluator must pass. Score is the max.

    Automatically adapts standard evaluators to session-aware signature.
    """
    adapted = [
        e if _is_session_aware(e) else adapt(e)
        for e in evaluators
    ]

    def evaluate(output: O, expected: E, session: SessionView) -> Score:
        scores = [e(output, expected, session) for e in adapted]
        passed = any(s.passed for s in scores)
        value = max(s.value for s in scores)
        reasons = [s.reason for s in scores if s.reason]
        return Score(value=value, passed=passed, reason="; ".join(reasons))
    return evaluate


def _is_session_aware(fn: Callable[..., Score]) -> bool:
    """Check if evaluator accepts session parameter."""
    import inspect
    sig = inspect.signature(fn)
    return len(sig.parameters) >= 3
```

### Usage Example

```python
from weakincentives.evals import (
    all_of, exact_match,
    tool_called, tool_not_called, all_tools_succeeded, token_usage_under,
)

# Compose output and behavioral assertions
evaluator = all_of(
    exact_match,                        # Output must match expected
    tool_called("search"),              # Must use search tool
    tool_not_called("dangerous_tool"),  # Must not use forbidden tool
    all_tools_succeeded(),              # No tool failures
    token_usage_under(5000),            # Stay under budget
)

eval_loop = EvalLoop(
    loop=main_loop,
    evaluator=evaluator,
    requests=requests,
)
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


@dataclass(slots=True, frozen=True)
class JudgeParams:
    """Parameters for the judge prompt."""
    criterion: str
    output: str
    expected: str


JUDGE_TEMPLATE = PromptTemplate[JudgeOutput](
    ns="wink.evals",
    key="llm-judge",
    name="llm_judge",
    sections=[
        MarkdownSection[JudgeParams](
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
) -> Evaluator[str, str]:
    """Create evaluator that uses LLM to judge output.

    Args:
        adapter: Provider adapter configured for JudgeOutput
        criterion: What to evaluate (e.g., "factual accuracy", "clarity")

    Returns:
        Evaluator function that scores string outputs
    """
    def evaluate(output: str, expected: str) -> Score:
        prompt = Prompt(JUDGE_TEMPLATE).bind(JudgeParams(
            criterion=criterion,
            output=output,
            expected=expected,
        ))
        response = adapter.evaluate(prompt)
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
judge_adapter: OpenAIAdapter[JudgeOutput] = OpenAIAdapter(model="gpt-4o-mini")

evaluator = all_of(
    contains,  # Must contain expected answer
    llm_judge(judge_adapter, "Response is helpful and well-formatted"),
    llm_judge(judge_adapter, "No hallucinated information"),
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

### EvalRequest

Request to evaluate a single sample:

```python
@dataclass(slots=True, frozen=True)
class EvalRequest(Generic[InputT, ExpectedT]):
    """Request to evaluate a sample."""
    sample: Sample[InputT, ExpectedT]
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### EvalLoop

Mailbox-driven evaluation loop. Follows the same two-mailbox pattern as
`MainLoop` for distributed deployments:

```python
class EvalLoop(Generic[InputT, OutputT, ExpectedT]):
    """Mailbox-driven evaluation loop.

    Receives EvalRequest messages, executes through MainLoop, scores
    with evaluator, and sends EvalResult to results mailbox. Designed
    to run alongside MainLoop workers in distributed deployments.

    Supports both standard and session-aware evaluators. Session-aware
    evaluators receive a SessionView for behavioral assertions.
    """

    def __init__(
        self,
        *,
        loop: MainLoop[InputT, OutputT],
        evaluator: Evaluator[OutputT, ExpectedT] | SessionEvaluator[OutputT, ExpectedT],
        requests: Mailbox[EvalRequest[InputT, ExpectedT]],
        results: Mailbox[EvalResult],
    ) -> None:
        self._loop = loop
        self._evaluator = evaluator
        self._requests = requests
        self._results = results

    def run(self, *, max_iterations: int | None = None) -> None:
        """Process evaluation requests from mailbox.

        Polls the requests mailbox, evaluates each sample through
        MainLoop, and sends results to the results mailbox.

        Args:
            max_iterations: Stop after N iterations (None = run forever)
        """
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            for msg in self._requests.receive(
                visibility_timeout=300,  # 5 min - must exceed max execution time
                wait_time_seconds=20,    # Long poll for efficiency
            ):
                try:
                    result = self._evaluate_sample(msg.body)
                    self._results.send(result)
                    msg.acknowledge()
                except Exception as e:
                    self._handle_failure(msg, e)
            iterations += 1

    def _evaluate_sample(self, request: EvalRequest[InputT, ExpectedT]) -> EvalResult:
        """Execute and score a single sample."""
        sample = request.sample
        start = time.monotonic()

        response, session = self._loop.execute(sample.input)
        latency_ms = int((time.monotonic() - start) * 1000)

        # Invoke evaluator with session if session-aware
        if _is_session_aware(self._evaluator):
            score = self._evaluator(response.output, sample.expected, session)
        else:
            score = self._evaluator(response.output, sample.expected)

        return EvalResult(
            sample_id=sample.id,
            score=score,
            latency_ms=latency_ms,
        )

    def _handle_failure(
        self,
        msg: Message[EvalRequest[InputT, ExpectedT]],
        error: Exception,
    ) -> None:
        """Handle evaluation failure with backoff retry."""
        latency_ms = 0  # Unknown on failure
        try:
            self._results.send(EvalResult(
                sample_id=msg.body.sample.id,
                score=Score(value=0.0, passed=False, reason=str(error)),
                latency_ms=latency_ms,
                error=str(error),
            ))
            msg.acknowledge()  # Error result sent - don't retry
        except Exception:
            # Result send failed - nack for retry with backoff
            msg.nack(visibility_timeout=min(60 * msg.delivery_count, 900))
```

### Submitting Samples

Submit samples to the requests mailbox for evaluation:

```python
def submit_dataset(
    dataset: Dataset[InputT, ExpectedT],
    requests: Mailbox[EvalRequest[InputT, ExpectedT]],
) -> None:
    """Submit all samples in a dataset for evaluation."""
    for sample in dataset:
        requests.send(EvalRequest(sample=sample))
```

### Collecting Results

Collect results from the results mailbox:

```python
def collect_results(
    results: Mailbox[EvalResult],
    expected_count: int,
    *,
    timeout_seconds: float = 300,
) -> EvalReport:
    """Collect evaluation results into a report.

    Args:
        results: Mailbox to receive results from
        expected_count: Number of results to collect
        timeout_seconds: Maximum time to wait for all results

    Returns:
        EvalReport with all collected results
    """
    collected: list[EvalResult] = []
    deadline = time.time() + timeout_seconds

    while len(collected) < expected_count and time.time() < deadline:
        remaining = timeout_seconds - (deadline - time.time())
        wait_time = min(20, max(1, int(remaining)))

        for msg in results.receive(wait_time_seconds=wait_time):
            collected.append(msg.body)
            msg.acknowledge()

    return EvalReport(results=tuple(collected))
```

## Usage Examples

### Basic Evaluation

```python
from dataclasses import dataclass
from weakincentives import MainLoop, PromptTemplate, Prompt, Session
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import ControlDispatcher, InMemoryMailbox, Mailbox
from weakincentives.evals import (
    Dataset, EvalLoop, EvalRequest, EvalResult,
    exact_match, submit_dataset, collect_results,
)


@dataclass(slots=True, frozen=True)
class QAParams:
    """Parameters for the QA prompt."""
    question: str


# Define the MainLoop for QA
class QALoop(MainLoop[str, str]):
    def __init__(self, *, adapter: OpenAIAdapter[str], bus: ControlDispatcher) -> None:
        super().__init__(adapter=adapter, bus=bus)
        self._template = PromptTemplate[str](
            ns="qa",
            key="answer",
            name="qa",
            sections=[
                MarkdownSection[QAParams](
                    title="Question",
                    template="Answer concisely: $question",
                    key="q",
                ),
            ],
        )

    def initialize(self, question: str) -> tuple[Prompt[str], Session]:
        prompt = Prompt(self._template).bind(QAParams(question=question))
        session = Session(dispatcher=self._bus)
        return prompt, session


# Load dataset
dataset = Dataset.load(Path("tests/fixtures/qa.jsonl"), str, str)

# Create mailboxes
requests: Mailbox[EvalRequest[str, str]] = InMemoryMailbox(name="eval-requests")
results: Mailbox[EvalResult] = InMemoryMailbox(name="eval-results")

# Create MainLoop and EvalLoop
adapter: OpenAIAdapter[str] = OpenAIAdapter(model="gpt-4o")
bus = ControlDispatcher()
main_loop = QALoop(adapter=adapter, bus=bus)
eval_loop = EvalLoop(
    loop=main_loop,
    evaluator=exact_match,
    requests=requests,
    results=results,
)

# Submit samples and run worker
submit_dataset(dataset, requests)
eval_loop.run(max_iterations=1)

# Collect results
report = collect_results(results, expected_count=len(dataset))

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

# Create judge adapter
judge_adapter: OpenAIAdapter[JudgeOutput] = OpenAIAdapter(model="gpt-4o-mini")

# Compose multiple criteria
evaluator = all_of(
    contains,  # Must contain expected substring
    llm_judge(judge_adapter, "Factually accurate"),
    llm_judge(judge_adapter, "Well-structured response"),
)

# Create EvalLoop with composite evaluator
eval_loop = EvalLoop(
    loop=main_loop,
    evaluator=evaluator,
    requests=requests,
    results=results,
)

submit_dataset(dataset, requests)
eval_loop.run(max_iterations=1)
report = collect_results(results, expected_count=len(dataset))
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

# Submit and evaluate
submit_dataset(dataset, requests)
eval_loop.run(max_iterations=1)
report = collect_results(results, expected_count=len(dataset))
```

### Session-Aware Evaluation

Evaluate both outputs and agent behavior:

```python
from weakincentives.evals import (
    EvalLoop, all_of, exact_match,
    tool_called, tool_not_called, all_tools_succeeded,
    token_usage_under, slice_contains,
)

# Compose output and behavioral assertions
evaluator = all_of(
    exact_match,                             # Correct answer
    tool_called("calculator"),               # Must use calculator
    tool_not_called("web_search"),           # No external lookups for math
    all_tools_succeeded(),                   # No tool errors
    token_usage_under(2000),                 # Efficient execution
)

eval_loop = EvalLoop(
    loop=math_loop,
    evaluator=evaluator,
    requests=requests,
    results=results,
)

submit_dataset(math_dataset, requests)
eval_loop.run(max_iterations=1)
report = collect_results(results, expected_count=len(math_dataset))

# Report includes behavioral failure reasons
for result in report.failed_samples():
    print(f"Failed: {result.sample_id}")
    print(f"  Reason: {result.score.reason}")
```

### Custom Slice Assertions

Assert against application-specific state:

```python
from dataclasses import dataclass
from weakincentives.evals import slice_contains

@dataclass(slots=True, frozen=True)
class PlanStep:
    """Step in the agent's plan."""
    name: str
    status: str  # "pending", "completed", "failed"

# Assert plan was fully executed
evaluator = all_of(
    exact_match,
    slice_contains(
        PlanStep,
        lambda step: step.status == "completed",
        min_count=1,  # At least one step completed
    ),
)
```

## Distributed Deployment

### Worker Process

EvalLoop workers run alongside MainLoop workers, polling the requests mailbox:

```python
from redis import Redis
from weakincentives.runtime import RedisMailbox, ControlDispatcher
from weakincentives.evals import EvalLoop, EvalRequest, EvalResult, exact_match

# Redis-backed mailboxes for cross-process durability
redis_client = Redis(host="localhost", port=6379)
requests: Mailbox[EvalRequest[str, str]] = RedisMailbox(
    name="eval-requests",
    client=redis_client,
)
results: Mailbox[EvalResult] = RedisMailbox(
    name="eval-results",
    client=redis_client,
)

# Create MainLoop (same as basic example)
adapter: OpenAIAdapter[str] = OpenAIAdapter(model="gpt-4o")
bus = ControlDispatcher()
main_loop = QALoop(adapter=adapter, bus=bus)

# Create and run worker (runs forever)
eval_loop = EvalLoop(
    loop=main_loop,
    evaluator=exact_match,
    requests=requests,
    results=results,
)
eval_loop.run()  # Blocks, processing requests
```

### Client Process

Submit samples and collect results from a separate process:

```python
from redis import Redis
from weakincentives.runtime import RedisMailbox, Mailbox
from weakincentives.evals import (
    Dataset, EvalRequest, EvalResult, submit_dataset, collect_results,
)

# Connect to same Redis mailboxes
redis_client = Redis(host="localhost", port=6379)
requests: Mailbox[EvalRequest[str, str]] = RedisMailbox(
    name="eval-requests",
    client=redis_client,
)
results: Mailbox[EvalResult] = RedisMailbox(
    name="eval-results",
    client=redis_client,
)

# Submit dataset
dataset = Dataset.load(Path("qa.jsonl"), str, str)
submit_dataset(dataset, requests)

# Wait for results (workers process in background)
report = collect_results(
    results,
    expected_count=len(dataset),
    timeout_seconds=600,  # 10 minute timeout
)

print(f"Completed {report.total} evaluations")
print(f"Pass rate: {report.pass_rate:.1%}")
```

### Multiple Workers

Scale horizontally by running multiple EvalLoop workers:

```
                    ┌─────────────────┐
                    │ Requests Queue  │
                    │ (Redis/SQS)     │
                    └────────┬────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  EvalLoop   │   │  EvalLoop   │   │  EvalLoop   │
    │  Worker #1  │   │  Worker #2  │   │  Worker #3  │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Results Queue   │
                    │ (Redis/SQS)     │
                    └─────────────────┘
```

Visibility timeout ensures each sample is processed by exactly one worker.
Failed evaluations retry automatically with exponential backoff.

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

### Testing Session Evaluators

Session evaluators require a mock session with the appropriate slices:

```python
from unittest.mock import Mock
from weakincentives.evals import tool_called, all_tools_succeeded, token_usage_under
from weakincentives.runtime.events import ToolInvoked, PromptExecuted, TokenUsage


def make_mock_session(tool_invocations: list[ToolInvoked]) -> Mock:
    """Create a mock session with ToolInvoked slice."""
    session = Mock()
    slice_view = Mock()
    slice_view.all.return_value = tuple(tool_invocations)
    session.__getitem__ = Mock(return_value=slice_view)
    return session


def test_tool_called_pass():
    invocation = ToolInvoked(
        prompt_name="test",
        adapter="openai",
        name="search",
        params={"query": "test"},
        result={"success": True},
        session_id=None,
        created_at=datetime.now(UTC),
    )
    session = make_mock_session([invocation])

    evaluator = tool_called("search")
    score = evaluator(None, None, session)

    assert score.passed is True
    assert "called 1 time" in score.reason


def test_tool_called_fail():
    session = make_mock_session([])  # No invocations

    evaluator = tool_called("search")
    score = evaluator(None, None, session)

    assert score.passed is False
    assert "called 0 time" in score.reason


def test_all_tools_succeeded_with_failure():
    invocations = [
        ToolInvoked(
            prompt_name="test",
            adapter="openai",
            name="search",
            params={},
            result={"success": False, "error": "not found"},
            session_id=None,
            created_at=datetime.now(UTC),
        ),
    ]
    session = make_mock_session(invocations)

    evaluator = all_tools_succeeded()
    score = evaluator(None, None, session)

    assert score.passed is False
    assert "search" in score.reason
```

## Limitations

- **Sequential execution** - MainLoop is synchronous; samples run one at a time
- **No caching** - Repeated samples re-execute; add caching at adapter level
- **No checkpoints** - Cannot resume interrupted runs
- **Single loop** - Each `EvalLoop.execute()` uses one MainLoop instance
