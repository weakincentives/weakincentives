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

The library provides basic evaluators and combinators for common scoring tasks.

**Implementation:** `src/weakincentives/evals/_evaluators.py`

| Evaluator | Purpose | Signature |
|-----------|---------|-----------|
| `exact_match` | Strict equality check | `(output: T, expected: T) -> Score` |
| `contains` | Substring presence check | `(output: str, expected: str) -> Score` |
| `all_of` | All evaluators must pass, score is mean | `(*evaluators) -> Evaluator` |
| `any_of` | At least one must pass, score is max | `(*evaluators) -> Evaluator` |

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

**Implementation:** `src/weakincentives/evals/_session_evaluators.py`

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

| Evaluator | Purpose | What It Checks |
|-----------|---------|----------------|
| `tool_called(name)` | Assert tool was invoked | Tool appears in ToolInvoked slice |
| `tool_not_called(name)` | Assert tool was NOT invoked | Tool does not appear in ToolInvoked slice |
| `tool_call_count(name, min_count, max_count)` | Assert tool call count within bounds | Number of ToolInvoked events for tool |
| `all_tools_succeeded()` | Assert all tool calls returned success | No ToolInvoked.result has success=False |
| `token_usage_under(max_tokens)` | Assert total token usage under budget | Sum of input_tokens + output_tokens across PromptExecuted events |
| `slice_contains(slice_type, predicate, min_count)` | Assert custom slice contains items | Slice contains at least min_count matching items |

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

The `all_of` and `any_of` combinators work with both evaluator types,
automatically adapting standard evaluators to session-aware signatures.

### Composition Example

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
```

## LLM-as-Judge

For subjective criteria, use an LLM to score outputs. Rather than asking for
numerical scores (which LLMs calibrate poorly), the judge selects from a fixed
set of rating labels that map to values.

**Implementation:** `src/weakincentives/evals/_judge.py`

### Design

- **Rating Scale:** Categorical labels (`excellent`, `good`, `fair`, `poor`, `wrong`) map to values (1.0, 0.75, 0.5, 0.25, 0.0)
- **Pass Threshold:** `excellent` and `good` ratings count as passing
- **Structured Output:** `JudgeOutput` dataclass with `rating: Rating` and `reason: str`
- **Judge Prompt:** Template presents criterion, output, reference, and rating scale
- **Factory Function:** `llm_judge(adapter, criterion)` returns an `Evaluator[str, str]`

### Usage Pattern

```python
# Use a smaller/cheaper model for judging
judge_adapter: OpenAIAdapter[JudgeOutput] = OpenAIAdapter(model="gpt-4o-mini")

evaluator = all_of(
    contains,  # Must contain expected answer
    llm_judge(judge_adapter, "Response is helpful and well-formatted"),
    llm_judge(judge_adapter, "No hallucinated information"),
)
```

The judge adapter must be configured with the `JudgeOutput` type for structured
output parsing.

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

**Implementation:** `src/weakincentives/evals/_loop.py`

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
        ...

    def run(self, *, max_iterations: int | None = None) -> None:
        """Process evaluation requests from mailbox.

        Polls the requests mailbox, evaluates each sample through
        MainLoop, and sends results to the results mailbox.

        Args:
            max_iterations: Stop after N iterations (None = run forever)
        """
        ...
```

### Helper Functions

**Implementation:** `src/weakincentives/evals/_helpers.py`

```python
def submit_dataset(
    dataset: Dataset[InputT, ExpectedT],
    requests: Mailbox[EvalRequest[InputT, ExpectedT]],
) -> None:
    """Submit all samples in a dataset for evaluation."""
    for sample in dataset:
        requests.send(EvalRequest(sample=sample))


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
    ...
```

## Usage Examples

Complete working examples demonstrate the evaluation framework in practice.

**Reference Implementation:** `code_reviewer_example.py`

This example shows:

- Creating a custom MainLoop subclass with domain-specific prompts
- Loading datasets from JSONL files
- Composing multiple evaluators (exact match, session-aware assertions)
- Running EvalLoop with max_iterations for local development
- Submitting datasets and collecting results
- Inspecting EvalReport metrics (pass rate, mean score, latency)
- Reviewing failed samples with reasons

For multi-criteria evaluation with LLM-as-judge, see the `llm_judge` factory
in `src/weakincentives/evals/_judge.py`.

For session-aware behavioral assertions (tool usage, token budgets, custom
slices), see session evaluator examples in
`tests/evals/test_session_evaluators.py`.

## Distributed Deployment

EvalLoop supports distributed evaluation using Redis or SQS mailboxes.

### Architecture Pattern

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

### Deployment Modes

**Worker Process:**

- Runs `EvalLoop.run()` with no max_iterations (blocks forever)
- Polls requests mailbox with long polling (20s wait_time)
- Executes each sample through MainLoop
- Sends EvalResult to results mailbox
- Acknowledges successful completions, nacks failures with backoff

**Client Process:**

- Submits samples via `submit_dataset(dataset, requests)`
- Collects results via `collect_results(results, expected_count, timeout_seconds)`
- Aggregates into EvalReport for analysis

**Scaling:**

- Run multiple worker processes for horizontal scaling
- Visibility timeout ensures exactly-once processing per sample
- Failed evaluations retry automatically with exponential backoff
- Workers can run on different machines for distributed load

**Implementation Example:**

For Redis-backed distributed evaluation, configure RedisMailbox for both
requests and results queues. See `specs/MAILBOX.md` for mailbox configuration
and `code_reviewer_example.py` for local development patterns.

For production deployments with health checks and graceful shutdown, see
`specs/LIFECYCLE.md` for LoopGroup coordination and `specs/HEALTH.md` for
watchdog configuration.

## Testing Evaluators

Evaluators are pure functions—test them directly without session infrastructure:

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

Session evaluators require a mock session with the appropriate slices.
See `tests/evals/test_session_evaluators.py` for complete examples using
mock sessions with `ToolInvoked`, `PromptExecuted`, and custom slices.

## Limitations

- **Sequential execution** - MainLoop is synchronous; samples run one at a time
- **No caching** - Repeated samples re-execute; add caching at adapter level
- **No checkpoints** - Cannot resume interrupted runs
- **Single loop** - Each `EvalLoop.execute()` uses one MainLoop instance

## Related Specifications

- `specs/DLQ.md` - Dead letter queue configuration for failed samples
- `specs/MAIN_LOOP.md` - MainLoop orchestration that EvalLoop wraps
- `specs/MAILBOX.md` - Mailbox protocol for distributed evaluation
- `specs/LIFECYCLE.md` - LoopGroup for coordinating multiple evaluation workers
- `specs/HEALTH.md` - Health checks and watchdog for production deployments
