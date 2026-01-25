# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core types for the evaluation framework."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from ..runtime.session import SessionProtocol, SessionViewProtocol

from ..experiment import Experiment
from ..serde import parse


@dataclass(slots=True, frozen=True)
class Sample[InputT, ExpectedT]:
    """Single evaluation input.

    Pairs an input with its expected output for evaluation.

    Example:
        >>> sample = Sample(id="1", input="What is 2+2?", expected="4")
    """

    id: str
    """Unique identifier for this sample."""

    input: InputT
    """The input to evaluate."""

    expected: ExpectedT
    """The expected output to compare against."""


@dataclass(slots=True, frozen=True)
class Score:
    """Result of scoring one output.

    The ``value`` field enables ranking and aggregation (0.0 to 1.0 normalized).
    The ``passed`` field provides a binary threshold for pass rates.

    Example:
        >>> score = Score(value=0.8, passed=True, reason="Minor formatting issue")
    """

    value: float
    """Normalized score from 0.0 to 1.0."""

    passed: bool
    """Binary pass/fail indicator."""

    reason: str = ""
    """Explanation (useful for LLM judges)."""


# Type alias for evaluator functions
Evaluator = Callable[[object, object], Score]
"""An evaluator is any callable: (output, expected) -> Score.

Evaluators are pure functions - no side effects, no state.
"""

# Type alias for session-aware evaluator functions
SessionEvaluator = Callable[
    [object, object, "SessionProtocol | SessionViewProtocol"], Score
]
"""A session-aware evaluator: (output, expected, session) -> Score.

Session-aware evaluators receive a SessionProtocol or SessionViewProtocol
providing access to session state for behavioral assertions (tool calls,
token usage, custom slices).
"""


def _coerce[T](value: object, target: type[T]) -> T:
    """Coerce JSON value to target type.

    Primitives (str, int, float, bool) pass through directly.
    Mappings are parsed as dataclasses via serde.parse.

    Args:
        value: The JSON-deserialized value.
        target: The target type to coerce to.

    Returns:
        The coerced value.

    Raises:
        TypeError: If the value cannot be coerced to the target type.
    """
    if target in {str, int, float, bool}:
        if not isinstance(value, target):
            msg = f"expected {target.__name__}, got {type(value).__name__}"
            raise TypeError(msg)
        return value
    if isinstance(value, Mapping):
        data: Mapping[str, object] = value  # type: ignore[assignment]
        return parse(target, data)
    msg = f"cannot coerce {type(value).__name__} to {target.__name__}"
    raise TypeError(msg)


@dataclass(slots=True, frozen=True)
class Dataset[InputT, ExpectedT]:
    """Immutable collection of evaluation samples.

    Provides a clean API for loading and accessing evaluation data.
    Supports JSONL loading and programmatic construction.

    Example:
        >>> dataset = Dataset.load(Path("qa.jsonl"), str, str)
        >>> for sample in dataset:
        ...     print(sample.input)
    """

    samples: tuple[Sample[InputT, ExpectedT], ...]
    """The samples in this dataset."""

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
            path: Path to JSONL file.
            input_type: Type for deserializing input field.
            expected_type: Type for deserializing expected field.

        Returns:
            Dataset containing all samples from the file.

        Example JSONL format:
            {"id": "1", "input": "What is 2+2?", "expected": "4"}
            {"id": "2", "input": "Capital of France?", "expected": "Paris"}
        """
        samples: list[Sample[I, E]] = []
        with path.open() as f:
            for line in f:
                obj = json.loads(line)
                samples.append(
                    Sample(
                        id=obj["id"],
                        input=_coerce(obj["input"], input_type),
                        expected=_coerce(obj["expected"], expected_type),
                    )
                )
        return Dataset(samples=tuple(samples))


@dataclass(slots=True, frozen=True)
class EvalResult:
    """Result for one sample under an experiment.

    Captures the score, latency, experiment name, and any error that occurred
    during evaluation.

    Example:
        >>> result = EvalResult(
        ...     sample_id="1",
        ...     experiment_name="baseline",
        ...     score=Score(value=1.0, passed=True),
        ...     latency_ms=150,
        ... )
    """

    sample_id: str
    """Identifier of the evaluated sample."""

    experiment_name: str
    """Name of the experiment this result was evaluated under."""

    score: Score
    """The score from the evaluator."""

    latency_ms: int
    """Time taken to evaluate in milliseconds."""

    error: str | None = None
    """Error message if evaluation failed."""

    @property
    def success(self) -> bool:
        """True if no error occurred."""
        return self.error is None


@dataclass(slots=True, frozen=True)
class ExperimentComparison:
    """Statistical comparison between two experiments.

    Provides metrics for comparing a treatment experiment against a baseline.

    Example:
        >>> comparison = report.compare_experiments("baseline", "v2-prompts")
        >>> print(f"Delta: {comparison.pass_rate_delta:+.1%}")
    """

    baseline_name: str
    """Name of the baseline experiment."""

    treatment_name: str
    """Name of the treatment experiment."""

    baseline_results: tuple[EvalResult, ...]
    """Results from the baseline experiment."""

    treatment_results: tuple[EvalResult, ...]
    """Results from the treatment experiment."""

    @property
    def baseline_pass_rate(self) -> float:
        """Pass rate for the baseline experiment."""
        successful = [r for r in self.baseline_results if r.error is None]
        if not successful:
            return 0.0
        return sum(1 for r in successful if r.score.passed) / len(successful)

    @property
    def treatment_pass_rate(self) -> float:
        """Pass rate for the treatment experiment."""
        successful = [r for r in self.treatment_results if r.error is None]
        if not successful:
            return 0.0
        return sum(1 for r in successful if r.score.passed) / len(successful)

    @property
    def pass_rate_delta(self) -> float:
        """Treatment pass rate minus baseline pass rate."""
        return self.treatment_pass_rate - self.baseline_pass_rate

    @property
    def relative_improvement(self) -> float | None:
        """Percentage improvement over baseline. None if baseline is 0."""
        if self.baseline_pass_rate == 0:
            return None
        return self.pass_rate_delta / self.baseline_pass_rate


@dataclass(slots=True, frozen=True)
class EvalReport:
    """Aggregate evaluation results with computed metrics and experiment breakdown.

    Provides properties for pass rate, mean score, mean latency,
    and access to failed samples. Supports grouping results by experiment
    for A/B test analysis.

    Example:
        >>> report = EvalReport(results=tuple(results))
        >>> print(f"Pass rate: {report.pass_rate:.1%}")
        >>> print(f"Mean score: {report.mean_score:.2f}")
        >>> for name, rate in report.pass_rate_by_experiment().items():
        ...     print(f"{name}: {rate:.1%}")
    """

    results: tuple[EvalResult, ...]
    """All evaluation results."""

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

    def by_experiment(self) -> dict[str, tuple[EvalResult, ...]]:
        """Group results by experiment name.

        Returns:
            Dictionary mapping experiment name to tuple of results.

        Example:
            >>> for name, results in report.by_experiment().items():
            ...     print(f"{name}: {len(results)} samples")
        """
        groups: dict[str, list[EvalResult]] = {}
        for result in self.results:
            groups.setdefault(result.experiment_name, []).append(result)
        return {k: tuple(v) for k, v in groups.items()}

    def pass_rate_by_experiment(self) -> dict[str, float]:
        """Compute pass rate for each experiment.

        Returns:
            Dictionary mapping experiment name to pass rate.

        Example:
            >>> for name, rate in report.pass_rate_by_experiment().items():
            ...     print(f"{name}: {rate:.1%}")
        """
        rates: dict[str, float] = {}
        for name, results in self.by_experiment().items():
            successful = [r for r in results if r.error is None]
            if successful:
                rates[name] = sum(1 for r in successful if r.score.passed) / len(
                    successful
                )
            else:
                rates[name] = 0.0
        return rates

    def mean_score_by_experiment(self) -> dict[str, float]:
        """Compute mean score for each experiment.

        Returns:
            Dictionary mapping experiment name to mean score.

        Example:
            >>> for name, score in report.mean_score_by_experiment().items():
            ...     print(f"{name}: {score:.2f}")
        """
        scores: dict[str, float] = {}
        for name, results in self.by_experiment().items():
            successful = [r for r in results if r.error is None]
            if successful:
                scores[name] = sum(r.score.value for r in successful) / len(successful)
            else:
                scores[name] = 0.0
        return scores

    def compare_experiments(
        self,
        baseline: str,
        treatment: str,
    ) -> ExperimentComparison:
        """Compare two experiments statistically.

        Args:
            baseline: Name of the baseline experiment.
            treatment: Name of the treatment experiment.

        Returns:
            ExperimentComparison with metrics for both experiments.

        Example:
            >>> comparison = report.compare_experiments("baseline", "v2-prompts")
            >>> print(f"Baseline: {comparison.baseline_pass_rate:.1%}")
            >>> print(f"Treatment: {comparison.treatment_pass_rate:.1%}")
            >>> print(f"Delta: {comparison.pass_rate_delta:+.1%}")
        """
        by_exp = self.by_experiment()
        baseline_results = by_exp.get(baseline, ())
        treatment_results = by_exp.get(treatment, ())
        return ExperimentComparison(
            baseline_name=baseline,
            treatment_name=treatment,
            baseline_results=baseline_results,
            treatment_results=treatment_results,
        )


@dataclass(slots=True, frozen=True)
class EvalRequest[InputT, ExpectedT]:
    """Request to evaluate a sample under an experiment.

    Wraps a sample with an experiment configuration for evaluation.
    The experiment field is required to ensure every evaluation is
    associated with a specific experimental configuration.
    """

    sample: Sample[InputT, ExpectedT]
    """The sample to evaluate."""

    experiment: Experiment
    """The experiment to evaluate under. Required for eval requests."""

    request_id: UUID = field(default_factory=uuid4)
    """Unique request identifier."""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When the request was created."""


__all__ = [
    "Dataset",
    "EvalReport",
    "EvalRequest",
    "EvalResult",
    "Evaluator",
    "ExperimentComparison",
    "Sample",
    "Score",
    "SessionEvaluator",
]
