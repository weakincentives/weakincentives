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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

# Rating scale for LLM-as-judge evaluations
Rating = Literal["excellent", "good", "fair", "poor", "wrong"]

RATING_VALUES: dict[Rating, float] = {
    "excellent": 1.0,  # Fully meets criterion
    "good": 0.75,  # Meets criterion with minor issues
    "fair": 0.5,  # Partially meets criterion
    "poor": 0.25,  # Mostly fails criterion
    "wrong": 0.0,  # Completely fails criterion
}

PASSING_RATINGS: frozenset[Rating] = cast(
    frozenset[Rating], frozenset({"excellent", "good"})
)


@dataclass(slots=True, frozen=True)
class Sample[InputT, ExpectedT]:
    """Single evaluation input."""

    id: str
    input: InputT
    expected: ExpectedT


@dataclass(slots=True, frozen=True)
class Score:
    """Result of scoring one output."""

    value: float  # 0.0 to 1.0, normalized
    passed: bool  # Binary pass/fail
    reason: str = ""  # Explanation (useful for LLM judges)


@dataclass(slots=True, frozen=True)
class JudgeOutput:
    """Structured output from judge prompt."""

    rating: Rating  # Categorical label
    reason: str  # Brief explanation


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


@dataclass(slots=True, frozen=True)
class SampleEvaluated:
    """Emitted after each sample is evaluated."""

    sample_id: str
    result: EvalResult


# Type alias for evaluator functions
Evaluator = Callable[[object, object], Score]


__all__ = [
    "PASSING_RATINGS",
    "RATING_VALUES",
    "EvalReport",
    "EvalResult",
    "Evaluator",
    "JudgeOutput",
    "Rating",
    "Sample",
    "SampleEvaluated",
    "Score",
]
