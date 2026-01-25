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

"""LLM-as-Judge evaluation support.

Provides categorical rating-based evaluation using an LLM to judge outputs.
Rather than asking for numerical scores (which LLMs calibrate poorly),
the judge selects from a fixed set of rating labels that map to values.

Key components:
- ``Rating`` - Literal type for categorical labels (excellent, good, fair, poor, wrong)
- ``RATING_VALUES`` - Mapping from labels to normalized scores (0.0 to 1.0)
- ``PASSING_RATINGS`` - Set of ratings that count as passing (excellent, good)
- ``JudgeOutput`` - Structured output dataclass from judge prompt
- ``JudgeParams`` - Parameters for the judge prompt template
- ``JUDGE_TEMPLATE`` - Pre-built prompt template for LLM judging
- ``llm_judge()`` - Factory function to create LLM-based evaluators

Example usage:
    >>> from weakincentives.adapters.openai import OpenAIAdapter
    >>> adapter = OpenAIAdapter[JudgeOutput](model="gpt-4o-mini")
    >>> evaluator = llm_judge(adapter, "factual accuracy")
    >>> score = evaluator("Paris is the capital.", "Paris")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..prompt import MarkdownSection, Prompt, PromptTemplate
from ..runtime import Session
from ._types import Score

if TYPE_CHECKING:
    from ..adapters.core import ProviderAdapter


Rating = Literal["excellent", "good", "fair", "poor", "wrong"]
"""Categorical rating labels for LLM-as-judge evaluation.

These labels provide a discrete scale that LLMs can reliably select from,
avoiding the calibration issues that arise with direct numerical scoring.
"""

RATING_VALUES: dict[Rating, float] = {
    "excellent": 1.0,  # Fully meets criterion
    "good": 0.75,  # Meets criterion with minor issues
    "fair": 0.5,  # Partially meets criterion
    "poor": 0.25,  # Mostly fails criterion
    "wrong": 0.0,  # Completely fails criterion
}
"""Mapping from rating labels to normalized score values (0.0 to 1.0).

Use this to convert categorical ratings to numeric scores for aggregation:
    >>> score_value = RATING_VALUES["good"]  # 0.75
"""

_PASSING: set[Rating] = {"excellent", "good"}
PASSING_RATINGS: frozenset[Rating] = frozenset(_PASSING)
"""Ratings that count as passing for pass/fail determination.

Contains "excellent" and "good". A rating is considered passing if it
appears in this set.
"""


@dataclass(slots=True, frozen=True)
class JudgeOutput:
    """Structured output from LLM judge evaluation.

    This dataclass defines the expected structured output format when using
    an LLM as a judge. The rating is a categorical label that maps to a
    numeric score via ``RATING_VALUES``.

    Example:
        >>> output = JudgeOutput(rating="good", reason="Mostly correct with minor issues")
        >>> score_value = RATING_VALUES[output.rating]  # 0.75
    """

    rating: Rating
    """Categorical label from the rating scale (excellent, good, fair, poor, wrong)."""

    reason: str
    """Brief explanation justifying the rating decision."""


@dataclass(slots=True, frozen=True)
class JudgeParams:
    """Parameters for the LLM judge prompt template.

    These parameters are bound to ``JUDGE_TEMPLATE`` to construct the
    evaluation prompt. The criterion defines what aspect to evaluate,
    while output and expected provide the comparison targets.

    Example:
        >>> params = JudgeParams(
        ...     criterion="factual accuracy",
        ...     output="The capital of France is Paris.",
        ...     expected="Paris is the capital of France.",
        ... )
    """

    criterion: str
    """The evaluation criterion (e.g., "factual accuracy", "clarity")."""

    output: str
    """The model output to evaluate."""

    expected: str
    """The reference/expected answer for comparison."""


JUDGE_TEMPLATE: PromptTemplate[JudgeOutput] = PromptTemplate[JudgeOutput](
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
"""Pre-built prompt template for LLM-as-judge evaluation.

This template is used by ``llm_judge()`` to construct evaluation prompts.
It expects ``JudgeParams`` for binding and produces ``JudgeOutput``.
The template instructs the LLM to select a categorical rating and provide
a brief explanation.
"""


def llm_judge(
    adapter: ProviderAdapter[JudgeOutput],
    criterion: str,
) -> Callable[[str, str], Score]:
    """Create evaluator that uses LLM to judge output.

    Creates an evaluator function that uses an LLM to score string outputs
    against a reference answer based on a specified criterion.

    Args:
        adapter: Provider adapter configured for JudgeOutput structured output.
        criterion: What to evaluate (e.g., "factual accuracy", "clarity").

    Returns:
        Evaluator function that scores string outputs.

    Example:
        >>> from weakincentives.adapters.openai import OpenAIAdapter
        >>> judge_adapter: OpenAIAdapter[JudgeOutput] = OpenAIAdapter(model="gpt-4o-mini")
        >>> evaluator = llm_judge(judge_adapter, "factual accuracy")
        >>> score = evaluator("The capital of France is Paris.", "Paris")
    """

    def evaluate(output: str, expected: str) -> Score:
        prompt = Prompt(JUDGE_TEMPLATE).bind(
            JudgeParams(
                criterion=criterion,
                output=output,
                expected=expected,
            )
        )
        session = Session()
        response = adapter.evaluate(prompt, session=session)
        if response.output is None:
            return Score(
                value=0.0,
                passed=False,
                reason="LLM judge returned no structured output",
            )
        rating = response.output.rating
        return Score(
            value=RATING_VALUES[rating],
            passed=rating in PASSING_RATINGS,
            reason=response.output.reason,
        )

    return evaluate


__all__ = [  # noqa: RUF022
    "JUDGE_TEMPLATE",
    "JudgeOutput",
    "JudgeParams",
    "PASSING_RATINGS",
    "RATING_VALUES",
    "Rating",
    "llm_judge",
]
