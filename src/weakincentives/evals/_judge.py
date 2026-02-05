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
"""Categorical rating labels for LLM-as-judge evaluation."""

RATING_VALUES: dict[Rating, float] = {
    "excellent": 1.0,  # Fully meets criterion
    "good": 0.75,  # Meets criterion with minor issues
    "fair": 0.5,  # Partially meets criterion
    "poor": 0.25,  # Mostly fails criterion
    "wrong": 0.0,  # Completely fails criterion
}
"""Mapping from rating labels to normalized score values."""

_PASSING: set[Rating] = {"excellent", "good"}
PASSING_RATINGS: frozenset[Rating] = frozenset(_PASSING)
"""Ratings that count as passing."""


@dataclass(slots=True, frozen=True)
class JudgeOutput:
    """Structured output from judge prompt."""

    rating: Rating
    """Categorical label from the rating scale."""

    reason: str
    """Brief explanation of the rating."""


@dataclass(slots=True, frozen=True)
class JudgeParams:
    """Parameters for the judge prompt."""

    criterion: str
    """The criterion to evaluate against."""

    output: str
    """The output to evaluate."""

    expected: str
    """The expected/reference answer."""


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
"""Prompt template for LLM-as-judge evaluation."""


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
        >>> from weakincentives.adapters import ProviderAdapter
        >>> judge_adapter: ProviderAdapter[JudgeOutput] = create_judge_adapter()
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
