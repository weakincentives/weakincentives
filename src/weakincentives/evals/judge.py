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

"""LLM-as-judge evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..prompt import MarkdownSection, Prompt
from ..prompt.prompt import PromptTemplate
from ..runtime import Session
from ._types import PASSING_RATINGS, RATING_VALUES, JudgeOutput, Score

if TYPE_CHECKING:
    from ..adapters.core import ProviderAdapter
    from ..runtime.events._types import TelemetryBus


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


def llm_judge(
    adapter: ProviderAdapter[JudgeOutput],
    criterion: str,
    *,
    bus: TelemetryBus,
) -> Callable[[str, str], Score]:
    """Create evaluator that uses LLM to judge output.

    Args:
        adapter: Provider adapter configured for JudgeOutput
        criterion: What to evaluate (e.g., "factual accuracy", "clarity")
        bus: TelemetryBus for creating judge sessions

    Returns:
        Evaluator function that scores string outputs
    """

    def evaluate(output: str, expected: str) -> Score:
        prompt = Prompt(JUDGE_TEMPLATE).bind(
            JudgeParams(criterion=criterion, output=output, expected=expected)
        )
        session = Session(bus=bus, tags={"judge_criterion": criterion})
        response = adapter.evaluate(prompt, session=session)
        if response.output is None:
            return Score(value=0.0, passed=False, reason="No output from judge")
        rating = response.output.rating
        return Score(
            value=RATING_VALUES[rating],
            passed=rating in PASSING_RATINGS,
            reason=response.output.reason,
        )

    return evaluate


__all__ = ["JUDGE_TEMPLATE", "JudgeParams", "llm_judge"]
