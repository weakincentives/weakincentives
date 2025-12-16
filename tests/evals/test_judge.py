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

"""Tests for LLM-as-judge functionality."""

from __future__ import annotations

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.evals import (
    JUDGE_TEMPLATE,
    PASSING_RATINGS,
    RATING_VALUES,
    JudgeOutput,
    JudgeParams,
    llm_judge,
)
from weakincentives.prompt import Prompt
from weakincentives.runtime import InProcessEventBus
from weakincentives.runtime.session.protocols import SessionProtocol


class _MockJudgeAdapter(ProviderAdapter[JudgeOutput]):
    """Mock adapter that returns configured judge outputs."""

    def __init__(self, output: JudgeOutput | None = None) -> None:
        self._output = output or JudgeOutput(rating="good", reason="Looks good")
        self._last_prompt: Prompt[JudgeOutput] | None = None

    def evaluate(
        self,
        prompt: Prompt[JudgeOutput],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponse[JudgeOutput]:
        del session, deadline, budget, budget_tracker
        self._last_prompt = prompt
        return PromptResponse(
            prompt_name="llm_judge",
            text=f"{self._output.rating}: {self._output.reason}",
            output=self._output,
        )


class _NoneOutputAdapter(ProviderAdapter[JudgeOutput]):
    """Mock adapter that returns None output."""

    def evaluate(
        self,
        prompt: Prompt[JudgeOutput],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponse[JudgeOutput]:
        del prompt, session, deadline, budget, budget_tracker
        return PromptResponse(
            prompt_name="llm_judge",
            text=None,
            output=None,
        )


# =============================================================================
# JUDGE_TEMPLATE Tests
# =============================================================================


def test_judge_template_namespace() -> None:
    """JUDGE_TEMPLATE has correct namespace."""
    assert JUDGE_TEMPLATE.ns == "wink.evals"
    assert JUDGE_TEMPLATE.key == "llm-judge"
    assert JUDGE_TEMPLATE.name == "llm_judge"


def test_judge_template_has_sections() -> None:
    """JUDGE_TEMPLATE has task section."""
    assert len(JUDGE_TEMPLATE.sections) > 0


def test_judge_template_structured_output() -> None:
    """JUDGE_TEMPLATE is specialized for JudgeOutput."""
    assert JUDGE_TEMPLATE.structured_output is not None
    assert JUDGE_TEMPLATE.structured_output.dataclass_type is JudgeOutput


# =============================================================================
# JudgeParams Tests
# =============================================================================


def test_judge_params_creation() -> None:
    """JudgeParams stores criterion, output, expected."""
    params = JudgeParams(
        criterion="factual accuracy",
        output="Paris is in France",
        expected="Paris is the capital of France",
    )
    assert params.criterion == "factual accuracy"
    assert params.output == "Paris is in France"
    assert params.expected == "Paris is the capital of France"


# =============================================================================
# llm_judge Factory Tests
# =============================================================================


def test_llm_judge_excellent_rating() -> None:
    """llm_judge returns correct score for excellent rating."""
    bus = InProcessEventBus()
    output = JudgeOutput(rating="excellent", reason="Perfect")
    adapter = _MockJudgeAdapter(output)

    evaluator = llm_judge(adapter, "factual accuracy", bus=bus)
    score = evaluator("test output", "test expected")

    assert score.value == RATING_VALUES["excellent"]
    assert score.passed is True
    assert score.reason == "Perfect"


def test_llm_judge_good_rating() -> None:
    """llm_judge returns passing score for good rating."""
    bus = InProcessEventBus()
    output = JudgeOutput(rating="good", reason="Minor issues")
    adapter = _MockJudgeAdapter(output)

    evaluator = llm_judge(adapter, "clarity", bus=bus)
    score = evaluator("output", "expected")

    assert score.value == RATING_VALUES["good"]
    assert score.passed is True  # good is passing


def test_llm_judge_fair_rating() -> None:
    """llm_judge returns failing score for fair rating."""
    bus = InProcessEventBus()
    output = JudgeOutput(rating="fair", reason="Partial match")
    adapter = _MockJudgeAdapter(output)

    evaluator = llm_judge(adapter, "accuracy", bus=bus)
    score = evaluator("output", "expected")

    assert score.value == RATING_VALUES["fair"]
    assert score.passed is False  # fair is not passing


def test_llm_judge_poor_rating() -> None:
    """llm_judge returns failing score for poor rating."""
    bus = InProcessEventBus()
    output = JudgeOutput(rating="poor", reason="Mostly wrong")
    adapter = _MockJudgeAdapter(output)

    evaluator = llm_judge(adapter, "accuracy", bus=bus)
    score = evaluator("output", "expected")

    assert score.value == RATING_VALUES["poor"]
    assert score.passed is False


def test_llm_judge_wrong_rating() -> None:
    """llm_judge returns zero score for wrong rating."""
    bus = InProcessEventBus()
    output = JudgeOutput(rating="wrong", reason="Completely incorrect")
    adapter = _MockJudgeAdapter(output)

    evaluator = llm_judge(adapter, "accuracy", bus=bus)
    score = evaluator("output", "expected")

    assert score.value == RATING_VALUES["wrong"]
    assert score.value == 0.0
    assert score.passed is False


def test_llm_judge_passes_criterion() -> None:
    """llm_judge passes criterion to prompt."""
    bus = InProcessEventBus()
    adapter = _MockJudgeAdapter()

    evaluator = llm_judge(adapter, "factual accuracy", bus=bus)
    evaluator("output text", "expected text")

    # Check the criterion was passed in params
    assert adapter._last_prompt is not None
    params = adapter._last_prompt.params
    judge_params = next(p for p in params if isinstance(p, JudgeParams))
    assert judge_params.criterion == "factual accuracy"


def test_llm_judge_passes_output_and_expected() -> None:
    """llm_judge passes output and expected to prompt."""
    bus = InProcessEventBus()
    adapter = _MockJudgeAdapter()

    evaluator = llm_judge(adapter, "test criterion", bus=bus)
    evaluator("my output", "my expected")

    assert adapter._last_prompt is not None
    params = adapter._last_prompt.params
    judge_params = next(p for p in params if isinstance(p, JudgeParams))
    assert judge_params.output == "my output"
    assert judge_params.expected == "my expected"


def test_llm_judge_none_output() -> None:
    """llm_judge handles None output from adapter."""
    bus = InProcessEventBus()
    adapter = _NoneOutputAdapter()

    evaluator = llm_judge(adapter, "test criterion", bus=bus)
    score = evaluator("output", "expected")

    assert score.value == 0.0
    assert score.passed is False
    assert "No output from judge" in score.reason


# =============================================================================
# Rating Constants Integration Tests
# =============================================================================


def test_passing_ratings_match_values() -> None:
    """PASSING_RATINGS contains ratings with value >= 0.75."""
    for rating in PASSING_RATINGS:
        assert RATING_VALUES[rating] >= 0.75


def test_all_ratings_have_values() -> None:
    """All Rating literals have corresponding values."""
    expected_ratings = {"excellent", "good", "fair", "poor", "wrong"}
    assert set(RATING_VALUES.keys()) == expected_ratings
