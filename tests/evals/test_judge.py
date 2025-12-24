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

"""Tests for LLM-as-Judge functionality."""

from __future__ import annotations

from typing import Any

import pytest

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
from weakincentives.prompt.tool import ResourceRegistry
from weakincentives.runtime.session import SessionProtocol

# =============================================================================
# Rating Constants Tests
# =============================================================================


def test_rating_values_complete() -> None:
    """RATING_VALUES has all expected ratings."""
    assert "excellent" in RATING_VALUES
    assert "good" in RATING_VALUES
    assert "fair" in RATING_VALUES
    assert "poor" in RATING_VALUES
    assert "wrong" in RATING_VALUES


def test_rating_values_ordered() -> None:
    """RATING_VALUES are ordered by quality."""
    assert RATING_VALUES["excellent"] > RATING_VALUES["good"]
    assert RATING_VALUES["good"] > RATING_VALUES["fair"]
    assert RATING_VALUES["fair"] > RATING_VALUES["poor"]
    assert RATING_VALUES["poor"] > RATING_VALUES["wrong"]


def test_rating_values_normalized() -> None:
    """RATING_VALUES are between 0.0 and 1.0."""
    for value in RATING_VALUES.values():
        assert 0.0 <= value <= 1.0


def test_passing_ratings() -> None:
    """PASSING_RATINGS contains correct ratings."""
    assert "excellent" in PASSING_RATINGS
    assert "good" in PASSING_RATINGS
    assert "fair" not in PASSING_RATINGS
    assert "poor" not in PASSING_RATINGS
    assert "wrong" not in PASSING_RATINGS


# =============================================================================
# JudgeOutput Tests
# =============================================================================


def test_judge_output_creation() -> None:
    """JudgeOutput can be created."""
    output = JudgeOutput(rating="excellent", reason="Perfect answer")
    assert output.rating == "excellent"
    assert output.reason == "Perfect answer"


def test_judge_output_is_frozen() -> None:
    """JudgeOutput is immutable."""
    output = JudgeOutput(rating="good", reason="Nice")
    with pytest.raises(AttributeError):
        output.rating = "excellent"  # type: ignore[misc]


# =============================================================================
# JudgeParams Tests
# =============================================================================


def test_judge_params_creation() -> None:
    """JudgeParams can be created."""
    params = JudgeParams(
        criterion="factual accuracy",
        output="Paris is the capital of France.",
        expected="Paris",
    )
    assert params.criterion == "factual accuracy"
    assert params.output == "Paris is the capital of France."
    assert params.expected == "Paris"


def test_judge_params_is_frozen() -> None:
    """JudgeParams is immutable."""
    params = JudgeParams(criterion="test", output="a", expected="b")
    with pytest.raises(AttributeError):
        params.criterion = "changed"  # type: ignore[misc]


# =============================================================================
# JUDGE_TEMPLATE Tests
# =============================================================================


def test_judge_template_structure() -> None:
    """JUDGE_TEMPLATE has correct structure."""
    assert JUDGE_TEMPLATE.ns == "wink.evals"
    assert JUDGE_TEMPLATE.key == "llm-judge"
    assert JUDGE_TEMPLATE.name == "llm_judge"
    assert len(JUDGE_TEMPLATE.sections) == 1


def test_judge_template_renders() -> None:
    """JUDGE_TEMPLATE can be rendered with params."""
    prompt = Prompt(JUDGE_TEMPLATE).bind(
        JudgeParams(
            criterion="factual accuracy",
            output="Paris is the capital.",
            expected="Paris",
        )
    )
    rendered = prompt.render()
    text = rendered.text
    assert "factual accuracy" in text
    assert "Paris is the capital." in text
    assert "Paris" in text
    assert "excellent" in text
    assert "good" in text


# =============================================================================
# llm_judge Tests
# =============================================================================


class _MockJudgeAdapter(ProviderAdapter[JudgeOutput]):
    """Mock adapter for testing llm_judge."""

    def __init__(
        self,
        *,
        rating: str = "good",
        reason: str = "Test reason",
        return_none: bool = False,
    ) -> None:
        self._rating = rating
        self._reason = reason
        self._return_none = return_none
        self.call_count = 0

    def evaluate(
        self,
        prompt: Prompt[JudgeOutput],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        resources: ResourceRegistry | None = None,
    ) -> PromptResponse[JudgeOutput]:
        del prompt, session, deadline, budget, budget_tracker, resources
        self.call_count += 1
        if self._return_none:
            return PromptResponse(
                prompt_name="llm_judge",
                text=None,
                output=None,
            )
        return PromptResponse(
            prompt_name="llm_judge",
            text="Rating: good",
            output=JudgeOutput(rating=self._rating, reason=self._reason),  # type: ignore[arg-type]
        )


def test_llm_judge_excellent() -> None:
    """llm_judge returns correct score for excellent rating."""
    adapter = _MockJudgeAdapter(rating="excellent", reason="Perfect")
    evaluator = llm_judge(adapter, "accuracy")

    score = evaluator("correct answer", "expected")

    assert score.value == 1.0
    assert score.passed is True
    assert score.reason == "Perfect"
    assert adapter.call_count == 1


def test_llm_judge_good() -> None:
    """llm_judge returns correct score for good rating."""
    adapter = _MockJudgeAdapter(rating="good", reason="Minor issues")
    evaluator = llm_judge(adapter, "accuracy")

    score = evaluator("mostly correct", "expected")

    assert score.value == 0.75
    assert score.passed is True
    assert score.reason == "Minor issues"


def test_llm_judge_fair() -> None:
    """llm_judge returns correct score for fair rating."""
    adapter = _MockJudgeAdapter(rating="fair", reason="Partially correct")
    evaluator = llm_judge(adapter, "accuracy")

    score = evaluator("half right", "expected")

    assert score.value == 0.5
    assert score.passed is False


def test_llm_judge_poor() -> None:
    """llm_judge returns correct score for poor rating."""
    adapter = _MockJudgeAdapter(rating="poor", reason="Mostly wrong")
    evaluator = llm_judge(adapter, "accuracy")

    score = evaluator("bad answer", "expected")

    assert score.value == 0.25
    assert score.passed is False


def test_llm_judge_wrong() -> None:
    """llm_judge returns correct score for wrong rating."""
    adapter = _MockJudgeAdapter(rating="wrong", reason="Completely incorrect")
    evaluator = llm_judge(adapter, "accuracy")

    score = evaluator("wrong answer", "expected")

    assert score.value == 0.0
    assert score.passed is False


def test_llm_judge_handles_none_output() -> None:
    """llm_judge handles None output gracefully."""
    adapter = _MockJudgeAdapter(return_none=True)
    evaluator = llm_judge(adapter, "accuracy")

    score = evaluator("answer", "expected")

    assert score.value == 0.0
    assert score.passed is False
    assert "no structured output" in score.reason


def test_llm_judge_uses_criterion() -> None:
    """llm_judge uses provided criterion in evaluation."""

    class _CaptureAdapter(ProviderAdapter[JudgeOutput]):
        captured_prompt: Prompt[Any] | None = None

        def evaluate(
            self,
            prompt: Prompt[JudgeOutput],
            *,
            session: SessionProtocol,
            deadline: Deadline | None = None,
            budget: Budget | None = None,
            budget_tracker: BudgetTracker | None = None,
            resources: ResourceRegistry | None = None,
        ) -> PromptResponse[JudgeOutput]:
            del session, deadline, budget, budget_tracker, resources
            self.captured_prompt = prompt
            return PromptResponse(
                prompt_name="llm_judge",
                text="ok",
                output=JudgeOutput(rating="good", reason="ok"),
            )

    adapter = _CaptureAdapter()
    evaluator = llm_judge(adapter, "custom criterion here")

    _ = evaluator("output", "expected")

    assert adapter.captured_prompt is not None
    rendered = adapter.captured_prompt.render().text
    assert "custom criterion here" in rendered
