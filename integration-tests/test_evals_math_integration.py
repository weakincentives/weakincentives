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

"""Integration tests for the evals module with Asteval math operations.

This test suite verifies the evaluation framework by using an LLM to solve
mathematical problems using Python as a calculator via the Asteval tool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import override

import pytest

from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.contrib.tools import AstevalSection
from weakincentives.evals import (
    Dataset,
    EvalLoop,
    EvalRequest,
    EvalResult,
    Sample,
    Score,
    collect_results,
    submit_dataset,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime import InMemoryMailbox, MainLoop, Session

pytest.importorskip("openai")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ,
        reason="OPENAI_API_KEY not set; skipping OpenAI integration tests.",
    ),
    pytest.mark.timeout(300),  # Math evals may take time (10 samples x ~30s each)
]

_MODEL_ENV_VAR = "OPENAI_TEST_MODEL"
_DEFAULT_MODEL = "gpt-4.1-mini"
_PROMPT_NS = "integration/evals-math"

# Evaluation constants
_NUMERIC_TOLERANCE = 0.001
_EXPECTED_SAMPLE_COUNT = 10
_MIN_PASS_RATE = 0.5


# =============================================================================
# Types
# =============================================================================


@dataclass(slots=True, frozen=True)
class MathProblem:
    """A math problem to solve."""

    question: str


@dataclass(slots=True, frozen=True)
class MathAnswer:
    """The answer to a math problem."""

    answer: str


@dataclass(slots=True, frozen=True)
class _InstructionParams:
    """Parameters for the instruction section."""

    question: str


# =============================================================================
# MainLoop Implementation
# =============================================================================


class MathSolverLoop(MainLoop[MathProblem, MathAnswer]):
    """MainLoop that solves math problems using Asteval."""

    _session: Session
    _template: PromptTemplate[MathAnswer]

    def __init__(
        self,
        *,
        adapter: OpenAIAdapter[MathAnswer],
        requests: InMemoryMailbox[object, object],
    ) -> None:
        super().__init__(adapter=adapter, requests=requests)  # type: ignore[arg-type]

        # Create persistent session for the loop
        self._session = Session(tags={"loop": "math-solver"})

        # Build template with Asteval section
        asteval_section = AstevalSection(session=self._session, accepts_overrides=True)
        self._template = PromptTemplate[MathAnswer](
            ns=_PROMPT_NS,
            key="math-solver",
            sections=[
                MarkdownSection[_InstructionParams](
                    title="Instructions",
                    template="""You are a math assistant. Solve the following problem using Python as a calculator.

**Problem:** $question

Use the evaluate_python tool to compute the answer. After computing, respond with ONLY the final numeric answer (no units, no explanation).

For example, if the problem is "What is 2 + 2?", you would:
1. Use evaluate_python with code: "2 + 2"
2. Respond with just: "4"
""",
                    key="instructions",
                ),
                asteval_section,
            ],
        )

    @override
    def prepare(self, request: MathProblem) -> tuple[Prompt[MathAnswer], Session]:
        prompt = Prompt(self._template).bind(
            _InstructionParams(question=request.question)
        )
        return prompt, self._session


# =============================================================================
# Dataset
# =============================================================================


def _create_math_dataset() -> Dataset[MathProblem, str]:
    """Create a dataset of 10 math problems."""
    samples: tuple[Sample[MathProblem, str], ...] = (
        Sample(
            id="add-1",
            input=MathProblem(question="What is 123 + 456?"),
            expected="579",
        ),
        Sample(
            id="sub-1",
            input=MathProblem(question="What is 1000 - 237?"),
            expected="763",
        ),
        Sample(
            id="mul-1",
            input=MathProblem(question="What is 17 * 23?"),
            expected="391",
        ),
        Sample(
            id="div-1",
            input=MathProblem(question="What is 144 / 12?"),
            expected="12",
        ),
        Sample(
            id="power-1",
            input=MathProblem(question="What is 2 to the power of 10?"),
            expected="1024",
        ),
        Sample(
            id="sqrt-1",
            input=MathProblem(question="What is the square root of 256?"),
            expected="16",
        ),
        Sample(
            id="complex-1",
            input=MathProblem(question="What is (15 + 25) * 3?"),
            expected="120",
        ),
        Sample(
            id="complex-2",
            input=MathProblem(question="What is 100 - (5 * 12)?"),
            expected="40",
        ),
        Sample(
            id="floor-1",
            input=MathProblem(
                question="What is 17 divided by 5, rounded down to nearest integer?"
            ),
            expected="3",
        ),
        Sample(
            id="mod-1",
            input=MathProblem(
                question="What is the remainder when 47 is divided by 7?"
            ),
            expected="5",
        ),
    )
    return Dataset(samples=samples)


# =============================================================================
# Evaluator
# =============================================================================


def _math_evaluator(output: MathAnswer, expected: str) -> Score:
    """Evaluate if the math answer matches expected value.

    Handles various numeric formats (e.g., "12.0" == "12").
    """
    # Normalize strings first
    actual_str = output.answer.strip()
    expected_str = expected.strip()

    try:
        # Try numeric comparison first
        actual_num = float(actual_str)
        expected_num = float(expected_str)

        if abs(actual_num - expected_num) < _NUMERIC_TOLERANCE:
            return Score(value=1.0, passed=True, reason="Exact match")

        return Score(
            value=0.0,
            passed=False,
            reason=f"Expected {expected_str}, got {actual_str}",
        )
    except ValueError:
        # Fall back to string comparison if not numeric
        if actual_str == expected_str:
            return Score(value=1.0, passed=True, reason="String match")
        return Score(
            value=0.0,
            passed=False,
            reason=f"Expected '{expected_str}', got '{actual_str}'",
        )


# =============================================================================
# Tests
# =============================================================================


@pytest.fixture(scope="module")
def openai_model() -> str:
    """Return the model name used for integration tests."""
    return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)


@pytest.fixture(scope="module")
def adapter(openai_model: str) -> OpenAIAdapter[MathAnswer]:
    """Create an OpenAI adapter for math solving."""
    return OpenAIAdapter(model=openai_model)


def test_math_eval_single_sample(adapter: OpenAIAdapter[MathAnswer]) -> None:
    """Test a single math problem evaluation."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[MathProblem, str], EvalResult] = (
        InMemoryMailbox(
            name="eval-requests",
            reply_resolver=lambda name: results if name == "results" else None,
        )
    )
    dummy_requests: InMemoryMailbox[object, object] = InMemoryMailbox(
        name="dummy-requests"
    )

    try:
        main_loop = MathSolverLoop(
            adapter=adapter,
            requests=dummy_requests,
        )

        eval_loop: EvalLoop[MathProblem, MathAnswer, str] = EvalLoop(
            loop=main_loop,
            evaluator=_math_evaluator,
            requests=requests,
        )

        # Single sample: 2 + 2 = 4
        sample = Sample(
            id="simple-add",
            input=MathProblem(question="What is 2 + 2?"),
            expected="4",
        )
        _ = requests.send(EvalRequest(sample=sample), reply_to="results")

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.sample_id == "simple-add"
        # Log result for debugging
        print(f"Result: score={result.score}, error={result.error}")
        msgs[0].acknowledge()

    finally:
        requests.close()
        results.close()
        dummy_requests.close()


def test_math_eval_full_dataset(adapter: OpenAIAdapter[MathAnswer]) -> None:
    """Test the full math dataset evaluation."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[MathProblem, str], EvalResult] = (
        InMemoryMailbox(
            name="eval-requests",
            reply_resolver=lambda name: results if name == "results" else None,
        )
    )
    dummy_requests: InMemoryMailbox[object, object] = InMemoryMailbox(
        name="dummy-requests"
    )

    try:
        main_loop = MathSolverLoop(
            adapter=adapter,
            requests=dummy_requests,
        )

        eval_loop: EvalLoop[MathProblem, MathAnswer, str] = EvalLoop(
            loop=main_loop,
            evaluator=_math_evaluator,
            requests=requests,
        )

        # Create and submit the full dataset with reply_to
        dataset = _create_math_dataset()
        submit_dataset(dataset, requests, reply_to="results")

        # Run evaluations (10 samples, give some headroom for iterations)
        eval_loop.run(max_iterations=15)

        # Collect results
        report = collect_results(
            results, expected_count=_EXPECTED_SAMPLE_COUNT, timeout_seconds=60
        )

        # Log results for debugging
        print("\nEval Report:")
        print(f"  Total: {report.total}")
        print(f"  Successful: {report.successful}")
        print(f"  Pass rate: {report.pass_rate:.2%}")
        print(f"  Mean latency: {report.mean_latency_ms}ms")

        if report.failed_samples():
            print("\nFailed samples:")
            for fail in report.failed_samples():
                print(f"  - {fail.sample_id}: {fail.score.reason or fail.error}")

        # We expect most math problems to be solved correctly
        # Allow some tolerance for LLM variability
        assert report.total == _EXPECTED_SAMPLE_COUNT, (
            f"Expected {_EXPECTED_SAMPLE_COUNT} results, got {report.total}"
        )
        assert report.pass_rate >= _MIN_PASS_RATE, (
            f"Expected >={_MIN_PASS_RATE:.0%} pass rate, got {report.pass_rate:.2%}"
        )

    finally:
        requests.close()
        results.close()
        dummy_requests.close()
