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

"""Integration tests for the evals module with math operations.

This test suite verifies the evaluation framework by using an LLM to solve
mathematical problems via the Claude Agent SDK's native Bash tool.

The workspace section provides a temporary directory and the SDK provides
native tools (Bash, Read, Write, etc.) so the agent can run
``python3 -c "..."`` to compute answers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, override

import pytest

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    get_default_model,
)
from weakincentives.evals import (
    BASELINE,
    Dataset,
    EvalLoop,
    EvalRequest,
    EvalResult,
    Sample,
    Score,
    collect_results,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime import AgentLoop, InMemoryMailbox, Session
from weakincentives.runtime.agent_loop import AgentLoopRequest, AgentLoopResult

pytest.importorskip("claude_agent_sdk")


def _is_bedrock_mode() -> bool:
    """Check if running in Bedrock mode based on environment."""
    return os.getenv("CLAUDE_CODE_USE_BEDROCK") == "1" and "AWS_REGION" in os.environ


def _has_credentials() -> bool:
    """Check if Bedrock or Anthropic API credentials are available."""
    return _is_bedrock_mode() or "ANTHROPIC_API_KEY" in os.environ


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _has_credentials(),
        reason="Neither CLAUDE_CODE_USE_BEDROCK+AWS_REGION nor ANTHROPIC_API_KEY set.",
    ),
    pytest.mark.timeout(300),  # Math evals may take time (10 samples x ~30s each)
]

_MODEL_ENV_VAR: Final[str] = "CLAUDE_AGENT_SDK_TEST_MODEL"
_PROMPT_NS: Final[str] = "integration/evals-math"

# Evaluation constants
_NUMERIC_TOLERANCE = 0.001
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
# AgentLoop Implementation
# =============================================================================


class MathSolverLoop(AgentLoop[MathProblem, MathAnswer]):
    """AgentLoop that solves math problems using the SDK's native Bash tool."""

    _session: Session
    _template: PromptTemplate[MathAnswer]

    def __init__(
        self,
        *,
        adapter: ClaudeAgentSDKAdapter[MathAnswer],
        requests: InMemoryMailbox[
            AgentLoopRequest[MathProblem], AgentLoopResult[MathAnswer]
        ],
        session: Session,
        workspace: ClaudeAgentWorkspaceSection,
    ) -> None:
        super().__init__(adapter=adapter, requests=requests)

        self._session = session
        self._template = PromptTemplate[MathAnswer](
            ns=_PROMPT_NS,
            key="math-solver",
            sections=[
                workspace,
                MarkdownSection[_InstructionParams](
                    title="Instructions",
                    template="""You are a math assistant. Solve the following problem using Python as a calculator.

**Problem:** $question

Use the Bash tool to run a Python one-liner that computes the answer. For example:
- To compute 2 + 2, run: python3 -c "print(2 + 2)"
- To compute sqrt(256), run: python3 -c "import math; print(int(math.sqrt(256)))"

After computing, respond with ONLY the final numeric answer (no units, no explanation).
""",
                    key="instructions",
                ),
            ],
        )

    @override
    def prepare(
        self,
        request: MathProblem,
        *,
        experiment: object = None,
    ) -> tuple[Prompt[MathAnswer], Session]:
        _ = experiment
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


def _math_evaluator(output: object, expected: object) -> Score:
    """Evaluate if the math answer matches expected value.

    Handles various numeric formats (e.g., "12.0" == "12").
    """
    # Type assertions - Evaluator requires (object, object) -> Score signature
    assert isinstance(output, MathAnswer)
    assert isinstance(expected, str)

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
# Helpers
# =============================================================================


def _get_model() -> str:
    """Return the model name used for integration tests."""
    return os.environ.get(_MODEL_ENV_VAR, get_default_model())


def _make_adapter(cwd: str) -> ClaudeAgentSDKAdapter[MathAnswer]:
    """Create a Claude Agent SDK adapter for math solving."""
    return ClaudeAgentSDKAdapter(
        model=_get_model(),
        client_config=ClaudeAgentSDKClientConfig(
            permission_mode="bypassPermissions",
            cwd=cwd,
        ),
    )


# =============================================================================
# Tests
# =============================================================================


def test_math_eval_single_sample() -> None:
    """Test a single math problem evaluation."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[MathProblem, str], EvalResult] = (
        InMemoryMailbox(name="eval-requests")
    )
    dummy_requests: InMemoryMailbox[
        AgentLoopRequest[MathProblem], AgentLoopResult[MathAnswer]
    ] = InMemoryMailbox(name="dummy-requests")

    session = Session(tags={"loop": "math-solver"})
    workspace = ClaudeAgentWorkspaceSection(session=session)

    try:
        adapter = _make_adapter(str(workspace.temp_dir))
        agent_loop = MathSolverLoop(
            adapter=adapter,
            requests=dummy_requests,
            session=session,
            workspace=workspace,
        )

        eval_loop: EvalLoop[MathProblem, MathAnswer, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_math_evaluator,
            requests=requests,
        )

        sample = Sample(
            id="simple-add",
            input=MathProblem(question="What is 2 + 2?"),
            expected="4",
        )
        _ = requests.send(
            EvalRequest(sample=sample, experiment=BASELINE), reply_to=results
        )

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.sample_id == "simple-add"
        print(f"Result: score={result.score}, error={result.error}")
        msgs[0].acknowledge()

    finally:
        workspace.cleanup()
        requests.close()
        results.close()
        dummy_requests.close()


def test_math_eval_full_dataset() -> None:
    """Test the full math dataset evaluation."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[MathProblem, str], EvalResult] = (
        InMemoryMailbox(name="eval-requests")
    )
    dummy_requests: InMemoryMailbox[
        AgentLoopRequest[MathProblem], AgentLoopResult[MathAnswer]
    ] = InMemoryMailbox(name="dummy-requests")

    session = Session(tags={"loop": "math-solver"})
    workspace = ClaudeAgentWorkspaceSection(session=session)

    try:
        adapter = _make_adapter(str(workspace.temp_dir))
        agent_loop = MathSolverLoop(
            adapter=adapter,
            requests=dummy_requests,
            session=session,
            workspace=workspace,
        )

        eval_loop: EvalLoop[MathProblem, MathAnswer, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_math_evaluator,
            requests=requests,
        )

        dataset = _create_math_dataset()
        for sample in dataset:
            _ = requests.send(
                EvalRequest(sample=sample, experiment=BASELINE), reply_to=results
            )

        sample_count = len(dataset)
        eval_loop.run(max_iterations=sample_count)

        report = collect_results(
            results, expected_count=sample_count, timeout_seconds=60
        )

        print("\nEval Report:")
        print(f"  Total: {report.total}")
        print(f"  Successful: {report.successful}")
        print(f"  Pass rate: {report.pass_rate:.2%}")
        print(f"  Mean latency: {report.mean_latency_ms}ms")

        if report.failed_samples():
            print("\nFailed samples:")
            for fail in report.failed_samples():
                print(f"  - {fail.sample_id}: {fail.score.reason or fail.error}")

        assert report.total == sample_count, (
            f"Expected {sample_count} results, got {report.total}"
        )
        assert report.pass_rate >= _MIN_PASS_RATE, (
            f"Expected >={_MIN_PASS_RATE:.0%} pass rate, got {report.pass_rate:.2%}"
        )

    finally:
        workspace.cleanup()
        requests.close()
        results.close()
        dummy_requests.close()
