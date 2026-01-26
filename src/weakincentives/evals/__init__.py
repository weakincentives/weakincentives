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

"""Evaluation framework for WINK agents.

This module provides a complete evaluation framework for testing and measuring
agent performance. Built on top of AgentLoop, it adds datasets, scoring, and
aggregated reporting while leveraging the existing mailbox infrastructure for
distributed evaluation.

Architecture Overview
---------------------

The evaluation framework follows a message-driven architecture:

1. **Samples** are loaded into a **Dataset** from JSONL files
2. Samples are wrapped in **EvalRequest** messages and sent to a mailbox
3. **EvalLoop** receives requests, executes through AgentLoop, and scores outputs
4. Results are aggregated into an **EvalReport** with computed metrics

This design enables both local testing and distributed evaluation across
multiple workers.


Core Types
----------

Sample : dataclass
    A single evaluation input pairing an input with its expected output.
    Each sample has a unique ``id``, an ``input`` value, and an ``expected``
    value to compare against.

Dataset : dataclass
    Immutable collection of samples with JSONL loading support. Iterable
    and indexable. Use ``Dataset.load(path, input_type, expected_type)``
    to load from JSONL files.

Score : dataclass
    Result of scoring one output. Contains:
    - ``value``: Normalized score from 0.0 to 1.0
    - ``passed``: Binary pass/fail indicator
    - ``reason``: Optional explanation (useful for LLM judges)

EvalResult : dataclass
    Result for one sample under an experiment. Contains the score, latency,
    experiment name, and optional error message.

EvalReport : dataclass
    Aggregate evaluation results with computed metrics:
    - ``pass_rate``: Fraction of successful samples that passed
    - ``mean_score``: Mean score across successful samples
    - ``mean_latency_ms``: Mean latency per sample
    - ``failed_samples()``: Get samples that did not pass
    - ``by_experiment()``: Group results by experiment name
    - ``compare_experiments(baseline, treatment)``: Statistical comparison


Evaluator Types
---------------

Two evaluator signatures are supported:

Evaluator : Callable[[object, object], Score]
    Standard evaluator taking ``(output, expected)`` and returning a Score.
    Pure functions with no side effects.

SessionEvaluator : Callable[[object, object, SessionProtocol], Score]
    Session-aware evaluator taking ``(output, expected, session)`` and
    returning a Score. Enables behavioral assertions by inspecting session
    state (tool calls, token usage, custom slices).


Built-in Evaluators
-------------------

exact_match(output, expected) -> Score
    Strict equality check using Python ``==``. Returns 1.0 on match, 0.0
    otherwise.

contains(output, expected) -> Score
    Substring presence check. Returns 1.0 if expected is in output, 0.0
    otherwise. Both arguments must be strings.


Session-Aware Evaluators
------------------------

These evaluators inspect session state for behavioral assertions:

tool_called(name: str) -> SessionEvaluator
    Assert that a specific tool was called at least once. Passes if the
    tool appears in the session's ToolInvoked events.

tool_not_called(name: str) -> SessionEvaluator
    Assert that a specific tool was never called. Useful for ensuring
    dangerous or unnecessary tools were avoided.

tool_call_count(name, min_count=0, max_count=None) -> SessionEvaluator
    Assert tool call count is within bounds. Use for rate-limiting
    assertions or ensuring efficient tool usage.

all_tools_succeeded() -> SessionEvaluator
    Assert all tool invocations succeeded. Checks the ``success`` field
    in each ToolInvoked.result dict.

token_usage_under(max_tokens: int) -> SessionEvaluator
    Assert total token usage is under budget. Sums input_tokens and
    output_tokens across all PromptExecuted events.

slice_contains(slice_type, predicate, min_count=1) -> SessionEvaluator
    Assert a custom slice contains items matching a predicate. Enables
    assertions on any session state tracked via reducers.


Evaluator Combinators
---------------------

Combine multiple evaluators into composite scoring:

all_of(*evaluators) -> SessionEvaluator
    All evaluators must pass. Score is the mean of individual scores.
    Automatically adapts standard evaluators to session-aware signature.

any_of(*evaluators) -> SessionEvaluator
    At least one evaluator must pass. Score is the maximum of individual
    scores. Automatically adapts standard evaluators.

adapt(evaluator: Evaluator) -> SessionEvaluator
    Explicitly convert a standard evaluator to session-aware signature.
    The session parameter is ignored.

is_session_aware(fn) -> bool
    Check if an evaluator accepts a session parameter. Used internally
    by combinators to determine adaptation needs.


LLM-as-Judge
------------

Use an LLM to judge outputs against a criterion:

llm_judge(adapter, criterion) -> Evaluator
    Create an evaluator that uses an LLM to score outputs. The LLM selects
    from categorical ratings rather than numerical scores for better
    calibration.

Rating : Literal["excellent", "good", "fair", "poor", "wrong"]
    Categorical rating labels for LLM-as-judge evaluation.

RATING_VALUES : dict[Rating, float]
    Mapping from rating labels to normalized scores:
    - excellent: 1.0 (fully meets criterion)
    - good: 0.75 (meets with minor issues)
    - fair: 0.5 (partially meets)
    - poor: 0.25 (mostly fails)
    - wrong: 0.0 (completely fails)

PASSING_RATINGS : frozenset[Rating]
    Ratings that count as passing: {"excellent", "good"}

JudgeParams : dataclass
    Parameters for the judge prompt: criterion, output, expected.

JudgeOutput : dataclass
    Structured output from judge: rating and reason.

JUDGE_TEMPLATE : PromptTemplate
    The prompt template used for LLM-as-judge evaluation.


Evaluation Loop
---------------

EvalLoop orchestrates evaluation using mailbox-driven messaging:

EvalLoop : class
    Mailbox-driven evaluation loop. Receives EvalRequest messages,
    executes through AgentLoop, scores with the evaluator, and sends
    EvalResult via reply. Designed for distributed deployments.

    Supports both standard and session-aware evaluators. The session
    is automatically passed to session-aware evaluators.

    Key features:
    - Automatic lease extension during long evaluations
    - Debug bundle creation for sample-level debugging
    - Dead letter queue support for failed evaluations
    - Graceful shutdown with in-flight completion

EvalLoopConfig : dataclass
    Configuration for EvalLoop:
    - ``lease_extender``: Automatic message visibility extension
    - ``debug_bundle_dir``: Enable debug bundle creation


Helper Functions
----------------

submit_dataset(dataset, experiment, requests) -> int
    Submit all samples in a dataset for evaluation under a single
    experiment. Returns the number of samples submitted.

submit_experiments(dataset, experiments, requests) -> int
    Submit dataset under multiple experiments for A/B testing.
    Submits each sample under each experiment.

collect_results(results, expected_count, timeout_seconds=300) -> EvalReport
    Collect evaluation results into a report. Polls the results mailbox
    until all expected results are collected or timeout expires.


Experiment Support
------------------

Run evaluations under different experimental conditions:

Experiment : dataclass
    An experimental configuration with a name and optional overrides.
    Use experiments to compare different prompts, models, or settings.

BASELINE : Experiment
    Pre-defined baseline experiment with name "baseline".

CONTROL : Experiment
    Pre-defined control experiment with name "control".


Examples
--------

Basic evaluation with exact matching::

    from pathlib import Path
    from weakincentives.evals import (
        BASELINE,
        Dataset,
        EvalLoop,
        collect_results,
        exact_match,
        submit_dataset,
    )

    # Load evaluation dataset
    dataset = Dataset.load(Path("qa.jsonl"), str, str)

    # Create evaluation loop
    eval_loop = EvalLoop(
        loop=agent_loop,
        evaluator=exact_match,
        requests=requests_mailbox,
    )

    # Submit samples and run evaluation
    submit_dataset(dataset, BASELINE, requests_mailbox)
    eval_loop.run(max_iterations=1)

    # Collect and analyze results
    report = collect_results(results_mailbox, expected_count=len(dataset))
    print(f"Pass rate: {report.pass_rate:.1%}")
    print(f"Mean score: {report.mean_score:.2f}")
    print(f"Mean latency: {report.mean_latency_ms:.0f}ms")


Behavioral assertions with session-aware evaluators::

    from weakincentives.evals import (
        all_of,
        all_tools_succeeded,
        exact_match,
        token_usage_under,
        tool_called,
        tool_not_called,
    )

    # Combine multiple assertions
    evaluator = all_of(
        exact_match,                      # Output must match expected
        tool_called("search"),            # Must use the search tool
        tool_not_called("dangerous_op"),  # Must not use dangerous operations
        all_tools_succeeded(),            # All tool calls must succeed
        token_usage_under(5000),          # Stay under token budget
    )


A/B testing with multiple experiments::

    from weakincentives.evals import (
        BASELINE,
        Experiment,
        collect_results,
        submit_experiments,
    )

    # Define experiments
    baseline = BASELINE
    treatment = Experiment(name="v2-prompts", overrides_tag="v2")

    # Submit under both experiments
    total = submit_experiments(dataset, [baseline, treatment], requests_mailbox)
    eval_loop.run(max_iterations=1)

    # Compare results
    report = collect_results(results_mailbox, expected_count=total)
    comparison = report.compare_experiments("baseline", "v2-prompts")
    print(f"Baseline pass rate: {comparison.baseline_pass_rate:.1%}")
    print(f"Treatment pass rate: {comparison.treatment_pass_rate:.1%}")
    print(f"Delta: {comparison.pass_rate_delta:+.1%}")


LLM-as-judge evaluation::

    from weakincentives.adapters.openai import OpenAIAdapter
    from weakincentives.evals import JudgeOutput, llm_judge

    # Create judge evaluator
    judge_adapter = OpenAIAdapter[JudgeOutput](model="gpt-4o-mini")
    evaluator = llm_judge(judge_adapter, "factual accuracy and clarity")

    # Use in evaluation
    score = evaluator(
        output="The capital of France is Paris, located on the Seine River.",
        expected="Paris",
    )
    print(f"Rating: {score.value}, Reason: {score.reason}")


Custom slice assertions::

    from weakincentives.evals import slice_contains

    # Assert that planning steps completed
    evaluator = slice_contains(
        PlanStep,
        lambda step: step.status == "completed",
        min_count=3,
    )


JSONL Dataset Format
--------------------

Datasets are loaded from JSONL files where each line is a JSON object
with ``id``, ``input``, and ``expected`` fields::

    {"id": "1", "input": "What is 2+2?", "expected": "4"}
    {"id": "2", "input": "Capital of France?", "expected": "Paris"}
    {"id": "3", "input": {"query": "weather"}, "expected": {"temp": 72}}

For complex types, the ``input`` and ``expected`` fields are deserialized
as dataclasses via serde.parse.
"""

from __future__ import annotations

from ..experiment import BASELINE, CONTROL, Experiment
from ._evaluators import adapt, all_of, any_of, contains, exact_match, is_session_aware
from ._helpers import collect_results, submit_dataset, submit_experiments
from ._judge import (
    JUDGE_TEMPLATE,
    PASSING_RATINGS,
    RATING_VALUES,
    JudgeOutput,
    JudgeParams,
    Rating,
    llm_judge,
)
from ._loop import EvalLoop, EvalLoopConfig
from ._session_evaluators import (
    all_tools_succeeded,
    slice_contains,
    token_usage_under,
    tool_call_count,
    tool_called,
    tool_not_called,
)
from ._types import (
    Dataset,
    EvalReport,
    EvalRequest,
    EvalResult,
    Evaluator,
    ExperimentComparison,
    Sample,
    Score,
    SessionEvaluator,
)

__all__ = [  # noqa: RUF022
    "BASELINE",
    "CONTROL",
    "Dataset",
    "EvalLoop",
    "EvalLoopConfig",
    "EvalReport",
    "EvalRequest",
    "EvalResult",
    "Evaluator",
    "Experiment",
    "ExperimentComparison",
    "JUDGE_TEMPLATE",
    "JudgeOutput",
    "JudgeParams",
    "PASSING_RATINGS",
    "RATING_VALUES",
    "Rating",
    "Sample",
    "Score",
    "SessionEvaluator",
    "adapt",
    "all_of",
    "all_tools_succeeded",
    "any_of",
    "collect_results",
    "contains",
    "exact_match",
    "is_session_aware",
    "llm_judge",
    "slice_contains",
    "submit_dataset",
    "submit_experiments",
    "token_usage_under",
    "tool_call_count",
    "tool_called",
    "tool_not_called",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
