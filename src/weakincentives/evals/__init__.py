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

"""Evaluation framework for testing agent outputs.

This module provides a minimal evaluation framework built on MainLoop.
MainLoop handles orchestration; this module adds datasets and scoring.

Basic usage with ``run_eval`` (simple, synchronous)::

    from weakincentives.evals import Sample, run_eval, exact_match, load_jsonl

    # Build dataset programmatically
    dataset = tuple(
        Sample(id=str(i), input=f"What is {a}+{b}?", expected=str(a+b))
        for i, (a, b) in enumerate([(1, 1), (2, 3)])
    )

    # Or load from JSONL
    dataset = load_jsonl(Path("tests/fixtures/qa.jsonl"), str, str)

    # Run evaluation
    report = run_eval(loop, dataset, exact_match)
    print(f"Pass rate: {report.pass_rate:.1%}")

Event-driven usage with ``EvalLoop`` (for cluster deployment)::

    from weakincentives.evals import EvalLoop, EvalLoopRequest, EvalLoopCompleted

    # Create EvalLoop with MainLoop, evaluator, and bus
    eval_loop = EvalLoop(loop=main_loop, evaluator=exact_match, bus=bus)

    # Subscribe to completion events
    bus.subscribe(EvalLoopCompleted, handle_completed)

    # Submit evaluation request via bus
    bus.publish(EvalLoopRequest(dataset=dataset))

LLM-as-judge for subjective criteria::

    from weakincentives.evals import llm_judge, all_of, contains

    evaluator = all_of(
        contains,
        llm_judge(adapter, "Factually accurate", bus=bus),
    )
    report = run_eval(loop, dataset, evaluator)
"""

from __future__ import annotations

from ._types import (
    PASSING_RATINGS,
    RATING_VALUES,
    EvalReport,
    EvalResult,
    Evaluator,
    JudgeOutput,
    Rating,
    Sample,
    SampleEvaluated,
    Score,
)
from .dataset import load_jsonl
from .eval_loop import (
    EvalLoop,
    EvalLoopCompleted,
    EvalLoopConfig,
    EvalLoopFailed,
    EvalLoopRequest,
)
from .evaluators import (
    all_of,
    any_of,
    contains,
    exact_match,
    json_subset,
    within_tolerance,
)
from .judge import JUDGE_TEMPLATE, JudgeParams, llm_judge
from .runner import run_eval

__all__ = [
    "JUDGE_TEMPLATE",
    "PASSING_RATINGS",
    "RATING_VALUES",
    "EvalLoop",
    "EvalLoopCompleted",
    "EvalLoopConfig",
    "EvalLoopFailed",
    "EvalLoopRequest",
    "EvalReport",
    "EvalResult",
    "Evaluator",
    "JudgeOutput",
    "JudgeParams",
    "Rating",
    "Sample",
    "SampleEvaluated",
    "Score",
    "all_of",
    "any_of",
    "contains",
    "exact_match",
    "json_subset",
    "llm_judge",
    "load_jsonl",
    "run_eval",
    "within_tolerance",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
