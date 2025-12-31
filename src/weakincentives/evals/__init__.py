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

Minimal evaluation framework built on MainLoop. MainLoop handles orchestration;
this module adds datasets and scoring.

Core components:
- **Sample/Dataset**: Immutable evaluation data with JSONL loading
- **Score/Evaluator**: Scoring primitives and type alias
- **EvalLoop**: Mailbox-driven evaluation orchestration
- **EvalReport**: Aggregate results with metrics

Built-in evaluators:
- **exact_match**: Strict equality check
- **contains**: Substring presence check

Combinators:
- **all_of**: All evaluators must pass (mean score)
- **any_of**: At least one must pass (max score)

Session assertions:
- **session_has**: Assert slice has expected item count
- **session_latest**: Assert latest item matches predicate
- **session_contains**: Assert any item matches predicate
- **session_all**: Assert all items match predicate
- **all_session_assertions**: All assertions must pass (mean score)
- **any_session_assertions**: At least one must pass (max score)

LLM-as-Judge:
- **llm_judge**: Create evaluator using LLM to judge outputs
- **JudgeOutput/JudgeParams**: Structured types for judge prompt
- **Rating/RATING_VALUES/PASSING_RATINGS**: Rating scale constants

Example:
    >>> from weakincentives.evals import (
    ...     Dataset, EvalLoop, exact_match, submit_dataset, collect_results,
    ... )
    >>> dataset = Dataset.load(Path("qa.jsonl"), str, str)
    >>> eval_loop = EvalLoop(
    ...     loop=main_loop,
    ...     evaluator=exact_match,
    ...     requests=requests_mailbox,
    ...     results=results_mailbox,
    ... )
    >>> submit_dataset(dataset, requests_mailbox)
    >>> eval_loop.run(max_iterations=1)
    >>> report = collect_results(results_mailbox, expected_count=len(dataset))
    >>> print(f"Pass rate: {report.pass_rate:.1%}")
"""

from __future__ import annotations

from ._assertions import (
    SessionAssertion,
    all_session_assertions,
    any_session_assertions,
    session_all,
    session_contains,
    session_has,
    session_latest,
)
from ._evaluators import all_of, any_of, contains, exact_match
from ._helpers import collect_results, submit_dataset
from ._judge import (
    JUDGE_TEMPLATE,
    PASSING_RATINGS,
    RATING_VALUES,
    JudgeOutput,
    JudgeParams,
    Rating,
    llm_judge,
)
from ._loop import EvalLoop
from ._types import (
    Dataset,
    EvalReport,
    EvalRequest,
    EvalResult,
    Evaluator,
    Sample,
    Score,
)

__all__ = [  # noqa: RUF022
    "Dataset",
    "EvalLoop",
    "EvalReport",
    "EvalRequest",
    "EvalResult",
    "Evaluator",
    "JUDGE_TEMPLATE",
    "JudgeOutput",
    "JudgeParams",
    "PASSING_RATINGS",
    "RATING_VALUES",
    "Rating",
    "Sample",
    "Score",
    "SessionAssertion",
    "all_of",
    "all_session_assertions",
    "any_of",
    "any_session_assertions",
    "collect_results",
    "contains",
    "exact_match",
    "llm_judge",
    "session_all",
    "session_contains",
    "session_has",
    "session_latest",
    "submit_dataset",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
