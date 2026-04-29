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

"""Evaluation framework for agent behaviour.

Build a :class:`Dataset` of :class:`EvalCase` values, declare an
:class:`Evaluator` that scores a single case against the agent loop's
output, and call :func:`run_evaluation` to produce a :class:`EvalReport`.
The framework is deliberately small: it composes with the existing
``runtime`` and ``transcript`` extras.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol, final, runtime_checkable

from weakincentives.adapters import NoopAdapter
from weakincentives.core import Prompt, Session, ToolInvoked
from weakincentives.runtime import AgentLoop, AgentLoopResult
from weakincentives.transcript import Transcript, TranscriptListener

__all__ = [
    "AllToolsSucceeded",
    "Contains",
    "Dataset",
    "EvalCase",
    "EvalReport",
    "EvalScore",
    "Evaluator",
    "ExactMatch",
    "JudgeProtocol",
    "LlmJudge",
    "PromptFactory",
    "RegexMatch",
    "ToolCalled",
    "build_noop_factory",
    "run_evaluation",
]


# =============================================================================
# Dataset / case types
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class EvalCase:
    """One row in an evaluation dataset."""

    name: str
    inputs: Mapping[str, object] = field(default_factory=lambda: {})
    expected: object | None = None
    metadata: Mapping[str, object] = field(default_factory=lambda: {})


@final
@dataclass(frozen=True, slots=True)
class Dataset:
    """An ordered collection of :class:`EvalCase` values."""

    cases: tuple[EvalCase, ...]

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)


# =============================================================================
# Evaluator protocol + result types
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class EvalScore:
    """Outcome of scoring a single case."""

    passed: bool
    score: float = 0.0
    detail: str = ""


@final
@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate of :class:`EvalScore` values across a dataset."""

    cases: tuple[EvalCase, ...]
    scores: tuple[EvalScore, ...]

    @property
    def passed(self) -> int:
        return sum(1 for score in self.scores if score.passed)

    @property
    def total(self) -> int:
        return len(self.scores)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@runtime_checkable
class Evaluator(Protocol):
    """Score a single :class:`EvalCase` against an agent run."""

    def score(
        self, case: EvalCase, result: AgentLoopResult, transcript: Transcript
    ) -> EvalScore: ...


# =============================================================================
# Built-in evaluators
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class ExactMatch:
    """Pass when ``result.final_response.text`` equals ``case.expected``."""

    def score(
        self, case: EvalCase, result: AgentLoopResult, transcript: Transcript
    ) -> EvalScore:
        del transcript
        actual = result.final_response.text
        expected = case.expected
        ok = actual == expected
        return EvalScore(
            passed=ok,
            score=1.0 if ok else 0.0,
            detail=f"actual={actual!r} expected={expected!r}",
        )


@final
@dataclass(frozen=True, slots=True)
class Contains:
    """Pass when ``case.expected`` (a string) is a substring of the response."""

    def score(
        self, case: EvalCase, result: AgentLoopResult, transcript: Transcript
    ) -> EvalScore:
        del transcript
        expected = case.expected
        if not isinstance(expected, str):
            return EvalScore(passed=False, detail="expected must be a string")
        ok = expected in result.final_response.text
        return EvalScore(passed=ok, score=1.0 if ok else 0.0)


@final
@dataclass(frozen=True, slots=True)
class ToolCalled:
    """Pass when ``tool_name`` was invoked at least once."""

    tool_name: str

    def score(
        self,
        case: EvalCase,
        result: AgentLoopResult,
        transcript: Transcript,
    ) -> EvalScore:
        del case, result
        for entry in transcript.entries():
            if entry.kind != ToolInvoked.__name__:
                continue
            if entry.payload.get("name") == self.tool_name:
                return EvalScore(passed=True, score=1.0)
        return EvalScore(
            passed=False,
            detail=f"tool {self.tool_name!r} was not invoked",
        )


@final
@dataclass(frozen=True, slots=True)
class AllToolsSucceeded:
    """Pass when every tool call in the run reported success."""

    def score(
        self,
        case: EvalCase,
        result: AgentLoopResult,
        transcript: Transcript,
    ) -> EvalScore:
        del case, transcript
        if not result.tool_results:
            return EvalScore(passed=True, score=1.0, detail="no tool calls")
        failures = [r for r in result.tool_results if not r.success]
        ok = not failures
        return EvalScore(
            passed=ok,
            score=1.0 if ok else 0.0,
            detail=f"{len(failures)} tool failures",
        )


@final
@dataclass(frozen=True, slots=True)
class RegexMatch:
    """Pass when ``result.final_response.text`` matches a regex.

    The regex comes from ``case.expected``; cases that don't supply a
    string fail the score with a clear detail message.
    """

    flags: int = 0

    def score(
        self,
        case: EvalCase,
        result: AgentLoopResult,
        transcript: Transcript,
    ) -> EvalScore:
        del transcript
        pattern = case.expected
        if not isinstance(pattern, str):
            return EvalScore(passed=False, detail="expected must be a regex string")
        match = re.search(pattern, result.final_response.text, self.flags)
        return EvalScore(
            passed=match is not None,
            score=1.0 if match is not None else 0.0,
            detail=f"pattern={pattern!r}",
        )


@runtime_checkable
class JudgeProtocol(Protocol):
    """Synchronous LLM-judge facade used by :class:`LlmJudge`."""

    def evaluate(
        self,
        *,
        case: EvalCase,
        response: str,
        transcript: Transcript,
    ) -> EvalScore: ...


@final
@dataclass(frozen=True, slots=True)
class LlmJudge:
    """Delegate scoring to a :class:`JudgeProtocol` implementation.

    Tests typically supply a callable wrapped in a tiny class; production
    callers wire in a real model-backed judge.
    """

    judge: JudgeProtocol

    def score(
        self,
        case: EvalCase,
        result: AgentLoopResult,
        transcript: Transcript,
    ) -> EvalScore:
        return self.judge.evaluate(
            case=case,
            response=result.final_response.text,
            transcript=transcript,
        )


# =============================================================================
# Runner
# =============================================================================


PromptFactory = Callable[[EvalCase], Prompt]


def build_noop_factory(prompt: Prompt) -> PromptFactory:
    """Return a :class:`PromptFactory` that ignores the case and returns ``prompt``."""

    def factory(case: EvalCase) -> Prompt:
        del case
        return prompt

    return factory


def run_evaluation(
    *,
    dataset: Dataset,
    prompt_factory: PromptFactory,
    adapter_for: Callable[[EvalCase], NoopAdapter],
    evaluator: Evaluator,
) -> EvalReport:
    """Run ``adapter_for(case)`` for every case and score the result.

    Each case gets its own :class:`Session` and :class:`Transcript`, so the
    evaluator sees only events from the case under test. The test author
    builds an adapter (typically a scripted :class:`NoopAdapter`) per case
    so the framework stays adapter-agnostic.
    """
    scores: list[EvalScore] = []
    for case in dataset.cases:
        session = Session()
        transcript = Transcript()
        listener = TranscriptListener(transcript=transcript)
        session.subscribe(listener)
        adapter = adapter_for(case)
        loop = AgentLoop(adapter=adapter, session=session)
        prompt = prompt_factory(case)
        result = loop.run(prompt)
        scores.append(evaluator.score(case, result, transcript))
    return EvalReport(cases=dataset.cases, scores=tuple(scores))
