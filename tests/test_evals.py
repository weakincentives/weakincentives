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

"""Tests for ``weakincentives.evals``."""

from __future__ import annotations

from typing import cast

from weakincentives.adapters import NoopAdapter, ScriptedResponse
from weakincentives.core import (
    MarkdownSection,
    Prompt,
    Tool,
    ToolCall,
    ToolContext,
    ToolResult,
)
from weakincentives.evals import (
    AllToolsSucceeded,
    Contains,
    Dataset,
    EvalCase,
    EvalReport,
    EvalScore,
    Evaluator,
    ExactMatch,
    ToolCalled,
    build_noop_factory,
    run_evaluation,
)


def _echo(params: object, context: ToolContext) -> ToolResult[str]:
    del context
    if isinstance(params, dict):
        text = cast("dict[str, object]", params).get("text", "")
    else:
        text = ""
    return ToolResult.ok(str(text))


def _make_prompt() -> Prompt:
    section = MarkdownSection[None](
        title="T",
        key="t",
        tools=(Tool(name="echo", description="echo", handler=_echo),),
    )
    return Prompt(ns="ns", key="k", sections=(section,))


def test_dataset_supports_iter_and_len() -> None:
    dataset = Dataset(
        cases=(
            EvalCase(name="a"),
            EvalCase(name="b"),
        )
    )
    assert len(dataset) == 2
    names = [c.name for c in dataset]
    assert names == ["a", "b"]


def test_eval_report_metrics() -> None:
    report = EvalReport(
        cases=(EvalCase(name="x"), EvalCase(name="y")),
        scores=(
            EvalScore(passed=True),
            EvalScore(passed=False),
        ),
    )
    assert report.passed == 1
    assert report.total == 2
    assert report.pass_rate == 0.5


def test_eval_report_pass_rate_empty_dataset() -> None:
    report = EvalReport(cases=(), scores=())
    assert report.pass_rate == 0.0


def test_exact_match_evaluator() -> None:
    evaluator = ExactMatch()
    assert isinstance(evaluator, Evaluator)
    dataset = Dataset(cases=(EvalCase(name="a", expected="hello"),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(_make_prompt()),
        adapter_for=lambda _: NoopAdapter(
            responses=(ScriptedResponse(text="hello", finish_reason="stop"),)
        ),
        evaluator=evaluator,
    )
    assert report.passed == 1


def test_exact_match_failure_records_detail() -> None:
    evaluator = ExactMatch()
    dataset = Dataset(cases=(EvalCase(name="a", expected="goodbye"),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(_make_prompt()),
        adapter_for=lambda _: NoopAdapter(
            responses=(ScriptedResponse(text="hello", finish_reason="stop"),)
        ),
        evaluator=evaluator,
    )
    assert report.scores[0].passed is False
    assert "expected='goodbye'" in report.scores[0].detail


def test_contains_evaluator() -> None:
    evaluator = Contains()
    dataset = Dataset(cases=(EvalCase(name="a", expected="quick"),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(_make_prompt()),
        adapter_for=lambda _: NoopAdapter(
            responses=(
                ScriptedResponse(text="the quick brown fox", finish_reason="stop"),
            )
        ),
        evaluator=evaluator,
    )
    assert report.passed == 1


def test_contains_evaluator_rejects_non_string_expected() -> None:
    evaluator = Contains()
    dataset = Dataset(cases=(EvalCase(name="a", expected=123),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(_make_prompt()),
        adapter_for=lambda _: NoopAdapter(
            responses=(ScriptedResponse(text="hi", finish_reason="stop"),)
        ),
        evaluator=evaluator,
    )
    assert report.scores[0].passed is False


def test_tool_called_evaluator() -> None:
    evaluator = ToolCalled(tool_name="echo")
    dataset = Dataset(cases=(EvalCase(name="a"),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(_make_prompt()),
        adapter_for=lambda _: NoopAdapter(
            responses=(
                ScriptedResponse(
                    text="thinking",
                    tool_calls=(ToolCall(name="echo", arguments={"text": "hi"}),),
                ),
                ScriptedResponse(text="ok", finish_reason="stop"),
            )
        ),
        evaluator=evaluator,
    )
    assert report.passed == 1


def test_tool_called_evaluator_skips_unrelated_tool_invocations() -> None:
    """Cover the branch where the loop sees a different tool's invocation."""
    section = MarkdownSection[None](
        title="T",
        key="t",
        tools=(
            Tool(name="echo", description="echo", handler=_echo),
            Tool(name="other", description="o", handler=_echo),
        ),
    )
    prompt = Prompt(ns="ns", key="k", sections=(section,))
    evaluator = ToolCalled(tool_name="missing")
    dataset = Dataset(cases=(EvalCase(name="a"),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(prompt),
        adapter_for=lambda _: NoopAdapter(
            responses=(
                ScriptedResponse(
                    text="thinking",
                    tool_calls=(
                        ToolCall(name="echo", arguments={"text": "x"}),
                        ToolCall(name="other", arguments={"text": "y"}),
                    ),
                ),
                ScriptedResponse(text="ok", finish_reason="stop"),
            )
        ),
        evaluator=evaluator,
    )
    assert report.scores[0].passed is False


def test_tool_called_evaluator_misses_when_tool_not_invoked() -> None:
    evaluator = ToolCalled(tool_name="echo")
    dataset = Dataset(cases=(EvalCase(name="a"),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(_make_prompt()),
        adapter_for=lambda _: NoopAdapter(
            responses=(ScriptedResponse(text="silent", finish_reason="stop"),)
        ),
        evaluator=evaluator,
    )
    assert report.scores[0].passed is False


def test_all_tools_succeeded_with_no_calls() -> None:
    evaluator = AllToolsSucceeded()
    dataset = Dataset(cases=(EvalCase(name="a"),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(_make_prompt()),
        adapter_for=lambda _: NoopAdapter(
            responses=(ScriptedResponse(text="silent", finish_reason="stop"),)
        ),
        evaluator=evaluator,
    )
    assert report.passed == 1


def test_all_tools_succeeded_when_all_pass() -> None:
    evaluator = AllToolsSucceeded()
    dataset = Dataset(cases=(EvalCase(name="a"),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(_make_prompt()),
        adapter_for=lambda _: NoopAdapter(
            responses=(
                ScriptedResponse(
                    text="thinking",
                    tool_calls=(ToolCall(name="echo", arguments={"text": "hi"}),),
                ),
                ScriptedResponse(text="ok", finish_reason="stop"),
            )
        ),
        evaluator=evaluator,
    )
    assert report.passed == 1


def test_all_tools_succeeded_records_failures() -> None:
    def failing(params: object, context: ToolContext) -> ToolResult[None]:
        del params, context
        return ToolResult.error("nope")

    section = MarkdownSection[None](
        title="T",
        key="t",
        tools=(Tool(name="boom", description="fails", handler=failing),),
    )
    prompt = Prompt(ns="ns", key="k", sections=(section,))

    evaluator = AllToolsSucceeded()
    dataset = Dataset(cases=(EvalCase(name="a"),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(prompt),
        adapter_for=lambda _: NoopAdapter(
            responses=(
                ScriptedResponse(
                    text="thinking",
                    tool_calls=(ToolCall(name="boom", arguments={}),),
                ),
                ScriptedResponse(text="ok", finish_reason="stop"),
            )
        ),
        evaluator=evaluator,
    )
    assert report.scores[0].passed is False
    assert "tool failures" in report.scores[0].detail


def test_eval_score_satisfies_protocol_runtime_check() -> None:
    assert isinstance(ToolCalled(tool_name="echo"), Evaluator)


# ---------------------------------------------------------------------------
# RegexMatch
# ---------------------------------------------------------------------------


def test_regex_match_passes_when_pattern_found() -> None:
    from weakincentives.evals import RegexMatch

    evaluator = RegexMatch()
    dataset = Dataset(cases=(EvalCase(name="a", expected=r"\b42\b"),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(_make_prompt()),
        adapter_for=lambda _: NoopAdapter(
            responses=(
                ScriptedResponse(text="answer is 42 done", finish_reason="stop"),
            )
        ),
        evaluator=evaluator,
    )
    assert report.passed == 1


def test_regex_match_fails_for_non_string_expected() -> None:
    from weakincentives.evals import RegexMatch

    evaluator = RegexMatch()
    dataset = Dataset(cases=(EvalCase(name="a", expected=123),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(_make_prompt()),
        adapter_for=lambda _: NoopAdapter(
            responses=(ScriptedResponse(text="x", finish_reason="stop"),)
        ),
        evaluator=evaluator,
    )
    assert report.scores[0].passed is False


def test_regex_match_fails_when_pattern_missing() -> None:
    from weakincentives.evals import RegexMatch

    evaluator = RegexMatch()
    dataset = Dataset(cases=(EvalCase(name="a", expected=r"^missing$"),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(_make_prompt()),
        adapter_for=lambda _: NoopAdapter(
            responses=(ScriptedResponse(text="hello", finish_reason="stop"),)
        ),
        evaluator=evaluator,
    )
    assert report.scores[0].passed is False


def test_llm_judge_delegates_to_protocol() -> None:
    from weakincentives.evals import EvalScore, LlmJudge
    from weakincentives.transcript import Transcript

    class TruthyJudge:
        def evaluate(
            self,
            *,
            case: EvalCase,
            response: str,
            transcript: Transcript,
        ) -> EvalScore:
            del case, response, transcript
            return EvalScore(passed=True, score=0.9, detail="judge-says-yes")

    evaluator = LlmJudge(judge=TruthyJudge())
    dataset = Dataset(cases=(EvalCase(name="a"),))
    report = run_evaluation(
        dataset=dataset,
        prompt_factory=build_noop_factory(_make_prompt()),
        adapter_for=lambda _: NoopAdapter(
            responses=(ScriptedResponse(text="hello", finish_reason="stop"),)
        ),
        evaluator=evaluator,
    )
    assert report.passed == 1
    assert report.scores[0].detail == "judge-says-yes"


def test_judge_protocol_is_runtime_checkable() -> None:
    from weakincentives.evals import EvalScore, JudgeProtocol
    from weakincentives.transcript import Transcript

    class Anon:
        def evaluate(
            self,
            *,
            case: EvalCase,
            response: str,
            transcript: Transcript,
        ) -> EvalScore:
            del case, response, transcript
            return EvalScore(passed=False)

    assert isinstance(Anon(), JudgeProtocol)
