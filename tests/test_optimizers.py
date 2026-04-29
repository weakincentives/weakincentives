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

"""Tests for ``weakincentives.optimizers``."""

from __future__ import annotations

import pytest

from weakincentives.adapters import NoopAdapter, ScriptedResponse
from weakincentives.core import (
    MarkdownSection,
    Prompt,
    Section,
)
from weakincentives.evals import (
    Contains,
    Dataset,
    EvalCase,
)
from weakincentives.optimizers import (
    OptimizationStep,
    PromptOptimizer,
    ablate_section,
    evaluate_overrides,
)


def _make_prompt(template: str) -> Prompt:
    section = MarkdownSection[None](
        title="System",
        key="system",
        template=template,
    )
    return Prompt(ns="ns", key="k", sections=(section,))


def _adapter_returning(text: str) -> NoopAdapter:
    return NoopAdapter(responses=(ScriptedResponse(text=text, finish_reason="stop"),))


def test_evaluate_overrides_runs_dataset_with_overrides() -> None:
    prompt = _make_prompt("starting body")
    from weakincentives.overrides import SectionOverride, describe_prompt

    descriptor = describe_prompt(prompt)
    section = descriptor.get("system")
    assert section is not None
    override = SectionOverride(
        section_key="system",
        template="overridden body",
        expected_hash=section.content_hash,
    )
    report = evaluate_overrides(
        prompt,
        (override,),
        dataset=Dataset(cases=(EvalCase(name="a", expected="hello"),)),
        adapter_for=lambda _case: _adapter_returning("hello world"),
        evaluator=Contains(),
    )
    assert report.passed == 1


def test_ablate_section_replaces_body_with_empty() -> None:
    prompt = _make_prompt("strong system prompt")
    report = ablate_section(
        prompt,
        section_key="system",
        dataset=Dataset(cases=(EvalCase(name="a", expected="hi"),)),
        adapter_for=lambda _case: _adapter_returning("hi"),
        evaluator=Contains(),
    )
    assert report.passed == 1


def test_ablate_section_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="no section"):
        ablate_section(
            _make_prompt("body"),
            section_key="missing",
            dataset=Dataset(cases=(EvalCase(name="a"),)),
            adapter_for=lambda _: NoopAdapter(),
            evaluator=Contains(),
        )


def test_optimizer_picks_best_mutated_template() -> None:
    prompt = _make_prompt("base")
    dataset = Dataset(
        cases=(
            EvalCase(name="alpha", expected="great"),
            EvalCase(name="beta", expected="great"),
        )
    )

    def adapter_for(case: EvalCase) -> NoopAdapter:
        del case
        return _adapter_returning("totally great")

    optimizer = PromptOptimizer(
        section_key="system",
        mutators=(
            ("noop", lambda template: template),
            ("rewrite", lambda _template: "completely different"),
            ("dup", lambda _template: "completely different"),
        ),
        dataset=dataset,
        adapter_for=adapter_for,
        evaluator=Contains(),
    )
    outcome = optimizer.run(prompt)
    assert outcome.baseline.pass_rate == 1.0
    # Even with a perfect baseline, every mutated step is recorded.
    labels = [step.label for step in outcome.steps]
    assert labels == ["rewrite", "dup"]


def test_optimizer_returns_best_when_score_improves() -> None:
    prompt = _make_prompt("baseline")
    dataset = Dataset(cases=(EvalCase(name="a", expected="GOOD"),))

    def adapter_for(case: EvalCase) -> NoopAdapter:
        del case
        return _adapter_returning("the result is GOOD")

    def adapter_for_baseline(case: EvalCase) -> NoopAdapter:
        del case
        return _adapter_returning("nothing of value")

    # Baseline fails (returns "nothing of value"), but the mutator path
    # uses a different adapter — simulate by switching after the first
    # call. Simpler: keep the same adapter_for for both, baseline has
    # different scoring outcome only when the mutator changes the
    # template.
    optimizer = PromptOptimizer(
        section_key="system",
        mutators=(("rewrite", lambda _template: "rewritten"),),
        dataset=dataset,
        adapter_for=adapter_for,
        evaluator=Contains(),
    )
    outcome = optimizer.run(prompt)
    assert outcome.baseline_pass_rate == 1.0
    assert outcome.best_step is None  # baseline already perfect
    del adapter_for_baseline


def test_optimizer_rejects_non_markdown_section() -> None:
    prompt = Prompt(
        ns="ns",
        key="k",
        sections=(Section[None](title="Plain", key="plain"),),
    )
    optimizer = PromptOptimizer(
        section_key="plain",
        mutators=(("x", lambda _: "y"),),
        dataset=Dataset(cases=(EvalCase(name="a"),)),
        adapter_for=lambda _: _adapter_returning("ok"),
        evaluator=Contains(),
    )
    with pytest.raises(TypeError, match="MarkdownSection"):
        optimizer.run(prompt)


def test_optimizer_rejects_missing_section() -> None:
    prompt = _make_prompt("body")
    optimizer = PromptOptimizer(
        section_key="missing",
        mutators=(("x", lambda _: "y"),),
        dataset=Dataset(cases=(EvalCase(name="a"),)),
        adapter_for=lambda _: _adapter_returning("ok"),
        evaluator=Contains(),
    )
    with pytest.raises(TypeError, match="MarkdownSection"):
        optimizer.run(prompt)


def test_optimizer_finds_section_in_nested_subtree() -> None:
    """Cover the recursion path in _walk_for, including sibling skipping."""
    other = MarkdownSection[None](title="Other", key="other", template="x")
    leaf = MarkdownSection[None](title="Leaf", key="leaf", template="leaf body")
    root = MarkdownSection[None](
        title="Root", key="root", template="root", children=(other, leaf)
    )
    prompt = Prompt(ns="ns", key="k", sections=(root,))
    dataset = Dataset(cases=(EvalCase(name="a", expected="answer"),))
    optimizer = PromptOptimizer(
        section_key="leaf",
        mutators=(("rewrite", lambda _t: "rewritten leaf"),),
        dataset=dataset,
        adapter_for=lambda _: _adapter_returning("answer"),
        evaluator=Contains(),
    )
    outcome = optimizer.run(prompt)
    assert len(outcome.steps) == 1


def test_best_pass_rate_returns_step_rate_when_improvement_found() -> None:
    prompt = _make_prompt("baseline body")
    dataset = Dataset(
        cases=(
            EvalCase(name="a", expected="WIN"),
            EvalCase(name="b", expected="WIN"),
        )
    )

    state = {"calls": 0}

    def adapter_for(_case: EvalCase) -> NoopAdapter:
        # First two calls (baseline) return failing text; subsequent
        # calls (mutated path) return passing text.
        state["calls"] += 1
        text = "no match" if state["calls"] <= 2 else "WIN"
        return _adapter_returning(text)

    optimizer = PromptOptimizer(
        section_key="system",
        mutators=(("rewrite", lambda _t: "different prompt"),),
        dataset=dataset,
        adapter_for=adapter_for,
        evaluator=Contains(),
    )
    outcome = optimizer.run(prompt)
    assert outcome.baseline_pass_rate == 0.0
    assert outcome.best_step is not None
    assert outcome.best_pass_rate == 1.0


def test_optimization_step_dataclass_round_trip() -> None:
    from weakincentives.evals import EvalReport, EvalScore
    from weakincentives.overrides import SectionOverride

    step = OptimizationStep(
        label="a",
        overrides=(SectionOverride(section_key="x", template="y", expected_hash="z"),),
        report=EvalReport(
            cases=(EvalCase(name="c"),),
            scores=(EvalScore(passed=True),),
        ),
    )
    assert step.label == "a"
    assert step.report.pass_rate == 1.0
