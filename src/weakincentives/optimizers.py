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

"""Prompt optimization primitives.

Optimizers iterate on a :class:`Prompt` by running it against a
:class:`Dataset`, scoring every case with an :class:`Evaluator`, and
producing a candidate :class:`SectionOverride` set that improves the
overall score. This module ships small, stdlib-only strategies; richer
LLM-backed optimizers belong in extras built on top.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import final

from weakincentives.adapters import NoopAdapter
from weakincentives.core import MarkdownSection, Prompt, Section
from weakincentives.evals import (
    Dataset,
    EvalCase,
    EvalReport,
    Evaluator,
    PromptFactory,
    run_evaluation,
)
from weakincentives.overrides import (
    OverrideStore,
    SectionOverride,
    apply_overrides,
    describe_prompt,
)

__all__ = [
    "OptimizationOutcome",
    "OptimizationStep",
    "PromptOptimizer",
    "TemplateMutator",
    "ablate_section",
    "evaluate_overrides",
]


@final
@dataclass(frozen=True, slots=True)
class OptimizationStep:
    """A single ``before → after`` optimization comparison."""

    label: str
    overrides: tuple[SectionOverride, ...]
    report: EvalReport


@final
@dataclass(frozen=True, slots=True)
class OptimizationOutcome:
    """Summary of a full optimizer run."""

    baseline: EvalReport
    steps: tuple[OptimizationStep, ...]
    best_step: OptimizationStep | None

    @property
    def baseline_pass_rate(self) -> float:
        return self.baseline.pass_rate

    @property
    def best_pass_rate(self) -> float:
        return self.best_step.report.pass_rate if self.best_step else 0.0


# =============================================================================
# Helpers
# =============================================================================


TemplateMutator = Callable[[str], str]


def evaluate_overrides(
    prompt: Prompt,
    overrides: Sequence[SectionOverride],
    *,
    dataset: Dataset,
    adapter_for: Callable[[EvalCase], NoopAdapter],
    evaluator: Evaluator,
) -> EvalReport:
    """Apply ``overrides`` to ``prompt`` and run the dataset against it."""
    store = OverrideStore()
    for override in overrides:
        store.upsert(override)
    rebuilt = apply_overrides(prompt, store)

    def factory(case: EvalCase) -> Prompt:
        del case
        return rebuilt

    return run_evaluation(
        dataset=dataset,
        prompt_factory=factory,
        adapter_for=adapter_for,
        evaluator=evaluator,
    )


def ablate_section(
    prompt: Prompt,
    *,
    section_key: str,
    dataset: Dataset,
    adapter_for: Callable[[EvalCase], NoopAdapter],
    evaluator: Evaluator,
) -> EvalReport:
    """Replace the named section's body with empty text and re-evaluate.

    A useful "is this section pulling its weight?" check: if scores stay
    the same after blanking the section, the section is doing nothing.
    """
    descriptor = describe_prompt(prompt)
    target = descriptor.get(section_key)
    if target is None:
        raise ValueError(f"prompt has no section {section_key!r}")
    override = SectionOverride(
        section_key=section_key,
        template="",
        expected_hash=target.content_hash,
    )
    return evaluate_overrides(
        prompt,
        (override,),
        dataset=dataset,
        adapter_for=adapter_for,
        evaluator=evaluator,
    )


# =============================================================================
# Optimizer
# =============================================================================


@dataclass(slots=True)
class PromptOptimizer:
    """Try a list of :class:`TemplateMutator`s against one section.

    Each mutator receives the current section template and returns the
    next candidate. The optimizer runs the dataset against the baseline
    and against every mutated prompt, then surfaces the best-scoring
    candidate. Mutators that return the same template are skipped.
    """

    section_key: str
    mutators: tuple[tuple[str, TemplateMutator], ...]
    dataset: Dataset
    adapter_for: Callable[[EvalCase], NoopAdapter]
    evaluator: Evaluator

    def run(self, prompt: Prompt) -> OptimizationOutcome:
        baseline_factory: PromptFactory = lambda _case: prompt  # noqa: E731
        baseline = run_evaluation(
            dataset=self.dataset,
            prompt_factory=baseline_factory,
            adapter_for=self.adapter_for,
            evaluator=self.evaluator,
        )
        original_template = _section_template(prompt, self.section_key)
        # _section_template raised already if the section was missing; the
        # descriptor lookup below is guaranteed to find a match.
        descriptor = describe_prompt(prompt)
        target = descriptor.get(self.section_key)
        assert target is not None
        steps: list[OptimizationStep] = []
        for label, mutator in self.mutators:
            candidate = mutator(original_template)
            if candidate == original_template:
                continue
            override = SectionOverride(
                section_key=self.section_key,
                template=candidate,
                expected_hash=target.content_hash,
            )
            report = evaluate_overrides(
                prompt,
                (override,),
                dataset=self.dataset,
                adapter_for=self.adapter_for,
                evaluator=self.evaluator,
            )
            steps.append(
                OptimizationStep(label=label, overrides=(override,), report=report)
            )
        best = _pick_best(baseline, steps)
        return OptimizationOutcome(
            baseline=baseline, steps=tuple(steps), best_step=best
        )


def _section_template(prompt: Prompt, key: str) -> str:
    target = _find_section(prompt, key)
    if not isinstance(target, MarkdownSection):
        raise TypeError(f"section {key!r} is not a MarkdownSection (cannot mutate)")
    return target.template


def _find_section(prompt: Prompt, key: str) -> Section[object] | None:
    for section in prompt.sections:
        found = _walk_for(section, key)
        if found is not None:
            return found
    return None


def _walk_for(section: Section[object], key: str) -> Section[object] | None:
    if section.key == key:
        return section
    for child in section.children:
        found = _walk_for(child, key)
        if found is not None:
            return found
    return None


def _pick_best(
    baseline: EvalReport, steps: Sequence[OptimizationStep]
) -> OptimizationStep | None:
    best: OptimizationStep | None = None
    best_rate = baseline.pass_rate
    for step in steps:
        if step.report.pass_rate > best_rate:
            best_rate = step.report.pass_rate
            best = step
    return best
