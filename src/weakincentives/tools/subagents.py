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

"""Tooling for dispatching work to parallel subagents."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import SupportsIndex, cast, overload

from ..events import InProcessEventBus
from ..prompt._types import SupportsDataclass
from ..prompt.composition import (
    DelegationPrompt,
    DelegationSummaryParams,
    ParentPromptParams,
    RecapParams,
)
from ..prompt.prompt import Prompt, RenderedPrompt
from ..prompt.tool import Tool, ToolContext
from ..prompt.tool_result import ToolResult


@dataclass(slots=True)
class SubagentDispatch:
    """Single subagent execution request."""

    summary: DelegationSummaryParams
    recap_lines: tuple[str, ...]


@dataclass(slots=True)
class DispatchSubagentsParams:
    """Payload constructed by the model when dispatching subagents."""

    dispatches: Sequence[SubagentDispatch]


@dataclass(slots=True)
class SubagentResult(SupportsDataclass):
    """Outcome of an individual subagent execution."""

    dispatch: SubagentDispatch
    output: object | None
    success: bool
    error: str | None = None


@dataclass(slots=True)
class SubagentResults(SupportsDataclass):
    """Sequence wrapper for subagent outcomes."""

    items: tuple[SubagentResult, ...]

    def __iter__(self) -> Iterator[SubagentResult]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    @overload
    def __getitem__(self, index: SupportsIndex, /) -> SubagentResult: ...

    @overload
    def __getitem__(self, index: slice, /) -> tuple[SubagentResult, ...]: ...

    def __getitem__(
        self, index: SupportsIndex | slice, /
    ) -> SubagentResult | tuple[SubagentResult, ...]:
        return self.items[index]


@dataclass(slots=True)
class _DefaultParentOutput(SupportsDataclass):
    """Fallback parent output type when none is provided."""

    value: object | None = None


@dataclass(slots=True)
class _DefaultDelegationOutput(SupportsDataclass):
    """Fallback delegation output type used for child prompts."""

    value: object | None = None


@dataclass(slots=True)
class _DispatchWork:
    dispatch: SubagentDispatch
    bus: InProcessEventBus
    session: object | None


def _default_max_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(32, cpu_count + 4)


def _normalize_dispatch(dispatch: SubagentDispatch) -> SubagentDispatch:
    return SubagentDispatch(
        summary=dispatch.summary,
        recap_lines=tuple(dispatch.recap_lines),
    )


def _prepare_work_items(
    dispatches: Sequence[SubagentDispatch],
    *,
    context: ToolContext,
) -> tuple[_DispatchWork, ...]:
    session = context.session
    work_items: list[_DispatchWork] = []
    for dispatch in dispatches:
        normalized = _normalize_dispatch(dispatch)
        bus = InProcessEventBus()
        clone: object | None = None
        if session is not None:
            clone_method = getattr(session, "clone", None)
            if clone_method is None:
                message = "Session does not support cloning."
                raise RuntimeError(message)
            clone = clone_method(bus=bus)
        work_items.append(_DispatchWork(dispatch=normalized, bus=bus, session=clone))
    return tuple(work_items)


def _build_delegation_prompt(
    *,
    context: ToolContext,
    rendered_parent: RenderedPrompt[_DefaultParentOutput],
    recap_lines: tuple[str, ...],
) -> DelegationPrompt[_DefaultParentOutput, _DefaultDelegationOutput]:
    prompt_type = DelegationPrompt[_DefaultParentOutput, _DefaultDelegationOutput]
    include_response_format = rendered_parent.container is not None
    parent_prompt = cast(Prompt[_DefaultParentOutput], context.prompt)
    return prompt_type(
        parent_prompt=parent_prompt,
        rendered_parent=rendered_parent,
        include_response_format=include_response_format,
        recap_lines=recap_lines,
    )


def _evaluate_dispatch(
    work: _DispatchWork,
    *,
    context: ToolContext,
    rendered_parent: RenderedPrompt[_DefaultParentOutput],
) -> SubagentResult:
    prompt_wrapper = _build_delegation_prompt(
        context=context,
        rendered_parent=rendered_parent,
        recap_lines=work.dispatch.recap_lines,
    )
    try:
        response = context.adapter.evaluate(
            prompt_wrapper.prompt,
            work.dispatch.summary,
            ParentPromptParams(body=rendered_parent.text),
            RecapParams(bullets=work.dispatch.recap_lines),
            bus=work.bus,
            session=work.session,
        )
    except Exception as error:
        return SubagentResult(
            dispatch=work.dispatch,
            output=None,
            success=False,
            error=str(error),
        )
    return SubagentResult(
        dispatch=work.dispatch,
        output=response.output,
        success=True,
    )


def _execute_dispatches(
    work_items: Sequence[_DispatchWork],
    *,
    context: ToolContext,
    rendered_parent: RenderedPrompt[_DefaultParentOutput],
) -> tuple[SubagentResult, ...]:
    if not work_items:
        return ()

    if len(work_items) == 1:
        return (
            _evaluate_dispatch(
                work_items[0], context=context, rendered_parent=rendered_parent
            ),
        )

    max_workers = min(len(work_items), _default_max_workers())
    results: list[SubagentResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _evaluate_dispatch,
                work,
                context=context,
                rendered_parent=rendered_parent,
            )
            for work in work_items
        ]

        for work, future in zip(work_items, futures, strict=True):
            try:
                result = future.result()
            except Exception as error:  # pragma: no cover - defensive path
                results.append(
                    SubagentResult(
                        dispatch=work.dispatch,
                        output=None,
                        success=False,
                        error=str(error),
                    )
                )
            else:
                results.append(result)

    return tuple(results)


def _handle_dispatch_subagents(
    params: DispatchSubagentsParams,
    *,
    context: ToolContext,
) -> ToolResult[SubagentResults]:
    rendered_parent = context.rendered_prompt
    if rendered_parent is None:
        return ToolResult(
            message="dispatch_subagents requires a rendered parent prompt.",
            value=None,
            success=False,
        )

    try:
        work_items = _prepare_work_items(tuple(params.dispatches), context=context)
    except Exception as error:  # pragma: no cover - defensive path
        return ToolResult(message=str(error), value=None, success=False)

    typed_parent = cast(RenderedPrompt[_DefaultParentOutput], rendered_parent)
    results = _execute_dispatches(
        work_items,
        context=context,
        rendered_parent=typed_parent,
    )
    count = len(results)
    suffix = "s" if count != 1 else ""
    message = f"Dispatched {count} subagent{suffix}."
    payload = SubagentResults(items=results)
    return ToolResult(value=payload, message=message, success=True)


dispatch_subagents = Tool[
    DispatchSubagentsParams,
    SubagentResults,
](
    name="dispatch_subagents",
    description="Dispatch parallel subagents with recap checkpoints.",
    handler=_handle_dispatch_subagents,
    accepts_overrides=False,
)


__all__ = [
    "DispatchSubagentsParams",
    "SubagentDispatch",
    "SubagentResult",
    "dispatch_subagents",
]
