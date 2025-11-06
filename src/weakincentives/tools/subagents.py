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

"""Tooling for dispatching subagents in parallel."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, is_dataclass
from enum import Enum, auto
from typing import Any, Final, cast

from ..adapters.core import PromptResponse
from ..events import EventBus, InProcessEventBus
from ..prompt.composition import DelegationParams, DelegationPrompt, RecapParams
from ..prompt.prompt import Prompt, RenderedPrompt
from ..prompt.tool import Tool, ToolContext
from ..prompt.tool_result import ToolResult
from ..serde import dump
from ..session.protocols import SessionProtocol


def _default_max_workers() -> int:
    with ThreadPoolExecutor() as executor:
        return executor._max_workers


_DEFAULT_MAX_WORKERS: Final[int] = _default_max_workers()


@dataclass(slots=True)
class DispatchSubagentsParams:
    """Parameters describing the delegations to execute."""

    delegations: tuple[DelegationParams, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.delegations = tuple(self.delegations)


@dataclass(slots=True)
class SubagentResult:
    """Outcome captured for an individual delegation."""

    output: str
    success: bool
    error: str | None = None


class SubagentIsolationLevel(Enum):
    """Enumerate the isolation guarantees offered to child runs."""

    NO_ISOLATION = auto()
    FULL_ISOLATION = auto()


@dataclass(slots=True)
class _ChildRuntime:
    """Execution context provisioned for a child delegation."""

    session: SessionProtocol
    bus: EventBus


def _extract_output_text(response: PromptResponse[Any]) -> str:
    if response.text:
        return response.text
    if response.output is not None:
        try:
            rendered = dump(response.output, exclude_none=True)
            return json.dumps(rendered, ensure_ascii=False)
        except TypeError:
            return str(response.output)
    return ""


def _build_error(message: str) -> str:
    cleaned = message.strip()
    return cleaned or "Subagent execution failed"


def _prepare_child_runtimes(
    *,
    context: ToolContext,
    count: int,
    isolation_level: SubagentIsolationLevel,
) -> tuple[_ChildRuntime, ...]:
    if isolation_level is SubagentIsolationLevel.NO_ISOLATION:
        return tuple(
            _ChildRuntime(session=context.session, bus=context.event_bus)
            for _ in range(count)
        )

    runtimes: list[_ChildRuntime] = []
    for _ in range(count):
        bus = InProcessEventBus()
        try:
            session_clone = context.session.clone(bus=bus)
        except Exception as error:  # pragma: no cover - defensive
            raise RuntimeError(
                "dispatch_subagents could not clone the session for full isolation"
            ) from error
        runtimes.append(_ChildRuntime(session=session_clone, bus=bus))
    return tuple(runtimes)


def _dispatch_subagents(
    params: DispatchSubagentsParams,
    *,
    context: ToolContext,
    isolation_level: SubagentIsolationLevel,
) -> ToolResult[tuple[SubagentResult, ...]]:
    rendered_parent = cast(RenderedPrompt[Any] | None, context.rendered_prompt)
    if rendered_parent is None:
        return ToolResult(
            message="dispatch_subagents requires the parent prompt to be rendered.",
            value=None,
            success=False,
        )

    parent_prompt = cast(Prompt[Any], context.prompt)
    parent_output_type = rendered_parent.output_type
    if not isinstance(parent_output_type, type) or not is_dataclass(parent_output_type):
        return ToolResult(
            message="Parent prompt must declare a dataclass output type for delegation.",
            value=None,
            success=False,
        )

    delegation_prompt_cls: type[DelegationPrompt[Any, Any]] = (
        DelegationPrompt.__class_getitem__((parent_output_type, parent_output_type))
    )
    delegation_prompt = delegation_prompt_cls(
        parent_prompt,
        rendered_parent,
        include_response_format=rendered_parent.container is not None,
    )

    delegations = tuple(params.delegations)
    if not delegations:
        empty_results = cast(tuple[SubagentResult, ...], ())
        return ToolResult(
            message="No delegations supplied.",
            value=empty_results,
        )

    adapter = context.adapter
    parse_output = rendered_parent.container is not None

    try:
        runtimes = _prepare_child_runtimes(
            context=context,
            count=len(delegations),
            isolation_level=isolation_level,
        )
    except RuntimeError as error:
        return ToolResult(
            message=_build_error(str(error)),
            value=None,
            success=False,
        )

    def _run_child(
        delegation: DelegationParams, runtime: _ChildRuntime
    ) -> SubagentResult:
        recap = RecapParams(bullets=delegation.recap_lines)
        try:
            response = adapter.evaluate(
                delegation_prompt.prompt,
                delegation,
                recap,
                parse_output=parse_output,
                bus=runtime.bus,
                session=runtime.session,
            )
        except Exception as error:  # pragma: no cover - defensive
            return SubagentResult(
                output="",
                success=False,
                error=_build_error(str(error)),
            )
        return SubagentResult(
            output=_extract_output_text(response),
            success=True,
        )

    max_workers = min(len(delegations), _DEFAULT_MAX_WORKERS) or 1
    results: list[SubagentResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_child, delegation, runtime)
            for delegation, runtime in zip(delegations, runtimes, strict=True)
        ]
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:  # pragma: no cover - defensive
                results.append(
                    SubagentResult(
                        output="",
                        success=False,
                        error=_build_error(str(error)),
                    )
                )

    return ToolResult(
        message=f"Dispatched {len(results)} subagents.",
        value=tuple(results),
    )


def build_dispatch_subagents_tool(
    *, isolation_level: SubagentIsolationLevel = SubagentIsolationLevel.NO_ISOLATION
) -> Tool[DispatchSubagentsParams, tuple[SubagentResult, ...]]:
    """Return a ``dispatch_subagents`` tool configured for the requested isolation."""

    def _handler(
        params: DispatchSubagentsParams, *, context: ToolContext
    ) -> ToolResult[tuple[SubagentResult, ...]]:
        return _dispatch_subagents(
            params,
            context=context,
            isolation_level=isolation_level,
        )

    return Tool[DispatchSubagentsParams, tuple[SubagentResult, ...]](
        name="dispatch_subagents",
        description="Run delegated child prompts in parallel.",
        handler=_handler,
        accepts_overrides=False,
    )


dispatch_subagents = build_dispatch_subagents_tool()


__all__ = [
    "DispatchSubagentsParams",
    "SubagentIsolationLevel",
    "SubagentResult",
    "build_dispatch_subagents_tool",
    "dispatch_subagents",
]
