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
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, is_dataclass
from enum import Enum, auto
from typing import Any, Final, cast, override

from ..adapters.core import PromptResponse
from ..prompt.composition import DelegationParams, DelegationPrompt, RecapParams
from ..prompt.errors import PromptRenderError
from ..prompt.prompt import Prompt, RenderedPrompt
from ..prompt.section import Section
from ..prompt.tool import Tool, ToolContext
from ..prompt.tool_result import ToolResult
from ..runtime.events import InProcessEventBus
from ..runtime.events._types import EventBus
from ..runtime.session.protocols import SessionProtocol
from ..serde import dump


class SubagentIsolationLevel(Enum):
    """Isolation modes describing how children interact with parent state."""

    NO_ISOLATION = auto()
    FULL_ISOLATION = auto()


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


def _clone_session(
    session: SessionProtocol,
    *,
    bus: EventBus,
) -> SessionProtocol | None:
    clone_method = getattr(session, "clone", None)
    if not callable(clone_method):
        return None
    return cast(SessionProtocol, clone_method(bus=bus))


def _prepare_child_contexts(
    *,
    delegations: Iterable[DelegationParams],
    session: SessionProtocol,
    bus: EventBus,
    isolation_level: SubagentIsolationLevel,
) -> tuple[tuple[SessionProtocol, EventBus], ...] | str:
    if isolation_level is SubagentIsolationLevel.NO_ISOLATION:
        return tuple((session, bus) for _ in delegations)

    child_pairs: list[tuple[SessionProtocol, EventBus]] = []
    for _ in delegations:
        child_bus = InProcessEventBus()
        try:
            cloned = _clone_session(session, bus=child_bus)
        except Exception as error:  # pragma: no cover - defensive
            return _build_error(str(error))
        if cloned is None:
            return "Parent session does not support cloning for full isolation."
        child_pairs.append((cloned, child_bus))
    return tuple(child_pairs)


def build_dispatch_subagents_tool(
    *, isolation_level: SubagentIsolationLevel = SubagentIsolationLevel.NO_ISOLATION
) -> Tool[DispatchSubagentsParams, tuple[SubagentResult, ...]]:
    """Return a configured dispatch tool bound to the desired isolation level."""

    def _dispatch_subagents(
        params: DispatchSubagentsParams,
        *,
        context: ToolContext,
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
        if not isinstance(parent_output_type, type) or not is_dataclass(
            parent_output_type
        ):
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

        contexts = _prepare_child_contexts(
            delegations=delegations,
            session=context.session,
            bus=context.event_bus,
            isolation_level=isolation_level,
        )
        if isinstance(contexts, str):
            return ToolResult(
                message=contexts,
                value=None,
                success=False,
            )

        adapter = context.adapter
        parse_output = rendered_parent.container is not None

        def _run_child(
            payload: tuple[DelegationParams, SessionProtocol, EventBus],
        ) -> SubagentResult:
            delegation, child_session, child_bus = payload
            recap = RecapParams(bullets=delegation.recap_lines)
            try:
                response = adapter.evaluate(
                    delegation_prompt.prompt,
                    delegation,
                    recap,
                    parse_output=parse_output,
                    bus=child_bus,
                    session=child_session,
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

        payloads = [
            (delegation, child_session, child_bus)
            for delegation, (child_session, child_bus) in zip(
                delegations, contexts, strict=True
            )
        ]

        max_workers = min(len(payloads), _DEFAULT_MAX_WORKERS) or 1
        results: list[SubagentResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_child, payload) for payload in payloads]
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

    return Tool[DispatchSubagentsParams, tuple[SubagentResult, ...]](
        name="dispatch_subagents",
        description="Run delegated child prompts in parallel.",
        handler=_dispatch_subagents,
        accepts_overrides=False,
    )


@dataclass(slots=True)
class _SubagentsSectionParams:
    """Placeholder params container for the subagents section."""

    pass


_DELEGATION_BODY: Final[str] = (
    "Use `dispatch_subagents` to offload work that can proceed in parallel.\n"
    "Each delegation must include recap bullet points so the parent can audit\n"
    "the child plan. Prefer dispatching concurrent tasks over running them\n"
    "sequentially yourself."
)


class SubagentsSection(Section[_SubagentsSectionParams]):
    """Explain the delegation workflow and expose the dispatch tool."""

    def __init__(
        self,
        *,
        isolation_level: SubagentIsolationLevel | None = None,
    ) -> None:
        effective_level = (
            isolation_level
            if isolation_level is not None
            else SubagentIsolationLevel.NO_ISOLATION
        )
        tool = build_dispatch_subagents_tool(isolation_level=effective_level)
        super().__init__(
            title="Delegation",
            key="subagents",
            default_params=_SubagentsSectionParams(),
            tools=(tool,),
            accepts_overrides=False,
        )

    @override
    def render(self, params: _SubagentsSectionParams | None, depth: int) -> str:
        if params is None:
            raise PromptRenderError(
                "Subagents section requires parameters.",
                dataclass_type=_SubagentsSectionParams,
            )
        heading = "#" * (depth + 2)
        return f"{heading} {self.title}\n\n{_DELEGATION_BODY}"


dispatch_subagents = build_dispatch_subagents_tool()


__all__ = [
    "DispatchSubagentsParams",
    "SubagentIsolationLevel",
    "SubagentResult",
    "SubagentsSection",
    "build_dispatch_subagents_tool",
    "dispatch_subagents",
]
