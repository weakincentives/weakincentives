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

"""Prompt section exposing the subagent dispatch tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, override

from .section import Section

if TYPE_CHECKING:
    from ..tools.subagents import SubagentIsolationLevel


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
        from ..tools.subagents import (
            SubagentIsolationLevel,
            build_dispatch_subagents_tool,
        )

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
    def render(self, params: _SubagentsSectionParams, depth: int) -> str:
        del params
        heading = "#" * (depth + 2)
        return f"{heading} {self.title}\n\n{_DELEGATION_BODY}"


__all__ = ["SubagentsSection"]
