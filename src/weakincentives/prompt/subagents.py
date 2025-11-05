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

"""Prompt section exposing the subagent delegation tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, override

from .section import Section


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

    def __init__(self) -> None:
        from ..tools.subagents import dispatch_subagents

        super().__init__(
            title="Delegation",
            key="subagents",
            default_params=_SubagentsSectionParams(),
            tools=(dispatch_subagents,),
            accepts_overrides=False,
        )

    @override
    def render(self, params: _SubagentsSectionParams, depth: int) -> str:
        heading = "#" * (depth + 2)
        return f"{heading} {self.title}\n\n{_DELEGATION_BODY}"


__all__ = ["SubagentsSection"]
