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

"""Prompt section that enforces parallel subagent dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

from .section import Section


@dataclass(slots=True)
class SubagentsParams:
    """Placeholder params to keep the section API consistent."""


class SubagentsSection(Section[SubagentsParams]):
    """Embed instructions and tooling for parallel delegation."""

    def __init__(self) -> None:
        from ..tools.subagents import dispatch_subagents

        super().__init__(
            title="Parallel Subagents",
            key="parallel-subagents",
            default_params=SubagentsParams(),
            tools=(dispatch_subagents,),
            accepts_overrides=False,
        )

    @override
    def render(self, params: SubagentsParams, depth: int) -> str:
        del params
        heading = "#" * (depth + 2)
        prefix = f"{heading} Parallel Subagents"
        return (
            f"{prefix}\n\n"
            "Use the `dispatch_subagents` tool to run parallel work. "
            "Whenever your execution plan exposes tasks that can happen at the same time, "
            "you **MUST** call `dispatch_subagents` instead of attempting to run the steps sequentially.\n\n"
            "When dispatching, provide a short summary for each child run and enumerate recap bullet points "
            "that describe the checkpoints the child must hit. Every delegated child must produce a recap block; "
            "omit nothing.\n\n"
            "Build the dispatch payload at call time—this section never includes default tasks. "
            "Always prefer the tool over serial execution once parallelism is available."
        )


__all__ = ["SubagentsParams", "SubagentsSection"]
