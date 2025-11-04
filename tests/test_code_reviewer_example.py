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

"""Tests for the interactive code reviewer example."""

from __future__ import annotations

import code_reviewer_example
from weakincentives.prompt.subagents import SubagentsSection
from weakincentives.session import Session
from weakincentives.tools.subagents import dispatch_subagents


def test_build_sunfish_prompt_includes_subagents_section() -> None:
    prompt = code_reviewer_example.build_sunfish_prompt(Session())

    node = next(
        (
            entry.section
            for entry in prompt.sections
            if isinstance(entry.section, SubagentsSection)
        ),
        None,
    )

    assert isinstance(node, SubagentsSection)
    assert node.tools() == (dispatch_subagents,)
