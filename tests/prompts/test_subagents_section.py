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

"""Coverage tests for the SubagentsSection helper."""

from weakincentives.prompt.subagents import SubagentsSection
from weakincentives.tools.subagents import dispatch_subagents


def test_subagents_section_render_mentions_tool() -> None:
    section = SubagentsSection()
    assert section.tools() == (dispatch_subagents,)

    params = section.param_type()
    rendered = section.render(params, depth=0)

    assert "dispatch_subagents" in rendered
    assert "parallel" in rendered.lower()
