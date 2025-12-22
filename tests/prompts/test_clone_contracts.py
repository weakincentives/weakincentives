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

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.contrib.tools.asteval import AstevalSection
from weakincentives.contrib.tools.planning import PlanningStrategy, PlanningToolsSection
from weakincentives.contrib.tools.vfs import VfsToolsSection
from weakincentives.prompt import MarkdownSection
from weakincentives.runtime.session import Session


@dataclass
class _SampleParams:
    value: str = "sample"


def test_markdown_section_clone_deep_copies_children_and_defaults() -> None:
    child = MarkdownSection[_SampleParams](
        title="Child", template="${value}", key="child", default_params=_SampleParams()
    )
    section = MarkdownSection[_SampleParams](
        title="Parent",
        template="${value}",
        key="parent",
        default_params=_SampleParams(value="original"),
        children=[child],
    )

    clone = section.clone()

    assert clone is not section
    assert clone.default_params is not section.default_params
    assert clone.children and clone.children[0] is not child


def test_tool_sections_clone_to_new_sessions() -> None:
    original_session = Session()
    vfs = VfsToolsSection(session=original_session)
    planning = PlanningToolsSection(
        session=original_session, strategy=PlanningStrategy.REACT
    )
    asteval = AstevalSection(session=original_session)

    new_session = Session()

    vfs_clone = vfs.clone(session=new_session)
    planning_clone = planning.clone(session=new_session)
    asteval_clone = asteval.clone(session=new_session)

    assert vfs_clone.session is new_session
    assert planning_clone.session is new_session
    assert asteval_clone.session is new_session


def test_tool_clones_validate_session_and_bus() -> None:
    base_session = Session()
    vfs = VfsToolsSection(session=base_session)
    planning = PlanningToolsSection(
        session=base_session, strategy=PlanningStrategy.REACT
    )
    asteval = AstevalSection(session=base_session)

    with pytest.raises(TypeError):
        _ = vfs.clone()
    with pytest.raises(TypeError):
        _ = planning.clone()
    with pytest.raises(TypeError):
        _ = asteval.clone()

    other_session = Session()
    with pytest.raises(TypeError):
        _ = vfs.clone(session=base_session, bus=other_session.dispatcher)
