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

from weakincentives.prompt import Section


@dataclass
class ExampleParams:
    value: str = "hello"


class ExampleSection(Section[ExampleParams]):
    def render(self, params: ExampleParams, depth: int) -> str:
        return f"Rendered {params.value} at depth {depth}"


def test_example_section_render_outputs_value() -> None:
    section = ExampleSection(title="Demo", key="demo")
    rendered = section.render(ExampleParams(value="hi"), depth=2)

    assert rendered == "Rendered hi at depth 2"


def test_section_defaults_children_and_enabled() -> None:
    section = ExampleSection(title="Demo", key="demo")

    assert section.children == ()
    assert section.is_enabled(ExampleParams()) is True
    assert section.param_type is ExampleParams


def test_section_key_normalization_and_validation() -> None:
    section = ExampleSection(title="Spacing Out", key=" spacing.out ")

    assert section.key == "spacing.out"

    with pytest.raises(ValueError):
        ExampleSection(title="Invalid", key="")

    with pytest.raises(ValueError):
        ExampleSection(title="Bad", key="Invalid Key")


def test_section_original_body_template_default_is_none() -> None:
    section = ExampleSection(title="Has No Template", key="no-template")

    assert section.original_body_template() is None


def test_section_allows_custom_children_and_enabled() -> None:
    child = ExampleSection(title="Child", key="child")

    def toggle(params: ExampleParams) -> bool:
        return params.value == "go"

    section = ExampleSection(
        title="Parent",
        key="parent",
        children=[child],
        enabled=toggle,
    )

    assert section.children == (child,)
    assert section.is_enabled(ExampleParams(value="stop")) is False
    assert section.is_enabled(ExampleParams(value="go")) is True


class PlainSection(Section):
    def render(self, params: object, depth: int) -> str:
        return ""


def test_plain_section_render_stubs_result() -> None:
    instance = PlainSection.__new__(PlainSection)
    assert PlainSection.render(instance, object(), 0) == ""


def test_section_requires_specialized_type_parameter() -> None:
    with pytest.raises(TypeError):
        PlainSection(title="Plain", key="plain")


def test_section_rejects_multiple_type_arguments() -> None:
    with pytest.raises(TypeError):
        Section.__class_getitem__((int, str))
