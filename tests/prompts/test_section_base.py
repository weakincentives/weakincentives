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

from weakincentives.prompts import Section


@dataclass
class ExampleParams:
    value: str = "hello"


class ExampleSection(Section[ExampleParams]):
    def render(
        self, params: ExampleParams, depth: int
    ) -> str:  # pragma: no cover - abstract behavior exercised elsewhere
        return f"Rendered {params.value} at depth {depth}"


def test_section_defaults_children_and_enabled():
    section = ExampleSection(title="Demo")

    assert section.children == ()
    assert section.is_enabled(ExampleParams()) is True
    assert section.params is ExampleParams


def test_section_key_defaults_to_slug_and_fallback():
    spaced = ExampleSection(title="Spacing Out")
    blank = ExampleSection(title="   ")

    assert spaced.key == "spacing-out"
    assert blank.key == "section"


def test_section_original_body_template_default_is_none():
    section = ExampleSection(title="Has No Template")

    assert section.original_body_template() is None


def test_section_allows_custom_children_and_enabled():
    child = ExampleSection(title="Child")

    def toggle(params: ExampleParams) -> bool:
        return params.value == "go"

    section = ExampleSection(
        title="Parent",
        children=[child],
        enabled=toggle,
    )

    assert section.children == (child,)
    assert section.is_enabled(ExampleParams(value="stop")) is False
    assert section.is_enabled(ExampleParams(value="go")) is True


class PlainSection(Section):
    def render(
        self, params: object, depth: int
    ) -> str:  # pragma: no cover - exercise instantiation guard
        return ""


def test_section_requires_specialized_type_parameter() -> None:
    with pytest.raises(TypeError):
        PlainSection(title="Plain")


def test_section_rejects_multiple_type_arguments() -> None:
    with pytest.raises(TypeError):
        Section.__class_getitem__((int, str))  # type: ignore[call-arg]
