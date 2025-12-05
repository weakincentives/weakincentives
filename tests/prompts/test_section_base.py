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

from weakincentives.prompt import Section, SectionVisibility
from weakincentives.prompt._generic_params_specializer import GenericParamsSpecializer
from weakincentives.prompt._normalization import (
    COMPONENT_KEY_PATTERN,
    normalize_component_key,
)


@dataclass
class ExampleParams:
    value: str = "hello"


@dataclass
class GenericParams:
    value: str = "generic"


class ExampleSection(Section[ExampleParams]):
    def render(
        self,
        params: ExampleParams,
        depth: int,
        number: str,
        *,
        visibility: SectionVisibility | None = None,
    ) -> str:
        del number, visibility
        _ = self.title
        return f"Rendered {params.value} at depth {depth}"

    def clone(self, **kwargs: object) -> ExampleSection:
        cloned_children = tuple(child.clone(**kwargs) for child in self.children)
        return ExampleSection(title=self.title, key=self.key, children=cloned_children)


def test_example_section_render_outputs_value() -> None:
    section = ExampleSection(title="Demo", key="demo")
    rendered = section.render(ExampleParams(value="hi"), depth=2, number="1")

    assert rendered == "Rendered hi at depth 2"


def test_section_defaults_children_and_enabled() -> None:
    section = ExampleSection(title="Demo", key="demo")

    assert section.children == ()
    assert section.is_enabled(ExampleParams()) is True
    assert section.param_type is ExampleParams


def test_section_key_normalization_and_validation() -> None:
    section = ExampleSection(title="Spacing Out", key=" spacing.out ")

    assert section.key == "spacing.out"

    with pytest.raises(ValueError) as excinfo:
        ExampleSection(title="Invalid", key="")

    assert str(excinfo.value) == "Section key must be a non-empty string."

    with pytest.raises(ValueError) as excinfo:
        normalize_component_key("", owner="Section")

    assert str(excinfo.value) == "Section key must be a non-empty string."

    with pytest.raises(ValueError) as excinfo:
        ExampleSection(title="Bad", key="Invalid Key")

    expected_message = f"Section key must match {COMPONENT_KEY_PATTERN.pattern}."
    assert str(excinfo.value) == expected_message

    with pytest.raises(ValueError) as excinfo:
        normalize_component_key("Invalid Key", owner="Section")

    assert str(excinfo.value) == expected_message


def test_section_original_body_template_default_is_none() -> None:
    section = ExampleSection(title="Has No Template", key="no-template")

    assert section.original_body_template() is None


def test_section_default_summary_is_none() -> None:
    section = ExampleSection(title="No Summary", key="no-summary")

    assert section.summary is None


def test_section_default_visibility_is_full() -> None:
    section = ExampleSection(title="Default Visibility", key="default-visibility")

    assert section.visibility == SectionVisibility.FULL


def test_section_allows_custom_summary_and_visibility() -> None:
    section = PlainSection(
        title="Custom",
        key="custom",
        summary="Brief description",
        visibility=SectionVisibility.SUMMARY,
    )

    assert section.summary == "Brief description"
    assert section.visibility == SectionVisibility.SUMMARY


def test_section_effective_visibility_returns_default_when_no_override() -> None:
    section = PlainSection(
        title="Default",
        key="default",
        summary="Summary text",
        visibility=SectionVisibility.FULL,
    )

    assert section.effective_visibility() == SectionVisibility.FULL


def test_section_effective_visibility_returns_override_when_provided() -> None:
    section = PlainSection(
        title="Override",
        key="override",
        summary="Summary text",
        visibility=SectionVisibility.FULL,
    )

    assert (
        section.effective_visibility(SectionVisibility.SUMMARY)
        == SectionVisibility.SUMMARY
    )


def test_section_effective_visibility_fallback_to_full_without_summary() -> None:
    section = PlainSection(title="No Summary", key="no-summary")

    # Default is FULL when no summary
    assert section.effective_visibility() == SectionVisibility.FULL
    # Falls back to FULL when requesting SUMMARY but no summary is set
    assert (
        section.effective_visibility(SectionVisibility.SUMMARY)
        == SectionVisibility.FULL
    )


def test_section_original_summary_template_returns_summary() -> None:
    section = PlainSection(
        title="With Summary",
        key="with-summary",
        summary="This is the summary template",
    )

    assert section.original_summary_template() == "This is the summary template"


def test_section_original_summary_template_returns_none_when_not_set() -> None:
    section = PlainSection(title="No Summary", key="no-summary")

    assert section.original_summary_template() is None


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
    def render(
        self,
        params: object,
        depth: int,
        number: str,
        *,
        visibility: SectionVisibility | None = None,
    ) -> str:
        del params, depth, number, visibility
        _ = getattr(self, "key", None)
        return ""

    def clone(self, **kwargs: object) -> PlainSection:
        return PlainSection(title=self.title, key=self.key, enabled=self._enabled)


def test_plain_section_render_stubs_result() -> None:
    instance = PlainSection.__new__(PlainSection)
    assert PlainSection.render(instance, object(), 0, "1") == ""


def test_plain_section_allows_absence_of_params() -> None:
    section = PlainSection(title="Plain", key="plain")

    assert section.param_type is None
    assert section.is_enabled(None) is True
    assert section.render(None, 1, "1") == ""


def test_plain_section_accepts_parameterless_enabled_callable() -> None:
    section = PlainSection(title="Plain", key="plain", enabled=lambda: False)

    assert section.is_enabled(None) is False


def test_plain_section_parameterless_enabled_handles_argument() -> None:
    recorded: list[object | None] = []

    def enabled(value: object | None) -> bool:
        recorded.append(value)
        return value is None

    section = PlainSection(title="Plain", key="plain", enabled=enabled)

    assert section.is_enabled(None) is True
    assert recorded == [None]


def test_plain_section_parameterless_enabled_handles_non_inspectable_callable() -> None:
    section = PlainSection(title="Plain", key="plain", enabled=bool)

    assert section.is_enabled(None) is False


def test_section_without_params_rejects_defaults() -> None:
    with pytest.raises(TypeError):
        PlainSection(title="Plain", key="plain", default_params=object())


def test_section_rejects_multiple_type_arguments() -> None:
    with pytest.raises(TypeError) as excinfo:
        Section.__class_getitem__((int, str))

    assert str(excinfo.value) == "Section[...] expects a single type argument."


def test_section_specialization_preserves_metadata() -> None:
    specialized = Section[ExampleParams]

    assert specialized.__module__ == Section.__module__
    assert specialized.__qualname__ == Section.__qualname__
    assert specialized.__name__ == Section.__name__


def test_generic_params_specializer_defaults_to_class_name() -> None:
    with pytest.raises(TypeError) as excinfo:
        bad_args = (GenericParams, GenericParams)
        GenericParamsSpecializer.__class_getitem__(bad_args)

    assert (
        str(excinfo.value)
        == "GenericParamsSpecializer[...] expects a single type argument."
    )


def test_generic_params_specializer_fallbacks_to_component_name() -> None:
    class NamelessSpecializer(GenericParamsSpecializer[GenericParams]):
        pass

    NamelessSpecializer.__name__ = ""

    with pytest.raises(TypeError) as excinfo:
        bad_args = (GenericParams, GenericParams)
        NamelessSpecializer.__class_getitem__(bad_args)

    assert str(excinfo.value) == "Component[...] expects a single type argument."
