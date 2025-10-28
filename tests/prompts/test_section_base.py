from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.prompts import Section, SupportsDataclass
from typing import cast


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


def test_section_allows_custom_children_and_enabled():
    child = ExampleSection(title="Child")

    def toggle(params: ExampleParams) -> bool:
        return params.value == "go"

    section = ExampleSection(
        title="Parent",
        children=cast(list[Section[SupportsDataclass]], [child]),
        enabled=toggle,
    )

    assert section.children == (child,)
    assert section.is_enabled(ExampleParams(value="stop")) is False
    assert section.is_enabled(ExampleParams(value="go")) is True


class PlainSection(Section):  # type: ignore[type-var]
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
