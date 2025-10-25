from __future__ import annotations

from dataclasses import dataclass

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
    section = ExampleSection(title="Demo", params=ExampleParams)

    assert section.children == ()
    assert section.is_enabled(ExampleParams()) is True
    assert section.params is ExampleParams


def test_section_allows_custom_children_and_enabled():
    child = ExampleSection(title="Child", params=ExampleParams)

    def toggle(params: ExampleParams) -> bool:
        return params.value == "go"

    section = ExampleSection(
        title="Parent",
        params=ExampleParams,
        children=[child],
        enabled=toggle,
    )

    assert section.children == (child,)
    assert section.is_enabled(ExampleParams(value="stop")) is False
    assert section.is_enabled(ExampleParams(value="go")) is True
