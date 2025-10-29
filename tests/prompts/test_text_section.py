from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from weakincentives.prompts import PromptRenderError, TextSection


@dataclass
class GreetingParams:
    greeting: str


def test_text_section_renders_heading_and_body() -> None:
    section = TextSection[GreetingParams](
        title="Greeting",
        body="""
            Greeting:
            ${greeting}
        """,
    )

    output = section.render(GreetingParams(greeting="hello"), depth=0)

    assert output == "## Greeting\n\nGreeting:\nhello"


def test_text_section_performs_strict_substitution() -> None:
    @dataclass
    class PlaceholderParams:
        value: str

    section = TextSection[PlaceholderParams](
        title="Placeholder Demo",
        body="Value: ${value}",
    )

    output = section.render(PlaceholderParams(value="42"), depth=1)

    assert output == "### Placeholder Demo\n\nValue: 42"


def test_text_section_supports_slotted_dataclass_params() -> None:
    @dataclass(slots=True)
    class SlottedParams:
        value: str

    section = TextSection[SlottedParams](
        title="Slots",
        body="Slot value: ${value}",
    )

    output = section.render(SlottedParams(value="ok"), depth=0)

    assert output == "## Slots\n\nSlot value: ok"


def test_text_section_rejects_non_dataclass_params() -> None:
    section = TextSection[SimpleNamespace](
        title="Reject",
        body="Value: ${value}",
    )

    with pytest.raises(PromptRenderError) as error_info:
        section.render(SimpleNamespace(value="nope"), depth=0)

    error = error_info.value
    assert isinstance(error, PromptRenderError)
    assert error.dataclass_type is SimpleNamespace
