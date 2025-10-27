from __future__ import annotations

from dataclasses import dataclass

from weakincentives.prompts import TextSection


@dataclass
class GreetingParams:
    greeting: str


def test_text_section_renders_heading_and_body():
    section = TextSection[GreetingParams](
        title="Greeting",
        body="""
            Greeting:
            ${greeting}
        """,
    )

    output = section.render(GreetingParams(greeting="hello"), depth=0)

    assert output == "## Greeting\n\nGreeting:\nhello"


def test_text_section_performs_safe_substitution():
    @dataclass
    class PlaceholderParams:
        value: str

    section = TextSection[PlaceholderParams](
        title="Placeholder Demo",
        body="Value: ${value}",
    )

    output = section.render(PlaceholderParams(value="42"), depth=1)

    assert output == "### Placeholder Demo\n\nValue: 42"
