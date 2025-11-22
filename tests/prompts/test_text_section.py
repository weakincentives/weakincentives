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
from types import SimpleNamespace

import pytest

from weakincentives.prompt import MarkdownSection, PromptRenderError


@dataclass
class GreetingParams:
    greeting: str


def test_text_section_renders_heading_and_body() -> None:
    section = MarkdownSection[GreetingParams](
        title="Greeting",
        template="""
            Greeting:
            ${greeting}
        """,
        key="greeting",
    )

    output = section.render(GreetingParams(greeting="hello"), depth=0, number="1")

    assert output == "## 1. Greeting\n\nGreeting:\nhello"


def test_text_section_performs_strict_substitution() -> None:
    @dataclass
    class PlaceholderParams:
        value: str

    section = MarkdownSection[PlaceholderParams](
        title="Placeholder Demo",
        template="Value: ${value}",
        key="placeholder-demo",
    )

    output = section.render(PlaceholderParams(value="42"), depth=1, number="1.1")

    assert output == "### 1.1. Placeholder Demo\n\nValue: 42"


def test_text_section_supports_slotted_dataclass_params() -> None:
    @dataclass(slots=True)
    class SlottedParams:
        value: str

    section = MarkdownSection[SlottedParams](
        title="Slots",
        template="Slot value: ${value}",
        key="slots",
    )

    output = section.render(SlottedParams(value="ok"), depth=0, number="1")

    assert output == "## 1. Slots\n\nSlot value: ok"


def test_text_section_rejects_non_dataclass_params() -> None:
    section = MarkdownSection[SimpleNamespace](
        title="Reject",
        template="Value: ${value}",
        key="reject",
    )

    with pytest.raises(PromptRenderError) as error_info:
        section.render(SimpleNamespace(value="nope"), depth=0, number="1")

    error = error_info.value
    assert isinstance(error, PromptRenderError)
    assert error.dataclass_type is SimpleNamespace
