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
from typing import cast

import pytest

from weakincentives.prompts import (
    OutputParseError,
    Prompt,
    PromptValidationError,
    TextSection,
    parse_output,
)


@dataclass
class Summary:
    title: str
    views: int
    featured: bool | None = None


@dataclass
class Guidance:
    topic: str


@dataclass
class ResultItem:
    title: str
    score: float


def _build_summary_prompt(
    *,
    inject_output_instructions: bool = True,
    allow_extra_keys: bool = False,
) -> Prompt[Summary]:
    task_section = TextSection[Guidance](
        title="Task",
        body="Summarize ${topic} and include view counts.",
    )
    return Prompt[Summary](
        key="summaries",
        name="summaries",
        sections=[task_section],
        inject_output_instructions=inject_output_instructions,
        allow_extra_keys=allow_extra_keys,
    )


def test_prompt_specialization_appends_response_format_block() -> None:
    prompt = _build_summary_prompt()

    rendered = prompt.render(Guidance(topic="Ada Lovelace"))

    assert rendered.output_type is Summary
    assert rendered.output_container == "object"
    assert rendered.allow_extra_keys is False
    assert rendered.text.endswith(
        "\n".join(
            [
                "## Response Format",
                "",
                "Return ONLY a single fenced JSON code block. Do not include any text",
                "before or after the block.",
                "",
                "The top-level JSON value MUST be an object that matches the fields",
                "of the expected schema. Do not add extra keys.",
            ]
        )
    )


def test_prompt_can_disable_response_format_injection() -> None:
    prompt = _build_summary_prompt(inject_output_instructions=False)

    rendered = prompt.render(Guidance(topic="Grace Hopper"))

    assert "## Response Format" not in rendered.text
    assert rendered.output_type is Summary


def test_prompt_render_can_skip_response_format_temporarily() -> None:
    prompt = _build_summary_prompt()

    rendered = prompt.render(
        Guidance(topic="Grace Hopper"),
        inject_output_instructions=False,
    )

    assert "## Response Format" not in rendered.text
    assert prompt.inject_output_instructions is True


def test_prompt_render_can_force_response_format_temporarily() -> None:
    prompt = _build_summary_prompt(inject_output_instructions=False)

    rendered = prompt.render(
        Guidance(topic="Grace Hopper"),
        inject_output_instructions=True,
    )

    assert "## Response Format" in rendered.text
    assert prompt.inject_output_instructions is False


def test_prompt_specialization_requires_dataclass() -> None:
    with pytest.raises(PromptValidationError) as exc:
        Prompt[str](key="invalid-output", sections=[])

    error = cast(PromptValidationError, exc.value)
    assert error.dataclass_type is str


def test_parse_output_handles_json_code_block() -> None:
    prompt = _build_summary_prompt()
    rendered = prompt.render(Guidance(topic="Ada"))

    reply = """All set.\n```json\n{\n  \"title\": \"Ada\",\n  \"views\": \"42\",\n  \"featured\": \"true\"\n}\n```"""

    parsed = parse_output(reply, rendered)

    assert isinstance(parsed, Summary)
    assert parsed.title == "Ada"
    assert parsed.views == 42
    assert parsed.featured is True


def test_parse_output_rejects_extra_keys_by_default() -> None:
    prompt = _build_summary_prompt()
    rendered = prompt.render(Guidance(topic="Ada"))

    reply = """```json\n{\n  \"title\": \"Ada\",\n  \"views\": 10,\n  \"notes\": \"extra\"\n}\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_output(reply, rendered)

    assert "Extra keys" in str(exc.value)


def test_parse_output_allows_extra_keys_when_configured() -> None:
    prompt = _build_summary_prompt(allow_extra_keys=True)
    rendered = prompt.render(Guidance(topic="Ada"))

    reply = """```json\n{\n  \"title\": \"Ada\",\n  \"views\": 10,\n  \"notes\": \"extra\"\n}\n```"""

    parsed = parse_output(reply, rendered)

    assert isinstance(parsed, Summary)
    assert parsed.title == "Ada"
    assert parsed.views == 10
    assert rendered.allow_extra_keys is True

    assert rendered.text.endswith(
        "\n".join(
            [
                "## Response Format",
                "",
                "Return ONLY a single fenced JSON code block. Do not include any text",
                "before or after the block.",
                "",
                "The top-level JSON value MUST be an object that matches the fields",
                "of the expected schema.",
            ]
        )
    )


def test_parse_output_validates_container_type() -> None:
    prompt = _build_summary_prompt()
    rendered = prompt.render(Guidance(topic="Ada"))

    reply = """```json\n[{\n  \"title\": \"Ada\",\n  \"views\": 10\n}]\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_output(reply, rendered)

    assert "Expected top-level JSON object" in str(exc.value)


def test_parse_output_supports_array_container() -> None:
    task_section = TextSection[Guidance](
        title="Task",
        body="Return search results.",
    )
    prompt = Prompt[list[ResultItem]](
        key="search-array-support",
        name="search",
        sections=[task_section],
    )
    rendered = prompt.render(Guidance(topic="Ada"))

    reply = """```json\n[{\n  \"title\": \"Ada\",\n  \"score\": \"0.9\"\n}]\n```"""

    parsed = parse_output(reply, rendered)

    assert isinstance(parsed, list)
    assert parsed[0].title == "Ada"
    assert parsed[0].score == pytest.approx(0.9)
    assert rendered.output_container == "array"
    assert "an array" in rendered.text


def test_parse_output_requires_wrapped_array_key() -> None:
    task_section = TextSection[Guidance](
        title="Task",
        body="Return search results.",
    )
    prompt = Prompt[list[ResultItem]](
        key="search-array-missing-key",
        name="search",
        sections=[task_section],
    )
    rendered = prompt.render(Guidance(topic="Ada"))

    reply = """```json\n{\n  \"results\": []\n}\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_output(reply, rendered)

    assert "Expected top-level JSON array" in str(exc.value)


def test_parse_output_requires_wrapped_array_list_value() -> None:
    task_section = TextSection[Guidance](
        title="Task",
        body="Return search results.",
    )
    prompt = Prompt[list[ResultItem]](
        key="search-array-non-list",
        name="search",
        sections=[task_section],
    )
    rendered = prompt.render(Guidance(topic="Ada"))

    reply = """```json\n{\n  \"items\": {\"bad\": true}\n}\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_output(reply, rendered)

    assert "Expected top-level JSON array" in str(exc.value)


def test_parse_output_falls_back_to_embedded_json() -> None:
    prompt = _build_summary_prompt()
    rendered = prompt.render(Guidance(topic="Ada"))

    reply = 'The payload is {"title": "Ada", "views": 7}'

    parsed = parse_output(reply, rendered)

    assert parsed.views == 7


def test_parse_output_requires_specialized_prompt() -> None:
    task_section = TextSection[Guidance](
        title="Task",
        body="Return guidance.",
    )
    prompt = Prompt(key="plain-guidance", name="plain", sections=[task_section])
    rendered = prompt.render(Guidance(topic="Ada"))

    with pytest.raises(OutputParseError):
        parse_output("{}", rendered)


def test_parse_output_array_requires_array_container() -> None:
    task_section = TextSection[Guidance](
        title="Task",
        body="Return search results.",
    )
    prompt = Prompt[list[ResultItem]](
        key="search-array",
        name="search",
        sections=[task_section],
    )
    rendered = prompt.render(Guidance(topic="Ada"))

    reply = """```json\n{\n  \"title\": \"Ada\"\n}\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_output(reply, rendered)

    assert "Expected top-level JSON array" in str(exc.value)


def test_parse_output_array_requires_object_items() -> None:
    task_section = TextSection[Guidance](title="Task", body="Return search results.")
    prompt = Prompt[list[ResultItem]](
        key="search-array-items",
        name="search",
        sections=[task_section],
    )
    rendered = prompt.render(Guidance(topic="Ada"))

    reply = """```json\n[\n  \"not an object\"\n]\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_output(reply, rendered)

    assert "Array item at index 0 is not an object." in str(exc.value)


def test_parse_output_array_reports_item_validation_error() -> None:
    task_section = TextSection[Guidance](title="Task", body="Return search results.")
    prompt = Prompt[list[ResultItem]](
        key="search-array-validation",
        name="search",
        sections=[task_section],
    )
    rendered = prompt.render(Guidance(topic="Ada"))

    reply = """```json\n[{\n  \"title\": \"Ada\"\n}]\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_output(reply, rendered)

    assert "Missing required field" in str(exc.value)


def test_parse_output_reports_invalid_fenced_block() -> None:
    prompt = _build_summary_prompt()
    rendered = prompt.render(Guidance(topic="Ada"))

    reply = """```json\n{ invalid json }\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_output(reply, rendered)

    assert "Failed to decode JSON from fenced code block." in str(exc.value)


def test_parse_output_requires_json_payload() -> None:
    prompt = _build_summary_prompt()
    rendered = prompt.render(Guidance(topic="Ada"))

    with pytest.raises(OutputParseError) as exc:
        parse_output("No structured data", rendered)

    assert "No JSON object or array" in str(exc.value)
