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
from typing import Literal, cast

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    OutputParseError,
    Prompt,
    PromptTemplate,
    PromptValidationError,
    parse_structured_output,
)
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.structured_output import StructuredOutputConfig


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
    allow_extra_keys: bool = False,
) -> PromptTemplate[Summary]:
    task_section = MarkdownSection[Guidance](
        title="Task",
        template="Summarize ${topic} and include view counts.",
        key="task",
    )
    return PromptTemplate[Summary](
        ns="tests/prompts",
        key="summaries",
        name="summaries",
        sections=[task_section],
        allow_extra_keys=allow_extra_keys,
    )


def test_prompt_specialization_exposes_output_metadata() -> None:
    prompt = Prompt(_build_summary_prompt()).bind(Guidance(topic="Ada Lovelace"))

    rendered = prompt.render()

    assert rendered.output_type is Summary
    assert rendered.container == "object"
    assert rendered.allow_extra_keys is False


def test_template_structured_output_property() -> None:
    template = _build_summary_prompt()
    prompt = Prompt(template).bind(Guidance(topic="Ada"))

    assert template.structured_output is not None
    assert template.structured_output.dataclass_type is Summary
    assert prompt.structured_output is not None
    assert prompt.structured_output.dataclass_type is Summary


def test_prompt_specialization_requires_dataclass() -> None:
    with pytest.raises(PromptValidationError) as exc:
        PromptTemplate[str](ns="tests/prompts", key="invalid-output", sections=[])

    error = cast(PromptValidationError, exc.value)
    assert error.dataclass_type is str


def test_prompt_resolve_output_spec_requires_dataclass_type() -> None:
    class InvalidOutputPrompt(PromptTemplate):
        _output_dataclass_candidate = "oops"
        _output_container_spec = "object"

    with pytest.raises(PromptValidationError) as exc:
        InvalidOutputPrompt(ns="tests/prompts", key="invalid-output", sections=())

    error = cast(PromptValidationError, exc.value)
    assert error.dataclass_type is str


def test_parse_structured_output_handles_json_code_block() -> None:
    prompt = Prompt(_build_summary_prompt()).bind(Guidance(topic="Ada"))
    rendered = prompt.render()

    reply = """All set.\n```json\n{\n  "title": "Ada",\n  "views": "42",\n  "featured": "true"\n}\n```"""

    parsed = parse_structured_output(reply, rendered)

    assert isinstance(parsed, Summary)
    assert parsed.title == "Ada"
    assert parsed.views == 42
    assert parsed.featured is True


def test_parse_structured_output_rejects_extra_keys_by_default() -> None:
    prompt = Prompt(_build_summary_prompt()).bind(Guidance(topic="Ada"))
    rendered = prompt.render()

    reply = (
        """```json\n{\n  "title": "Ada",\n  "views": 10,\n  "notes": "extra"\n}\n```"""
    )

    with pytest.raises(OutputParseError) as exc:
        parse_structured_output(reply, rendered)

    assert "Extra keys" in str(exc.value)


def test_parse_structured_output_allows_extra_keys_when_configured() -> None:
    prompt = Prompt(_build_summary_prompt(allow_extra_keys=True)).bind(
        Guidance(topic="Ada")
    )
    rendered = prompt.render()

    reply = (
        """```json\n{\n  "title": "Ada",\n  "views": 10,\n  "notes": "extra"\n}\n```"""
    )

    parsed = parse_structured_output(reply, rendered)

    assert isinstance(parsed, Summary)
    assert parsed.title == "Ada"
    assert parsed.views == 10
    assert rendered.allow_extra_keys is True


def test_parse_structured_output_validates_container_type() -> None:
    prompt = Prompt(_build_summary_prompt()).bind(Guidance(topic="Ada"))
    rendered = prompt.render()

    reply = """```json\n[{\n  "title": "Ada",\n  "views": 10\n}]\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_structured_output(reply, rendered)

    assert "Expected top-level JSON object" in str(exc.value)


def test_parse_structured_output_supports_array_container() -> None:
    task_section = MarkdownSection[Guidance](
        title="Task",
        template="Return search results.",
        key="task",
    )
    template = PromptTemplate[list[ResultItem]](
        ns="tests/prompts",
        key="search-array-support",
        name="search",
        sections=[task_section],
    )
    rendered = Prompt(template).bind(Guidance(topic="Ada")).render()

    reply = """```json\n[{\n  "title": "Ada",\n  "score": "0.9"\n}]\n```"""

    parsed = parse_structured_output(reply, rendered)

    assert isinstance(parsed, list)
    assert parsed[0].title == "Ada"
    assert parsed[0].score == pytest.approx(0.9)
    assert rendered.container == "array"


def test_parse_structured_output_requires_wrapped_array_key() -> None:
    task_section = MarkdownSection[Guidance](
        title="Task",
        template="Return search results.",
        key="task",
    )
    template = PromptTemplate[list[ResultItem]](
        ns="tests/prompts",
        key="search-array-missing-key",
        name="search",
        sections=[task_section],
    )
    rendered = Prompt(template).bind(Guidance(topic="Ada")).render()

    reply = """```json\n{\n  "results": []\n}\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_structured_output(reply, rendered)

    assert "Expected top-level JSON array" in str(exc.value)


def test_parse_structured_output_requires_wrapped_array_list_value() -> None:
    task_section = MarkdownSection[Guidance](
        title="Task",
        template="Return search results.",
        key="task",
    )
    prompt = PromptTemplate[list[ResultItem]](
        ns="tests/prompts",
        key="search-array-non-list",
        name="search",
        sections=[task_section],
    )
    rendered = Prompt(prompt).bind(Guidance(topic="Ada")).render()

    reply = """```json\n{\n  "items": {"bad": true}\n}\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_structured_output(reply, rendered)

    assert "Expected top-level JSON array" in str(exc.value)


def test_parse_structured_output_falls_back_to_embedded_json() -> None:
    prompt = Prompt(_build_summary_prompt()).bind(Guidance(topic="Ada"))
    rendered = prompt.render()

    reply = 'The payload is {"title": "Ada", "views": 7}'

    parsed = parse_structured_output(reply, rendered)

    assert parsed.views == 7


def test_parse_structured_output_skips_invalid_prefix_chars() -> None:
    prompt = Prompt(_build_summary_prompt()).bind(Guidance(topic="Ada"))
    rendered = prompt.render()

    reply = '[oops]{"title": "Ada", "views": 11}'

    parsed = parse_structured_output(reply, rendered)

    assert parsed.title == "Ada"
    assert parsed.views == 11


def test_parse_structured_output_requires_specialized_prompt() -> None:
    task_section = MarkdownSection[Guidance](
        title="Task",
        template="Return guidance.",
        key="task",
    )
    template = PromptTemplate(
        ns="tests/prompts",
        key="plain-guidance",
        name="plain",
        sections=[task_section],
    )
    rendered = Prompt(template).bind(Guidance(topic="Ada")).render()

    with pytest.raises(OutputParseError):
        parse_structured_output("{}", rendered)


def test_parse_structured_output_array_requires_array_container() -> None:
    task_section = MarkdownSection[Guidance](
        title="Task",
        template="Return search results.",
        key="task",
    )
    template = PromptTemplate[list[ResultItem]](
        ns="tests/prompts",
        key="search-array",
        name="search",
        sections=[task_section],
    )
    rendered = Prompt(template).bind(Guidance(topic="Ada")).render()

    reply = """```json\n{\n  "title": "Ada"\n}\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_structured_output(reply, rendered)

    assert "Expected top-level JSON array" in str(exc.value)


def test_parse_structured_output_array_requires_object_items() -> None:
    task_section = MarkdownSection[Guidance](
        title="Task", template="Return search results.", key="task"
    )
    template = PromptTemplate[list[ResultItem]](
        ns="tests/prompts",
        key="search-array-items",
        name="search",
        sections=[task_section],
    )
    rendered = Prompt(template).bind(Guidance(topic="Ada")).render()

    reply = """```json\n[\n  "not an object"\n]\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_structured_output(reply, rendered)

    assert "Array item at index 0 is not an object." in str(exc.value)


def test_parse_structured_output_array_reports_item_validation_error() -> None:
    task_section = MarkdownSection[Guidance](
        title="Task", template="Return search results.", key="task"
    )
    template = PromptTemplate[list[ResultItem]](
        ns="tests/prompts",
        key="search-array-validation",
        name="search",
        sections=[task_section],
    )
    rendered = Prompt(template).bind(Guidance(topic="Ada")).render()

    reply = """```json\n[{\n  "title": "Ada"\n}]\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_structured_output(reply, rendered)

    assert "Missing required field" in str(exc.value)


def test_parse_structured_output_reports_invalid_fenced_block() -> None:
    prompt = Prompt(_build_summary_prompt()).bind(Guidance(topic="Ada"))
    rendered = prompt.render()

    reply = """```json\n{ invalid json }\n```"""

    with pytest.raises(OutputParseError) as exc:
        parse_structured_output(reply, rendered)

    assert "Failed to decode JSON from fenced code block." in str(exc.value)


def test_parse_structured_output_requires_json_payload() -> None:
    prompt = Prompt(_build_summary_prompt()).bind(Guidance(topic="Ada"))
    rendered = prompt.render()

    with pytest.raises(OutputParseError) as exc:
        parse_structured_output("No structured data", rendered)

    assert "No JSON object or array" in str(exc.value)


def test_parse_structured_output_rejects_unknown_container() -> None:
    rendered = RenderedPrompt[Summary](
        text="",
        structured_output=StructuredOutputConfig(
            dataclass_type=Summary,
            container=cast(Literal["object", "array"], "invalid"),
            allow_extra_keys=False,
        ),
    )

    with pytest.raises(OutputParseError) as exc:
        parse_structured_output("{}", rendered)

    assert "Unknown output container" in str(exc.value)


def test_parse_structured_output_empty_whitespace_only() -> None:
    """Test branch 113->119: when text.strip() is empty, skip json.loads and use decoder."""
    prompt = Prompt(_build_summary_prompt()).bind(Guidance(topic="Ada"))
    rendered = prompt.render()

    # Text with only whitespace - stripped will be empty, triggering branch 113->119
    reply = "   \n   \t   "

    with pytest.raises(OutputParseError) as exc:
        parse_structured_output(reply, rendered)

    assert "No JSON object or array" in str(exc.value)
